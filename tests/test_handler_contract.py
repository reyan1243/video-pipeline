"""Dry run of the RunPod handler's contract, without a GPU.

rp_handler is the one component that has never executed anywhere — it only runs
on a RunPod worker. These tests stub torch/transformers/runpod/boto3 so the
module imports on any machine, then exercise everything that does not need a
GPU: input validation, the guards that must reject a job *before* GPU time is
spent, key derivation, and the ok/false contract the backend branches on.

What is deliberately NOT covered: anything downstream of `select_seed`. That
needs real weights.
"""

import hashlib
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


def _install_stubs() -> None:
    """Minimal stand-ins for the heavy imports, installed before rp_handler."""
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: False, empty_cache=lambda: None,
        reset_peak_memory_stats=lambda: None, memory_allocated=lambda: 0,
        max_memory_allocated=lambda: 0,
    )
    torch.uint8 = "uint8"
    torch.bfloat16 = "bfloat16"
    torch.no_grad = lambda: mock.MagicMock()
    torch.inference_mode = lambda: mock.MagicMock()
    torch.autocast = lambda *a, **k: mock.MagicMock()
    torch.backends = types.SimpleNamespace(
        cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=False)),
        cudnn=types.SimpleNamespace(allow_tf32=False),
        mps=types.SimpleNamespace(is_available=lambda: False),
    )
    sys.modules["torch"] = torch

    transformers = types.ModuleType("transformers")
    for name in ("AutoProcessor", "AutoModelForZeroShotObjectDetection", "Sam2Model",
                 "Sam2Processor", "Sam2VideoModel", "Sam2VideoProcessor"):
        setattr(transformers, name, mock.MagicMock())
    sys.modules["transformers"] = transformers

    runpod = types.ModuleType("runpod")
    runpod.serverless = types.SimpleNamespace(
        start=lambda *a, **k: None, progress_update=lambda *a, **k: None
    )
    sys.modules["runpod"] = runpod

    boto3 = types.ModuleType("boto3")
    boto3.client = mock.MagicMock()
    sys.modules["boto3"] = boto3
    transfer = types.ModuleType("boto3.s3.transfer")
    transfer.TransferConfig = mock.MagicMock()
    sys.modules["boto3.s3"] = types.ModuleType("boto3.s3")
    sys.modules["boto3.s3.transfer"] = transfer

    botocore = types.ModuleType("botocore")
    client_mod = types.ModuleType("botocore.client")
    client_mod.Config = mock.MagicMock()
    exc_mod = types.ModuleType("botocore.exceptions")

    class ClientError(Exception):
        pass

    exc_mod.ClientError = ClientError
    sys.modules["botocore"] = botocore
    sys.modules["botocore.client"] = client_mod
    sys.modules["botocore.exceptions"] = exc_mod


_install_stubs()
# The handler refuses to start without a usable GPU — on a worker, a CPU
# fallback silently burns the execution timeout. These tests are the one
# legitimate CPU-only caller, so they opt in explicitly.
os.environ["ALLOW_CPU"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import arch_guard  # noqa: E402
import rp_handler  # noqa: E402


class TestInputValidation(unittest.TestCase):
    def test_missing_video_url_is_a_soft_failure(self):
        # Must be ok:false, never a raise — a truthy "error" key marks the whole
        # RunPod job FAILED, which the backend cannot distinguish from a crash.
        out = rp_handler.handler({"id": "j1", "input": {}})
        self.assertIs(out["ok"], False)
        self.assertEqual(out["code"], "missing_input")

    def test_empty_input_object_does_not_raise(self):
        self.assertIs(rp_handler.handler({"id": "j2"})["ok"], False)


class TestOutputKey(unittest.TestCase):
    def test_same_inputs_give_the_same_key(self):
        p = {"prompt": "person", "start": None, "end": 10.0}
        self.assertEqual(rp_handler._output_key("etag:123", p), rp_handler._output_key("etag:123", p))

    def test_key_changes_with_params(self):
        a = rp_handler._output_key("etag:123", {"prompt": "person"})
        b = rp_handler._output_key("etag:123", {"prompt": "dog"})
        self.assertNotEqual(a, b)

    def test_key_changes_with_source(self):
        p = {"prompt": "person"}
        self.assertNotEqual(rp_handler._output_key("etag:1", p), rp_handler._output_key("etag:2", p))

    def test_param_order_does_not_matter(self):
        # Keys are canonicalised with sort_keys, so a dict built in a different
        # order must not produce a cache miss.
        a = rp_handler._output_key("e", {"prompt": "person", "close": 7})
        b = rp_handler._output_key("e", {"close": 7, "prompt": "person"})
        self.assertEqual(a, b)

    def test_key_lands_under_the_configured_prefix(self):
        self.assertTrue(rp_handler._output_key("e", {}).startswith(rp_handler.S3_PREFIX + "/"))
        self.assertTrue(rp_handler._output_key("e", {}).endswith("person_matte.mp4"))


class TestRejectionContract(unittest.TestCase):
    def test_rejected_never_sets_error(self):
        # The single most important shape in the whole contract.
        out = rp_handler._rejected("too_long", "clip is 400s", duration=400.0)
        self.assertNotIn("error", out)
        self.assertIs(out["ok"], False)
        self.assertEqual(out["code"], "too_long")
        self.assertEqual(out["duration"], 400.0)


class TestGuards(unittest.TestCase):
    """The guards exist so an impossible job is refused before GPU time is spent."""

    def setUp(self):
        self.workdir = Path("/tmp")

    def _run(self, probe, metadata=None):
        with mock.patch.object(rp_handler, "_remote_identity", return_value=None), \
             mock.patch.object(rp_handler, "_download", return_value="deadbeef"), \
             mock.patch.object(rp_handler, "_already_done", return_value=False), \
             mock.patch.object(rp_handler, "_trim", side_effect=lambda s, d, a, b: s), \
             mock.patch.object(rp_handler, "_probe", return_value=probe), \
             mock.patch.object(rp_handler, "VideoSource") as vs:
            vs.return_value.load_metadata.return_value = metadata
            return rp_handler.handler({"id": "j", "input": {"video_url": "https://x/v.mp4"}})

    def test_over_length_clip_is_refused(self):
        out = self._run({"width": 1080, "height": 1920, "duration": 9999.0})
        self.assertIs(out["ok"], False)
        self.assertEqual(out["code"], "too_long")
        self.assertIn("start_time", out["reason"])

    def test_oversized_job_is_refused_before_tracking(self):
        meta = types.SimpleNamespace(frame_count=10**6, width=3840, height=2160, fps=30.0)
        out = self._run({"width": 3840, "height": 2160, "duration": 10.0}, meta)
        self.assertIs(out["ok"], False)
        self.assertEqual(out["code"], "too_large")
        self.assertIn("resolution", out)

    def test_a_clip_inside_the_limits_gets_past_the_guards(self):
        # Proves the guards are not simply rejecting everything: this one fails
        # later, at the mocked-out seed stage, not at a guard.
        meta = types.SimpleNamespace(frame_count=300, width=1080, height=1920, fps=30.0)
        out = self._run({"width": 1080, "height": 1920, "duration": 10.0}, meta)
        self.assertNotIn(out.get("code"), ("too_long", "too_large"))


class TestEdgeCases(unittest.TestCase):
    def test_negative_start_time_is_refused(self):
        out = rp_handler.handler({"id": "j", "input": {"video_url": "https://x/v.mp4", "start_time": -5}})
        self.assertEqual(out["code"], "invalid_range")

    def test_end_before_start_is_refused(self):
        out = rp_handler.handler({"id": "j", "input": {
            "video_url": "https://x/v.mp4", "start_time": 30, "end_time": 10}})
        self.assertEqual(out["code"], "invalid_range")

    def test_expired_presigned_url_is_actionable(self):
        # The most predictable production failure: a presigned URL that expired
        # while the job waited in RunPod's queue. Must name the cause.
        import urllib.error
        err = urllib.error.HTTPError("https://x/v.mp4", 403, "Forbidden", {}, None)
        with mock.patch.object(rp_handler, "_remote_identity", side_effect=err):
            out = rp_handler.handler({"id": "j", "input": {"video_url": "https://x/v.mp4"}})
        self.assertEqual(out["code"], "source_unavailable")
        self.assertIn("expired", out["reason"])

    def test_unreachable_source_is_a_soft_failure(self):
        import urllib.error
        with mock.patch.object(rp_handler, "_remote_identity",
                               side_effect=urllib.error.URLError("no route to host")):
            out = rp_handler.handler({"id": "j", "input": {"video_url": "https://x/v.mp4"}})
        self.assertEqual(out["code"], "source_unavailable")

    def test_file_with_no_video_stream_is_refused(self):
        import subprocess as sp
        with mock.patch.object(rp_handler.subprocess, "run") as run:
            run.return_value = sp.CompletedProcess([], 0, stdout='{"streams":[],"format":{}}', stderr="")
            with self.assertRaises(ValueError) as caught:
                rp_handler._probe(Path("/tmp/whatever.mp4"))
        self.assertIn("no video stream", str(caught.exception))


class TestRemoteIdentity(unittest.TestCase):
    def test_unreachable_host_returns_none_rather_than_raising(self):
        # Falling back to hashing the download must always be possible.
        self.assertIsNone(rp_handler._remote_identity("http://127.0.0.1:1/nope.mp4"))


class TestPresignedUpload(unittest.TestCase):
    def test_put_sends_the_signed_content_type(self):
        # A presigned PUT rejects the signature if Content-Type differs.
        captured = {}

        class FakeResponse:
            status = 200
            def __enter__(self): return self
            def __exit__(self, *a): return False

        def fake_open(request, timeout=None):
            captured["method"] = request.get_method()
            captured["type"] = request.headers.get("Content-type")
            captured["len"] = len(request.data)
            return FakeResponse()

        payload = Path("/tmp/_matte_probe.mp4")
        payload.write_bytes(b"x" * 2048)
        self.addCleanup(payload.unlink, missing_ok=True)

        with mock.patch.object(rp_handler._OPENER, "open", side_effect=fake_open):
            rp_handler._put_presigned(payload, "https://bucket/signed", "video/mp4")

        self.assertEqual(captured["method"], "PUT")
        self.assertEqual(captured["type"], "video/mp4")
        self.assertEqual(captured["len"], 2048)

    def test_rejected_upload_raises(self):
        class FakeResponse:
            status = 403
            def __enter__(self): return self
            def __exit__(self, *a): return False

        payload = Path("/tmp/_matte_probe2.mp4")
        payload.write_bytes(b"y")
        self.addCleanup(payload.unlink, missing_ok=True)

        with mock.patch.object(rp_handler._OPENER, "open", return_value=FakeResponse()):
            with self.assertRaises(RuntimeError):
                rp_handler._put_presigned(payload, "https://bucket/signed")


class TestWorkerIsolation(unittest.TestCase):
    def test_handler_serialises_jobs(self):
        # The stages are module-level singletons whose fields are mutated per
        # request, so overlapping jobs would silently corrupt each other.
        self.assertFalse(rp_handler._JOB_LOCK.locked())
        with rp_handler._JOB_LOCK:
            self.assertTrue(rp_handler._JOB_LOCK.locked())

    def test_models_are_module_level_singletons(self):
        # If these were built per request, every job would pay the model load
        # that RunPod bills as worker start time.
        for name in ("_detector", "_segmenter", "_tracker"):
            self.assertIsNotNone(getattr(rp_handler, name, None), name)


class TestArchParsing(unittest.TestCase):
    """`sm_120` is 12.0, not 1.20 — the minor version is the last digit."""

    def test_two_digit_arch(self):
        self.assertEqual(arch_guard.parse_arch("sm_86"), (8, 6))

    def test_three_digit_blackwell_arch(self):
        self.assertEqual(arch_guard.parse_arch("sm_120"), (12, 0))
        self.assertEqual(arch_guard.parse_arch("sm_100"), (10, 0))

    def test_arch_conditional_suffix_is_stripped(self):
        self.assertEqual(arch_guard.parse_arch("sm_90a"), (9, 0))

    def test_ptx_entries_are_ignored(self):
        self.assertIsNone(arch_guard.parse_arch("compute_120"))


class TestArchCoverage(unittest.TestCase):
    CU124 = ["sm_50", "sm_60", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"]
    CU128 = ["sm_70", "sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"]

    def test_forward_compatible_within_a_major(self):
        # The whole reason the 4090 kept working: it is sm_89, nothing compiles
        # for sm_89, and the sm_86 cubin covers it.
        self.assertTrue(arch_guard.covers(["sm_86"], (8, 9)))
        self.assertTrue(arch_guard.covers(["sm_80"], (8, 6)))

    def test_not_backward_compatible(self):
        self.assertFalse(arch_guard.covers(["sm_86"], (8, 0)))

    def test_does_not_cross_major_versions(self):
        # sm_90 kernels cannot carry a Blackwell card. This is the outage.
        self.assertFalse(arch_guard.covers(self.CU124, (12, 0)))
        self.assertTrue(arch_guard.covers(self.CU128, (12, 0)))

    def test_every_required_capability_is_covered_by_cu128(self):
        for cap in arch_guard.REQUIRED:
            self.assertTrue(arch_guard.covers(self.CU128, cap), cap)

    def test_cu124_misses_exactly_the_blackwell_capabilities(self):
        uncovered = {c for c in arch_guard.REQUIRED if not arch_guard.covers(self.CU124, c)}
        self.assertEqual(uncovered, {(10, 0), (12, 0)})

    def test_empty_arch_list_covers_nothing(self):
        # torch.cuda.get_arch_list() returns [] with no GPU visible. Reading the
        # list that way on a CPU-only image builder reports every arch missing
        # and fails the build — which is exactly what it did.
        self.assertFalse(arch_guard.covers([], (8, 9)))

    def test_compiled_archs_does_not_go_through_get_arch_list(self):
        # The GPU-free path: _cuda_getArchFlags is a compile-time macro with no
        # is_available() guard. If compiled_archs ever falls back to
        # get_arch_list on a builder, the build breaks again.
        flags = "sm_80 sm_86 sm_90 sm_120"
        fake_C = types.SimpleNamespace(_cuda_getArchFlags=lambda: flags)
        exploded = mock.Mock(side_effect=AssertionError("must not be called"))
        with mock.patch.object(arch_guard.torch, "_C", fake_C, create=True), \
                mock.patch.object(arch_guard.torch, "cuda",
                                  types.SimpleNamespace(get_arch_list=exploded)):
            self.assertEqual(arch_guard.compiled_archs(), flags.split())
        exploded.assert_not_called()


class TestDeviceSelection(unittest.TestCase):
    """The guard that turns a silent CPU fallback into a startup crash."""

    CU124 = ["sm_50", "sm_60", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"]
    CU128 = ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120", "compute_120"]

    def _run(self, arch_list, capability, name="Test GPU"):
        cuda = types.SimpleNamespace(
            is_available=lambda: True,
            get_device_capability=lambda: capability,
            get_device_name=lambda *a: name,
        )
        with mock.patch.object(rp_handler, "compiled_archs", lambda: arch_list), \
                mock.patch.object(rp_handler.torch, "cuda", cuda), \
                mock.patch.object(rp_handler.torch, "__version__", "x", create=True), \
                mock.patch.object(rp_handler.torch, "version",
                                  types.SimpleNamespace(cuda="x"), create=True):
            return rp_handler._select_device()

    def test_blackwell_on_cu124_is_refused(self):
        # The production failure: RTX 5090 is sm_120, outside cu124's list.
        with self.assertRaises(RuntimeError) as ctx:
            self._run(self.CU124, (12, 0), name="NVIDIA GeForce RTX 5090")
        message = str(ctx.exception)
        self.assertIn("sm_120", message)
        self.assertIn("RTX 5090", message)

    def test_blackwell_on_cu128_is_accepted(self):
        self.assertEqual(self._run(self.CU128, (12, 0)), "cuda")

    def test_ada_is_accepted_via_the_sm_86_cubin(self):
        # 4090/L40S are sm_89 and no build compiles for it explicitly; they run
        # on the sm_86 binary because cubins are forward-compatible within a
        # major version. This is why the 4090 kept working through the outage.
        self.assertEqual(self._run(self.CU124, (8, 9)), "cuda")
        self.assertEqual(self._run(self.CU128, (8, 9)), "cuda")

    def test_hopper_is_accepted(self):
        self.assertEqual(self._run(self.CU124, (9, 0)), "cuda")

    def test_arch_below_the_lowest_compiled_minor_is_refused(self):
        # sm_70 hardware cannot run an sm_75 cubin; compatibility is forward only.
        with self.assertRaises(RuntimeError):
            self._run(self.CU128, (7, 0))

    def test_no_gpu_is_refused_rather_than_falling_back(self):
        cuda = types.SimpleNamespace(is_available=lambda: False)
        with mock.patch.object(rp_handler, "_ALLOW_CPU", False), \
                mock.patch.object(rp_handler.torch, "cuda", cuda):
            with self.assertRaises(RuntimeError) as ctx:
                rp_handler._select_device()
        self.assertIn("CPU", str(ctx.exception))

    def test_cuda_init_failure_is_reraised_with_context(self):
        def boom():
            raise RuntimeError("no kernel image is available for execution on the device")

        cuda = types.SimpleNamespace(is_available=boom)
        with mock.patch.object(rp_handler.torch, "cuda", cuda):
            with self.assertRaises(RuntimeError) as ctx:
                rp_handler._select_device()
        self.assertIn("no kernel image", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
