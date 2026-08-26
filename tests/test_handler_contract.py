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

    def _run(self, probe, metadata=None, **extra_input):
        job_input = {"video_url": "https://x/v.mp4", **extra_input}
        with mock.patch.object(rp_handler, "_remote_identity", return_value=None), \
             mock.patch.object(rp_handler, "_download", return_value="deadbeef"), \
             mock.patch.object(rp_handler, "_already_done", return_value=False), \
             mock.patch.object(rp_handler, "_trim", side_effect=lambda s, d, a, b: s), \
             mock.patch.object(rp_handler, "_probe", return_value=probe), \
             mock.patch.object(rp_handler, "VideoSource") as vs:
            vs.return_value.load_metadata.return_value = metadata
            return rp_handler.handler({"id": "j", "input": job_input})

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


class TestProjectedMaskBytes(unittest.TestCase):
    """Masks are stored downscaled, so the projection must be too.

    `tracker.track` calls `downscale_mask` the moment a mask leaves the GPU, so
    what accumulates in host RAM is the *capped* size, never the source size.
    """

    UHD = (3840, 2160)

    def test_source_resolution_when_uncapped(self):
        self.assertEqual(
            rp_handler._projected_mask_bytes(*self.UHD, frame_count=10, max_mask_height=0),
            10 * 3840 * 2160,
        )

    def test_cap_scales_both_dimensions(self):
        # 2160 -> 720 is a third of the height, so a third of the width too:
        # 9x fewer bytes, not 3x. Capping only the height would understate it.
        self.assertEqual(
            rp_handler._projected_mask_bytes(*self.UHD, frame_count=10, max_mask_height=720),
            10 * 1280 * 720,
        )

    def test_cap_above_source_height_is_a_no_op(self):
        self.assertEqual(
            rp_handler._projected_mask_bytes(1280, 720, frame_count=10, max_mask_height=2160),
            10 * 1280 * 720,
        )

    def test_matches_downscale_mask_for_odd_aspect_ratios(self):
        # The projection has to agree with the real thing or the guard is
        # protecting a number nothing produces.
        import numpy as np

        from mask_ops import downscale_mask

        width, height, cap = 1001, 733, 300
        actual = downscale_mask(np.zeros((height, width), dtype=np.uint8), cap)
        self.assertEqual(
            rp_handler._projected_mask_bytes(width, height, frame_count=1, max_mask_height=cap),
            actual.shape[0] * actual.shape[1],
        )


class TestLargestFittingHeight(unittest.TestCase):
    """The rejection should name a height that works, not guess '720p'."""

    def test_suggests_a_height_that_actually_fits(self):
        suggested = rp_handler._largest_fitting_height(3840, 2160, frame_count=4900)
        self.assertIsNotNone(suggested)
        self.assertLessEqual(
            rp_handler._projected_mask_bytes(3840, 2160, 4900, suggested),
            rp_handler.MAX_MASK_BYTES,
        )

    def test_returns_none_when_no_height_helps(self):
        # Downscaling cannot save a clip this long; the caller needs to trim.
        self.assertIsNone(rp_handler._largest_fitting_height(3840, 2160, frame_count=10**8))


class TestMaskMemoryGuard(TestGuards):
    """4K regression: the guard ignored the parameter that fixes it.

    `max_mask_height` was applied to the tracker 13 lines *after* the guard ran,
    so a 4K job was refused on its source-resolution footprint even when the
    request had already asked for masks small enough to fit.
    """

    UHD_4900 = types.SimpleNamespace(frame_count=4900, width=3840, height=2160, fps=30.0)
    PROBE = {"width": 3840, "height": 2160, "duration": 163.0}

    def test_uncapped_4k_is_auto_capped_rather_than_refused(self):
        # Was a rejection. A 4K source now gets the default cap applied for it.
        out = self._run(self.PROBE, self.UHD_4900)
        self.assertNotEqual(out.get("code"), "too_large")
        self.assertEqual(rp_handler._tracker.config.max_mask_height, 720)

    def test_4k_is_accepted_when_the_cap_makes_it_fit(self):
        # 4900 frames at 1280x720 is ~4.2 GB, well inside the 8 GB limit.
        out = self._run(self.PROBE, self.UHD_4900, max_mask_height=720)
        self.assertNotEqual(out.get("code"), "too_large")

    def test_4k_is_still_refused_when_the_cap_is_too_generous(self):
        # 4900 frames at 1920x1080 is ~9.5 GB — over the limit even downscaled.
        out = self._run(self.PROBE, self.UHD_4900, max_mask_height=1080)
        self.assertIs(out["ok"], False)
        self.assertEqual(out["code"], "too_large")

    def test_rejection_names_a_height_that_would_fit(self):
        # An explicit cap wins, so an explicit cap that does not fit still gets
        # refused — and the message must name one that would.
        out = self._run(self.PROBE, self.UHD_4900, max_mask_height=1080)
        self.assertIn("max_mask_height", out["reason"])
        self.assertIn(str(out["suggested_max_mask_height"]), out["reason"])

    def test_rejection_reports_the_downscaled_projection(self):
        # Reporting the source-resolution figure when a cap was requested tells
        # the caller to fix something they already did.
        out = self._run(self.PROBE, self.UHD_4900, max_mask_height=1080)
        self.assertAlmostEqual(out["projected_gb"], 9.5, places=1)


class TestAutoMaskHeight(unittest.TestCase):
    """Resolution picks the default cap; the memory budget picks the floor.

    Frame rate never appears here on purpose — it is already inside
    `frame_count`, so a 60fps clip is handled by the same budget check that
    handles a long 30fps one.
    """

    UHD = (3840, 2160)
    HD = (1920, 1080)

    def test_explicit_request_wins_even_when_it_does_not_fit(self):
        # The caller asked for 1080 on a clip where it overflows. That is their
        # call to make; the guard refuses it rather than silently substituting.
        self.assertEqual(
            rp_handler._auto_mask_height(*self.UHD, frame_count=4867, requested=1080), 1080
        )

    def test_explicit_request_wins_over_the_large_source_default(self):
        self.assertEqual(
            rp_handler._auto_mask_height(*self.UHD, frame_count=100, requested=1440), 1440
        )

    def test_4k_gets_the_default_cap(self):
        self.assertEqual(
            rp_handler._auto_mask_height(*self.UHD, frame_count=1217, requested=0), 720
        )

    def test_1080p_is_left_at_source(self):
        self.assertEqual(
            rp_handler._auto_mask_height(*self.HD, frame_count=300, requested=0), 0
        )

    def test_long_60fps_4k_drops_below_the_default_cap(self):
        # 300s at 60fps is 18000 frames; 720p would still need 16.6 GB.
        applied = rp_handler._auto_mask_height(*self.UHD, frame_count=18000, requested=0)
        self.assertLess(applied, 720)
        self.assertLessEqual(
            rp_handler._projected_mask_bytes(*self.UHD, 18000, applied),
            rp_handler.MAX_MASK_BYTES,
        )

    def test_long_1080p_is_rescued_even_though_it_is_not_a_large_source(self):
        # The other half: high frame count on a source below the 4K threshold.
        applied = rp_handler._auto_mask_height(*self.HD, frame_count=9000, requested=0)
        self.assertGreater(applied, 0)
        self.assertLessEqual(
            rp_handler._projected_mask_bytes(*self.HD, 9000, applied),
            rp_handler.MAX_MASK_BYTES,
        )

    def test_falls_back_to_the_default_cap_when_nothing_fits(self):
        # Nothing rescues this, so the guard must still get a chance to refuse.
        applied = rp_handler._auto_mask_height(*self.UHD, frame_count=10**8, requested=0)
        self.assertEqual(applied, 720)
        self.assertGreater(
            rp_handler._projected_mask_bytes(*self.UHD, 10**8, applied),
            rp_handler.MAX_MASK_BYTES,
        )


class TestAutoCapThroughHandler(TestGuards):
    """The reported 60fps 4K case, end to end."""

    PROBE_4K = {"width": 3840, "height": 2160, "duration": 20.3}

    def test_20s_60fps_4k_is_accepted_and_capped_at_720(self):
        meta = types.SimpleNamespace(frame_count=1217, width=3840, height=2160, fps=60.0)
        out = self._run(self.PROBE_4K, meta)
        self.assertNotEqual(out.get("code"), "too_large")
        self.assertEqual(rp_handler._tracker.config.max_mask_height, 720)

    def test_300s_60fps_4k_is_accepted_below_the_default_cap(self):
        meta = types.SimpleNamespace(frame_count=18000, width=3840, height=2160, fps=60.0)
        out = self._run({**self.PROBE_4K, "duration": 300.0}, meta)
        self.assertNotEqual(out.get("code"), "too_large")
        self.assertLess(rp_handler._tracker.config.max_mask_height, 720)

    def test_explicit_zero_still_requests_a_full_resolution_matte(self):
        # Passing 0 explicitly is indistinguishable from omitting it, so the
        # documented escape hatch for a source-resolution matte is a small clip.
        meta = types.SimpleNamespace(frame_count=100, width=1280, height=720, fps=30.0)
        out = self._run({"width": 1280, "height": 720, "duration": 3.3}, meta)
        self.assertNotEqual(out.get("code"), "too_large")
        self.assertEqual(rp_handler._tracker.config.max_mask_height, 0)

    def test_applied_cap_is_reported_to_the_caller(self):
        # The matte ships at mask resolution (exporter.py), so a caller that did
        # not choose the cap still has to learn which one was chosen for it.
        meta = types.SimpleNamespace(frame_count=1217, width=3840, height=2160, fps=60.0)
        seed = types.SimpleNamespace(frame_index=7, iou_score=0.98)
        tracking = types.SimpleNamespace(masks={}, coverage=lambda n: 1.0)
        export = types.SimpleNamespace(matte_path="/tmp/m.mp4", width=1280, height=720)
        with mock.patch.object(rp_handler._segmenter, "select_seed", return_value=seed), \
             mock.patch.object(rp_handler._tracker, "track", return_value=tracking), \
             mock.patch.object(rp_handler, "clean_masks"), \
             mock.patch.object(rp_handler, "LayerExporter") as exporter, \
             mock.patch.object(rp_handler, "_upload", return_value="https://signed"):
            exporter.return_value.export.return_value = export
            out = self._run(self.PROBE_4K, meta)
        self.assertIs(out["ok"], True)
        self.assertEqual(out["applied_max_mask_height"], 720)
        self.assertEqual((out["width"], out["height"]), (1280, 720))


class TestBatchedRanges(unittest.TestCase):
    """Several portions in ONE job.

    A caller masking four scattered captions used to send four jobs: four worker
    starts, four downloads of the same video, for the same GPU work. These prove
    the fixed cost is now paid once and that one bad portion cannot take the
    others down with it.
    """

    META = types.SimpleNamespace(frame_count=300, width=1080, height=1920, fps=30.0)
    PROBE = {"width": 1080, "height": 1920, "duration": 10.0}

    def _run(self, job_input, download=None, per_range=None):
        download = download or mock.MagicMock(return_value="deadbeef")
        with mock.patch.object(rp_handler, "_remote_identity", return_value=None), \
             mock.patch.object(rp_handler, "_download", download), \
             mock.patch.object(rp_handler, "_already_done", return_value=False), \
             mock.patch.object(rp_handler, "_trim", side_effect=lambda s, d, a, b: s), \
             mock.patch.object(rp_handler, "_probe", return_value=self.PROBE), \
             mock.patch.object(rp_handler, "VideoSource") as vs:
            vs.return_value.load_metadata.return_value = self.META
            if per_range is not None:
                with mock.patch.object(rp_handler, "_process_range", side_effect=per_range):
                    return rp_handler.handler({"id": "j", "input": job_input}), download
            return rp_handler.handler({"id": "j", "input": job_input}), download

    def test_downloads_the_video_once_for_every_range(self):
        # The whole point: the fixed cost is paid once, not per portion.
        out, download = self._run(
            {
                "video_url": "https://x/v.mp4",
                "ranges": [
                    {"start_time": 0, "end_time": 2, "matte_key": "a.mp4"},
                    {"start_time": 8, "end_time": 9, "matte_key": "b.mp4"},
                    {"start_time": 20, "end_time": 24, "matte_key": "c.mp4"},
                ],
            },
            per_range=lambda *a, **k: {"ok": True, "matte_key": "x"},
        )
        self.assertEqual(download.call_count, 1)
        self.assertEqual(len(out["results"]), 3)

    def test_one_bad_portion_does_not_sink_the_others(self):
        # Partial success on purpose: a subject who walks out of frame during one
        # portion must not throw away GPU time already spent on the rest.
        calls = iter(
            [
                {"ok": True, "matte_key": "a.mp4"},
                {"ok": False, "code": "incomplete_coverage", "matte_key": "b.mp4"},
                {"ok": True, "matte_key": "c.mp4"},
            ]
        )
        out, _ = self._run(
            {
                "video_url": "https://x/v.mp4",
                "ranges": [{"matte_key": k} for k in ("a.mp4", "b.mp4", "c.mp4")],
            },
            per_range=lambda *a, **k: next(calls),
        )
        self.assertIs(out["ok"], True)
        self.assertEqual([r["ok"] for r in out["results"]], [True, False, True])

    def test_every_portion_failing_reports_the_job_as_failed(self):
        out, _ = self._run(
            {"video_url": "https://x/v.mp4", "ranges": [{"matte_key": "a"}, {"matte_key": "b"}]},
            per_range=lambda *a, **k: {"ok": False, "code": "incomplete_coverage"},
        )
        self.assertIs(out["ok"], False)

    def test_a_range_carries_its_own_matte_key_through_a_rejection(self):
        # The caller settles each portion against its own row, and the key is the
        # only thing tying a result back to one. A rejection without it is unusable.
        out, _ = self._run(
            {
                "video_url": "https://x/v.mp4",
                "ranges": [{"start_time": 0, "end_time": 2, "matte_key": "wanted.mp4"}],
            },
            per_range=None,
        )
        self.assertEqual(out["results"][0].get("matte_key"), "wanted.mp4")

    def test_invalid_ranges_are_refused_before_the_download(self):
        # An invalid range is a caller bug, not something the footage can settle.
        # Paying for a 67 MB fetch to discover it would be pure waste.
        out, download = self._run(
            {"video_url": "https://x/v.mp4", "ranges": [{"start_time": 30, "end_time": 10}]}
        )
        self.assertEqual(out["code"], "invalid_range")
        self.assertEqual(download.call_count, 0)

    def test_a_good_range_still_runs_beside_an_invalid_one(self):
        out, download = self._run(
            {
                "video_url": "https://x/v.mp4",
                "ranges": [{"start_time": 30, "end_time": 10}, {"start_time": 0, "end_time": 2}],
            },
            per_range=lambda *a, **k: {"ok": True},
        )
        self.assertEqual(download.call_count, 1)
        self.assertEqual(len(out["results"]), 2)

    def test_a_crash_in_one_portion_keeps_the_others(self):
        # The mattes for the good portions are already uploaded by the time a
        # later one blows up. Letting the exception escape would throw away GPU
        # time that has been spent and cannot be recovered.
        def side_effect(job, source, workdir, index, rng, *a, **k):
            if index == 1:
                raise RuntimeError("cuda blew up")
            return {"ok": True, "matte_key": rng["matte_key"]}

        with mock.patch.object(rp_handler, "_process_range", side_effect=side_effect):
            out, _ = self._run(
                {
                    "video_url": "https://x/v.mp4",
                    "ranges": [{"matte_key": k} for k in ("a", "b", "c")],
                }
            )

        self.assertIs(out["ok"], True)
        self.assertEqual([bool(r.get("ok")) for r in out["results"]], [True, False, True])
        # The crash is reported as an error for that portion alone, still keyed
        # so the caller can settle the right row.
        self.assertIn("cuda blew up", out["results"][1]["error"])
        self.assertEqual(out["results"][1]["matte_key"], "b")

    def test_the_old_single_range_shape_still_works(self):
        # Deploy skew runs in both directions: an older caller must keep working
        # against a newer worker.
        out, _ = self._run(
            {"video_url": "https://x/v.mp4", "start_time": 1, "end_time": 3, "matte_key": "legacy.mp4"},
            per_range=lambda *a, **k: {"ok": True, "matte_key": "legacy.mp4", "width": 1080},
        )
        self.assertIs(out["ok"], True)
        # Flattened to the old top-level shape as well as the new results array.
        self.assertEqual(out["matte_key"], "legacy.mp4")
        self.assertEqual(out["width"], 1080)
        self.assertEqual(len(out["results"]), 1)

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
