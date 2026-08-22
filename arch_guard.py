"""Does this torch build have kernels for this GPU?

The failure being guarded against, seen in production on Blackwell workers:

    RuntimeError: CUDA initialization failed: Failed to initialize GPU 0:
    CUDA error: no kernel image is available for execution on the device

It is an *architecture* mismatch, not a driver one. torch ships cubins for a
fixed set of compute capabilities and nothing else runs. The endpoint's CUDA
filter gates the host driver and says nothing about this.

Imported by `rp_handler` for the per-worker check, and run as `__main__` by
`Dockerfile.serverless` for the build-time check, so the compatibility rule is
written once.
"""

from __future__ import annotations

import torch

# Capabilities the endpoint can actually land on, and the card that motivates
# each. Keep in sync with the GPU types selected in DEPLOY.md §3.
REQUIRED: dict[tuple[int, int], str] = {
    (8, 6): "A6000 / A40",
    (8, 9): "RTX 4090 / L40S",
    (9, 0): "H100",
    (10, 0): "B200",
    (12, 0): "RTX 5090 / RTX PRO 6000",
}


def parse_arch(arch: str) -> tuple[int, int] | None:
    """``"sm_86"`` -> ``(8, 6)``, ``"sm_120"`` -> ``(12, 0)``, ``"sm_90a"`` -> ``(9, 0)``.

    The minor version is always the last digit, so the major is everything
    before it — ``sm_120`` is 12.0, not 1.20. Arch-conditional suffixes
    (``90a``) are trailing letters and get stripped. Returns None for
    ``compute_*`` PTX entries and anything unparseable.
    """
    if not arch.startswith("sm_"):
        return None
    digits = arch[3:].rstrip("abcdef")
    if len(digits) < 2 or not digits.isdigit():
        return None
    return int(digits[:-1]), int(digits[-1])


def covers(archs: list[str], capability: tuple[int, int]) -> bool:
    """Can any cubin in `archs` run on a device of `capability`?

    CUDA binaries are forward-compatible *within* a major version, so an sm_86
    cubin runs on sm_89 — which is why a 4090 works on builds that never
    mention sm_89, and why literal string membership is the wrong test. The
    guarantee does not cross majors: nothing in sm_9x can carry an sm_12x card.
    """
    major, minor = capability
    for arch in archs:
        parsed = parse_arch(arch)
        if parsed and parsed[0] == major and parsed[1] <= minor:
            return True
    return False


def compiled_archs() -> list[str]:
    """The arch list, readable without a GPU.

    `torch.cuda.get_arch_list()` returns `[]` when `is_available()` is False
    (torch/cuda/__init__.py), so on a CPU-only image builder it reports that
    *every* arch is missing. `_cuda_getArchFlags` is the compile-time macro
    underneath it and has no such guard, which is what makes a build-time check
    possible at all.
    """
    getter = getattr(torch._C, "_cuda_getArchFlags", None)
    if getter is not None:
        flags = getter()
        if flags:
            return flags.split()
    return torch.cuda.get_arch_list()


def _main() -> int:
    import torchvision

    archs = compiled_archs()
    # Printed before anything can fail: on a hosted build these lines are the
    # only record of what the base image actually shipped.
    print(f"torch {torch.__version__}  cuda {torch.version.cuda}  "
          f"torchvision {torchvision.__version__}", flush=True)
    print(f"arch list: {archs}", flush=True)

    if not torch.version.cuda:
        print("FAIL: torch has no CUDA support at all", flush=True)
        return 1
    if not archs:
        print("FAIL: could not read the arch list; cannot verify GPU coverage", flush=True)
        return 1

    uncovered = {cap: gpu for cap, gpu in REQUIRED.items() if not covers(archs, cap)}
    for (major, minor), gpu in sorted(REQUIRED.items()):
        mark = "MISSING" if (major, minor) in uncovered else "ok"
        print(f"  {f'sm_{major}{minor}':<7} {gpu:<24} {mark}", flush=True)

    if uncovered:
        names = ", ".join(
            f"sm_{a}{b} ({gpu})" for (a, b), gpu in sorted(uncovered.items())
        )
        # Into the assertion text, not just stdout: RunPod's hosted build log
        # surfaces the failing command but not always the step's output.
        raise AssertionError(
            f"base image has no kernels for {names}. Arch list is {archs}. "
            f"Either use a base image that covers them, or drop those GPU types "
            f"from the endpoint and from REQUIRED here."
        )

    print("arch coverage OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
