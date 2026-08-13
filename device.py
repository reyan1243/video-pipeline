"""Device selection and precision policy, shared by every model-backed stage."""

from __future__ import annotations

import contextlib
import os

import torch

# Escape hatch. Autocast is not a tuning knob on CUDA — see `autocast` below —
# so this exists for debugging a suspected numerical issue, not for normal use.
_AUTOCAST_DISABLED = os.environ.get("VIDEO_PIPELINE_NO_AUTOCAST", "").strip().lower() in {
    "1",
    "true",
    "yes",
}


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def enable_fast_matmul() -> None:
    """Allow TF32 on Ampere+ tensor cores.

    TF32 keeps fp32's exponent range and drops mantissa bits that a segmentation
    mask cannot resolve anyway. PyTorch's own SAM2 work measured mIoU 0.997-0.998
    against full fp32 with this plus fp16 enabled.
    """
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def autocast(device: str):
    """bfloat16 autocast on CUDA, a no-op everywhere else.

    This is not an optimisation on CUDA — it is required for correctness.
    `transformers`' SAM2 video model deliberately stores its memory-attention
    features as bfloat16 ("for consistency with the original implementation"),
    but this pipeline loads the model in float32 and nothing casts them back, so
    every frame that uses memory attention dies with:

        RuntimeError: mat1 and mat2 must have the same dtype, but got BFloat16 and Float

    The reference SAM2 implementation runs its whole forward pass under bf16
    autocast, which is the consistency that comment refers to. Conditioning
    frames skip memory attention entirely, so a one-frame smoke test passes and
    hides this — it only bites on real clips.

    MPS and CPU take the disabled branch and behave exactly as before.
    """
    if _AUTOCAST_DISABLED or device != "cuda":
        return contextlib.nullcontext()
    return torch.autocast("cuda", dtype=torch.bfloat16)


def weights_dtype(device: str, prefer_bfloat16: bool):
    """Dtype to load model weights in.

    bfloat16 halves the ~2.7 GB of weights this pipeline reads from disk and
    uploads to the GPU, which is the whole of the pre-run startup cost — billed
    on every serverless cold start. Activations already run in bf16 under
    `autocast`, and SAM2's reference implementation runs bf16 throughout, so the
    numerics are the ones the model was designed around.

    Off by default regardless: it is a real change to what is loaded, and unlike
    autocast it is not required for correctness.
    """
    if not prefer_bfloat16 or device != "cuda":
        return None
    return torch.bfloat16


def load_pretrained(model_cls, model_id: str, device: str, prefer_bfloat16: bool):
    """`from_pretrained(...).to(device).eval()` with the dtype kwarg sorted out.

    transformers renamed `torch_dtype` to `dtype` and warns loudly on the old
    name, but the repo supports >=4.56 where only the old name exists. Try the
    new one, fall back on TypeError.
    """
    dtype = weights_dtype(device, prefer_bfloat16)
    if dtype is None:
        model = model_cls.from_pretrained(model_id)
    else:
        try:
            model = model_cls.from_pretrained(model_id, dtype=dtype)
        except TypeError:
            model = model_cls.from_pretrained(model_id, torch_dtype=dtype)
    return model.to(device).eval()
