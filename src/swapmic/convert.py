"""RVC (Retrieval-based Voice Conversion) wrapper.

We don't reimplement RVC. We orchestrate it. The user trains a voice model
(.pth + optional .index) once with any standard RVC trainer (RVC-WebUI,
Mangio-RVC, rvc-python). SwapMic then calls into it for inference.

Two backends are supported, in priority order:
  1. rvc-python  (pip install rvc-python)            ← preferred, pure-python
  2. RVC CLI script path passed via env SWAPMIC_RVC_CLI  ← shell-out fallback

Either way, the function signature is the same: hand it a vocals WAV + a
trained model, get back a converted vocals WAV in your voice.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ConversionParams:
    """Knobs that actually matter for vocal-cover quality."""
    pitch_shift: int = 0          # semitones; +12 male→female, -12 female→male
    index_rate: float = 0.75      # 0..1, how strongly to lean on the .index
    filter_radius: int = 3        # median filter on f0; 3 is a sane default
    rms_mix_rate: float = 0.25    # 0=use source loudness, 1=use model loudness
    protect: float = 0.33         # protect voiceless consonants (0..0.5)
    f0_method: str = "rmvpe"      # rmvpe is the modern best


def convert(
    vocals_path: str | Path,
    model_path: str | Path,
    out_path: str | Path,
    index_path: str | Path | None = None,
    params: ConversionParams | None = None,
) -> Path:
    """Run RVC inference. Returns the path to the converted vocal WAV."""
    vocals_path = Path(vocals_path).resolve()
    model_path = Path(model_path).resolve()
    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    params = params or ConversionParams()

    if not vocals_path.exists():
        raise FileNotFoundError(f"vocals not found: {vocals_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"RVC model not found: {model_path}")

    if _has_rvc_python():
        return _convert_rvc_python(vocals_path, model_path, out_path, index_path, params)

    cli = os.environ.get("SWAPMIC_RVC_CLI")
    if cli:
        return _convert_shellout(vocals_path, model_path, out_path, index_path, params, cli)

    raise RuntimeError(
        "No RVC backend found. Install one of:\n"
        "  pip install rvc-python\n"
        "  …or set SWAPMIC_RVC_CLI to a path that runs RVC inference."
    )


def _has_rvc_python() -> bool:
    try:
        import rvc_python  # noqa: F401
        return True
    except Exception:
        return False


def _convert_rvc_python(
    vocals_path: Path,
    model_path: Path,
    out_path: Path,
    index_path: Path | None,
    params: ConversionParams,
) -> Path:
    from rvc_python.infer import RVCInference

    rvc = RVCInference(device="cuda:0")
    rvc.load_model(str(model_path), index_path=str(index_path) if index_path else None)
    rvc.set_params(
        f0up_key=params.pitch_shift,
        f0method=params.f0_method,
        index_rate=params.index_rate,
        filter_radius=params.filter_radius,
        rms_mix_rate=params.rms_mix_rate,
        protect=params.protect,
    )
    rvc.infer_file(str(vocals_path), str(out_path))
    return out_path


def _convert_shellout(
    vocals_path: Path,
    model_path: Path,
    out_path: Path,
    index_path: Path | None,
    params: ConversionParams,
    cli: str,
) -> Path:
    """Generic shell-out for users with their own RVC install.

    The CLI must accept these flags, in this order:
      <cli> --input <wav> --model <pth> --output <wav> --pitch <int>
            --index-rate <float> --protect <float> [--index <index>]
    """
    cmd = [
        cli,
        "--input", str(vocals_path),
        "--model", str(model_path),
        "--output", str(out_path),
        "--pitch", str(params.pitch_shift),
        "--index-rate", f"{params.index_rate}",
        "--protect", f"{params.protect}",
    ]
    if index_path:
        cmd += ["--index", str(index_path)]

    if not shutil.which(cli) and not Path(cli).exists():
        raise FileNotFoundError(f"RVC CLI not found: {cli}")

    subprocess.run(cmd, check=True)
    return out_path
