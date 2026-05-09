"""Sum the converted vocal back into the instrumental at sane levels.

The hard parts are: matching sample rates, matching the vocal's loudness to
roughly where the original sat in the mix, and keeping headroom so the final
file doesn't clip. No magic — just the boring stuff done right.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass
class RemixParams:
    vocal_gain_db: float = 0.0       # post-match trim
    instrumental_gain_db: float = 0.0
    target_peak_dbfs: float = -1.0   # ceiling for the final mix
    match_loudness_to: str | None = "reference"  # "reference" | None
    crossfade_ms: int = 0            # 0 = straight sum


def _load(path: str | Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), always_2d=True, dtype="float32")
    return audio, sr


def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio
    # SciPy is heavy; use a simple poly resampler from soxr if present, else fail loud.
    try:
        import soxr
        return soxr.resample(audio, src_sr, dst_sr).astype(np.float32)
    except ImportError as e:
        raise RuntimeError(
            f"Sample rate mismatch ({src_sr} vs {dst_sr}) and soxr is not installed. "
            "Install it: pip install soxr"
        ) from e


def _to_stereo(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        audio = audio[:, None]
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    return audio[:, :2]


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-12)


def _db_to_lin(db: float) -> float:
    return float(10 ** (db / 20.0))


def remix(
    converted_vocal_path: str | Path,
    instrumental_path: str | Path,
    out_path: str | Path,
    reference_vocal_path: str | Path | None = None,
    params: RemixParams | None = None,
) -> Path:
    """Sum converted vocal + instrumental into a single 24-bit WAV.

    If ``reference_vocal_path`` is given (the original lead vocal from the
    Demucs split), the converted vocal is loudness-matched to it before
    summing, so the cover sits in the mix where the original did.
    """
    params = params or RemixParams()
    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    voc, voc_sr = _load(converted_vocal_path)
    instr, instr_sr = _load(instrumental_path)

    target_sr = max(voc_sr, instr_sr)
    voc = _to_stereo(_resample(voc, voc_sr, target_sr))
    instr = _to_stereo(_resample(instr, instr_sr, target_sr))

    if params.match_loudness_to == "reference" and reference_vocal_path:
        ref, ref_sr = _load(reference_vocal_path)
        ref = _to_stereo(_resample(ref, ref_sr, target_sr))
        ratio = _rms(ref) / _rms(voc)
        voc = voc * ratio

    voc = voc * _db_to_lin(params.vocal_gain_db)
    instr = instr * _db_to_lin(params.instrumental_gain_db)

    n = max(len(voc), len(instr))
    if len(voc) < n:
        voc = np.pad(voc, ((0, n - len(voc)), (0, 0)))
    if len(instr) < n:
        instr = np.pad(instr, ((0, n - len(instr)), (0, 0)))

    mix = voc + instr

    peak = float(np.max(np.abs(mix)))
    ceiling = _db_to_lin(params.target_peak_dbfs)
    if peak > ceiling:
        mix = mix * (ceiling / peak)

    sf.write(out_path, mix.astype(np.float32), target_sr, subtype="PCM_24")
    return out_path
