"""Vocal/instrumental separation via Demucs.

Wraps demucs.api.Separator into a tiny call surface: give a song, get back
(vocals_path, instrumental_path) as 24-bit WAV.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


DEFAULT_MODEL = "htdemucs"  # 4-stem; we collapse non-vocal stems → instrumental


@dataclass
class SeparationResult:
    vocals: Path
    instrumental: Path
    sample_rate: int


def separate(
    song_path: str | Path,
    out_dir: str | Path,
    model: str = DEFAULT_MODEL,
    device: str | None = None,
) -> SeparationResult:
    """Split a song into (vocals, instrumental) WAV files.

    Returns the on-disk paths. Output is 24-bit WAV at the model's native rate
    (44.1 kHz for htdemucs).
    """
    from demucs.api import Separator  # imported lazily; demucs is heavy

    song_path = Path(song_path).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    separator = Separator(model=model, device=device)
    _, sources = separator.separate_audio_file(song_path)

    sr = separator.samplerate
    vocals_t = sources["vocals"]
    instrumental_t = sum(t for name, t in sources.items() if name != "vocals")

    vocals_np = vocals_t.cpu().numpy().T.astype(np.float32)
    instr_np = instrumental_t.cpu().numpy().T.astype(np.float32)

    vocals_path = out_dir / f"{song_path.stem}.vocals.wav"
    instr_path = out_dir / f"{song_path.stem}.instrumental.wav"

    sf.write(vocals_path, vocals_np, sr, subtype="PCM_24")
    sf.write(instr_path, instr_np, sr, subtype="PCM_24")

    return SeparationResult(vocals=vocals_path, instrumental=instr_path, sample_rate=sr)
