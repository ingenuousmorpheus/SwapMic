"""Smoke tests for the remix step. No models, no GPU."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from swapmic.remix import RemixParams, remix


def _write_sine(path: Path, freq: float, seconds: float, sr: int = 44100) -> None:
    t = np.arange(int(seconds * sr)) / sr
    audio = 0.3 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    sf.write(path, np.column_stack([audio, audio]), sr, subtype="PCM_24")


def test_remix_produces_24bit_wav(tmp_path: Path) -> None:
    voc = tmp_path / "voc.wav"
    instr = tmp_path / "instr.wav"
    out = tmp_path / "mix.wav"
    _write_sine(voc, 440.0, 1.0)
    _write_sine(instr, 220.0, 1.0)

    result = remix(voc, instr, out, params=RemixParams(target_peak_dbfs=-1.0))

    assert result.exists()
    info = sf.info(str(result))
    assert info.subtype == "PCM_24"
    assert info.samplerate == 44100

    mix, _ = sf.read(str(result))
    assert float(np.max(np.abs(mix))) <= 10 ** (-1.0 / 20.0) + 1e-3


def test_remix_handles_length_mismatch(tmp_path: Path) -> None:
    voc = tmp_path / "voc.wav"
    instr = tmp_path / "instr.wav"
    out = tmp_path / "mix.wav"
    _write_sine(voc, 440.0, 1.0)
    _write_sine(instr, 220.0, 1.5)

    remix(voc, instr, out)
    mix, sr = sf.read(str(out))
    assert len(mix) == int(1.5 * sr)
