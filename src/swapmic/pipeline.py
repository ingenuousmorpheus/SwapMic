"""End-to-end pipeline: song in → song with your voice out."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import convert as _convert
from . import remix as _remix
from . import separate as _separate


@dataclass
class SwapResult:
    final: Path
    vocals: Path
    instrumental: Path
    converted_vocals: Path


@dataclass
class SwapConfig:
    model_path: Path
    index_path: Path | None = None
    out_dir: Path = field(default_factory=lambda: Path("output"))
    keep_intermediates: bool = True
    convert_params: _convert.ConversionParams = field(default_factory=_convert.ConversionParams)
    remix_params: _remix.RemixParams = field(default_factory=_remix.RemixParams)
    demucs_model: str = _separate.DEFAULT_MODEL
    device: str | None = None


def swap(
    song_path: str | Path,
    config: SwapConfig,
    progress: Callable[[str], None] | None = None,
) -> SwapResult:
    """Run the full Demucs → RVC → remix pipeline."""
    song_path = Path(song_path).resolve()
    out_dir = Path(config.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    log = progress or (lambda _: None)

    log("Separating vocals from instrumental (Demucs)...")
    sep = _separate.separate(
        song_path,
        out_dir / "stems",
        model=config.demucs_model,
        device=config.device,
    )

    log("Converting vocals to your voice (RVC)...")
    converted = _convert.convert(
        vocals_path=sep.vocals,
        model_path=config.model_path,
        out_path=out_dir / "stems" / f"{song_path.stem}.vocals.you.wav",
        index_path=config.index_path,
        params=config.convert_params,
    )

    log("Mixing converted vocal back into the instrumental...")
    final = _remix.remix(
        converted_vocal_path=converted,
        instrumental_path=sep.instrumental,
        out_path=out_dir / f"{song_path.stem}.swapmic.wav",
        reference_vocal_path=sep.vocals,
        params=config.remix_params,
    )

    log(f"Done → {final}")
    return SwapResult(
        final=final,
        vocals=sep.vocals,
        instrumental=sep.instrumental,
        converted_vocals=converted,
    )
