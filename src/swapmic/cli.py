"""SwapMic command-line interface.

Usage:
    swapmic swap <song.wav> --model voice.pth [--index voice.index] [--pitch 0]
    swapmic separate <song.wav> [--out-dir output/stems]
    swapmic version
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .convert import ConversionParams
from .pipeline import SwapConfig, swap
from .remix import RemixParams
from .separate import separate as run_separate


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="swapmic",
        description="Swap the lead vocal of any song with your own voice.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("swap", help="Run the full pipeline on a song.")
    s.add_argument("song", type=Path, help="Input song (.wav, .mp3, .flac, .m4a).")
    s.add_argument("--model", type=Path, required=True, help="Trained RVC .pth file.")
    s.add_argument("--index", type=Path, default=None, help="RVC .index file (optional).")
    s.add_argument("--out-dir", type=Path, default=Path("output"), help="Where to write results.")
    s.add_argument("--pitch", type=int, default=0, help="Pitch shift in semitones.")
    s.add_argument("--index-rate", type=float, default=0.75)
    s.add_argument("--protect", type=float, default=0.33)
    s.add_argument("--f0", default="rmvpe", choices=["rmvpe", "harvest", "crepe", "pm"])
    s.add_argument("--vocal-gain-db", type=float, default=0.0)
    s.add_argument("--instrumental-gain-db", type=float, default=0.0)
    s.add_argument("--peak-dbfs", type=float, default=-1.0, help="Final mix ceiling.")
    s.add_argument("--demucs-model", default="htdemucs")
    s.add_argument("--device", default=None, help="cuda | cpu (auto-detected if omitted).")

    sp = sub.add_parser("separate", help="Run only the Demucs separation step.")
    sp.add_argument("song", type=Path)
    sp.add_argument("--out-dir", type=Path, default=Path("output/stems"))
    sp.add_argument("--demucs-model", default="htdemucs")
    sp.add_argument("--device", default=None)

    sub.add_parser("version", help="Print SwapMic version and exit.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "version":
        print(f"swapmic {__version__}")
        return 0

    if args.command == "separate":
        result = run_separate(args.song, args.out_dir, model=args.demucs_model, device=args.device)
        print(f"vocals       → {result.vocals}")
        print(f"instrumental → {result.instrumental}")
        return 0

    if args.command == "swap":
        config = SwapConfig(
            model_path=args.model,
            index_path=args.index,
            out_dir=args.out_dir,
            convert_params=ConversionParams(
                pitch_shift=args.pitch,
                index_rate=args.index_rate,
                protect=args.protect,
                f0_method=args.f0,
            ),
            remix_params=RemixParams(
                vocal_gain_db=args.vocal_gain_db,
                instrumental_gain_db=args.instrumental_gain_db,
                target_peak_dbfs=args.peak_dbfs,
            ),
            demucs_model=args.demucs_model,
            device=args.device,
        )
        result = swap(args.song, config, progress=lambda m: print(f"[swapmic] {m}"))
        print(f"\nfinal → {result.final}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
