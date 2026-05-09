# Training Your Voice for SwapMic

SwapMic itself doesn't train. It runs inference. You train your voice once, with any RVC trainer, then point SwapMic at the resulting `.pth` (and `.index`).

## What you need

- **15–20 minutes** of clean audio of you singing or talking. The cleaner the source, the cleaner the swap.
- A GPU. An RTX 4070 / 4080 / 4090 trains a usable model in ~30–90 minutes.
- An RVC training tool. Any of these work:
  - [Mangio-RVC-Fork](https://github.com/Mangio621/Mangio-RVC-Fork) — community fork, friendly UI
  - [RVC-Project/Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) — upstream
  - [`rvc-python`](https://pypi.org/project/rvc-python/) — Python-only path, smaller surface

## Recording tips

- Sing/talk in the **same pitch range** you want to convert into. Train on belting if you want to swap belted leads.
- **Vary** your performance. Soft, loud, low, high, sustained, staccato. The model learns from the spread.
- **Mono, 44.1 or 48 kHz, no effects.** Dry vocal only. No reverb, no compression, no auto-tune.
- Splice into 5–15s clips. Many trainers expect short slices.

## After training

You'll end up with two files:

- `my_voice.pth` — the model weights (~50–100 MB)
- `my_voice.index` — the speaker index (timbre lookup table; optional but improves fidelity)

Hand them to SwapMic:

```bash
swapmic swap song.wav --model my_voice.pth --index my_voice.index
```

That's it. The same `.pth` works on every song from then on.
