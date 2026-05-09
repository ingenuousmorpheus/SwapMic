# SwapMic Examples

Drop sample songs and trained voice models into this directory and try them.
Audio files are gitignored by default — nothing here will be committed.

## Suggested workflow

1. Train an RVC model of your voice once with any RVC trainer.
2. Drop `my_voice.pth` (and `my_voice.index` if you have it) here.
3. Drop a Suno export `my_song.wav` here.
4. Run from the repo root:

   ```bash
   swapmic swap examples/my_song.wav \
     --model examples/my_voice.pth \
     --index examples/my_voice.index \
     --out-dir examples/output
   ```

5. Listen to `examples/output/my_song.swapmic.wav`.
