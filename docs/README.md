# docs/ — the safety nets

This folder holds two demo safety nets, both produced during your LIVE rehearsal:

- **`cached_adapter.b64`** — the trained LoRA adapter, written automatically after a
  successful LIVE train. Ticking "use cached adapter" (or `WALL_USE_CACHED=1`) skips live
  training and loads this, so Act 2 finishes instantly and the swarm still fires.
- **`fallback.mp4`** — a recording of one clean successful LIVE run. If a live step hangs
  more than ~10s on the day, cut to this video and keep narrating (see `../demo.md`).

Both require a real GPU run, so they aren't committed — generate them in rehearsal.

---

## Recording the fallback video

`fallback.mp4` is the demo's last-resort safety net.
If any live step hangs more than ~10s on the day, cut to this video and keep narrating
(see `../demo.md`).

It is **not committed yet** because it can only be recorded from a real GPU run, which
needs your RunPod account + credits. Record it during your rehearsal:

## How to record it

1. Get a clean LIVE run working end to end:
   ```bash
   export WALL_MOCK=0
   flash login
   python wall.py --warm      # wait until both endpoints report "warm"
   python app.py
   ```
2. Open `http://localhost:8000` full-screen.
3. Screen-record (OBS, or Win+G Game Bar on Windows) while you:
   - paint once (let the wall fully fill),
   - switch to Embed, embed the docs, run one search.
4. Save the recording here as `docs/fallback.mp4`.
5. Commit it:
   ```bash
   git add -f docs/fallback.mp4
   git commit -m "Add fallback demo video"
   ```

Keep it short (under ~60s) and make sure the **Cost** figure is legible — that's the
closing shot.

> A MOCK run (`WALL_MOCK=1`) is a usable stand-in if you cannot get GPU time before the
> event: it looks identical on screen. Prefer a real LIVE recording if you can.
