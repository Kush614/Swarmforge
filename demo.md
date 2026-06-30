# Demo run-of-show — Teach & Swarm

Optimize for two things: **it runs**, and **every capability is visible on the
projector**. If any live step hangs more than ~10s, flip the safety net (or cut to
`docs/fallback.mp4`) and keep narrating. A smooth recording beats a stalled live run.

---

## Pre-demo (start 5+ minutes before)

```bash
# 1. real photos of your concept
#    -> drop 5-10 images in sample_images/

export WALL_MOCK=0
flash login
python wall.py --warm          # warms renderer (turbo + SD1.5) AND the LoRA trainer
```

While it warms:

- [ ] Set `COST_PER_WORKER_SEC` in `config.py` from today's RunPod pricing.
- [ ] Start the tunnel + export it: `cloudflared tunnel --url http://localhost:8000`
      then `export WALL_PUBLIC_URL=https://<tunnel>.trycloudflare.com`.
- [ ] `python app.py`, open the projector at the tunnel URL (or `localhost:8000`).
- [ ] **Do one full LIVE rehearsal run.** This (a) proves it works and (b) writes
      `docs/cached_adapter.b64` — your safety net.
- [ ] Screen-record that rehearsal as `docs/fallback.mp4`.
- [ ] Open the RunPod serverless dashboard in a tab (shows the 4 endpoints + live workers).

**Safety net:** if the live train runs long on stage, tick **"use cached adapter"** in
the UI (or set `WALL_USE_CACHED=1`) — Act 2 finishes instantly with the rehearsed adapter
and the swarm still fires. If everything is flaky, run the whole demo in MOCK
(`WALL_MOCK=1`); it looks identical on the projector.

---

## Live, in order

**0. Cold open.** "Last month Civitai trained 868,069 LoRAs on RunPod — each one
previewed with generated images, all of it bursty, unpredictable, scale-to-zero. That's
RunPod's creator input layer. We built a miniature of it you can watch run, live, in
twenty seconds — and we feed it with real web data via Bright Data."

**1. Act 1 — Before.** Make sure you're on the **Teach & Paint** tab. Hit **Teach &
Paint**. The filmstrip renders the base model's idea of `sks dog`. "That's the base model.
It has no clue what `sks dog` is. Those are nonsense."

**2. Act 2 — Train (the production story).** The training overlay counts up. Narrate:
"We just kicked off a LoRA fine-tune as a **queued GPU job** on RunPod. Data in — a few
photos — model out, in ~90 seconds. This is the thing you can't do on a closed API."
Point at the dashboard: one worker doing the training burst.

**3. Act 3 — After (the spectacle).** The overlay clears and the wall paints in. "Same
renderer, now loaded with the adapter we trained 90 seconds ago, fanned out across N
GPUs — each painting one tile in parallel." Point at **Workers** spiking, then draining
to 0. "...and they're already gone."

**4. The room (QR).** Show the **QR**. "Scan it." The audience hits the same URL and can
trigger a run from their phones. Watch the worker counter spike and drain again.

**5. The swap (optional, if time).** Switch to the **Embed** tab → **Embed docs** →
search. "Same fan-out — plan, scatter, scale to zero — but each worker now embeds a
document instead of painting a tile. One function body swapped. That's a production
retrieval pipeline on the exact same machinery."

**6. Close.** Point at the big **Cost** figure and the Civitai line. "We trained a LoRA
and generated a wall of previews for $0.0x, and it's already gone. Civitai does eight
hundred and sixty-eight thousand of these a month on RunPod. You just watched one — the
unit that fuels an entire creative platform. That's the workload serverless GPU was made
for: bursty, unpredictable, scale-to-zero, pennies each."

---

## After

```bash
flash undeploy     # delete the endpoints — they cost money until you do
```

Confirm the serverless dashboard is empty.
