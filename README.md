# SwarmForge

**Point it at the open web and watch a swarm of GPUs erupt, do real work in parallel, and vanish for pennies.**

SwarmForge is a live demonstration of serverless GPU as a visible spectacle, built on two
clouds: **[Bright Data](https://brightdata.com)** ingests real web data (pages and media,
past bot-walls) and **[RunPod Flash](https://github.com/runpod/flash)** fans that data out
across many GPU workers that scale from zero to N and back to zero in seconds. A
projector-grade dark UI paints results live, with a running cost figure whose closing line
makes the scale-to-zero economics impossible to miss.

> One generic pipeline, made visible:
> **cheap CPU split → wide GPU fan-out → scale to zero.**

## The pitch

Every team renting GPUs pays for idle. A box sits warm overnight so it's ready for a burst
that lasts ten seconds. SwarmForge is the opposite: nothing runs until work arrives, then a
swarm erupts, finishes in seconds, and bills you for exactly those seconds. Running 25 GPUs
for 10 seconds costs the same as one GPU for 250 seconds — but it's done 25× faster, and
then it's *gone*.

The demo makes that economics impossible to miss. You point it at the live web, watch real
GPU workers spike from zero, do real work — generate images, embed pages, transcribe clips —
and drain back to zero, with a cost counter that closes on *"running this 24/7 would be
$X/mo; you paid $0.0Y."* It's the workload serverless GPU was built for: bursty,
unpredictable, embarrassingly parallel. The same pattern that powers
[Civitai's 868,069 LoRA trainings a month on RunPod](https://www.runpod.io/case-studies/civitai).

## Under the hood (tech)

- **RunPod Flash** — serverless GPU. Each worker is a Python `@Endpoint` with
  `workers=(0, N)`; no Dockerfile, no idle cost. Endpoints provision on first call and
  scale to zero on `idle_timeout`.
- **Bright Data** — web-scale ingestion. Web Unlocker / API fetches pages and media past
  bot-walls so the swarm has real data to chew on.
- **Models**: `sd-turbo` (single-step diffusion tiles), SD-1.5 + **PEFT LoRA** (DreamBooth
  fine-tune), `all-MiniLM-L6-v2` (embeddings), **faster-whisper** (transcription).
- **Orchestrator**: an `aiohttp` server streams results to the browser over Server-Sent
  Events; `asyncio.as_completed` drives the fan-out so tiles/clips appear the instant their
  worker finishes; a numpy cosine index serves search.
- **Resilience**: every run is recorded; a self-contained `offline.html` replays any mode
  with no server, no network, no GPU.

## Real-world use cases

This isn't a toy — each mode maps to a workload people pay for:

- **Custom image generation at scale** (Teach & Paint) — creator platforms (Civitai-style),
  brand/product art, per-customer fine-tuned style libraries. Train on demand, render a
  preview wall, ship the adapter.
- **Build a searchable knowledge base from the live web** (Embed) — point it at competitor
  sites, news, docs, or a Slack/Notion export; index thousands of chunks in one burst;
  serve a low-latency `/search`. The ingestion half of any RAG pipeline.
- **Footage / media logging** (Media) — video editors and journalists drop hours of clips,
  the swarm transcribes them in parallel, and you search a phrase to jump to the exact
  timecode. Also podcast/meeting search and broadcast compliance monitoring.
- **The general pattern** — any bursty, parallel GPU job that's wasteful on always-on
  infra: batch upscaling/background-removal, document OCR, embedding refreshes, mass
  transcription. Swap the function body, keep the orchestration.

```
  browser (paints tiles / logs clips)
        ▲  Server-Sent Events
        │
   app.py  ── local orchestrator + web server ──┐
        │                                        │ calls Flash endpoints
        │                          ┌─────────────┴──────────────┐
   Bright Data                tile-planner (CPU)          render_tile ×N (GPU)
   scrape(url) / images()     chunk-embedder ×N (GPU)     media-transcriber ×N (GPU)
        │                          lora-trainer (GPU)
        └── real web data ─────────► deployed on RunPod, workers=(0,N), idle→0
```

---

## What it does — four modes, one fan-out

| Mode | What you see | Endpoints used |
|------|--------------|----------------|
| **Teach & Paint** | Train a lightweight LoRA on a few images, then a swarm generates a **wall of preview images** of the new concept — the exact "train a model, preview with images" workload [Civitai runs 868,069×/month on RunPod](https://www.runpod.io/case-studies/civitai). | `lora-trainer`, `tile-renderer` |
| **Paint** | A grid paints itself, one fast turbo-diffusion tile per GPU worker, then scales to zero. | `tile-planner`, `tile-renderer` |
| **Embed** | Bright Data scrapes any URL → the swarm embeds it in parallel → instant semantic search. | `chunk-embedder` |
| **Media** (footage logger) | Drop audio/video clips → the swarm transcribes each in parallel (Whisper) → search a phrase and **jump to the exact timecode**. | `media-transcriber`, `chunk-embedder` |

All four run on the **same** fan-out driver — only the function body each worker runs
changes. That is the whole point: Flash is an orchestration layer, not just a deploy
shortcut.

### Two clouds, for real

- **Bright Data** is the ingestion layer (`brightdata.py`): `scrape(url)` → text chunks,
  `images(query)` → photos, fetched past bot-walls. Feeds Embed (scrape → search) and
  Teach (scrape → train).
- **RunPod Flash** is the compute engine (`wall.py`): every `@Endpoint` is a serverless
  GPU/CPU worker pool with `workers=(0, N)` that scales to zero when idle.

---

## Run modes

Set with `WALL_MOCK`:

- **MOCK** (`WALL_MOCK=1`, default) — render/transcribe/embed run **locally** (solid tiles,
  synthetic transcripts, random vectors). Zero cost, no GPU, no credentials, instant. This
  is the rehearsal mode *and* the on-stage safety net.
- **LIVE** (`WALL_MOCK=0`) — deploys and runs real RunPod GPU workers. Needs `flash login`
  (or `RUNPOD_API_KEY`) and quota.

Everything below the GPU boundary is identical in both modes, so what you rehearse in MOCK
is exactly what runs LIVE.

### Verified real output (LIVE)

The media swarm transcribed 5 real clips in parallel — genuine Whisper output:

```
[jfk.wav]      And so my fellow Americans ask not what your country can do for you,
               ask what you can do for your country.
[harvard.wav]  The stale smell of old beer lingers. It takes heat to bring out the odor...
[new-home-in-the-stars.wav]  We must find a new home in the stars.
```

---

## Offline replay — runs with the server closed

Every run is recorded. Two layers of "works without compute":

1. **▶ Replay button** (served app) — replays the last run from cache with zero GPU.
2. **`offline.html`** — a single self-contained file (built by `build_offline.py`) with all
   modes' results embedded. Open it by double-click (`file://`), no server, no internet,
   no GPU — every tab replays its recorded run. The bulletproof demo fallback.

```bash
python build_offline.py     # bundles static/cache/*.json into offline.html
# then just open offline.html in a browser
```

---

## Setup

Python 3.10–3.13. (Built and tested on Windows + Python 3.13.)

```bash
pip install -r requirements.txt      # local deps only: runpod-flash, aiohttp, numpy
cp .env.example .env                 # then fill in keys (see below)
```

The heavy GPU deps (torch, diffusers, peft, sentence-transformers, faster-whisper) are
**not** local — they're declared per-endpoint in `@Endpoint(dependencies=[...])` and
install on the remote RunPod workers.

### Environment (`.env`)

```bash
RUNPOD_API_KEY=...                 # runpod.io/console/user/settings  (or `flash login`)
BRIGHTDATA_API_TOKEN=...           # Bright Data API token
BRIGHTDATA_ZONE=web_unlocker1      # your Web Unlocker zone name (required for live scraping)
WALL_COST_PER_WORKER_SEC=0.00044   # set to your GPU's real RunPod price
```

Without Bright Data creds, scraping falls back to MOCK; the rest still runs.

---

## Running it

```bash
# MOCK (free, instant) — start here
python app.py                      # open http://localhost:8000

# LIVE (real GPUs)
flash login
$env:WALL_MOCK="0"                 # PowerShell ($env:); bash: export WALL_MOCK=0
python wall.py --warm              # pre-warm all endpoints in parallel (cold start = minutes)
python app.py

# De-risk a live run cheaply (a few cents): measures real cold start + cost
python smoke_live.py
```

CLI shortcuts:

```bash
python wall.py "a neon city at night"   # paint
python wall.py --mode=finetune "sks dog"# teach & paint
python wall.py --mode=embed             # embed sample_docs + demo search
python media_live.py                    # live transcribe the sample clips
```

---

## Known constraints (designed around)

- **Worker quota.** RunPod caps max workers summed across all *deployed* endpoints (this
  account: 10). `MAX_WORKERS` defaults to 8 so a single demo path fits; switching modes may
  need a `flash undeploy` between. Raise once quota allows.
- **Cold start is minutes**, not seconds, for torch/diffusers endpoints (the first call
  builds the worker + loads weights). Always `--warm` 5+ min before a live demo. CPU cold
  start is ~10s.
- **Endpoints are self-contained.** A remote worker runs only the function body — module
  globals like `config` are not shipped, so every value an endpoint needs is passed as an
  argument.
- **Idle billing.** `IDLE_TIMEOUT` (default 8s) scales workers to zero fast after a burst to
  protect credits.
- **Windows + UTF-8.** The flash SDK streams Unicode logs; `wall.py` forces UTF-8 so they
  print cleanly on Windows.

---

## Repo layout

```
wall.py            # Flash endpoints (planner/renderer/embedder/trainer/transcriber) + drivers + CLI
app.py             # local orchestrator + web server (SSE, /paint /teach /embed /ingest /transcribe /search)
brightdata.py      # Bright Data ingestion: scrape(url) -> text, images(query) -> photos
config.py          # every knob (fan-out width, GPU, models, cost/sec, mock + safety flags)
build_offline.py   # bundle cached runs -> self-contained offline.html
smoke_live.py      # one-shot LIVE de-risk (cold start + cost)
media_live.py      # live transcription proof
static/index.html  # projector UI (dark Gumroad theme): canvas + dashboard + QR + replay
static/cache/      # recorded runs per mode (for offline replay)
offline.html       # self-contained, no-server replay of all modes
sample_docs/       # text for embed mode
sample_images/     # drop training photos here for the fine-tune
demo.md            # live run-of-show
```

---

## Tuning (`config.py`)

`MAX_WORKERS` (8) · `GRID` (6) · `FINETUNE_GRID` (5) · `TRAIN_STEPS` (150) ·
`GPU_TYPE` (RTX 4090) · `MODEL_ID` (sd-turbo) · `EMBED_MODEL_ID` (MiniLM) ·
`WHISPER_MODEL` (base) · `COST_PER_WORKER_SEC` · `IDLE_TIMEOUT` (8) · `USE_CACHED`.
All are also env vars (`WALL_*`).

## Teardown (endpoints cost money until deleted)

```bash
flash undeploy --all --force
```

Then confirm the [serverless dashboard](https://www.runpod.io/console/serverless) is empty.

---

Built for a RunPod × Bright Data hackathon. Inspired by Civitai's RunPod-powered LoRA
training pipeline — SwarmForge is an original demo, not affiliated with Civitai.
