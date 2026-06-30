"""
config.py — central knobs for the Self-Painting Wall demo.

Everything tunable for the live demo lives here so you can change pacing,
drama, and cost without hunting through wall.py / app.py. See README.md §Tuning.
"""

import os

from runpod_flash import GpuType

# ---------------------------------------------------------------------------
# Fan-out width / drama
# ---------------------------------------------------------------------------

# MAX_WORKERS — the upper bound of the GPU worker pool (the (0, N) in workers=(0, N)).
# Start modest; raise only once a small run is stable (RunPod account worker-capacity
# limits get hit fast when fanning out wide). See SPEC §7.
# NOTE: RunPod accounts have a max-worker QUOTA summed across ALL deployed endpoints
# (this account = 10). render + embed use this value; the trainer adds 2 and the planner
# 1, so keep this <= ~8 unless your quota is higher. Raise once quota allows.
MAX_WORKERS = int(os.getenv("WALL_MAX_WORKERS", "8"))

# GRID — tiles per side. Total tiles = GRID * GRID.
#   GRID=5  -> 25 tiles (1:1 with MAX_WORKERS=25, no queueing — smoothest first run)
#   GRID=6  -> 36 tiles (good drama/reliability balance — default)
#   GRID=8  -> 64 tiles (most dramatic; workers each paint ~2-3 tiles)
GRID = int(os.getenv("WALL_GRID", "6"))

# TILE_PX — pixel size of each square tile the GPU renders. 512 is the sweet spot
# for sd-turbo (its native size) and keeps per-tile render fast.
TILE_PX = int(os.getenv("WALL_TILE_PX", "512"))

# Full canvas size, derived. Browser displays it scaled to fit the projector.
CANVAS_PX = GRID * TILE_PX

# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------

# GPU_TYPE — the renderer GPU. RTX 4090 (24 GB) is cheap, fast, and fits sd-turbo
# with room to spare. Bump only if you swap in a model that won't fit. See the GPU
# table in the runpod-flash README for other options (L4, 5090, A100, ...).
GPU_TYPE = GpuType.NVIDIA_GEFORCE_RTX_4090

# CPU spec for the cheap planner endpoint. "cpu3c-1-2" = 1 vCPU / 2 GB.
CPU_TYPE = os.getenv("WALL_CPU_TYPE", "cpu3c-1-2")

# ---------------------------------------------------------------------------
# Models (these run on the REMOTE worker — declared in @Endpoint(dependencies=...))
# ---------------------------------------------------------------------------

# MODEL_ID — diffusion model for tiles. sd-turbo is single-step (1 inference step,
# guidance 0.0) and ~fast enough that a tile lands in ~1-2 s once warm. Keep it small
# for demo speed; quality is secondary to the painting-in-parallel spectacle.
MODEL_ID = os.getenv("WALL_MODEL_ID", "stabilityai/sd-turbo")

# EMBED_MODEL_ID — sentence-transformers model for the "real application" embed mode.
# all-MiniLM-L6-v2 is tiny (384-dim) and fast — same workers=(0, N) shape as render_tile.
EMBED_MODEL_ID = os.getenv("WALL_EMBED_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")

# ---------------------------------------------------------------------------
# Cost dashboard
# ---------------------------------------------------------------------------

# COST_PER_WORKER_SEC — used by the dashboard's running cost estimate.
# TODO(human): set this from CURRENT RunPod serverless pricing for GPU_TYPE before the
# demo — pricing changes, so verify at https://www.runpod.io/pricing. As of writing,
# RTX 4090 Flex serverless is roughly $0.00031-0.00044 /sec. Default below is a
# deliberately conservative placeholder so the dashboard shows a plausible number.
COST_PER_WORKER_SEC = float(os.getenv("WALL_COST_PER_WORKER_SEC", "0.00044"))

# IDLE_TIMEOUT — seconds a worker stays alive (billing) after its last job before
# scaling to zero. LOW protects your credits during a tight-budget demo (workers vanish
# fast); flashboot makes the occasional re-spin cheap. Bump up only if you see workers
# cold-starting between beats.
IDLE_TIMEOUT = int(os.getenv("WALL_IDLE_TIMEOUT", "8"))

# ---------------------------------------------------------------------------
# Run mode
# ---------------------------------------------------------------------------

# MOCK — when true, render_tile / embed_chunk are computed LOCALLY (solid-color tiles,
# random unit vectors) instead of deploying GPU workers. Zero cost, no credentials,
# instant. This is Phase 1 of the build plan kept permanently as:
#   * a rehearsal / offline-demo mode, and
#   * the safe fallback if the venue network or RunPod is unreachable.
# Set WALL_MOCK=0 (and have `flash login` done) for the real GPU run.
MOCK = os.getenv("WALL_MOCK", "1") not in ("0", "false", "False", "")

# Server bind for the local orchestrator (app.py).
HOST = os.getenv("WALL_HOST", "0.0.0.0")
PORT = int(os.getenv("WALL_PORT", "8000"))

# PUBLIC_URL — the URL the QR code points at so a phone can trigger /paint.
# For the live demo this is typically a tunnel to this machine, e.g.
#   cloudflared tunnel --url http://localhost:8000
# TODO(human): set WALL_PUBLIC_URL to your tunnel URL before the QR beat. If unset,
# the QR falls back to this machine's LAN address (works if phones share the wifi).
PUBLIC_URL = os.getenv("WALL_PUBLIC_URL", "")

# Top-k results returned by /search in embed mode.
SEARCH_TOP_K = int(os.getenv("WALL_SEARCH_TOP_K", "5"))

# Folder of .txt/.md files to embed in embed mode.
DOCS_DIR = os.getenv("WALL_DOCS_DIR", "sample_docs")

# ---------------------------------------------------------------------------
# Fine-tune hybrid  (the headline demo: teach a concept live, then swarm-paint it)
# ---------------------------------------------------------------------------

# Base model for the fine-tune flow. LoRA/DreamBooth wants a trainable base, so this is
# SD 1.5 (the well-trodden path) rather than the turbo model used for pure paint mode.
# Both the "before" preview and the "after" swarm render on THIS base so the adapter
# actually applies. Slower per tile than sd-turbo (~20-30 steps); use a smaller grid.
FINETUNE_BASE = os.getenv("WALL_FINETUNE_BASE", "runwayml/stable-diffusion-v1-5")

# The concept token we teach. "sks" is the classic rare-token trick so the model has no
# prior for it — making the before/after obvious.
CONCEPT = os.getenv("WALL_CONCEPT", "sks dog")

# Prompt template woven around the concept for both preview and swarm.
PROMPT_TEMPLATE = os.getenv("WALL_PROMPT_TEMPLATE", "a photo of {concept}")

# LoRA training steps. Low = fast (good for a live ~60-90s beat), at some quality cost.
TRAIN_STEPS = int(os.getenv("WALL_TRAIN_STEPS", "150"))

# Folder of training images for the concept (drop 5-10 photos). Optional in MOCK.
IMAGES_DIR = os.getenv("WALL_IMAGES_DIR", "sample_images")

# Number of "before" base-model preview tiles in Act 1.
BASE_PREVIEW = int(os.getenv("WALL_BASE_PREVIEW", "6"))

# Grid used specifically for the fine-tune swarm (Act 3). Smaller than paint-mode GRID
# because SD 1.5 multi-step tiles are heavier than turbo tiles.
FINETUNE_GRID = int(os.getenv("WALL_FINETUNE_GRID", "5"))

# SAFETY NET: skip live training and load a cached adapter from rehearsal. Flip this on
# stage if live training runs long or fails — the swarm still fires. See README.
USE_CACHED = os.getenv("WALL_USE_CACHED", "0") in ("1", "true", "True")

# Where a freshly-trained adapter is cached (and loaded from when USE_CACHED).
CACHED_ADAPTER = os.getenv("WALL_CACHED_ADAPTER", "docs/cached_adapter.b64")

# ---------------------------------------------------------------------------
# Media mode  (video-production: transcribe a pile of clips in parallel, then
# search a phrase and jump to the exact clip + timecode)
# ---------------------------------------------------------------------------

# faster-whisper model size. "base" is a good speed/quality balance for a demo;
# "small"/"medium" are sharper but slower. Runs on the media-transcriber GPU endpoint.
WHISPER_MODEL = os.getenv("WALL_WHISPER_MODEL", "base")

# Default clips to transcribe when none are supplied (short public samples). Audio OR
# video both work — faster-whisper decodes the audio track of a video. Bright Data can
# also discover/serve media URLs past bot-walls; pass those in instead.
SAMPLE_MEDIA = [u for u in os.getenv("WALL_SAMPLE_MEDIA", ",".join([
    "https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav",
    "https://github.com/Uberi/speech_recognition/raw/master/examples/english.wav",
    "https://github.com/realpython/python-speech-recognition/raw/master/audio_files/harvard.wav",
    "https://github.com/realpython/python-speech-recognition/raw/master/audio_files/jackhammer.wav",
    "https://github.com/mozilla/DeepSpeech/raw/master/data/smoke_test/new-home-in-the-stars-16k.wav",
])).split(",") if u.strip()]
