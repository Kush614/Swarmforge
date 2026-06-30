"""
wall.py — the Self-Painting Wall orchestrator.

Contains the three Flash endpoints (one CPU planner, two GPU workers) and the async
fan-out drivers that call them. The whole point of this file is to make ONE generic
pipeline visible:

        cheap CPU split  ->  wide GPU fan-out  ->  scale back to zero

`paint()` renders an image as a grid of tiles, one worker per tile. `embed_folder()` /
`search()` reuse the *same* fan-out shape to embed a folder of documents — proving the
wall and a production retrieval pipeline are the same machinery. Compare the two drivers
below: they are deliberately near-identical.

Run modes:
  * MOCK (default)  — render_tile / embed_chunk run LOCALLY (solid tiles, random vectors).
                      Zero cost, no GPU, no credentials. Rehearsal + offline fallback.
  * LIVE            — set WALL_MOCK=0 and `flash login`. Calls deploy real RunPod workers.

CLI:
  python wall.py "a neon city at night"      # paint (console stream)
  python wall.py --mode=embed                # embed sample_docs/, then a demo search
  python wall.py --warm                       # pre-warm GPU workers before the demo
"""

import os
import sys

# Windows fix: the flash SDK streams remote logs containing Unicode (e.g. box-drawing
# chars). On Windows the default cp1252 console can't encode them and raises a spurious
# 'charmap' error. Force UTF-8 on our streams so live runs print cleanly.
os.environ.setdefault("PYTHONUTF8", "1")
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Direct calls to @Endpoint functions only work inside `flash dev` (which sets a flash
# context) OR when live provisioning is enabled. For standalone runs (python wall.py,
# python app.py) we opt into live provisioning so the SDK deploys + calls real workers.
# setdefault — so `flash deploy` (which sets this to "false") is never overridden.
os.environ.setdefault("FLASH_IS_LIVE_PROVISIONING", "true")

import asyncio
import base64
import random
import struct
import time
import zlib
from pathlib import Path
from typing import Awaitable, Callable, Optional

import config
from runpod_flash import Endpoint

# An emit callback receives one event dict and ships it somewhere (the SSE hub in app.py,
# or stdout in the CLI). Always awaited.
Emit = Callable[[dict], Awaitable[None]]


# ===========================================================================
# Flash endpoints  (deployed to RunPod; heavy imports live INSIDE each function)
# ===========================================================================

@Endpoint(name="tile-planner", cpu=config.CPU_TYPE, workers=(0, 1))
async def plan_tiles(width: int, height: int, grid: int) -> dict:
    """CPU endpoint — splits a canvas into a grid of tile specs.

    Trivial compute on purpose: it exists to demonstrate the cheap CPU preprocessing
    stage that feeds the expensive GPU fan-out. Many cheap CPU planners could feed one
    big GPU pool — the two scale independently.
    """
    tw = width // grid
    th = height // grid
    tiles = []
    for r in range(grid):
        for c in range(grid):
            tiles.append(
                {"id": r * grid + c, "x": c * tw, "y": r * th, "w": tw, "h": th}
            )
    return {"tiles": tiles, "tile_w": tw, "tile_h": th}


@Endpoint(
    name="tile-renderer",
    gpu=config.GPU_TYPE,
    workers=(0, config.MAX_WORKERS),          # (min, max): scale to zero when idle
    idle_timeout=config.IDLE_TIMEOUT,         # vanish fast after the burst (saves credits)
    dependencies=["torch", "diffusers", "transformers", "accelerate", "peft", "pillow"],
)
async def render_tile(tile: dict, prompt: str, seed: int, model_id: str,
                      adapter_b64: str = None, turbo: bool = True) -> dict:
    """GPU endpoint — render ONE image tile.

    Self-contained: every value it needs (model_id, turbo flag) is an ARGUMENT, because
    the remote worker runs only this function body — module-level imports like `config`
    are NOT available here. Two modes, same endpoint:
      * paint mode    — turbo model, single step.
      * fine-tune mode— SD1.5 + adapter_b64 -> load the freshly trained LoRA and render
                        the taught concept (the "after" swarm uses the model we just made).

    All heavy imports are inside the function. Pipelines are cached per model_id in a
    module global so warm workers reuse them.
    """
    import base64 as _b64
    import io
    import tempfile

    import torch
    from diffusers import AutoPipelineForText2Image

    global _PIPES, _LOADED_ADAPTER
    try:
        _PIPES
    except NameError:
        _PIPES = {}
        _LOADED_ADAPTER = {}
    if model_id not in _PIPES:
        _PIPES[model_id] = AutoPipelineForText2Image.from_pretrained(
            model_id, torch_dtype=torch.float16
        ).to("cuda")
        _LOADED_ADAPTER[model_id] = None
    pipe = _PIPES[model_id]

    # Load the LoRA adapter once per (base_model, adapter) — warm workers keep it loaded.
    if adapter_b64 and _LOADED_ADAPTER.get(model_id) != id(adapter_b64):
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            f.write(_b64.b64decode(adapter_b64))
            adapter_path = f.name
        try:
            pipe.unload_lora_weights()
        except Exception:
            pass
        pipe.load_lora_weights(adapter_path)
        _LOADED_ADAPTER[model_id] = id(adapter_b64)

    generator = torch.Generator("cuda").manual_seed(seed)
    image = pipe(
        prompt=prompt,
        num_inference_steps=1 if turbo else 25,
        guidance_scale=0.0 if turbo else 7.5,
        height=tile["h"],
        width=tile["w"],
        generator=generator,
    ).images[0]

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    png_b64 = _b64.b64encode(buf.getvalue()).decode("ascii")
    return {
        "id": tile["id"], "x": tile["x"], "y": tile["y"],
        "w": tile["w"], "h": tile["h"], "png_b64": png_b64,
    }


@Endpoint(
    name="lora-trainer",
    gpu=config.GPU_TYPE,
    workers=(0, 2),                           # one heavy queued job — the "burst" beat
    idle_timeout=config.IDLE_TIMEOUT,
    dependencies=["torch", "diffusers", "transformers", "accelerate", "peft",
                  "datasets", "pillow"],
    execution_timeout_ms=15 * 60 * 1000,      # training can run minutes; don't time out
)
async def train_lora(images_b64: list, concept: str, steps: int,
                     base_model: str, prompt_template: str) -> dict:
    """GPU endpoint — DreamBooth-style LoRA fine-tune on a handful of images.

    Self-contained (base_model + prompt_template are arguments — no remote `config`).
    Returns the trained adapter as base64 safetensors so the renderer can load it with
    no shared storage. Compact loop for a live demo: tune steps/LR before the day.
    """
    import base64 as _b64
    import io
    import tempfile

    import torch
    from diffusers import StableDiffusionPipeline
    from peft import LoraConfig, get_peft_model_state_dict
    from PIL import Image

    device = "cuda"
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model, torch_dtype=torch.float32, safety_checker=None
    ).to(device)
    unet, vae, text_encoder = pipe.unet, pipe.vae, pipe.text_encoder
    tokenizer, scheduler = pipe.tokenizer, pipe.scheduler
    vae.requires_grad_(False); text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # attach LoRA to the UNet attention layers
    lora = LoraConfig(r=8, lora_alpha=8,
                      target_modules=["to_q", "to_k", "to_v", "to_out.0"])
    unet.add_adapter(lora)
    params = [p for p in unet.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-4)

    imgs = [Image.open(io.BytesIO(_b64.b64decode(b))).convert("RGB").resize((512, 512))
            for b in images_b64]
    prompt = prompt_template.format(concept=concept)
    ids = tokenizer(prompt, padding="max_length", truncation=True,
                    max_length=tokenizer.model_max_length, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        enc = text_encoder(ids)[0]

    import numpy as np
    unet.train()
    for step in range(steps):
        img = imgs[step % len(imgs)]
        x = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().div(127.5).sub(1.0)
        x = x.unsqueeze(0).to(device)
        with torch.no_grad():
            latents = vae.encode(x).latent_dist.sample() * vae.config.scaling_factor
        noise = torch.randn_like(latents)
        t = torch.randint(0, scheduler.config.num_train_timesteps, (1,), device=device).long()
        noisy = scheduler.add_noise(latents, noise, t)
        pred = unet(noisy, t, encoder_hidden_states=enc).sample
        loss = torch.nn.functional.mse_loss(pred, noise)
        opt.zero_grad(); loss.backward(); opt.step()

    # export just the LoRA weights as safetensors
    from safetensors.torch import save_file
    state = get_peft_model_state_dict(unet)
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        save_file({k: v.contiguous().cpu() for k, v in state.items()}, f.name)
        adapter_bytes = open(f.name, "rb").read()
    return {"adapter_b64": _b64.b64encode(adapter_bytes).decode("ascii"),
            "concept": concept, "steps": steps, "final_loss": float(loss.item())}


@Endpoint(
    name="chunk-embedder",
    gpu=config.GPU_TYPE,
    workers=(0, config.MAX_WORKERS),          # SAME shape as render_tile — that's the point
    idle_timeout=config.IDLE_TIMEOUT,
    dependencies=["sentence-transformers"],
)
async def embed_chunk(text: str, model_id: str) -> dict:
    """GPU endpoint — embed ONE text chunk into a vector.

    Self-contained (model_id is an argument — no module-level `config` on the remote).
    Structurally identical to render_tile: same workers=(0, N), heavy import inside,
    one unit of work in, one result out. Only the body differs.
    """
    global _EMBED_MODEL
    try:
        _EMBED_MODEL
    except NameError:
        _EMBED_MODEL = None
    if _EMBED_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _EMBED_MODEL = SentenceTransformer(model_id)

    vector = _EMBED_MODEL.encode(text, normalize_embeddings=True)
    return {"vector": [float(x) for x in vector]}


@Endpoint(
    name="media-transcriber",
    gpu=config.GPU_TYPE,
    workers=(0, config.MAX_WORKERS),          # one clip per worker — the same fan-out
    idle_timeout=config.IDLE_TIMEOUT,
    dependencies=["faster-whisper"],
)
async def transcribe(media_url: str, model_id: str) -> dict:
    """GPU endpoint — transcribe ONE audio/video clip into timestamped segments.

    Self-contained (model_id is an argument). faster-whisper decodes the audio track of
    audio OR video, so this powers the footage-logger: each clip → its own worker → a
    searchable transcript with timecodes. Whisper model cached on the warm worker.
    """
    import tempfile
    import urllib.request

    from faster_whisper import WhisperModel

    global _WHISPER
    try:
        _WHISPER
    except NameError:
        _WHISPER = None
    if _WHISPER is None:
        _WHISPER = WhisperModel(model_id, device="cuda", compute_type="float16")

    path = tempfile.mktemp(suffix=".media")
    urllib.request.urlretrieve(media_url, path)   # public clip, or a Bright-Data URL
    segments, info = _WHISPER.transcribe(path, beam_size=1)
    segs = [{"start": round(s.start, 1), "end": round(s.end, 1), "text": s.text.strip()}
            for s in segments]
    return {"url": media_url, "duration": round(info.duration, 1), "segments": segs}


# ===========================================================================
# Mock implementations  (MOCK mode — local, instant, free, no credentials)
# ===========================================================================

def _solid_png(rgb: tuple, w: int, h: int) -> str:
    """Build a base64 PNG of a solid color with stdlib only (no Pillow locally)."""
    r, g, b = rgb
    raw = bytearray()
    row = bytes((r, g, b)) * w
    for _ in range(h):
        raw.append(0)            # filter type 0 per scanline
        raw.extend(row)

    def chunk(tag: bytes, data: bytes) -> bytes:
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c))

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)   # 8-bit, truecolor RGB
    idat = zlib.compress(bytes(raw), 6)
    png = sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")
    return base64.b64encode(png).decode("ascii")


def _hue_for(seed: int, prompt: str) -> tuple:
    """Deterministic pleasant color from seed+prompt so mock tiles look intentional."""
    rng = random.Random(f"{prompt}:{seed}")
    # bias toward saturated, mid-bright colors so the wall pops on a dark bg
    import colorsys
    hh, ss, vv = rng.random(), 0.55 + rng.random() * 0.4, 0.6 + rng.random() * 0.35
    r, g, b = colorsys.hsv_to_rgb(hh, ss, vv)
    return (int(r * 255), int(g * 255), int(b * 255))


async def _mock_render_tile(tile: dict, prompt: str, seed: int, model_id: str = None,
                            adapter_b64: str = None, turbo: bool = True) -> dict:
    # Simulate variable GPU render time so the canvas fills progressively and the
    # worker-count curve looks real on the dashboard.
    await asyncio.sleep(0.3 + random.random() * 1.4)
    rgb = _hue_for(seed, prompt)
    if adapter_b64 is None and not turbo:
        # "before": base model has no idea what the concept is — render muted/greyish
        # so the after/before contrast is obvious on the projector.
        g = sum(rgb) // 3
        rgb = (int((rgb[0] + g) / 2 * 0.55), int((rgb[1] + g) / 2 * 0.55),
               int((rgb[2] + g) / 2 * 0.55))
    return {
        "id": tile["id"], "x": tile["x"], "y": tile["y"],
        "w": tile["w"], "h": tile["h"],
        "png_b64": _solid_png(rgb, tile["w"], tile["h"]),
    }


async def _mock_train_lora(images_b64: list, concept: str, steps: int,
                           base_model: str = None, prompt_template: str = None) -> dict:
    # Simulate a fine-tune: a few seconds of "training" so the live beat has weight.
    # Returns a fake adapter token so the renderer's adapter path is exercised.
    await asyncio.sleep(min(6.0, 0.02 * steps))
    token = base64.b64encode(f"mock-lora:{concept}:{steps}".encode()).decode()
    return {"adapter_b64": token, "concept": concept, "steps": steps,
            "final_loss": round(0.18 + random.random() * 0.05, 4)}


_MOCK_LINES = [
    "Okay so the first thing we need to talk about is the budget.",
    "I think the second act drags a little, we should cut for pace.",
    "Roll the B-roll of the city at sunset right here.",
    "And that's a wrap on scene twelve, great work everyone.",
    "The lighting in this shot is exactly what we wanted.",
    "Can we get a closer mic on the interview subject next time.",
    "Let's match the color grade to the reference we pulled earlier.",
    "This is the soundbite we open the trailer with.",
]


async def _mock_transcribe(media_url: str, model_id: str = None) -> dict:
    # Simulate variable transcription time + return a few timestamped segments so the
    # footage-logger and timecode search are fully demoable offline.
    await asyncio.sleep(0.5 + random.random() * 1.8)
    rng = random.Random(media_url)
    n = 2 + rng.randint(0, 2)
    segs, t = [], 0.0
    for _ in range(n):
        dur = round(2 + rng.random() * 4, 1)
        segs.append({"start": round(t, 1), "end": round(t + dur, 1),
                     "text": rng.choice(_MOCK_LINES)})
        t += dur
    return {"url": media_url, "duration": round(t, 1), "segments": segs}


async def _mock_embed_chunk(text: str, model_id: str = None) -> dict:
    await asyncio.sleep(0.2 + random.random() * 0.8)
    rng = random.Random(text)
    import math
    v = [rng.gauss(0, 1) for _ in range(384)]
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return {"vector": [x / n for x in v]}


def _plan_tiles_local(width: int, height: int, grid: int) -> dict:
    tw, th = width // grid, height // grid
    tiles = [
        {"id": r * grid + c, "x": c * tw, "y": r * th, "w": tw, "h": th}
        for r in range(grid) for c in range(grid)
    ]
    return {"tiles": tiles, "tile_w": tw, "tile_h": th}


# ===========================================================================
# Cost / worker tracker  (drives the dashboard)
# ===========================================================================

class _Meter:
    """Integrates the active-worker curve over time to estimate cost.

    cost = COST_PER_WORKER_SEC * integral(active_workers dt). active_workers is the
    number of in-flight jobs, capped at MAX_WORKERS (you never run more workers than the
    pool). This produces the spike-then-drain shape and a penny-scale final bill.
    """

    def __init__(self):
        self.inflight = 0
        self.done = 0
        self.total = 0
        self.worker_seconds = 0.0
        self.t0 = time.monotonic()
        self._last = self.t0

    @property
    def active(self) -> int:
        return min(self.inflight, config.MAX_WORKERS)

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.t0

    @property
    def cost(self) -> float:
        return self.worker_seconds * config.COST_PER_WORKER_SEC

    def accrue(self):
        now = time.monotonic()
        self.worker_seconds += self.active * (now - self._last)
        self._last = now

    def stats(self, kind: str = "stats") -> dict:
        return {
            "type": kind, "workers": self.active, "done": self.done,
            "total": self.total, "elapsed": round(self.elapsed, 1),
            "cost": round(self.cost, 4),
            "worker_seconds": round(self.worker_seconds, 1),
            # what these workers would cost running 24/7 — the closing-line punch
            "always_on_month": round(
                config.COST_PER_WORKER_SEC * config.MAX_WORKERS * 2592000, 0),
        }


async def _run_with_meter(coro_fn, meter: _Meter):
    """Wrap one unit of work so the meter tracks it as a busy worker."""
    meter.inflight += 1
    try:
        return await coro_fn()
    finally:
        meter.inflight -= 1
        meter.done += 1


# ===========================================================================
# Driver #1 — paint  (image fan-out)
# ===========================================================================

async def paint(prompt: str, emit: Emit, seed_base: int = 0) -> dict:
    """Plan a grid, fan out one render_tile per cell, stream tiles back as they finish.

    This is the canonical fan-out driver. embed_folder() below mirrors it line for line.
    """
    meter = _Meter()

    # 1. cheap CPU split
    if config.MOCK:
        plan = _plan_tiles_local(config.CANVAS_PX, config.CANVAS_PX, config.GRID)
    else:
        plan = await plan_tiles(config.CANVAS_PX, config.CANVAS_PX, config.GRID)
    tiles = plan["tiles"]
    meter.total = len(tiles)

    await emit({
        "type": "start", "prompt": prompt, "total": meter.total,
        "grid": config.GRID, "canvas": config.CANVAS_PX, "tile_px": config.TILE_PX,
        "max_workers": config.MAX_WORKERS,
    })

    render = _mock_render_tile if config.MOCK else render_tile

    # One unit of work. Partial failure is contained HERE: a dead worker returns a
    # positioned placeholder tile (ok=False) so the wall paints a marker and never aborts.
    async def _one(t, i):
        try:
            r = await render(t, prompt, seed_base + i, config.MODEL_ID, None, True)
            r["ok"] = True
            return r
        except Exception as e:
            return {"id": t["id"], "x": t["x"], "y": t["y"],
                    "w": t["w"], "h": t["h"], "ok": False, "error": str(e)}

    # 2. fan out — every tile starts at once; the pool caps real concurrency at N
    jobs = [
        _run_with_meter(lambda t=t, i=i: _one(t, i), meter)
        for i, t in enumerate(tiles)
    ]

    # background ticker keeps the dashboard live (cost integrates continuously)
    ticker = asyncio.create_task(_ticker(emit, meter))

    # 3. consume as each worker finishes -> paint immediately
    for fut in asyncio.as_completed(jobs):
        tile = await fut
        tile["type"] = "tile"
        await emit(tile)
        meter.accrue()
        await emit(meter.stats())

    ticker.cancel()
    final = meter.stats("done")
    await emit(final)
    return final


async def _ticker(emit: Emit, meter: _Meter, interval: float = 0.25):
    """Emit live stats on a timer so elapsed/cost move even between tile completions."""
    try:
        while True:
            await asyncio.sleep(interval)
            meter.accrue()
            await emit(meter.stats())
    except asyncio.CancelledError:
        pass


# ===========================================================================
# Driver #2 — embed  (document fan-out)  — note the structural twin of paint()
# ===========================================================================

def _chunk_docs(docs_dir: str, words_per_chunk: int = 60) -> list:
    """Cheap CPU split: turn a folder of .txt/.md into a list of chunk specs.

    This is the embed-mode analogue of plan_tiles(): the cheap preprocessing stage.
    """
    chunks = []
    cid = 0
    for path in sorted(Path(docs_dir).glob("**/*")):
        if path.suffix.lower() not in (".txt", ".md"):
            continue
        words = path.read_text(encoding="utf-8", errors="ignore").split()
        for i in range(0, len(words), words_per_chunk):
            text = " ".join(words[i:i + words_per_chunk])
            if text.strip():
                chunks.append({"id": cid, "file": path.name, "text": text})
                cid += 1
    return chunks


async def embed_folder(docs_dir: str, emit: Emit) -> dict:
    """Embed every chunk of a document folder via the SAME fan-out as paint().

    Compare to paint(): plan -> fan out -> consume-as-completed -> scale to zero. The
    only thing that changed is the function body each worker runs. That is the entire
    pitch of the demo.
    """
    chunks = _chunk_docs(docs_dir)          # 1. cheap CPU split (plan_tiles' twin)
    return await _embed_chunks(chunks, emit, source=docs_dir)


async def _embed_chunks(chunks: list, emit: Emit, source: str = "") -> dict:
    """Shared embed fan-out used by embed_folder() AND Bright Data ingest()."""
    meter = _Meter()
    meter.total = len(chunks)
    await emit({"type": "embed_start", "total": meter.total, "docs_dir": source})

    embed = _mock_embed_chunk if config.MOCK else embed_chunk

    # Wrap so each result carries its chunk id — as_completed unorders results, and we
    # need to pair every vector back to its chunk when building the index.
    async def _embed_one(c):
        r = await embed(c["text"], config.EMBED_MODEL_ID)
        return {"id": c["id"], "vector": r["vector"]}

    # 2. fan out — one embed_chunk per chunk  (render_tile's twin)
    jobs = [_run_with_meter(lambda c=c: _embed_one(c), meter) for c in chunks]
    ticker = asyncio.create_task(_ticker(emit, meter))

    # 3. consume as completed, collect vectors into an in-memory index
    import numpy as np
    vectors = {}
    for fut in asyncio.as_completed(jobs):
        try:
            r = await fut
            vectors[r["id"]] = r["vector"]
        except Exception as e:
            await emit({"type": "log", "message": f"embed chunk failed: {e}"})
        meter.accrue()
        await emit(meter.stats("embed_stats"))

    ticker.cancel()

    dim = len(next(iter(vectors.values()))) if vectors else 384
    matrix = np.zeros((len(chunks), dim), dtype="float32")
    for cid, vec in vectors.items():
        matrix[cid] = vec

    index = {"matrix": matrix, "chunks": chunks}
    await emit(meter.stats("embed_done"))
    return index


async def search(query: str, index: dict, top_k: int = config.SEARCH_TOP_K) -> list:
    """Embed the query with the same worker, cosine-rank the index, return top-k chunks."""
    import numpy as np
    embed = _mock_embed_chunk if config.MOCK else embed_chunk
    q = np.asarray((await embed(query, config.EMBED_MODEL_ID))["vector"], dtype="float32")
    mat = index["matrix"]
    # vectors are L2-normalized at embed time, so dot == cosine similarity
    sims = mat @ q
    top = np.argsort(-sims)[:top_k]
    return [
        {
            "score": round(float(sims[i]), 3),
            "file": index["chunks"][i]["file"],
            "text": index["chunks"][i]["text"],
            "t": index["chunks"][i].get("t"),   # clip timecode (media mode), else None
        }
        for i in top
    ]


# ===========================================================================
# Driver #2b — Bright Data ingest  (scrape the web -> swarm-embed -> searchable)
# ===========================================================================

async def ingest_and_index(url: str, emit: Emit) -> dict:
    """Scrape a live URL via Bright Data, then embed it across the SAME fan-out.

    Two clouds, one pipeline: Bright Data fetches real web data past bot blocks, RunPod's
    GPU swarm turns it into a searchable vector index in seconds. The fan-out is identical
    to embed_folder() — only the SOURCE of the chunks changed.
    """
    import brightdata
    await emit({"type": "log", "message": f"Bright Data scraping {url} ({brightdata.status()['brightdata']})"})
    texts = await brightdata.scrape(url)                       # cheap ingest (CPU/proxy)
    chunks = [{"id": i, "file": url, "text": t} for i, t in enumerate(texts)]
    return await _embed_chunks(chunks, emit, source=url)       # swarm-embed


# ===========================================================================
# Driver #4 — media (video production: transcribe a pile of clips, then search
# a phrase and jump to the exact clip + timecode)
# ===========================================================================

async def transcribe_media(sources: list, emit: Emit) -> dict:
    """Fan out transcription over a list of clips, then make every segment searchable.

    Same fan-out as paint/embed — one clip per worker, scale to zero. Each finished clip
    streams its timestamped segments to the UI (the footage log), and every segment is
    embedded so you can search a phrase and jump to the moment it was said.
    """
    sources = sources or config.SAMPLE_MEDIA
    meter = _Meter()
    meter.total = len(sources)
    await emit({"type": "media_start", "total": meter.total})

    tx = _mock_transcribe if config.MOCK else transcribe

    async def _one(i, url):
        try:
            r = await tx(url, config.WHISPER_MODEL)
            r["ok"] = True
            return r
        except Exception as e:
            return {"url": url, "segments": [], "ok": False, "error": str(e)}

    jobs = [_run_with_meter(lambda i=i, u=u: _one(i, u), meter) for i, u in enumerate(sources)]
    ticker = asyncio.create_task(_ticker(emit, meter))

    # collect every segment as a searchable chunk carrying its clip + timecode
    chunks, cid = [], 0
    for fut in asyncio.as_completed(jobs):
        clip = await fut
        await emit({"type": "media_clip", "url": clip["url"], "ok": clip.get("ok", True),
                    "duration": clip.get("duration"), "segments": clip.get("segments", [])})
        for seg in clip.get("segments", []):
            chunks.append({"id": cid, "file": clip["url"], "text": seg["text"],
                           "t": seg["start"]})
            cid += 1
        meter.accrue()
        await emit(meter.stats("media_stats"))

    ticker.cancel()
    await emit({"type": "media_done"})

    # embed all segments so /search returns clip + timecode (re-uses the swarm)
    if chunks:
        index = await _embed_chunks(chunks, emit, source="footage")
    else:
        index = {"matrix": __import__("numpy").zeros((0, 384), dtype="float32"), "chunks": []}
    await emit(meter.stats("done"))
    return index


# ===========================================================================
# Driver #3 — fine-tune hybrid  (the headline: teach a concept, then swarm-paint it)
# ===========================================================================
#
# Three acts on one code path:
#   1. BEFORE  — base model renders the concept -> it has no idea (muted tiles).
#   2. TRAIN   — one queued GPU job fine-tunes a LoRA on a few photos (~60-90s).
#   3. AFTER   — the SAME renderer, now loaded with the fresh adapter, fans out across
#                N workers and paints a wall of the taught concept, then scales to zero.
#
# Acts 1 and 3 are just paint()'s fan-out with base_model/adapter set. Act 2 is the
# "data in, live model out" beat. A cached adapter is the on-stage safety net.

def _load_images_b64(images_dir: str) -> list:
    """Load training images as base64 (any jpg/png in the folder)."""
    paths = [p for p in sorted(Path(images_dir).glob("**/*"))
             if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")]
    if not paths:
        if config.MOCK:
            # mock training ignores content — hand it dummy bytes so the path runs
            return [_solid_png((180, 120, 90), 64, 64) for _ in range(4)]
        raise ValueError(
            f"no training images in {images_dir}/ — drop 5-10 photos of your concept "
            f"there (jpg/png), or set WALL_USE_CACHED=1 to use a cached adapter."
        )
    return [base64.b64encode(p.read_bytes()).decode("ascii") for p in paths]


def _save_adapter_cache(adapter_b64: str) -> None:
    p = Path(config.CACHED_ADAPTER)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(adapter_b64, encoding="ascii")


def _load_adapter_cache() -> Optional[str]:
    p = Path(config.CACHED_ADAPTER)
    return p.read_text(encoding="ascii") if p.exists() else None


async def _fanout_render(tiles, prompt, emit, meter, model_id, adapter, seed_base=0,
                         turbo=False):
    """Shared fan-out: render every tile (with optional adapter), stream as completed."""
    render = _mock_render_tile if config.MOCK else render_tile

    async def _one(t, i):
        try:
            r = await render(t, prompt, seed_base + i, model_id, adapter, turbo)
            r["ok"] = True
            return r
        except Exception as e:
            return {"id": t["id"], "x": t["x"], "y": t["y"],
                    "w": t["w"], "h": t["h"], "ok": False, "error": str(e)}

    jobs = [_run_with_meter(lambda t=t, i=i: _one(t, i), meter) for i, t in enumerate(tiles)]
    ticker = asyncio.create_task(_ticker(emit, meter))
    for fut in asyncio.as_completed(jobs):
        tile = await fut
        tile["type"] = "tile"
        await emit(tile)
        meter.accrue()
        await emit(meter.stats())
    ticker.cancel()


async def finetune_and_paint(concept: str, emit: Emit, images_dir: str = None,
                             image_source: str = "folder") -> dict:
    base = config.FINETUNE_BASE
    prompt = config.PROMPT_TEMPLATE.format(concept=concept)
    images_dir = images_dir or config.IMAGES_DIR

    # ---- ACT 1: BEFORE (base model knows nothing) ----
    await emit({"type": "ft_phase", "phase": "before", "concept": concept,
                "prompt": prompt, "n": config.BASE_PREVIEW})
    # a single row of preview tiles (a filmstrip), not a grid
    base_tiles = [{"id": i, "x": i * config.TILE_PX, "y": 0,
                   "w": config.TILE_PX, "h": config.TILE_PX}
                  for i in range(config.BASE_PREVIEW)]
    bmeter = _Meter(); bmeter.total = len(base_tiles)
    render = _mock_render_tile if config.MOCK else render_tile
    bjobs = [_run_with_meter(
        lambda t=t, i=i: render(t, prompt, 1000 + i, base, None, False), bmeter)
        for i, t in enumerate(base_tiles)]
    for k, fut in enumerate(asyncio.as_completed(bjobs)):
        r = await fut
        await emit({"type": "base_tile", "i": r["id"], "n": config.BASE_PREVIEW,
                    "png_b64": r.get("png_b64", "")})

    # ---- ACT 2: TRAIN (one queued GPU job: data in, live model out) ----
    est = 6.0 if config.MOCK else 90.0
    await emit({"type": "train_start", "concept": concept,
                "steps": config.TRAIN_STEPS, "est": est})

    async def _progress():
        t0 = time.monotonic()
        try:
            while True:
                await asyncio.sleep(0.3)
                frac = min(0.99, (time.monotonic() - t0) / est)
                await emit({"type": "train_progress",
                            "step": int(frac * config.TRAIN_STEPS),
                            "steps": config.TRAIN_STEPS,
                            "elapsed": round(time.monotonic() - t0, 1),
                            "workers": 1})
        except asyncio.CancelledError:
            pass

    prog = asyncio.create_task(_progress())
    adapter = None
    try:
        if config.USE_CACHED:
            adapter = _load_adapter_cache()
            if adapter:
                await emit({"type": "log", "message": "using cached adapter (safety net)"})
        if adapter is None:
            if image_source == "web":
                # pull training images straight off the web via Bright Data
                import base64 as _b64
                import brightdata
                await emit({"type": "log",
                            "message": f"Bright Data fetching images for '{concept}'"})
                raw = await brightdata.images(concept, n=8)
                images = [_b64.b64encode(b).decode("ascii") for b in raw]
            else:
                images = _load_images_b64(images_dir)
            train = _mock_train_lora if config.MOCK else train_lora
            result = await train(images, concept, config.TRAIN_STEPS,
                                 config.FINETUNE_BASE, config.PROMPT_TEMPLATE)
            adapter = result["adapter_b64"]
            if not config.MOCK:
                _save_adapter_cache(adapter)   # cache the real adapter for next time
    except Exception as e:
        prog.cancel()
        adapter = _load_adapter_cache()
        if adapter is None:
            await emit({"type": "log", "message": f"train failed and no cached adapter: {e}"})
            raise
        await emit({"type": "log", "message": f"train failed ({e}); fell back to cached adapter"})
    prog.cancel()
    await emit({"type": "train_done", "concept": concept})

    # ---- ACT 3: AFTER (swarm-paint the taught concept, then scale to zero) ----
    meter = _Meter()
    canvas = config.FINETUNE_GRID * config.TILE_PX
    tiles = _plan_tiles_local(canvas, canvas, config.FINETUNE_GRID)["tiles"]
    meter.total = len(tiles)
    await emit({"type": "start", "prompt": prompt, "total": meter.total,
                "grid": config.FINETUNE_GRID, "canvas": canvas, "tile_px": config.TILE_PX,
                "max_workers": config.MAX_WORKERS, "concept": concept})
    await _fanout_render(tiles, prompt, emit, meter, base, adapter, turbo=False)
    final = meter.stats("done")
    await emit(final)
    return final


# ===========================================================================
# Warmup  (pre-spin GPU workers so the audience-facing run is fast — SPEC §7)
# ===========================================================================

async def warmup(emit: Optional[Emit] = None) -> None:
    """Fire one throwaway job at each GPU endpoint to build the worker env / cache weights.

    No-op in MOCK mode. Run this 5+ minutes before the live demo.
    """
    async def _say(m):
        if emit:
            await emit({"type": "log", "message": m})
        print(m)

    if config.MOCK:
        await _say("MOCK mode — no warmup needed (nothing deploys).")
        return

    t = {"id": 0, "x": 0, "y": 0, "w": config.TILE_PX, "h": config.TILE_PX}

    # Warm all endpoints CONCURRENTLY so the (minutes-long) cold starts overlap instead
    # of stacking — critical when you only have a short window before the demo.
    async def _warm(label, coro):
        try:
            await coro
            await _say(f"  {label} warm.")
        except Exception as e:
            await _say(f"  {label} warmup FAILED: {e}")

    await _say("Warming all endpoints in parallel (cold starts overlap)...")
    await asyncio.gather(
        _warm("tile-renderer/turbo", render_tile(t, "warmup", 0, config.MODEL_ID, None, True)),
        _warm("tile-renderer/SD1.5", render_tile(t, "warmup", 0, config.FINETUNE_BASE, None, False)),
        _warm("lora-trainer", train_lora(_load_images_b64(config.IMAGES_DIR), config.CONCEPT, 1,
                                         config.FINETUNE_BASE, config.PROMPT_TEMPLATE)),
        _warm("chunk-embedder", embed_chunk("warmup", config.EMBED_MODEL_ID)),
    )
    await _say("warmup complete.")


# ===========================================================================
# CLI
# ===========================================================================

def _build_console_emit():
    async def emit(e: dict):
        t = e.get("type")
        if t == "start":
            print(f"painting {e['total']} tiles  (grid {e['grid']}x{e['grid']}, "
                  f"up to {e['max_workers']} workers)")
        elif t == "tile":
            mark = "ok" if e.get("ok") else "FAIL"
            tid = e.get("id", "?")
            print(f"  tile {tid:>3} {mark}")
        elif t in ("stats", "embed_stats"):
            print(f"\r  workers={e['workers']:>2} done={e['done']}/{e['total']} "
                  f"elapsed={e['elapsed']}s cost=${e['cost']:.4f}", end="", flush=True)
        elif t in ("done", "embed_done"):
            print(f"\nDONE  {e['done']}/{e['total']} in {e['elapsed']}s  "
                  f"cost=${e['cost']:.4f}")
        elif t == "embed_start":
            print(f"embedding {e['total']} chunks from {e['docs_dir']}/")
        elif t == "ft_phase":
            print(f"\n[ACT 1] BEFORE: base model renders {e['prompt']!r} (no idea yet)")
        elif t == "base_tile":
            print(f"  base preview {e['i']+1}/{e['n']}")
        elif t == "train_start":
            print(f"[ACT 2] TRAIN: fine-tuning LoRA on '{e['concept']}' "
                  f"({e['steps']} steps, ~{e['est']:.0f}s)")
        elif t == "train_progress":
            print(f"\r  step {e['step']}/{e['steps']}  {e['elapsed']}s", end="", flush=True)
        elif t == "train_done":
            print(f"\n[ACT 3] AFTER: swarm-painting '{e['concept']}' with the new adapter")
        elif t == "log":
            print(e["message"])
    return emit


def _main():
    import sys

    args = sys.argv[1:]
    mode = "paint"
    warm = False
    prompt_parts = []
    for a in args:
        if a in ("--warm", "-w"):
            warm = True
        elif a.startswith("--mode="):
            mode = a.split("=", 1)[1]
        else:
            prompt_parts.append(a)
    prompt = " ".join(prompt_parts) or "a neon city at night"

    emit = _build_console_emit()
    mode_label = "MOCK" if config.MOCK else "LIVE"
    print(f"[wall.py] mode={mode} run={mode_label}  "
          f"(set WALL_MOCK=0 for real GPU workers)\n")

    async def run():
        if warm:
            await warmup(emit)
        if mode == "embed":
            index = await embed_folder(config.DOCS_DIR, emit)
            for q in ("how does scale to zero save money?",
                      "what is a cold start?"):
                print(f"\nsearch: {q!r}")
                for hit in await search(q, index):
                    print(f"  [{hit['score']:+.3f}] {hit['file']}: {hit['text'][:70]}...")
        elif mode == "finetune":
            concept = prompt if prompt != "a neon city at night" else config.CONCEPT
            await finetune_and_paint(concept, emit)
        else:
            await paint(prompt, emit)

    asyncio.run(run())
    if not config.MOCK:
        print("\nReminder: live endpoints persist until you run `flash undeploy`.")


if __name__ == "__main__":
    _main()
