"""
app.py — the local orchestrator + projector web server.

This is the "local orchestrator" box in the SPEC §3 diagram. It runs on the machine
driving the projector, serves the self-painting canvas, and calls the Flash GPU/CPU
endpoints in wall.py. Completed tiles and dashboard stats stream to every connected
browser over Server-Sent Events (SSE).

Why a local server instead of an LB Flash endpoint for the API? An LB endpoint runs on
stateless serverless workers and can't hold a persistent SSE connection or an in-memory
tile broadcast queue for the projector. The reliable, spec-faithful design keeps the
*orchestration + UI* local and pushes only the heavy per-tile / per-chunk work to Flash.
For the QR "audience triggers a paint" beat, expose this server publicly with a tunnel
(see README) — both the projector and phones then hit the same URL.

Routes:
  GET  /         -> the projector front-end (static/index.html)
  GET  /events   -> SSE stream: one message per completed tile + live dashboard stats
  GET  /config   -> grid / canvas size / public URL / mock flag (front-end bootstrap)
  GET  /health   -> {"status": "ok"}
  POST /paint    -> {"prompt": ...}  kicks off a paint in the background
  POST /embed    -> embed sample_docs/ (streams to dashboard), builds the search index
  POST /search   -> {"query": ...}   returns top-k nearest chunks

Run:  python app.py        (open http://localhost:8000)
"""

import asyncio
import json
from pathlib import Path

from aiohttp import web

import config
import wall

STATIC = Path(__file__).parent / "static"


# ---------------------------------------------------------------------------
# SSE hub — fan one event out to every connected browser
# ---------------------------------------------------------------------------

class Hub:
    def __init__(self):
        self._subs: set[asyncio.Queue] = set()

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._subs.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subs.discard(q)

    async def publish(self, event: dict) -> None:
        for q in list(self._subs):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass  # a slow browser must never stall the paint


# ---------------------------------------------------------------------------
# Recorder — capture each run's events to static/last_run.json so the UI can
# preload + replay it with ZERO compute (bulletproof fallback if RunPod/the
# server isn't running).
# ---------------------------------------------------------------------------

_REC_START = {"start", "ft_phase", "embed_start", "media_start"}
_REC_END = {"done", "embed_done"}
CACHE = STATIC / "cache"


def _run_kind(events: list) -> str:
    types = {e.get("type") for e in events}
    if "ft_phase" in types:
        return "teach"
    if "media_start" in types:
        return "media"
    if "embed_start" in types and "start" not in types:
        return "embed"
    return "paint"


async def _publish_record(app: web.Application, event: dict) -> None:
    rec = app["rec"]
    t = event.get("type")
    if t in _REC_START and not rec.get("open"):
        rec["events"] = []
        rec["open"] = True
    rec["events"].append(event)
    await app["hub"].publish(event)
    if t in _REC_END:
        rec["open"] = False
        try:
            CACHE.mkdir(parents=True, exist_ok=True)
            blob = json.dumps(rec["events"])
            (STATIC / "last_run.json").write_text(blob, encoding="utf-8")
            # also cache by mode so the offline build can replay every tab
            (CACHE / f"{_run_kind(rec['events'])}.json").write_text(blob, encoding="utf-8")
        except Exception:
            pass


def _recorder(app: web.Application):
    async def emit(event: dict):
        await _publish_record(app, event)
    return emit


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

async def index(request: web.Request) -> web.Response:
    return web.FileResponse(STATIC / "index.html")


async def health(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok", "mock": config.MOCK})


async def get_config(request: web.Request) -> web.Response:
    return web.json_response({
        "grid": config.GRID,
        "canvas": config.CANVAS_PX,
        "tile_px": config.TILE_PX,
        "max_workers": config.MAX_WORKERS,
        "public_url": config.PUBLIC_URL,
        "mock": config.MOCK,
        "cost_per_worker_sec": config.COST_PER_WORKER_SEC,
        "concept": config.CONCEPT,
        "base_preview": config.BASE_PREVIEW,
        "finetune_grid": config.FINETUNE_GRID,
        "brightdata": __import__("brightdata").status()["brightdata"],
    })


async def events(request: web.Request) -> web.StreamResponse:
    """SSE stream the canvas subscribes to."""
    resp = web.StreamResponse(
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",   # disable proxy buffering (nginx/tunnels)
        }
    )
    await resp.prepare(request)
    hub: Hub = request.app["hub"]
    q = hub.subscribe()
    # tell a freshly-connected browser how to size its canvas
    await resp.write(_sse({"type": "hello", "grid": config.GRID,
                           "canvas": config.CANVAS_PX, "mock": config.MOCK}))
    try:
        while True:
            try:
                event = await asyncio.wait_for(q.get(), timeout=15)
                await resp.write(_sse(event))
            except asyncio.TimeoutError:
                await resp.write(b": ping\n\n")   # heartbeat keeps the connection open
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    finally:
        hub.unsubscribe(q)
    return resp


def _sse(event: dict) -> bytes:
    return f"data: {json.dumps(event)}\n\n".encode("utf-8")


async def paint_route(request: web.Request) -> web.Response:
    """Kick off a paint in the background; return immediately (phone-friendly)."""
    try:
        data = await request.json()
    except Exception:
        data = {}
    prompt = (data.get("prompt") or "a neon city at night").strip()

    lock: asyncio.Lock = request.app["paint_lock"]
    if lock.locked():
        return web.json_response({"status": "busy"}, status=429)

    hub: Hub = request.app["hub"]
    emit = _recorder(request.app)

    async def _job():
        async with lock:
            try:
                await wall.paint(prompt, emit)
            except Exception as e:
                await hub.publish({"type": "log", "message": f"paint error: {e}"})

    asyncio.create_task(_job())
    return web.json_response({"status": "painting", "prompt": prompt})


async def embed_route(request: web.Request) -> web.Response:
    """Embed the docs folder (streams to the dashboard) and build the search index."""
    lock: asyncio.Lock = request.app["paint_lock"]
    if lock.locked():
        return web.json_response({"status": "busy"}, status=429)
    hub: Hub = request.app["hub"]

    emit = _recorder(request.app)

    async def _job():
        async with lock:
            try:
                request.app["index"] = await wall.embed_folder(config.DOCS_DIR, emit)
            except Exception as e:
                await hub.publish({"type": "log", "message": f"embed error: {e}"})

    asyncio.create_task(_job())
    return web.json_response({"status": "embedding", "docs_dir": config.DOCS_DIR})


async def teach_route(request: web.Request) -> web.Response:
    """The headline demo: teach a concept (LoRA fine-tune), then swarm-paint it."""
    try:
        data = await request.json()
    except Exception:
        data = {}
    concept = (data.get("concept") or config.CONCEPT).strip()
    source = "web" if data.get("source") == "web" else "folder"

    lock: asyncio.Lock = request.app["paint_lock"]
    if lock.locked():
        return web.json_response({"status": "busy"}, status=429)
    hub: Hub = request.app["hub"]
    # per-run safety-net override: {"use_cached": true} skips live training
    config.USE_CACHED = bool(data.get("use_cached", config.USE_CACHED))

    emit = _recorder(request.app)

    async def _job():
        async with lock:
            try:
                await wall.finetune_and_paint(concept, emit, image_source=source)
            except Exception as e:
                await hub.publish({"type": "log", "message": f"teach error: {e}"})

    asyncio.create_task(_job())
    return web.json_response({"status": "teaching", "concept": concept,
                             "use_cached": config.USE_CACHED, "source": source})


async def ingest_route(request: web.Request) -> web.Response:
    """Bright Data scrape -> swarm-embed -> searchable. Body: {url}."""
    try:
        data = await request.json()
    except Exception:
        data = {}
    url = (data.get("url") or "").strip()
    if not url:
        return web.json_response({"error": "url is required"}, status=400)
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    lock: asyncio.Lock = request.app["paint_lock"]
    if lock.locked():
        return web.json_response({"status": "busy"}, status=429)
    hub: Hub = request.app["hub"]

    emit = _recorder(request.app)

    async def _job():
        async with lock:
            try:
                request.app["index"] = await wall.ingest_and_index(url, emit)
            except Exception as e:
                await hub.publish({"type": "log", "message": f"ingest error: {e}"})

    asyncio.create_task(_job())
    return web.json_response({"status": "ingesting", "url": url})


async def upload_route(request: web.Request) -> web.Response:
    """Accept training images (multipart) into IMAGES_DIR for the fine-tune."""
    from pathlib import Path as _P
    dest = _P(config.IMAGES_DIR)
    dest.mkdir(parents=True, exist_ok=True)
    saved = []
    reader = await request.multipart()
    async for part in reader:
        if part.filename:
            out = dest / _P(part.filename).name
            with open(out, "wb") as f:
                while chunk := await part.read_chunk():
                    f.write(chunk)
            saved.append(out.name)
    return web.json_response({"saved": saved, "dir": config.IMAGES_DIR})


async def transcribe_route(request: web.Request) -> web.Response:
    """Footage logger: transcribe clips across the swarm, index segments by timecode.

    Body: {sources: [url, ...]} (optional; defaults to config.SAMPLE_MEDIA)."""
    try:
        data = await request.json()
    except Exception:
        data = {}
    sources = data.get("sources") or []

    lock: asyncio.Lock = request.app["paint_lock"]
    if lock.locked():
        return web.json_response({"status": "busy"}, status=429)
    hub: Hub = request.app["hub"]
    emit = _recorder(request.app)

    async def _job():
        async with lock:
            try:
                request.app["index"] = await wall.transcribe_media(sources, emit)
            except Exception as e:
                await hub.publish({"type": "log", "message": f"transcribe error: {e}"})

    asyncio.create_task(_job())
    return web.json_response({"status": "transcribing",
                             "count": len(sources) or len(config.SAMPLE_MEDIA)})


async def search_route(request: web.Request) -> web.Response:
    """Embed the query and return top-k nearest chunks. Builds the index on first use."""
    try:
        data = await request.json()
    except Exception:
        data = {}
    query = (data.get("query") or "").strip()
    if not query:
        return web.json_response({"error": "query is required"}, status=400)

    hub: Hub = request.app["hub"]
    if request.app.get("index") is None:
        # lazily embed so /search works even if /embed was never clicked
        request.app["index"] = await wall.embed_folder(config.DOCS_DIR, hub.publish)

    results = await wall.search(query, request.app["index"])
    return web.json_response({"query": query, "results": results})


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def make_app() -> web.Application:
    app = web.Application()
    app["hub"] = Hub()
    app["paint_lock"] = asyncio.Lock()
    app["index"] = None
    app["rec"] = {"events": [], "open": False}   # last-run recording buffer

    app.add_routes([
        web.get("/", index),
        web.get("/health", health),
        web.get("/config", get_config),
        web.get("/events", events),
        web.post("/paint", paint_route),
        web.post("/embed", embed_route),
        web.post("/search", search_route),
        web.post("/teach", teach_route),
        web.post("/upload", upload_route),
        web.post("/ingest", ingest_route),
        web.post("/transcribe", transcribe_route),
    ])
    if STATIC.exists():
        app.router.add_static("/static/", STATIC)
    return app


def main():
    mode = "MOCK (free, local)" if config.MOCK else "LIVE (deploys RunPod workers)"
    print(f"""
  Self-Painting Wall  —  orchestrator up
  ----------------------------------------
  mode:        {mode}
  projector:   http://localhost:{config.PORT}
  grid:        {config.GRID}x{config.GRID} = {config.GRID*config.GRID} tiles
  max workers: {config.MAX_WORKERS}
  public URL:  {config.PUBLIC_URL or '(none — set WALL_PUBLIC_URL for the QR beat)'}

  Open the projector URL, then click Paint (or POST /paint).
  Set WALL_MOCK=0 and run `flash login` for real GPU workers.
""")
    web.run_app(make_app(), host=config.HOST, port=config.PORT, print=None)


if __name__ == "__main__":
    main()
