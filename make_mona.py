"""Build a 'jigsaw' paint cache: one coherent Mona Lisa split into GRID x GRID pieces
that assemble in random order (like a swarm painting one masterpiece piece by piece).
Writes static/cache/paint.json + teach.json + last_run.json — replayed by the wall."""
import base64
import io
import json
import random
from pathlib import Path

from PIL import Image

SRC = "_mona.jpg"
GRID = 8
TILE = 256
CANVAS = GRID * TILE          # 2048
RATE = 0.00044                # cost/worker-sec (matches config default)

im = Image.open(SRC).convert("RGB")
w, h = im.size
# center-ish square crop (shifted up to frame the face), then resize to canvas
side = min(w, h)
top = int((h - side) * 0.18)
im = im.crop((0, top, side, top + side)).resize((CANVAS, CANVAS), Image.LANCZOS)

tiles = []
for r in range(GRID):
    for c in range(GRID):
        piece = im.crop((c * TILE, r * TILE, c * TILE + TILE, r * TILE + TILE))
        buf = io.BytesIO(); piece.save(buf, format="PNG")
        tiles.append({
            "id": r * GRID + c, "x": c * TILE, "y": r * TILE, "w": TILE, "h": TILE,
            "png_b64": base64.b64encode(buf.getvalue()).decode("ascii"),
        })

order = list(range(len(tiles)))
random.Random(7).shuffle(order)     # random assembly = jigsaw

events = [{
    "type": "start", "prompt": "the Mona Lisa — assembled by a GPU swarm",
    "total": len(tiles), "grid": GRID, "canvas": CANVAS, "tile_px": TILE,
    "max_workers": 8, "concept": "mona lisa",
}]
total = len(tiles)
for k, idx in enumerate(order, 1):
    t = dict(tiles[idx]); t["type"] = "tile"; t["ok"] = True
    events.append(t)
    remaining = total - k
    workers = min(remaining + 1, 8)
    elapsed = round(k / total * 12.0, 1)
    worker_seconds = round(k * 1.5, 1)
    events.append({
        "type": "stats", "workers": max(workers, 0), "done": k, "total": total,
        "elapsed": elapsed, "cost": round(worker_seconds * RATE, 4),
        "worker_seconds": worker_seconds, "always_on_month": round(RATE * 8 * 2592000, 0),
    })
events.append({
    "type": "done", "workers": 0, "done": total, "total": total, "elapsed": 12.0,
    "cost": round(total * 1.5 * RATE, 4), "worker_seconds": round(total * 1.5, 1),
    "always_on_month": round(RATE * 8 * 2592000, 0),
})

blob = json.dumps(events)
for f in ("static/cache/paint.json", "static/cache/teach.json", "static/last_run.json"):
    Path(f).write_text(blob, encoding="utf-8")
print(f"wrote Mona Lisa jigsaw: {total} pieces, {len(blob)//1024} KB -> paint/teach/last")
