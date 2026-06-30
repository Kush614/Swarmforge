"""
build_offline.py — bundle the recorded runs into a single self-contained offline.html
that replays every mode with NO server and NO compute (openable by double-click / file://).

Run the app, do one run per tab (Paint / Teach / Embed / Media) so each caches to
static/cache/<mode>.json, then: python build_offline.py
"""
import json
from pathlib import Path

import config

STATIC = Path("static")
CACHE = STATIC / "cache"

runs = {}
if CACHE.exists():
    for f in sorted(CACHE.glob("*.json")):
        try:
            runs[f.stem] = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            pass
last = STATIC / "last_run.json"
if last.exists():
    runs["last"] = json.loads(last.read_text(encoding="utf-8"))
elif runs:
    runs["last"] = next(iter(runs.values()))

if not runs:
    raise SystemExit("No cached runs yet. Start the app and do one run per tab first.")

cfg = {
    "grid": config.GRID, "canvas": config.CANVAS_PX, "tile_px": config.TILE_PX,
    "max_workers": config.MAX_WORKERS, "concept": config.CONCEPT,
    "public_url": config.PUBLIC_URL, "base_preview": config.BASE_PREVIEW,
}

html = (STATIC / "index.html").read_text(encoding="utf-8")
inject = ("<script>window.OFFLINE=true;window.OFFLINE_RUNS=%s;window.OFFLINE_CFG=%s;</script>\n"
          % (json.dumps(runs), json.dumps(cfg)))
marker = '<script src="https://cdn.jsdelivr.net'
html = html.replace(marker, inject + marker, 1)

out = Path("offline.html")
out.write_text(html, encoding="utf-8")
print(f"wrote {out} ({out.stat().st_size // 1024} KB) — cached modes: {list(runs)}")
