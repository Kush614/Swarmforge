"""LIVE transcription of real internet video/audio links -> records REAL results to
static/cache/media.json so the Media tab replays real video + real transcripts + timecode
search. Run: WALL_MOCK=0 WALL_MAX_WORKERS=5 python media_live.py"""
import asyncio
import json
import os
from pathlib import Path

os.environ["WALL_MOCK"] = "0"
import config
import wall

events = []


async def emit(e):
    events.append(e)                      # record for the cache
    t = e.get("type")
    if t == "media_start":
        print(f"transcribing {e['total']} real clips across the swarm...\n")
    elif t == "media_clip":
        name = e["url"].split("/")[-1]
        text = " ".join(s["text"] for s in e.get("segments", [])) or "(no speech)"
        print(f"[{name}] {text}")
    elif t == "log":
        print("log:", e["message"])


async def go():
    await wall.transcribe_media(config.SAMPLE_MEDIA, emit)
    Path("static/cache/media.json").write_text(json.dumps(events), encoding="utf-8")
    print(f"\nDONE — wrote {len(events)} events to static/cache/media.json (REAL).")


asyncio.run(go())
