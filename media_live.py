"""Quick LIVE transcription proof — deploys the media-transcriber and prints real
transcripts of the sample speech clips. Run: WALL_MOCK=0 python media_live.py"""
import asyncio
import os

os.environ["WALL_MOCK"] = "0"
import config
import wall


async def emit(e):
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
    print("\nDONE — real transcripts above.")


asyncio.run(go())
