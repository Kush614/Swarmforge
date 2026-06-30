"""
smoke_live.py — de-risk the LIVE RunPod path before the demo.

Runs the cheapest real endpoint first (CPU planner), then a GPU tile (cold + warm), then
optionally a short LoRA train. Times each stage and prints a rough cost, so you learn the
real cold-start and spend before you're on stage — for a few cents.

Usage (must be LIVE):
    flash login                 # once, if not already
    set WALL_MOCK=0             (PowerShell: $env:WALL_MOCK="0")
    python smoke_live.py         # CPU + GPU tile
    python smoke_live.py --full  # also runs a 20-step LoRA train (heavier/slower)

The first GPU call BUILDS the worker (pulls torch/diffusers, loads weights) — expect
minutes, not seconds. That is exactly the cold start you're measuring.
"""

import asyncio
import sys
import time

import config
import wall


async def stage(name, coro):
    t0 = time.monotonic()
    try:
        r = await coro
        dt = time.monotonic() - t0
        print(f"  [OK]   {name}: {dt:6.1f}s   ~${dt * config.COST_PER_WORKER_SEC:.4f}")
        return r, dt
    except Exception as e:
        dt = time.monotonic() - t0
        print(f"  [FAIL] {name}: {type(e).__name__}: {e}  (after {dt:.1f}s)")
        return None, None


async def main():
    full = "--full" in sys.argv
    if config.MOCK:
        print("WALL_MOCK is on — this would only test the mock path. Set WALL_MOCK=0 "
              "and `flash login` first.")
        return

    print(f"LIVE smoke test  (GPU={config.GPU_TYPE.value}, "
          f"cost/sec=${config.COST_PER_WORKER_SEC})\n")
    tile = {"id": 0, "x": 0, "y": 0, "w": config.TILE_PX, "h": config.TILE_PX}

    print("1) CPU planner (cheap, fast — confirms auth + live provisioning):")
    await stage("plan_tiles", wall.plan_tiles(config.CANVAS_PX, config.CANVAS_PX, 2))

    print("\n2) GPU renderer / turbo (first call builds the worker — minutes):")
    await stage("render_tile cold", wall.render_tile(tile, "a red apple on a table", 0))
    await stage("render_tile warm", wall.render_tile(tile, "a blue apple on a table", 1))

    if full:
        print("\n3) GPU LoRA trainer (heaviest — short 20-step run):")
        try:
            imgs = wall._load_images_b64(config.IMAGES_DIR)
        except Exception as e:
            print(f"  (skipping train — {e})")
            imgs = None
        if imgs:
            await stage("train_lora(20)", wall.train_lora(imgs, config.CONCEPT, 20))

    print("\nDone. Remember to `flash undeploy` when finished — endpoints cost money "
          "until deleted.")


if __name__ == "__main__":
    asyncio.run(main())
