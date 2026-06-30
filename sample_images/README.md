# sample_images/ — training photos for the fine-tune

Drop **5–10 photos of the concept you want to teach** here (jpg/png/webp). For a clean
before/after, use a specific subject the base model can't already draw: a particular
person, pet, product, logo, or art style. Photograph it from a few angles, reasonably
well-lit, roughly square crops work best.

These feed the `lora-trainer` endpoint in LIVE mode (`WALL_MOCK=0`). The `CONCEPT` in
`config.py` (default `sks dog`) is the rare token the LoRA binds them to.

You can also upload images from the projector UI ("Upload" button on the Teach tab) —
they land in this folder.

> MOCK mode (`WALL_MOCK=1`) ignores image contents (training is simulated), so the demo
> runs even with this folder empty. Real photos only matter for a LIVE train.
