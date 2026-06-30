"""
brightdata.py — live web-data ingestion that feeds the RunPod fan-out.

This is the "data in" front of the pipeline. Two functions, both MOCK-first:

    scrape(url)        -> list[str]   text chunks  (-> swarm-embed -> /search)
    images(query, n)   -> list[bytes] image bytes  (-> lora-train -> swarm-paint)

The story: Bright Data fetches REAL web data (past bot blocks, at scale), RunPod's GPU
swarm processes it, and you get search / a fine-tuned model out — two clouds, one
pipeline, no toy sample data.

Credentials (env, see .env.example). If none are set we fall back to MOCK so the demo
always runs offline:
    BRIGHTDATA_PROXY      full proxy URL for Web Unlocker / SERP, e.g.
                          http://brd-customer-<id>-zone-<zone>:<pass>@brd.superproxy.io:33335
    BRIGHTDATA_SERP_PROXY optional separate proxy URL for a SERP zone (image search)
    WALL_BRIGHTDATA_MOCK  set to 1 to force MOCK even if credentials exist
"""

import os
import re
from html.parser import HTMLParser

PROXY = os.getenv("BRIGHTDATA_PROXY", "")
SERP_PROXY = os.getenv("BRIGHTDATA_SERP_PROXY", PROXY)
# API-token mode (Bright Data unified API: POST https://api.brightdata.com/request).
# Needs a token AND a zone name (the Web Unlocker zone you created in the dashboard).
API_TOKEN = os.getenv("BRIGHTDATA_API_TOKEN", "")
ZONE = os.getenv("BRIGHTDATA_ZONE", "")
SERP_ZONE = os.getenv("BRIGHTDATA_SERP_ZONE", ZONE)
API_URL = "https://api.brightdata.com/request"
FORCE_MOCK = os.getenv("WALL_BRIGHTDATA_MOCK", "0") in ("1", "true", "True")

# Live when EITHER a proxy is set OR an API token + zone are set.
_API_READY = bool(API_TOKEN and ZONE)
ENABLED = (bool(PROXY) or _API_READY) and not FORCE_MOCK


def status() -> dict:
    mode = "api" if _API_READY else ("proxy" if PROXY else "none")
    return {"brightdata": "live" if ENABLED else "mock",
            "mode": mode, "token_set": bool(API_TOKEN), "zone_set": bool(ZONE)}


async def _api_request(target_url: str, zone: str) -> str:
    """Fetch a URL through Bright Data's unified API (Bearer token + zone)."""
    import aiohttp
    headers = {"Authorization": f"Bearer {API_TOKEN}", "Content-Type": "application/json"}
    body = {"zone": zone, "url": target_url, "format": "raw"}
    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(timeout=timeout) as s:
        async with s.post(API_URL, json=body, headers=headers) as r:
            r.raise_for_status()
            return await r.text()


# ---------------------------------------------------------------------------
# HTML -> text
# ---------------------------------------------------------------------------

class _TextExtractor(HTMLParser):
    _SKIP = {"script", "style", "noscript", "head", "svg"}

    def __init__(self):
        super().__init__()
        self._skip = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            t = data.strip()
            if t:
                self.parts.append(t)


def _html_to_text(html: str) -> str:
    p = _TextExtractor()
    try:
        p.feed(html)
    except Exception:
        pass
    return re.sub(r"\s+", " ", " ".join(p.parts)).strip()


def _chunk(text: str, words_per_chunk: int = 60) -> list:
    words = text.split()
    out = []
    for i in range(0, len(words), words_per_chunk):
        c = " ".join(words[i:i + words_per_chunk]).strip()
        if c:
            out.append(c)
    return out


# ---------------------------------------------------------------------------
# scrape(url) -> text chunks
# ---------------------------------------------------------------------------

async def scrape(url: str, max_chunks: int = 200) -> list:
    """Fetch a URL through Bright Data Web Unlocker and return text chunks."""
    if not ENABLED:
        return _mock_scrape(url, max_chunks)

    if _API_READY:
        html = await _api_request(url, ZONE)
    else:
        import aiohttp
        # Web Unlocker proxy is a MITM (solves blocks / CAPTCHAs), so skip cert verify.
        timeout = aiohttp.ClientTimeout(total=60)
        headers = {"User-Agent": "Mozilla/5.0 (compatible; SwarmForge/1.0)"}
        async with aiohttp.ClientSession(timeout=timeout) as s:
            async with s.get(url, proxy=PROXY, ssl=False, headers=headers) as r:
                r.raise_for_status()
                html = await r.text()
    chunks = _chunk(_html_to_text(html))
    return chunks[:max_chunks] or _mock_scrape(url, max_chunks)


def _mock_scrape(url: str, max_chunks: int) -> list:
    """Offline stand-in: synthesize plausible page chunks for the URL's domain."""
    domain = re.sub(r"^https?://", "", url).split("/")[0] or "example.com"
    seeds = [
        f"{domain} is the source for this page. Bright Data fetched it past any bot "
        f"protection so the RunPod swarm could embed it in parallel.",
        f"Section about {domain}: serverless GPUs fan out one embedding job per chunk, "
        f"scale to zero when the queue drains, and bill per worker-second.",
        f"More from {domain}: the same machinery that paints image tiles also turns a "
        f"freshly scraped page into a searchable vector index in seconds.",
        f"{domain} notes that cold starts are minutes for heavy models, so you pre-warm "
        f"a worker before serving real traffic.",
        f"Cost on {domain}: twenty-five GPUs for ten seconds costs what one costs for "
        f"two hundred and fifty seconds, but finishes twenty-five times faster.",
    ]
    return (seeds * ((max_chunks // len(seeds)) + 1))[:min(max_chunks, 12)]


# ---------------------------------------------------------------------------
# images(query) -> image bytes  (for the fine-tune)
# ---------------------------------------------------------------------------

async def images(query: str, n: int = 8) -> list:
    """Pull n images for a query via Bright Data SERP (image search) + download.

    Uses the SERP *proxy* (binary image download). API-token-only setups stay MOCK for
    images (the unified API path here serves text scraping); set BRIGHTDATA_SERP_PROXY
    to enable real web images for the fine-tune.
    """
    if not ENABLED or not SERP_PROXY:
        return _mock_images(n)

    import aiohttp
    # Bright Data SERP: route a Google image search through the SERP proxy and read the
    # JSON results. brd_json=1 asks Bright Data to return parsed JSON instead of HTML.
    search = (f"https://www.google.com/search?tbm=isch&q="
              f"{aiohttp.helpers.quote(query)}&brd_json=1")
    timeout = aiohttp.ClientTimeout(total=60)
    urls = []
    try:
        async with aiohttp.ClientSession(timeout=timeout) as s:
            async with s.get(search, proxy=SERP_PROXY, ssl=False) as r:
                r.raise_for_status()
                data = await r.json(content_type=None)
            for item in (data.get("images") or [])[: n * 2]:
                u = item.get("image") or item.get("source") or item.get("link")
                if u:
                    urls.append(u)
            out = []
            for u in urls:
                if len(out) >= n:
                    break
                try:
                    async with s.get(u, proxy=PROXY, ssl=False) as ir:
                        if ir.status == 200:
                            out.append(await ir.read())
                except Exception:
                    continue
    except Exception:
        return _mock_images(n)
    return out or _mock_images(n)


def _mock_images(n: int) -> list:
    """Offline stand-in: tiny solid PNGs so the train path runs without real images."""
    import wall  # reuse the stdlib PNG writer
    import base64
    return [base64.b64decode(wall._solid_png((160, 110, 80), 64, 64)) for _ in range(n)]
