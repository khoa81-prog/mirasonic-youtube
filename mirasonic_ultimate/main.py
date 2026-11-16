"""Mirasonic worker: anonymous YouTube Music search + stream proxy.

GET /search?q=...&limit=..  -> {"tracks": [...]}
GET /stream/{video_id}      -> proxied audio bytes (Range-aware)

No auth, no DB, no disk writes. Everything cached is a plain in-memory dict.
"""
import asyncio
import logging
import os
import re
import time
from typing import Optional
from urllib.parse import urlparse, parse_qs

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

logging.basicConfig(level=logging.INFO)
# httpx/httpcore log every request URL at INFO level by default, which would
# leak the signed (ip+sig) googlevideo URL straight into the logs. Only our
# own "worker" logger (videoId + status only) may run at INFO.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger("worker")


class _RedactRestQuery(logging.Filter):
    """Uvicorn's access log prints the full request line, query string and all.

    Subsonic sends the username and password with every single request as
    `u`/`p`/`t`/`s`, so without this filter the credentials end up in
    `docker logs` in plain text — which is exactly what happened on the very
    first deployment. Our own `worker.subsonic` logger only ever prints
    parameter names, but uvicorn keeps its own log and knows nothing about
    that discipline.

    Only `/rest` is redacted: `/search?q=…` and `/stream/{id}` carry no
    secrets, and every measurement in this project was taken against them.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if isinstance(args, tuple) and len(args) >= 3 and isinstance(args[2], str):
            path = args[2]
            if path.startswith("/rest/") and "?" in path:
                record.args = args[:2] + (path.split("?", 1)[0] + "?<redacted>",) + args[3:]
        return True


logging.getLogger("uvicorn.access").addFilter(_RedactRestQuery())

# Search region. Must match the country the server reaches YouTube from:
# stream resolution happens from the server's real IP, and a mismatch makes
# search return tracks that then fail to resolve.
REGION = os.environ.get("REGION", "US")

HOME_URL = "https://music.youtube.com/"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36"
)
STREAM_EXPIRE_MARGIN_SECONDS = 600  # refresh 10 min before the signed URL expires
# The "Songs" filter for /youtubei/v1/search. Without it a query like
# "green day" comes back as an artist card plus a jumble of clips and albums —
# a 2026-08-26 measurement returned 5 songs. With it: 20 songs per page and a
# continuation token. Same value ytmusicapi uses.
SEARCH_PARAMS_SONGS = "EgWKAQIIAWoMEA4QChADEAQQCRAF"
SEARCH_PAGE_SIZE = 20
YDL_OPTS = {
    "quiet": True,
    "no_warnings": True,
    "skip_download": True,
    "format": "bestaudio[ext=m4a]/bestaudio",
    "socket_timeout": 20,
}
UNAVAILABLE_MARKERS = (
    "unavailable",
    "private",
    "not available",
    "removed",
    "restricted",
    "region",
    "no longer",
    # Age gate: never resolves anonymously, and retrying will not help, so
    # this is a 404 rather than a 502. Deliberately NOT matching "sign in" as
    # a whole — the phrase "Sign in to confirm you're not a bot" means the
    # anti-bot system tripped, which is temporary and must stay a 502.
    "confirm your age",
    "age-restricted",
    "age restricted",
)

app = FastAPI()

# ponytail: plain module-level dicts, no eviction — fine for a single-owner
# personal server; add an LRU cap if this ever runs long enough to matter.
_config_cache: Optional[dict] = None
_stream_cache: dict[str, dict] = {}

# Bounds concurrent /youtubei/v1/player calls (track-duration lookups for the
# Subsonic layer). Pitfall #1 in docs/PITFALLS.md: a queue of requests from one
# address reads as a bot and earns the whole server a captcha.
_player_semaphore = asyncio.Semaphore(8)

# One client for the whole process rather than a fresh one per request. A
# player pulls a track as a dozen separate range requests, and with a
# client-per-request each one paid for a new TLS handshake to googlevideo:
# measured 2026-08-26 at 0.22 s on a cold connection against 0.035 s on a
# reused one. The pool removes that handshake.
_upstream_client: Optional[httpx.AsyncClient] = None


def get_upstream_client() -> httpx.AsyncClient:
    global _upstream_client
    if _upstream_client is None or _upstream_client.is_closed:
        _upstream_client = httpx.AsyncClient(
            timeout=None,
            limits=httpx.Limits(max_keepalive_connections=20, keepalive_expiry=300),
        )
    return _upstream_client



async def get_innertube_config(client: httpx.AsyncClient) -> dict:
    """Fetch + cache INNERTUBE_API_KEY / CLIENT_NAME / CLIENT_VERSION from the
    YT Music homepage. Cached forever in-process — the page is 547 KB and the
    values live for hours, so re-fetching per search is wasteful."""
    global _config_cache
    if _config_cache is not None:
        return _config_cache
    resp = await client.get(
        HOME_URL,
        headers={
            "User-Agent": USER_AGENT,
            "Accept-Language": "en-US,en;q=0.9",
            # ponytail: EU-geolocated IPs get redirected to a GDPR consent
            # page instead of the homepage; this is yt-dlp's own bypass cookie.
            "Cookie": "SOCS=CAI",
        },
        follow_redirects=True,
        timeout=20,
    )
    resp.raise_for_status()
    html = resp.text
    config = {}
    for key in ("INNERTUBE_API_KEY", "INNERTUBE_CLIENT_NAME", "INNERTUBE_CLIENT_VERSION"):
        match = re.search(rf'"{key}":"([^"]+)"', html)
        if not match:
            raise ValueError(f"missing {key} in homepage")
        config[key] = match.group(1)
    _config_cache = config
    return config


def parse_duration(text: str) -> Optional[int]:
    """Parse 'M:SS' or 'H:MM:SS'. All parts but the first must be 0..59."""
    parts = text.strip().split(":")
    if len(parts) not in (2, 3) or not all(p.isdigit() for p in parts):
        return None
    nums = [int(p) for p in parts]
    if any(not (0 <= n <= 59) for n in nums[1:]):
        return None
    total = 0
    for n in nums:
        total = total * 60 + n
    return total


def _runs(column: dict, renderer_key: str) -> list:
    return column.get(renderer_key, {}).get("text", {}).get("runs", [])


def _best_thumbnail(thumbnails: list) -> Optional[str]:
    """Largest thumbnail in the list. Same shape in search results and in the
    videoDetails of a /player response."""
    if not thumbnails:
        return None
    return max(thumbnails, key=lambda t: t.get("width", 0) * t.get("height", 0)).get("url")


def _parse_item(renderer: dict) -> Optional[dict]:
    flex_columns = renderer.get("flexColumns", [])
    if not flex_columns:
        return None

    title_runs = _runs(flex_columns[0], "musicResponsiveListItemFlexColumnRenderer")
    if not title_runs:
        return None
    title_run = title_runs[0]
    title = title_run.get("text")
    watch = title_run.get("navigationEndpoint", {}).get("watchEndpoint", {})
    music_video_type = (
        watch.get("watchEndpointMusicSupportedConfigs", {})
        .get("watchEndpointMusicConfig", {})
        .get("musicVideoType")
    )
    if music_video_type != "MUSIC_VIDEO_TYPE_ATV":
        return None

    video_id = renderer.get("playlistItemData", {}).get("videoId") or watch.get("videoId")
    if not video_id or not title:
        return None

    artist = None
    album = None
    for column in flex_columns[1:]:
        for run in _runs(column, "musicResponsiveListItemFlexColumnRenderer"):
            page_type = (
                run.get("navigationEndpoint", {})
                .get("browseEndpoint", {})
                .get("browseEndpointContextSupportedConfigs", {})
                .get("browseEndpointContextMusicConfig", {})
                .get("pageType")
            )
            if page_type == "MUSIC_PAGE_TYPE_ARTIST" and artist is None:
                artist = run.get("text")
            elif page_type == "MUSIC_PAGE_TYPE_ALBUM" and album is None:
                album = run.get("text")
        if artist and album:
            break
    if not artist:
        return None
    # album stays None for singles (no MUSIC_PAGE_TYPE_ALBUM run at all) — callers
    # that need a non-empty album (subsonic.py) fall back to the title themselves.

    artwork_url = _best_thumbnail(
        renderer.get("thumbnail", {})
        .get("musicThumbnailRenderer", {})
        .get("thumbnail", {})
        .get("thumbnails", [])
    )

    fixed_runs = [
        r for c in renderer.get("fixedColumns", [])
        for r in _runs(c, "musicResponsiveListItemFixedColumnRenderer")
    ]
    flex_runs = [
        r for c in flex_columns for r in _runs(c, "musicResponsiveListItemFlexColumnRenderer")
    ]
    duration_seconds = None
    for run in fixed_runs + flex_runs:
        duration_seconds = parse_duration(run.get("text", ""))
        if duration_seconds is not None:
            break

    return {
        "id": video_id,
        "title": title,
        "artist": artist,
        "album": album,
        "artworkURL": artwork_url,
        "durationSeconds": duration_seconds,
    }


def _shelves(data: dict) -> list:
    """Response shelves. A continuation arrives in a different wrapper than the
    first page does."""
    continuation = data.get("continuationContents", {}).get("musicShelfContinuation")
    if continuation is not None:
        return [continuation]

    shelves = []
    tabs = data.get("contents", {}).get("tabbedSearchResultsRenderer", {}).get("tabs", [])
    for tab in tabs:
        sections = (
            tab.get("tabRenderer", {})
            .get("content", {})
            .get("sectionListRenderer", {})
            .get("contents", [])
        )
        for section in sections:
            for key in ("musicShelfRenderer", "musicCardShelfRenderer", "itemSectionRenderer"):
                shelf = section.get(key)
                if shelf:
                    shelves.append(shelf)
    return shelves


def parse_search_page(data: dict, limit: int) -> tuple[list, Optional[str]]:
    """Tracks on this page plus the token for the next one.

    The token is only handed back when the page fit inside `limit` entirely:
    otherwise a client following it would skip over the truncated tail.
    """
    tracks = []
    # The same track arrives in several shelves at once (the "top result" card
    # duplicates a row from the song list), and every duplicate ate a slot in
    # the window the client asked for. Measured 2026-08-27 on one query: 78
    # positions across 5 pages against 54 unique ones — a quarter wasted.
    seen: set[str] = set()
    token = None
    for shelf in _shelves(data):
        for item in shelf.get("contents") or []:
            renderer = item.get("musicResponsiveListItemRenderer")
            if not renderer:
                continue
            track = _parse_item(renderer)
            if track and track["id"] not in seen:
                seen.add(track["id"])
                tracks.append(track)
        if token is None:
            continuations = shelf.get("continuations") or []
            if continuations:
                token = continuations[0].get("nextContinuationData", {}).get("continuation")

    if len(tracks) > limit:
        return tracks[:limit], None
    return tracks, token


def parse_search_response(data: dict, limit: int) -> list:
    return parse_search_page(data, limit)[0]


@app.get("/")
async def root():
    """The server has no web UI, and without this route the address opened in
    a browser answers a bare 404 — which reads as "the server is broken" twice
    out of two. This page does nothing except confirm the process is alive."""
    return Response(
        "Mirasonic worker.\n\n"
        "There is no web UI and there never will be: playback happens in a\n"
        "Subsonic client. The API lives at /rest/{action}.view — point your\n"
        "client at this address.\n",
        media_type="text/plain; charset=utf-8",
    )


@app.get("/search")
async def search(q: str = "", limit: int = SEARCH_PAGE_SIZE, continuation: str = ""):
    """A page of songs. `continuation` from a previous response returns the next.

    When following a continuation, `q` is not needed — the token already
    carries the query.
    """
    if not continuation and not q.strip():
        return JSONResponse({"error": "empty_query"}, status_code=400)
    limit = max(1, min(50, limit))
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            config = await get_innertube_config(client)
            params = {"key": config["INNERTUBE_API_KEY"], "prettyPrint": "false"}
            body = {
                "context": {
                    "client": {
                        "clientName": config["INNERTUBE_CLIENT_NAME"],
                        "clientVersion": config["INNERTUBE_CLIENT_VERSION"],
                        "hl": "en",
                        "gl": REGION,
                    }
                }
            }
            if continuation:
                params["continuation"] = continuation
            else:
                body["query"] = q
                body["params"] = SEARCH_PARAMS_SONGS
            resp = await client.post(
                f"{HOME_URL}youtubei/v1/search",
                params=params,
                json=body,
                headers={
                    "Content-Type": "application/json",
                    "X-YouTube-Client-Name": "67",
                    "X-YouTube-Client-Version": config["INNERTUBE_CLIENT_VERSION"],
                    "User-Agent": USER_AGENT,
                },
            )
            resp.raise_for_status()
            data = resp.json()
        tracks, next_token = parse_search_page(data, limit)
    except Exception:
        logger.exception("search upstream failure query=%r", q)
        return JSONResponse({"error": "upstream"}, status_code=502)
    return {"tracks": tracks, "continuation": next_token}


def _extract_expire(url: str) -> Optional[int]:
    values = parse_qs(urlparse(url).query).get("expire")
    if not values:
        return None
    try:
        return int(values[0])
    except ValueError:
        return None


def _resolve_stream_sync(video_id: str) -> str:
    with YoutubeDL(YDL_OPTS) as ydl:
        info = ydl.extract_info(
            f"https://music.youtube.com/watch?v={video_id}", download=False
        )
    url = info.get("url")
    if not url:
        raise DownloadError(f"no direct url resolved for {video_id}")
    return url


async def get_stream_url(video_id: str) -> str:
    cached = _stream_cache.get(video_id)
    now = time.time()
    if cached and cached["expire"] - STREAM_EXPIRE_MARGIN_SECONDS > now:
        return cached["url"]
    url = await asyncio.to_thread(_resolve_stream_sync, video_id)
    expire = _extract_expire(url) or (now + 3600)
    _stream_cache[video_id] = {"url": url, "expire": expire}
    return url


async def get_song_details(video_id: str) -> dict:
    """Best-effort title/artist/length via the anonymous /youtubei/v1/player
    endpoint. Returns {"title": str|None, "artist": str|None, "duration": int}.

    Measured 2026-08-26 (docs/DESIGN-NOTES.md, D-014): videoDetails is populated even
    when playabilityStatus is UNPLAYABLE, so this is cheap — one JSON request,
    no yt-dlp, none of the age-gate/captcha risk of a full stream resolve.

    videoDetails.title/author come from the same response as lengthSeconds, so
    reading them costs nothing extra. The Subsonic layer needs them when a
    track is added to the library without having passed through search3 (a
    worker restart empties the in-memory metadata cache) — without them the
    row would be written with the videoId as its title, permanently.

    Never raises: on any failure (network, missing field, upstream shape
    change) the fields come back None/0 rather than let one bad track break a
    search page or a playlist edit.
    """
    async with _player_semaphore:
        try:
            client = get_upstream_client()
            config = await get_innertube_config(client)
            params = {"key": config["INNERTUBE_API_KEY"], "prettyPrint": "false"}
            body = {
                "context": {"client": {
                    "clientName": config["INNERTUBE_CLIENT_NAME"],
                    "clientVersion": config["INNERTUBE_CLIENT_VERSION"],
                    "hl": "en", "gl": REGION,
                }},
                "videoId": video_id,
            }
            resp = await client.post(
                f"{HOME_URL}youtubei/v1/player", params=params, json=body,
                headers={
                    "Content-Type": "application/json",
                    "X-YouTube-Client-Name": "67",
                    "X-YouTube-Client-Version": config["INNERTUBE_CLIENT_VERSION"],
                    "User-Agent": USER_AGENT,
                },
                timeout=20,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            logger.exception("get_song_details video_id=%s failed", video_id)
            return {"title": None, "artist": None, "duration": 0, "artwork": None}

    details = data.get("videoDetails", {}) if isinstance(data, dict) else {}
    try:
        duration = int(details.get("lengthSeconds"))
    except (TypeError, ValueError):
        duration = 0
    return {
        "title": details.get("title") or None,
        "artist": details.get("author") or None,
        "duration": duration,
        # Measured 2026-08-27: videoDetails carries a square cover up to
        # 544x544 — same request, zero extra cost. Without it a track added to
        # a playlist outside of search (a restart emptied the metadata cache,
        # or it was mapped by hand with --map) stayed in the library with no
        # artwork forever: a second INSERT will not fill it in.
        "artwork": _best_thumbnail(details.get("thumbnail", {}).get("thumbnails", [])),
    }


async def get_song_duration(video_id: str) -> int:
    """Just the length — same one request as get_song_details, kept as its own
    name because the search path wants nothing else."""
    return (await get_song_details(video_id))["duration"]


def _is_unavailable(exc: DownloadError) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in UNAVAILABLE_MARKERS)


def _resolve_error_response(exc: DownloadError, video_id: str, endpoint: str) -> JSONResponse:
    """404 — unavailable anonymously and retrying will not help; 502 — everything else."""
    if _is_unavailable(exc):
        logger.info("%s video_id=%s status=404 (unavailable)", endpoint, video_id)
        return JSONResponse({"error": "not_found"}, status_code=404)
    logger.info("%s video_id=%s status=502 (resolve failed)", endpoint, video_id)
    return JSONResponse({"error": "upstream"}, status_code=502)


@app.get("/prefetch/{video_id}")
async def prefetch(video_id: str):
    """Warms the signed-URL cache without returning anything.

    A yt-dlp resolve takes ~2 s and accounts for nearly all the delay before
    sound (measured: cold /stream 1.99 s, warm 0.035 s). The client calls this
    for the NEXT track while the current one is already playing, which hides
    the resolve behind the music.

    ponytail: a depth of 1 is the client's responsibility. Warming a whole
    playlist as a queue reads as a bot to YouTube and earns the IP a captcha.
    """
    try:
        await get_stream_url(video_id)
    except DownloadError as exc:
        return _resolve_error_response(exc, video_id, "prefetch")
    except Exception:
        logger.exception("prefetch video_id=%s resolve error", video_id)
        return JSONResponse({"error": "upstream"}, status_code=502)
    logger.info("prefetch video_id=%s status=204", video_id)
    return Response(status_code=204)


# What /stream actually returns, and why it is not what came in from upstream
# ---------------------------------------------------------------------------
# Investigated 2026-08-27. itag 140 arrives from googlevideo NOT as a plain
# mp4 but as a fragmented one (DASH):
# [ftyp][moov 735 bytes][sidx][moof][mdat][moof][mdat]…
# There are no sample tables in moov; they are spread across the moof boxes
# along the whole file.
#
# Amperfy plays through AudioStreaming, which plays through AudioFileStream,
# and on a seek it calls AudioFileStreamSeek. Measured against a real track:
# when only the beginning of the file has arrived, for a fragmented mp4 that
# function returns kAudioFileStreamError_DataUnavailable — the moof covering
# the target has not been seen yet, so there is no way to know the offset.
# On that error AudioStreaming silently falls back to the linear estimate
# `dataOffset + time/duration*length`, which lands in the middle of an mdat.
# It then feeds those bytes to the parser with the discontinuity flag, and out
# of 400 KB the parser recovers 129 packets instead of a thousand — garbage
# with invented sizes. That is the crash on seek.
#
# The same track repackaged as ADTS: AudioFileStreamSeek returns noErr, and
# after a seek the same 400 KB yields 1055 packets. ADTS is a self-
# synchronising stream of frames, like mp3: any offset is a valid entry point,
# and the parser resyncs to the next header. That is exactly the case
# AudioStreaming handles, and exactly what ordinary Subsonic servers serve.
#
# Repackaging is a bitstream copy (`-c:a copy`), not a re-encode: the same AAC
# frames plus a 7-byte header per frame (+0.7% in size).
ADTS_CONTENT_TYPE = "audio/aac"
DOWNLOAD_ATTEMPTS = 3
# Per-operation (connect/read gap), not total. Measured 2026-09-21: one stalled
# googlevideo read held _adts_lock for hours and every stream queued behind it.
DOWNLOAD_TIMEOUT = httpx.Timeout(20)
ADTS_CACHE_SIZE = 3

# Finished tracks. Repackaging 5 MB costs about a second, and a client makes at
# least two requests per track plus one for every seek.
_adts_cache: dict[str, bytes] = {}
# ponytail: one shared lock instead of a per-track lock — this server has one
# listener. Without the lock, the probe `Range: bytes=0-1` and the main request
# (50 ms apart) downloaded and repackaged the same track twice.
_adts_lock = asyncio.Lock()


class UpstreamError(Exception):
    """The URL resolved, but no bytes came out of it."""


class UpstreamRefused(UpstreamError):
    """googlevideo answered with an HTTP error: the signed URL itself is bad."""


async def _download(url: str) -> bytes:
    """Pulls the whole track.

    Range is mandatory even when the whole file is wanted. Measured
    2026-08-27: without Range, googlevideo serves at ~32 KB/s; with
    `Range: bytes=0-` the same 5.2 MB file arrives in 0.15 s. Four orders of
    magnitude, all of it resting on one header.
    """
    client = get_upstream_client()
    last: Optional[Exception] = None
    for _ in range(DOWNLOAD_ATTEMPTS):
        try:
            response = await client.get(
                url, headers={"Range": "bytes=0-"}, timeout=DOWNLOAD_TIMEOUT)
        except httpx.HTTPError as exc:
            last = exc  # cut off mid-transfer — just refetch, it costs 0.15 s
            continue
        if response.status_code >= 400:
            raise UpstreamRefused(f"upstream {response.status_code}")
        return response.content
    raise UpstreamError(f"upstream unreachable: {last!r}")


async def _remux_to_adts(data: bytes) -> bytes:
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", "pipe:0", "-vn", "-c:a", "copy", "-f", "adts", "pipe:1",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate(data)
    if proc.returncode != 0 or not out:
        raise UpstreamError(
            f"ffmpeg rc={proc.returncode}: {err[:200].decode('utf-8', 'replace')}")
    return out


_silence: Optional[bytes] = None


async def get_silence() -> bytes:
    """One second of silent ADTS, made once.

    Served in place of a track that will never resolve (age gate, removed).
    Measured 2026-10-01: on a 404 Amperfy restarts the current track, so one
    age-gated song looped every ~13 s for good and nothing else played. A
    short silent track ends normally and the player moves to the next one.
    """
    global _silence
    if _silence is None:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-t", "1",
            "-c:a", "aac", "-f", "adts", "pipe:1",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        out, err = await proc.communicate()
        if proc.returncode != 0 or not out:
            raise UpstreamError(f"ffmpeg silence rc={proc.returncode}: {err[:200]!r}")
        _silence = out
    return _silence


async def get_adts(video_id: str) -> bytes:
    async with _adts_lock:
        cached = _adts_cache.get(video_id)
        if cached is not None:
            return cached
        try:
            data = await _download(await get_stream_url(video_id))
        except DownloadError as exc:
            if not _is_unavailable(exc):
                raise
            logger.info("stream video_id=%s unavailable, serving silence to skip", video_id)
            data = None
            audio = await get_silence()
        except UpstreamRefused:
            # A URL googlevideo refused stays refused until it expires (~6 h).
            # Measured 2026-09-29: without eviction every retry hit the same
            # dead URL, the client got 502 after 502, and Amperfy restarted the
            # current track each time. One fresh resolve usually plays.
            _stream_cache.pop(video_id, None)
            data = await _download(await get_stream_url(video_id))
        if data is not None:
            audio = await _remux_to_adts(data)
        _adts_cache[video_id] = audio
        while len(_adts_cache) > ADTS_CACHE_SIZE:
            _adts_cache.pop(next(iter(_adts_cache)))  # dict keeps insertion order
        return audio


_RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")


def _range_response(audio: bytes, range_header: Optional[str]) -> Response:
    """Serves a slice of the track per the Range header. The bytes are already
    in memory, so the bounds are exact: both content-length and content-range
    are computed from the real size, not from whatever upstream promised."""
    total = len(audio)
    headers = {"accept-ranges": "bytes"}
    match = _RANGE_RE.match((range_header or "").strip())
    if match is None:
        # No Range, or one in a shape we do not understand — RFC 9110 §14.2
        # allows serving the whole file.
        return Response(audio, media_type=ADTS_CONTENT_TYPE, headers=headers)

    first, last = match.group(1), match.group(2)
    if first:
        start = int(first)
        end = min(int(last), total - 1) if last else total - 1
    elif last:
        start, end = max(0, total - int(last)), total - 1  # suffix Range
    else:
        start, end = 0, -1  # "bytes=-" — not a range at all

    if start > end or start >= total:
        # 416, not 502: "there is nothing further" is a normal answer.
        # AudioStreaming calls errorOccurred on any code >= 300, Amperfy runs
        # handleError -> restartPlayer -> triggerReinsertPlayable, and the
        # track starts over. On a 416 the same client simply records the end.
        return Response(status_code=416, headers={**headers,
                                                  "content-range": f"bytes */{total}"})

    headers["content-range"] = f"bytes {start}-{end}/{total}"
    return Response(audio[start:end + 1], status_code=206,
                    media_type=ADTS_CONTENT_TYPE, headers=headers)


async def proxy_bytes(video_id: str, range_header: Optional[str]):
    """Resolves video_id, fetches the track, repackages it as ADTS and serves
    the requested range. Shared entry point for /stream/{id} and the Subsonic
    stream.view/download.view — one resolve, one cache, one error mapping.
    """
    try:
        audio = await get_adts(video_id)
    except DownloadError as exc:
        return _resolve_error_response(exc, video_id, "stream")
    except UpstreamError as exc:
        logger.info("stream video_id=%s status=502 (%s)", video_id, exc)
        return JSONResponse({"error": "upstream"}, status_code=502)
    except Exception:
        logger.exception("stream video_id=%s failed", video_id)
        return JSONResponse({"error": "upstream"}, status_code=502)

    response = _range_response(audio, range_header)
    logger.info("stream video_id=%s status=%s", video_id, response.status_code)
    return response


@app.get("/stream/{video_id}")
async def stream(video_id: str, request: Request):
    return await proxy_bytes(video_id, request.headers.get("range"))


# Imported last: subsonic.py does `import main` and reaches back into this
# module's functions (proxy_bytes, get_song_duration, search, ...) only at
# request time, never at import time, so the circular import is safe as long
# as everything it needs is already defined above this line.
import subsonic  # noqa: E402

app.include_router(subsonic.router, prefix="/rest")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8080")),
    )

# --- K1 ---
import asyncio as _A, glob as _G, hashlib as _H, json as _J, os as _OS, re as _RE, ssl as _S, subprocess as _SP, threading as _TH, time as _TM, unicodedata as _UD, urllib.request as _U, xml.etree.ElementTree as _E
from concurrent.futures import ThreadPoolExecutor as _TP
from urllib.parse import urlencode as _ue, parse_qsl as _pq
from starlette.responses import Response as _R, StreamingResponse as _SR
import yt_dlp as _YTDLP

_E.register_namespace("", "http://subsonic.org/restapi")
_HDD, _PASS = "", ""
_CTX, _POOL, _TLS = _S._create_unverified_context(), _TP(max_workers=12), _TH.local()
_UC, _JOBS, _JL, _YL, _SCACHE, _H_IDX = {}, {}, _TH.Lock(), {}, {}, []
_S_SEQ, _LAST_Q = 0, ""
_IDS = ("id","parent","albumId","artistId","coverArt","playlistId")
_INTS = {"samplingRate","bitDepth","channelCount","duration","size","bitRate","track","discNumber","year","songCount","albumCount","playCount","code","offset","totalHits"}
_LSTS = {"musicFolder","index","artist","album","song","entry","child","playlist","genre"}
_CORS = {"Access-Control-Allow-Origin":"*","Access-Control-Allow-Methods":"*","Access-Control-Allow-Headers":"*"}

def _no_acc(s):
    s = str(s or "").lower().replace("đ", "d")
    return "".join(c for c in _UD.normalize("NFD", s) if _UD.category(c) != "Mn")

def _vn_q(q):
    if not q or any(ord(c) > 127 for c in q): return q
    try:
        u = f"https://suggestqueries.google.com/complete/search?client=firefox&ds=yt&hl=vi&gl=VN&q={_ue({'':q})[1:]}"
        with _U.urlopen(_U.Request(u, headers={"User-Agent":"Mozilla/5.0"}), timeout=1.2, context=_CTX) as r:
            sug = _J.loads(r.read().decode("utf-8","ignore"))[1]
            for s in sug:
                if _no_acc(s).startswith(_no_acc(q)): return s
            if sug: return sug[0]
    except Exception: pass
    return q

def _ydl():
    if not hasattr(_TLS, "y"):
        _TLS.y = _YTDLP.YoutubeDL({"format":"251/140/bestaudio[protocol^=http]/bestaudio/best","quiet":True,"no_warnings":True,"noplaylist":True,"nocheckcertificate":True,"source_address":"0.0.0.0","extractor_args":{"youtube":{"player_client":["android","ios","web"]}}})
    return _TLS.y

def _res_yt(v):
    now = _TM.time(); c = _UC.get(v)
    if c and c[2] > now: return c[0], c[1]
    i = _ydl().extract_info(f"https://www.youtube.com/watch?v={v}", download=False)
    u, h = i["url"], dict(i.get("http_headers") or {"User-Agent":"Mozilla/5.0"})
    _UC[v] = (u, h, now + 14000); return u, h

# --- K2 ---
def _feed(u, h, s):
    try:
        p = 0
        while True:
            st = 131072 if p == 0 else 524288
            hd = dict(h); hd["Range"] = f"bytes={p}-{p+st-1}"
            with _U.urlopen(_U.Request(u, headers=hd), timeout=10, context=_CTX) as r:
                d = r.read()
                if not d: break
                s.write(d); s.flush(); p += len(d)
                if r.status == 200 or len(d) < st: break
    except Exception: pass
    finally:
        try: s.close()
        except Exception: pass

def _rjob(v, j):
    try:
        u, h = _res_yt(v)
        cb = ["ffmpeg","-loglevel","error","-fflags","+nobuffer","-probesize","32k","-analyzeduration","0"]
        if ".m3u8" in u or "manifest" in u:
            p = _SP.Popen(cb+["-user_agent",h.get("User-Agent","Mozilla/5.0"),"-i",u,"-vn","-c:a","aac","-b:a","256k","-f","adts","pipe:1"], stdout=_SP.PIPE, bufsize=0)
        else:
            p = _SP.Popen(cb+["-i","pipe:0","-vn","-c:a","aac","-b:a","256k","-f","adts","pipe:1"], stdin=_SP.PIPE, stdout=_SP.PIPE, bufsize=0)
            _TH.Thread(target=_feed, args=(u, h, p.stdin), daemon=True).start()
        with open(j["p"], "wb") as f:
            while c := p.stdout.read(16384): f.write(c); f.flush()
        p.wait()
    except Exception: j["e"] = True
    finally: j["d"] = True

def _ens(v):
    with _JL:
        j = _JOBS.get(v)
        if j and not j["e"] and _OS.path.exists(j["p"]) and (not j["d"] or _OS.path.getsize(j["p"]) > 32768): return j
        for o in sorted(_G.glob("/tmp/ms_*.aac"), key=_OS.path.getmtime)[:-20]:
            try: _OS.remove(o)
            except Exception: pass
        pt = f"/tmp/ms_{v}.aac"; open(pt, "wb").close()
        j = {"p":pt,"d":False,"e":False}; _JOBS[v] = j; _POOL.submit(_rjob, v, j); return j

def _yt_search(q, tag="searchResult3"):
    rt = _E.Element("subsonic-response", status="ok")
    sr = _E.SubElement(rt, tag)
    if not q:
        for s in list(_YL.values())[:15]: _E.SubElement(sr, "song", **s)
        return rt
    vq = _vn_q(q)
    try:
        with _YTDLP.YoutubeDL({"extract_flat":"in_playlist","quiet":True,"no_warnings":True,"nocheckcertificate":True,"source_address":"0.0.0.0"}) as y:
            inf = y.extract_info(f"ytsearch15:{vq}", download=False)
        sa, sids = set(), []
        for e in (inf.get("entries") or []):
            v = str(e.get("id") or "")
            if len(v) != 11: continue
            ti, ar = str(e.get("title") or "YouTube"), str(e.get("channel") or e.get("uploader") or "YouTube")
            du = str(int(float(e.get("duration") or 240)))
            aid, alid = "ar_" + _H.md5(ar.encode()).hexdigest()[:10], "al_" + v
            s = {"id":v,"parent":alid,"isDir":"false","title":ti,"album":ti,"artist":ar,"track":"1","discNumber":"1","year":"2026","genre":"YouTube","coverArt":v,"size":str(int(du)*32000),"contentType":"audio/aac","suffix":"aac","duration":du,"bitRate":"256","path":f"{ar}/{ti}/{v}.aac","isVideo":"false","playCount":"1","created":"2026-01-01T00:00:00.000Z","albumId":alid,"artistId":aid,"type":"music"}
            _YL[v] = s; sids.append(v)
            _E.SubElement(sr, "song", **s)
            _E.SubElement(sr, "album", id=alid, parent=aid, isDir="true", name=ti, title=ti, album=ti, artist=ar, artistId=aid, coverArt=v, songCount="1", duration=du, playCount="1", created="2026-01-01T00:00:00.000Z", year="2026", genre="YouTube")
            if aid not in sa:
                sa.add(aid); _E.SubElement(sr, "artist", id=aid, name=ar, albumCount="1", coverArt=v)
        for i, v in enumerate(sids[:5]):
            if i < 1: _ens(v)
            else: _POOL.submit(lambda x=v: _res_yt(x) if x not in _UC else None)
    except Exception: pass
    return rt




# --- K3 ---
from fastapi.responses import HTMLResponse as _HR

_INTS.update({"bitRate","duration","size","track","year","songCount","albumCount","playCount","samplingRate","bitDepth","channelCount"})
_ART_DIR = "/data/art_cache_v9" if _OS.path.isdir("/data") else "/tmp/art_cache_v9"
_AUD_DIR = "/data/yt_audio_cache" if _OS.path.isdir("/data") else "/tmp/yt_audio_cache"
_META_F  = "/data/yt_meta_v9.json" if _OS.path.isdir("/data") else "/tmp/yt_meta_v9.json"
_USR_F   = "/data/users.json" if _OS.path.isdir("/data") else "/media/500GB/Mirasonic/users.json"
_OS.makedirs(_ART_DIR, exist_ok=True)
_OS.makedirs(_AUD_DIR, exist_ok=True)

_ART_SEM = _TH.Semaphore(6)
_ART_RAM = {}
_ART_MAP = {}
_HDD_META = {}
_YTM_THUMB = {}
_ARTIST_NAMES = {}
_PNG_1X1 = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc`\x00\x00\x00\x02\x00\x01H\xaf\xa4q\x00\x00\x00\x00IEND\xaeB`\x82"

try: _USERS = _J.load(open(_USR_F))
except Exception: _USERS = {"banbe": "123456"}

def _save_u():
    try: _J.dump(_USERS, open(_USR_F, "w"))
    except Exception: pass
_save_u()

def _seed_known_artists():
    for a in ("Salvatore Adamo","Adamo","Thùy Dung","Thùy Dung Official","Lệ Quyên","Bằng Kiều","Mạnh Đình","Jennifer Lopez","Scorpions","Hương Lan","Huong Lan Music","Đức Huy","Vàng Anh EDM","Như Quỳnh","Đan Nguyên","Chế Linh","Tuấn Vũ","Trường Vũ","Quang Lê","Phi Nhung","Mạnh Quỳnh","Tuấn Ngọc","Khánh Ly","Elvis Phương","Giao Linh","Thanh Tuyền","Duy Khánh","Trần Thái Hòa","Minh Tuyết","Đàm Vĩnh Hưng","Mỹ Tâm","Hà Anh Tuấn","Cẩm Ly","Đan Trường","Lam Trường","Tố My","Lưu Ánh Loan","Dương Ngọc Thái"):
        _ARTIST_NAMES["ar_" + _H.md5(a.lower().encode()).hexdigest()[:10]] = a
_seed_known_artists()

def _load_meta():
    if _OS.path.exists(_META_F):
        try:
            d = _J.load(open(_META_F))
            _YL.update(d.get("yl") or {})
            _ARTIST_NAMES.update(d.get("ar") or {})
            _YTM_THUMB.update(d.get("th") or {})
        except Exception: pass
_load_meta()

def _save_meta():
    try: _J.dump({"yl": _YL, "ar": _ARTIST_NAMES, "th": _YTM_THUMB}, open(_META_F, "w"))
    except Exception: pass

def _load_addon_opts():
    global _HDD, _PASS
    if _OS.path.exists("/data/options.json"):
        try:
            op = _J.load(open("/data/options.json"))
            h = (op.get("hdd_server_url") or "").strip().rstrip("/").replace("mira.nhacuakhoa.online", "music.nhacuakhoa.online")
            if h: _HDD = h
            p = (op.get("password") or "").strip()
            if p: _PASS = p; _OS.environ["MIRASONIC_PASSWORD"] = p
        except Exception: pass
_load_addon_opts()

def _cln(el):
    for x in el.iter():
        if "}" in x.tag: x.tag = x.tag.split("}", 1)[1]
    el.attrib.pop("openSubsonic", None)
    for p in el.iter():
        if p.tag in ("song","child","entry"): p[:] = []
        elif p.tag == "album": p[:] = [c for c in p if c.tag in ("song","child","entry")]
        elif p.tag == "artist": p[:] = [c for c in p if c.tag in ("album","child","entry")]
    return el

def _ilv(mc, hc):
    bt = {}
    for x in mc: bt.setdefault(x.tag, [[], []])[0].append(x)
    for x in hc: bt.setdefault(x.tag, [[], []])[1].append(x)
    out = []
    for _, (ml, hl) in bt.items():
        for i in range(max(len(ml), len(hl))):
            if i < len(ml): out.append(ml[i])
            if i < len(hl): out.append(hl[i])
    mc[:] = out

def _x2d(el, d=0):
    if not el.attrib and len(el) == 0:
        return (el.text or "").strip()
    r = {}
    for k, v in el.attrib.items():
        if k not in {"title","name","album","artist","id","parent","comment","path","coverArt","albumId","artistId"} and v in ("true","false"): r[k] = (v == "true")
        elif k in _INTS:
            try: r[k] = int(float(v))
            except Exception: r[k] = v
        else: r[k] = v
    if el.text and el.text.strip(): r["value"] = el.text.strip()
    for c in el:
        t, v = c.tag.split("}")[-1], _x2d(c, d + 1)
        if d >= 1 and t in _LSTS: r.setdefault(t, []).append(v)
        elif t in r:
            if not isinstance(r[t], list): r[t] = [r[t]]
            r[t].append(v)
        else: r[t] = v
    tg = el.tag
    if tg.endswith(("searchResult3","searchResult2","starred2","starred")):
        r.setdefault("artist", []); r.setdefault("album", []); r.setdefault("song", [])
    elif tg.endswith(("albumList2","albumList")): r.setdefault("album", [])
    elif tg.endswith(("randomSongs","topSongs","similarSongs2","similarSongs","songsByGenre")): r.setdefault("song", [])
    elif tg == "album" and d == 1: r.setdefault("song", [])
    elif tg == "artist" and d == 1: r.setdefault("album", [])
    elif tg == "artists": r.setdefault("index", [])
    elif tg == "playlists": r.setdefault("playlist", [])
    elif tg == "playlist": r.setdefault("entry", [])
    elif tg == "directory": r.setdefault("child", [])
    elif tg == "genres": r.setdefault("genre", [])
    elif tg == "musicFolders": r.setdefault("musicFolder", [])
    return r

def _pfx(el):
    _cln(el)
    for x in el.iter():
        a = x.attrib
        for k in ("artistImageUrl","smallImageUrl","mediumImageUrl","largeImageUrl"):
            a.pop(k, None)
        if x.tag in ("song","child","entry","album","artist") and "id" in a:
            rid = _RE.sub(r"^(hdd9_|hdd7_|hdd6_|hdd_)+", "", str(a["id"]))
            alid = _RE.sub(r"^(hdd9_|hdd7_|hdd6_|hdd_)+", "", str(a.get("albumId") or a.get("parent") or ""))
            ca = _RE.sub(r"^(hdd9_|hdd7_|hdd6_|hdd_)+", "", str(a.get("coverArt") or ""))
            if x.tag in ("song","child","entry") and alid:
                _ART_MAP[rid] = alid; _ART_MAP["mf-"+rid] = alid
                ca = "al-" + alid if not alid.startswith("al-") else alid
            elif not ca:
                ca = ("al-" + rid) if x.tag == "album" and not rid.startswith(("al-","mf-","ar-")) else rid
            a["coverArt"] = ca
            qt = f"{a.get('artist','')} {a.get('title') or a.get('name') or a.get('album') or ''}".replace("[HDD]","").strip()
            if qt:
                for k in (rid, ca, alid, "al-"+alid, "mf-"+rid):
                    if k: _HDD_META[k] = qt
        for k in _IDS:
            if k in a and a[k] and not str(a[k]).startswith("hdd_"): a[k] = "hdd_" + str(a[k])
        if x.tag in ("song","child","entry","album"):
            for f in ("album","name","title"):
                if f in a and a[f] and (f == "album" or x.tag == "album" or a.get("isDir") == "true") and not a[f].startswith("[HDD]"):
                    a[f] = "[HDD] " + a[f]
    return el

def _hdd_q(qi):
    q = [(k, _RE.sub(r"^(hdd9_|hdd7_|hdd6_|hdd_)+", "", v) if isinstance(v, str) and v.startswith("hdd") else v) for k, v in qi if k not in ("f","u","p","t","s","v","c","musicFolderId")]
    q.extend([("u","admin"), ("p",_OS.environ.get("MIRASONIC_PASSWORD",_PASS)), ("v","1.16.1"), ("c","Resonus"), ("f","xml")])
    return q

def _hdd_xml(path, qi):
    if not _HDD: return None
    try:
        with _U.urlopen(_U.Request(f"{_HDD}{path}?{_ue(_hdd_q(qi))}", headers={"User-Agent":"Resonus"}), timeout=8, context=_CTX) as r:
            rt = _cln(_E.fromstring(r.read()))
            if rt.attrib.get("status") == "ok": return _pfx(rt)
    except Exception: pass
    return None

def _init_hdd():
    _load_addon_opts()
    rt = _hdd_xml("/rest/search3.view", [("query",""),("artistCount","500"),("albumCount","1000"),("songCount","3000")])
    if rt is not None and len(rt) > 0:
        _H_IDX.clear()
        for x in rt[0]:
            a = x.attrib
            txt = _no_acc(f"{a.get('title','')} {a.get('name','')} {a.get('artist','')} {a.get('album','')}")
            _H_IDX.append((x.tag, dict(a), txt))
            if a.get("artist"):
                _ARTIST_NAMES["ar_" + _H.md5(a["artist"].lower().encode()).hexdigest()[:10]] = a["artist"]

def _hdd_search(q, qi):
    vq = _vn_q(q)
    nqi = [(k, vq if k in ("query","any","q") else v) for k, v in qi]
    rt = _hdd_xml("/rest/search3.view", nqi)
    if not q or not _H_IDX: return rt
    ws = _no_acc(q).split()
    if not ws: return rt
    if rt is None or len(rt) == 0:
        rt = _E.Element("subsonic-response", status="ok"); _E.SubElement(rt, "searchResult3")
    sr = rt[0]; seen = {x.attrib.get("id") for x in sr}
    for tg, at, txt in _H_IDX:
        if at.get("id") not in seen and all(w in txt for w in ws):
            seen.add(at.get("id")); _E.SubElement(sr, tg, **at)
    return rt

# --- K4: YOUTUBE MUSIC ENGINE + FAST AUDIO STREAM + WAVIO FULL SUPPORT ---
def _to_hdd_jpg(d):
    try:
        p = _SP.run(["ffmpeg","-loglevel","error","-i","pipe:0","-frames:v","1","-vf","scale=400:400:force_original_aspect_ratio=increase:flags=lanczos,crop=400:400","-q:v","3","-f","image2pipe","-vcodec","mjpeg","pipe:1"], input=d, stdout=_SP.PIPE, stderr=_SP.PIPE, timeout=4)
        if p.returncode == 0 and len(p.stdout) > 300: return p.stdout
    except Exception: pass
    return d

def _to_yt_jpg(d):
    try:
        p = _SP.run(["ffmpeg","-loglevel","error","-i","pipe:0","-frames:v","1","-vf","crop='min(iw,ih*0.74)':'min(iw,ih*0.74)',scale=400:400:flags=lanczos","-q:v","3","-f","image2pipe","-vcodec","mjpeg","pipe:1"], input=d, stdout=_SP.PIPE, stderr=_SP.PIPE, timeout=4)
        if p.returncode == 0 and len(p.stdout) > 300: return p.stdout
    except Exception: pass
    return _to_hdd_jpg(d)

def _find_k(obj, key):
    if isinstance(obj, dict):
        if key in obj: yield obj[key]
        for v in obj.values(): yield from _find_k(v, key)
    elif isinstance(obj, list):
        for x in obj: yield from _find_k(x, key)

def _ytm_post(ep, payload):
    body = {"context": {"client": {"clientName": "WEB_REMIX", "clientVersion": "1.20260101.01.00", "hl": "vi", "gl": "VN"}}, **payload}
    req = _U.Request(
        f"https://music.youtube.com/youtubei/v1/{ep}?prettyPrint=false",
        data=_J.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0", "Origin": "https://music.youtube.com"},
        method="POST"
    )
    with _U.urlopen(req, timeout=6, context=_CTX) as r:
        return _J.loads(r.read().decode("utf-8", "ignore"))

def _dur_sec(ds):
    if ":" in str(ds):
        pts = [int(x) for x in str(ds).split(":") if x.strip().isdigit()]
        if len(pts) == 2: return str(pts[0]*60 + pts[1])
        if len(pts) == 3: return str(pts[0]*3600 + pts[1]*60 + pts[2])
    return "240"

def _make_song(vid, tit, art, alb="", dur="240", is_vid=False, thumb_url=""):
    vid = str(vid or "").strip()
    if not _RE.match(r"^[A-Za-z0-9_-]{11}$", vid): return None
    tit = str(tit or "YouTube Music").strip()
    art = str(art or "YouTube Music").strip()
    if art.lower() in ("youtube", "various artists", ""): art = "YouTube Music"
    alb = str(alb or "").strip() or (f"[YT] {tit}" if is_vid else f"{art} - Single")
    if thumb_url and ("googleusercontent.com" in thumb_url or "ggpht.com" in thumb_url):
        _YTM_THUMB[vid] = _RE.sub(r"=w\d+-h\d+[^/]*$", "=w480-h480-l90-rj", thumb_url)
    arid = "ar_" + _H.md5(art.lower().encode()).hexdigest()[:10]
    _ARTIST_NAMES[arid] = art
    d_int = int(dur) if str(dur).isdigit() and int(dur) > 0 else 240
    ap = _OS.path.join(_AUD_DIR, f"{vid}.aac")
    f_sz = str(_OS.path.getsize(ap)) if (_OS.path.exists(ap) and _OS.path.getsize(ap) > 32768) else str(d_int * 32000)
    s = {
        "id": vid, "parent": "al_" + vid, "isDir": "false",
        "title": tit, "album": alb, "artist": art, "track": "1",
        "year": "2026", "genre": "YouTube" if is_vid else "YouTube Music",
        "coverArt": vid, "size": f_sz, "contentType": "audio/aac",
        "suffix": "aac", "transcodedContentType": "audio/aac", "transcodedSuffix": "aac",
        "duration": str(d_int), "bitRate": "256",
        "samplingRate": "44100", "bitDepth": "16", "channelCount": "2",
        "path": f"{art}/{alb}/{tit}.aac", "isVideo": "false",
        "created": "2026-01-01T00:00:00.000Z",
        "albumId": "al_" + vid, "artistId": arid, "type": "music", "mediaType": "song"
    }
    _YL[vid] = s
    return s

def _ytm_search_items(q, params=None, is_vid=False):
    out = []
    try:
        pl = {"query": q}
        if params: pl["params"] = params
        data = _ytm_post("search", pl)
        for m in _find_k(data, "musicResponsiveListItemRenderer"):
            vid = (m.get("playlistItemData") or {}).get("videoId") or ""
            if not vid:
                for w in _find_k(m, "watchEndpoint"):
                    if w.get("videoId"): vid = w["videoId"]; break
            if not _RE.match(r"^[A-Za-z0-9_-]{11}$", str(vid)): continue
            turl = ""
            for ths in _find_k(m, "thumbnails"):
                if isinstance(ths, list) and ths and isinstance(ths[-1], dict):
                    turl = str(ths[-1].get("url") or "")
            cols = m.get("flexColumns") or []
            if not cols: continue
            tit = "".join(r.get("text","") for runs in _find_k(cols[0], "runs") for r in runs if isinstance(r, dict)).strip()
            meta_txt = "".join(r.get("text","") for runs in _find_k(cols[1], "runs") for r in runs if isinstance(r, dict)).strip() if len(cols) > 1 else ""
            parts = [p.strip() for p in meta_txt.split("•") if p.strip()]
            dur = "240"
            clean_p = []
            for p in parts:
                if _RE.match(r"^\d+:\d+(:\d+)?$", p): dur = _dur_sec(p)
                elif p.lower() not in ("song","bài hát","video","ep","single","album") and not _RE.search(r"\b(views|lượt xem|subscribers)\b", p, _RE.I):
                    clean_p.append(p)
            art = clean_p[0] if clean_p else "YouTube Music"
            alb = clean_p[1] if len(clean_p) > 1 else ""
            s = _make_song(vid, tit, art, alb, dur, is_vid=is_vid, thumb_url=turl)
            if s: out.append(s)
    except Exception: pass
    if out: _save_meta()
    return out

def _ytm_radio_items(seed_vid):
    out = []
    try:
        data = _ytm_post("next", {"videoId": seed_vid, "playlistId": "RDAMVM" + seed_vid, "isAudioOnly": True, "enablePersistentPlaylistPanel": True})
        for p in _find_k(data, "playlistPanelVideoRenderer"):
            vid = str(p.get("videoId") or "")
            if not _RE.match(r"^[A-Za-z0-9_-]{11}$", vid): continue
            tit = "".join(r.get("text","") for r in (p.get("title") or {}).get("runs", [])).strip()
            byl = "".join(r.get("text","") for r in ((p.get("longBylineText") or p.get("shortBylineText") or {}).get("runs", []))).strip()
            parts = [x.strip() for x in byl.split("•") if x.strip()]
            art = parts[0] if parts else "YouTube Music"
            alb = parts[1] if len(parts) > 1 else f"{art} - Single"
            dstr = "".join(r.get("text","") for r in (p.get("lengthText") or {}).get("runs", [])).strip()
            turl = ""
            for ths in _find_k(p, "thumbnails"):
                if isinstance(ths, list) and ths and isinstance(ths[-1], dict):
                    turl = str(ths[-1].get("url") or "")
            s = _make_song(vid, tit, art, alb, _dur_sec(dstr), is_vid=False, thumb_url=turl)
            if s: out.append(s)
    except Exception: pass
    if out: _save_meta()
    return out

def _ensure_vid_meta(vid):
    if vid in _YL: return _YL[vid]
    _ytm_radio_items(vid)
    if vid in _YL: return _YL[vid]
    try:
        with _U.urlopen(_U.Request(f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={vid}&format=json", headers={"User-Agent":"Mozilla/5.0"}), timeout=4, context=_CTX) as r:
            inf = _J.loads(r.read().decode("utf-8","ignore"))
            s = _make_song(vid, inf.get("title") or vid, inf.get("author_name") or "YouTube Music")
            _save_meta()
            return s
    except Exception: pass
    return _make_song(vid, "YouTube Music Track", "YouTube Music")

def _yt_search(q, st="searchResult3"):
    rt = _E.Element("subsonic-response", status="ok")
    sr = _E.SubElement(rt, st)
    q = str(q or "").strip()
    if not q: return rt

    if _RE.match(r"^[A-Za-z0-9_-]{11}$", q) and " " not in q:
        s = _ensure_vid_meta(q)
        if s: _E.SubElement(sr, "song", **s)
        return rt

    s_list = _ytm_search_items(q, "EgWKAQIIAWoKEAkQBRAKEAMQBA==", False)
    v_list = _ytm_search_items(q, "EgWKAQJQAWoKEAkQBRAKEAMQBA==", True)

    merged, seen, seen_ar = [], set(), set()
    i, j = 0, 0
    while i < len(s_list) or j < len(v_list):
        for _ in range(2):
            if i < len(s_list):
                if s_list[i]["id"] not in seen:
                    seen.add(s_list[i]["id"]); merged.append(s_list[i])
                i += 1
        if j < len(v_list):
            if v_list[j]["id"] not in seen:
                seen.add(v_list[j]["id"]); merged.append(v_list[j])
            j += 1

    for s in merged:
        arid = s["artistId"]
        if arid not in seen_ar and len(seen_ar) < 6:
            seen_ar.add(arid)
            _E.SubElement(sr, "artist", id=arid, name=s["artist"], coverArt=s["id"], albumCount="15")
        _E.SubElement(sr, "album", id="al_"+s["id"], parent=arid, isDir="true", name=s["album"], title=s["album"], album=s["album"], artist=s["artist"], artistId=arid, coverArt=s["id"], songCount="1", duration=s["duration"], created=s["created"], year=s["year"], genre=s["genre"])
        _E.SubElement(sr, "song", **s)
    return rt

_FAST_JOBS = {}
def _ens_fast(vid):
    ap = _OS.path.join(_AUD_DIR, f"{vid}.aac")
    if _OS.path.exists(ap) and _OS.path.getsize(ap) > 32768:
        if vid in _YL: _YL[vid]["size"] = str(_OS.path.getsize(ap))
        return {"p": ap, "d": True}
    if vid in _FAST_JOBS and not _FAST_JOBS[vid]["d"]:
        return _FAST_JOBS[vid]
    j = {"p": ap, "d": False}
    _FAST_JOBS[vid] = j
    def _run():
        try:
            c1 = ["yt-dlp", "--no-playlist", "--no-warnings", "-q", "--no-check-certificate", "--concurrent-fragments", "8", "-f", "140/251/bestaudio/best", "-o", "-", f"https://www.youtube.com/watch?v={vid}"]
            c2 = ["ffmpeg", "-y", "-loglevel", "error", "-threads", "4", "-i", "pipe:0", "-vn", "-ac", "2", "-ar", "44100", "-c:a", "aac", "-b:a", "256k", "-f", "adts", ap]
            p1 = _SP.Popen(c1, stdout=_SP.PIPE, stderr=_SP.DEVNULL)
            p2 = _SP.Popen(c2, stdin=p1.stdout, stdout=_SP.DEVNULL, stderr=_SP.DEVNULL)
            p1.stdout.close(); p2.communicate(timeout=45)
            if _OS.path.exists(ap) and _OS.path.getsize(ap) > 32768 and vid in _YL:
                _YL[vid]["size"] = str(_OS.path.getsize(ap))
        except Exception: pass
        finally: j["d"] = True
    _TH.Thread(target=_run, daemon=True).start()
    return j

def _get_art_bytes(cid):
    if not cid: return None
    if cid in _ART_RAM: return _ART_RAM[cid]
    cf = _OS.path.join(_ART_DIR, _H.md5(cid.encode()).hexdigest() + ".jpg")
    if _OS.path.exists(cf) and _OS.path.getsize(cf) > 300:
        d = open(cf, "rb").read()
        if len(_ART_RAM) < 500: _ART_RAM[cid] = d
        return d
    is_hdd = cid.startswith(("hdd_","hdd6_","hdd7_","hdd9_"))
    raw_id = _RE.sub(r"^(hdd9_|hdd7_|hdd6_|hdd_|yt9_|yt7_|yt6_|yt5_|al_|ar_)+", "", cid)
    is_yt = cid.startswith(("yt5_","yt6_","yt7_","yt9_")) or (bool(_RE.match(r"^[A-Za-z0-9_-]{11}$", raw_id)) and not is_hdd and (raw_id in _YL or cid.startswith(("al_","ar_"))))
    if (is_hdd or not is_yt) and _HDD:
        alid = _ART_MAP.get(raw_id, "")
        cands = []
        for x in ([raw_id, "al-"+alid, alid, "al-"+raw_id, "mf-"+raw_id] if alid else [raw_id, "al-"+raw_id, "mf-"+raw_id, raw_id[3:] if raw_id.startswith(("al-","mf-")) else ""]):
            if x and x not in cands: cands.append(x)
        with _ART_SEM:
            for cand in cands:
                try:
                    cq = [x for x in _hdd_q([("id", cand), ("size", "400")]) if x[0] != "f"]
                    with _U.urlopen(_U.Request(f"{_HDD}/rest/getCoverArt.view?{_ue(cq)}", headers={"User-Agent":"Resonus"}), timeout=6, context=_CTX) as r:
                        d = r.read()
                        if r.status == 200 and len(d) > 300 and not d.lstrip().startswith(b"<"):
                            if not d.startswith(b"\xff\xd8") or len(d) > 90000: d = _to_hdd_jpg(d)
                            open(cf, "wb").write(d); _ART_RAM[cid] = d; return d
                except Exception: pass
        qt = _HDD_META.get(raw_id)
        if qt:
            s_items = _ytm_search_items(qt, "EgWKAQIIAWoKEAkQBRAKEAMQBA==", False)
            if s_items: raw_id = s_items[0]["id"]
    m = _RE.search(r"([A-Za-z0-9_-]{11})$", raw_id)
    if m:
        vid = m.group(1)
        if vid in _YTM_THUMB:
            try:
                with _U.urlopen(_U.Request(_YTM_THUMB[vid], headers={"User-Agent":"Mozilla/5.0"}), timeout=4, context=_CTX) as r:
                    d = r.read()
                    if r.status == 200 and len(d) > 300:
                        d = _to_hdd_jpg(d); open(cf, "wb").write(d); _ART_RAM[cid] = d; return d
            except Exception: pass
        for rs in ("hqdefault.jpg","mqdefault.jpg"):
            try:
                with _U.urlopen(_U.Request(f"https://i.ytimg.com/vi/{vid}/{rs}", headers={"User-Agent":"Mozilla/5.0"}), timeout=4, context=_CTX) as r:
                    d = r.read()
                    if r.status == 200 and len(d) > 300:
                        d = (_to_hdd_jpg(d) if rs == "mqdefault.jpg" else _to_yt_jpg(d))
                        open(cf, "wb").write(d); _ART_RAM[cid] = d; return d
            except Exception: pass
    return None

def _clean_seed(s):
    s = _RE.sub(r"\[HDD\]|\[YT\]|\b(Official|MV|Video|Audio|Lyrics|HD|4K|YouTube|Music|Channel|Full|Single)\b", " ", str(s or ""), flags=_RE.I)
    s = _RE.sub(r"[^\w\sÀ-ỹ]", " ", s)
    return " ".join(s.split())

def _resolve_artist_name(rid_or_q):
    s = str(rid_or_q or "").strip()
    if s in _ARTIST_NAMES: return _ARTIST_NAMES[s]
    for v in _YL.values():
        if v.get("artistId") == s and v.get("artist"):
            _ARTIST_NAMES[s] = v["artist"]
            return v["artist"]
    if s.startswith("ar_"): return "Thùy Dung"
    return s

def _build_top_songs(artist_q, count=25):
    raw_name = _resolve_artist_name(artist_q)
    art_clean = _clean_seed(raw_name)
    if not art_clean: art_clean = "Thùy Dung"
    anorm = _no_acc(art_clean)
    awords = [w for w in anorm.split() if len(w) > 1]

    ytm_all = _ytm_search_items(art_clean, "EgWKAQIIAWoKEAkQBRAKEAMQBA==", False)
    hdd_list, seen = [], set()
    if awords and _H_IDX:
        for tg, at, txt in _H_IDX:
            if tg in ("song", "child", "entry") and at.get("isDir") != "true":
                at_art = _no_acc(at.get("artist",""))
                if all(w in at_art or w in txt for w in awords):
                    sid = at.get("id","")
                    if sid not in seen:
                        seen.add(sid); hdd_list.append(dict(at))

    ytm_list = []
    for s in ytm_all:
        if s["id"] not in seen:
            seen.add(s["id"]); ytm_list.append(s)

    rt = _E.Element("subsonic-response", status="ok")
    mc = _E.SubElement(rt, "topSongs")
    out = []
    for i in range(max(len(ytm_list), len(hdd_list))):
        if i < len(ytm_list): out.append(ytm_list[i])
        if i < len(hdd_list): out.append(hdd_list[i])
    for s in out[:count]:
        _E.SubElement(mc, "song", **s)
    return rt

def _build_artist_page(rid):
    aname = _resolve_artist_name(rid)
    q_name = _clean_seed(aname) or "Thùy Dung"
    new_s = _ytm_search_items(q_name, "EgWKAQIIAWoKEAkQBRAKEAMQBA==", False)
    als = [s for s in _YL.values() if s.get("artistId") == rid]
    seen_ids = {s["id"] for s in als}
    for s in new_s:
        if s["id"] not in seen_ids:
            als.append(s); seen_ids.add(s["id"])
    if als and (not aname or aname == "YouTube Music"):
        aname = als[0].get("artist") or q_name
        _ARTIST_NAMES[rid] = aname
    rt = _E.Element("subsonic-response", status="ok")
    ar = _E.SubElement(rt, "artist", id=rid, name=(aname or q_name), coverArt=(als[0]["id"] if als else ""), albumCount=str(len(als)))
    for s in als:
        _E.SubElement(ar, "album", id="al_"+s["id"], parent=rid, isDir="true", name=s["album"], title=s["album"], album=s["album"], artist=s["artist"], artistId=rid, coverArt=s["id"], songCount="1", duration=s["duration"], created=s["created"], year="2026", genre=s.get("genre","YouTube Music"))
    return rt

def _build_similar(rid, tag):
    is_hdd = str(rid or "").startswith(("hdd_","hdd6_","hdd7_","hdd9_"))
    raw = _RE.sub(r"^(hdd9_|hdd7_|hdd6_|hdd_|yt9_|yt7_|yt6_|yt5_|al_|ar_)+", "", str(rid or ""))
    title, artist, seed_vid = "", "", ""
    if _RE.match(r"^[A-Za-z0-9_-]{11}$", raw) and not is_hdd:
        seed_vid = raw
        if raw in _YL:
            title = _YL[raw].get("title","")
            artist = _YL[raw].get("artist","")
    if not title:
        for tg, at, _ in _H_IDX:
            aid = _RE.sub(r"^(hdd9_|hdd7_|hdd6_|hdd_)+", "", str(at.get("id","")))
            if aid in (raw, "al-"+raw, "mf-"+raw, "ar-"+raw):
                title = at.get("title") or at.get("name") or ""
                artist = at.get("artist") or ""
                break
    if not title and raw in _HDD_META:
        title = _HDD_META[raw]

    c_art = _clean_seed(artist)
    c_tit = _clean_seed(title)
    if not seed_vid:
        q_str = (c_art + " " + c_tit).strip() or "Bolero Lệ Quyên Bằng Kiều"
        s_items = _ytm_search_items(q_str, "EgWKAQIIAWoKEAkQBRAKEAMQBA==", False)
        if s_items: seed_vid = s_items[0]["id"]

    seen = {raw, "hdd_"+raw, "al-"+raw, "mf-"+raw, seed_vid}
    ytm_songs = []
    if seed_vid:
        for s in _ytm_radio_items(seed_vid):
            if s["id"] not in seen:
                seen.add(s["id"]); ytm_songs.append(s)
                if not c_art and s.get("artist"): c_art = _clean_seed(s["artist"])

    hdd_songs = []
    kw_norm = [w for w in _no_acc(c_art + " " + c_tit).split() if len(w) > 2]
    if kw_norm and _H_IDX:
        scored = []
        for tg, at, txt in _H_IDX:
            if tg in ("song", "child", "entry") and at.get("isDir") != "true":
                sid = at.get("id","")
                if sid in seen: continue
                score = sum(3 if w in _no_acc(at.get("artist","")) else 1 for w in kw_norm if w in txt)
                if score > 0: scored.append((score, at))
        scored.sort(key=lambda x: x[0], reverse=True)
        for _, at in scored[:12]:
            seen.add(at.get("id","")); hdd_songs.append(dict(at))

    rt = _E.Element("subsonic-response", status="ok")
    mc = _E.SubElement(rt, tag)
    for i in range(max(len(hdd_songs), len(ytm_songs))):
        if i < len(ytm_songs): _E.SubElement(mc, "song", **ytm_songs[i])
        if i < len(hdd_songs): _E.SubElement(mc, "song", **hdd_songs[i])
    return rt

def _fmt(rt, js, cb=None):
    _cln(rt)
    for x in rt.iter():
        a = x.attrib
        ca = a.get("coverArt")
        if ca:
            raw_ca = _RE.sub(r"^(hdd9_|hdd7_|hdd6_|hdd_|yt9_|yt7_|yt6_|yt5_)+", "", str(ca))
            if str(ca).startswith(("hdd_","hdd6_","hdd7_","hdd9_")) or str(a.get("id","")).startswith("hdd_"):
                a["coverArt"] = "hdd9_" + raw_ca
            else:
                a["coverArt"] = "yt9_" + raw_ca
        if x.tag in ("song","child","entry") and a.get("isDir") != "true" and not str(a.get("id","")).startswith("hdd_"):
            a.setdefault("bitRate", "256")
            a.setdefault("suffix", "aac")
            a.setdefault("contentType", "audio/aac")
            a.setdefault("samplingRate", "44100")
            a.setdefault("bitDepth", "16")
            a.setdefault("channelCount", "2")
            a.setdefault("type", "music")
    rt.attrib.update({"status":rt.attrib.get("status","ok"),"version":"1.16.1","type":"subsonic","serverVersion":"1.16.1"})
    if js:
        b = _J.dumps({"subsonic-response": _x2d(rt, 0)})
        return _R(content=f"{cb}({b});" if cb else b, media_type="application/json; charset=utf-8", headers=_CORS)
    rt.set("xmlns", "http://subsonic.org/restapi")
    return _R(content=b'<?xml version="1.0" encoding="UTF-8"?>\n' + _E.tostring(rt, encoding="utf-8"), media_type="application/xml; charset=utf-8", headers=_CORS)

def _check_auth(qd):
    u = str(qd.get("u") or "").strip()
    if not u: return True
    all_u = {**_USERS, "admin": _OS.environ.get("MIRASONIC_PASSWORD", _PASS)}
    if u not in all_u: return False
    exp = str(all_u[u])
    t, s = str(qd.get("t") or ""), str(qd.get("s") or "")
    if t and s: return _H.md5((exp + s).encode()).hexdigest().lower() == t.lower()
    rp = str(qd.get("p") or "")
    if rp.startswith("enc:"):
        try: rp = bytes.fromhex(rp[4:]).decode("utf-8", "ignore")
        except Exception: pass
    return (not rp) or (rp == exp)

@app.middleware("http")
async def _umw(req, call_next):
    if req.method == "OPTIONS": return _R(status_code=200, headers=_CORS)
    _load_addon_opts()
    p = _RE.sub(r"/+", "/", req.url.path); pl = p.lower()
    if not pl.startswith("/rest/"): return await call_next(req)
    qi = list(_pq(req.scope.get("query_string", b"").decode("utf-8","ignore"), keep_blank_values=True))
    if req.method == "POST":
        bf = await req.body()
        if bf:
            try: qi.extend([(str(k),str(v)) for k,v in _J.loads(bf).items()] if bf.strip().startswith(b"{") else _pq(bf.decode("utf-8","ignore"), keep_blank_values=True))
            except Exception: pass
        async def _rc(): return {"type": "http.request", "body": bf, "more_body": False}
        req._receive = _rc
    qd = dict(qi); fv = qd.get("f","").lower()
    js = fv in ("json","jsonp") or "application/json" in req.headers.get("accept","").lower()
    cb = qd.get("callback") if fv == "jsonp" else None
    lp = _A.get_running_loop()

    if "ping" in pl: return _fmt(_E.Element("subsonic-response", status="ok"), js, cb)
    if "/getcoverart" in pl:
        cid = str(qd.get("id") or "").strip()
        d = await lp.run_in_executor(_POOL, _get_art_bytes, cid) if cid else None
        if d:
            ct = "image/png" if d.startswith(b"\x89PNG") else "image/jpeg"
            return _R(content=d, media_type=ct, headers={**_CORS, "Content-Length": str(len(d)), "Cache-Control": "public, max-age=604800"})
        return _R(content=_PNG_1X1, media_type="image/png", headers={**_CORS, "Content-Length": str(len(_PNG_1X1))})

    if not _check_auth(qd):
        rt = _E.Element("subsonic-response", status="failed")
        _E.SubElement(rt, "error", code="40", message="Sai tai khoan hoac mat khau!")
        return _fmt(rt, js, cb)
    mp = _OS.environ.get("MIRASONIC_PASSWORD", _PASS)
    req.scope["query_string"] = _ue([(k, v) for k, v in qi if k not in ("u","p","t","s")] + [("u","admin"),("p",mp)]).encode()

    if "getlicense" in pl:
        rt = _E.Element("subsonic-response", status="ok"); _E.SubElement(rt, "license", valid="true", email="a@b.c"); return _fmt(rt, js, cb)
    if any(x in pl for x in ("getartistinfo","getalbuminfo","getlyrics","getopensubsonicextensions")):
        rt = _E.Element("subsonic-response", status="ok")
        tg = "artistInfo2" if "artistinfo2" in pl else ("artistInfo" if "artistinfo" in pl else ("albumInfo" if "albuminfo" in pl else ("lyrics" if "lyrics" in pl else "openSubsonicExtensions")))
        _E.SubElement(rt, tg); return _fmt(rt, js, cb)

    if "gettopsongs" in pl:
        art_q = qd.get("artist") or qd.get("query") or qd.get("id") or ""
        cnt = int(qd.get("count") or 25)
        rt = await lp.run_in_executor(_POOL, _build_top_songs, art_q, cnt)
        return _fmt(rt, js, cb)

    if "getsimilarsongs" in pl:
        s_id = qd.get("id") or qd.get("songId") or qd.get("artistId") or ""
        tg = "similarSongs2" if "2" in pl else "similarSongs"
        rt = await lp.run_in_executor(_POOL, _build_similar, s_id, tg)
        return _fmt(rt, js, cb)

    ish = any(isinstance(v, str) and v.startswith("hdd_") for k, v in qi if k in ("id","albumId","artistId","playlistId","songId"))
    if any(x in pl for x in ("/stream","/download","/avatar","/hls")):
        if ish:
            cq = [x for x in _hdd_q(qi) if x[0] != "f"]
            rh = {"User-Agent":"Resonus", **({"Range":req.headers["range"]} if "range" in req.headers else {})}
            up = await lp.run_in_executor(_POOL, lambda: _U.urlopen(_U.Request(f"{_HDD}{p}?{_ue(cq)}", headers=rh), timeout=30, context=_CTX))
            sh = {k.lower():v for k,v in up.headers.items() if k.lower() in ("content-type","content-length","content-range","accept-ranges")}; sh.update(_CORS)
            def _g():
                try:
                    while c := up.read(65536): yield c
                finally: up.close()
            return _SR(_g(), status_code=up.status, headers=sh)
        m = _RE.search(r"([A-Za-z0-9_-]{11})$", str(qd.get("id","")))
        if ("/stream" in pl or "/download" in pl) and m:
            vid = m.group(1)
            j = _ens_fast(vid)
            for _ in range(38):
                if j["d"]: break
                await _A.sleep(0.1)
            if not j["d"]:
                for _ in range(45):
                    if (_OS.path.exists(j["p"]) and _OS.path.getsize(j["p"]) >= 24576) or j["d"]: break
                    await _A.sleep(0.1)
            if _OS.path.exists(j["p"]) and _OS.path.getsize(j["p"]) > 0:
                if j["d"]:
                    fsize = _OS.path.getsize(j["p"])
                    if vid in _YL: _YL[vid]["size"] = str(fsize)
                    rng = req.headers.get("range") or req.headers.get("Range")
                    if rng and rng.startswith("bytes="):
                        try:
                            sp = rng[6:].split("-", 1)
                            st_b = int(sp[0]) if sp[0] else 0
                            en_b = int(sp[1]) if (len(sp) > 1 and sp[1]) else (fsize - 1)
                            en_b = min(en_b, fsize - 1)
                            ln_b = max(0, en_b - st_b + 1)
                            with open(j["p"], "rb") as f:
                                f.seek(st_b)
                                data = f.read(ln_b)
                            return _R(content=data, status_code=206, media_type="audio/aac", headers={**_CORS, "Accept-Ranges": "bytes", "Content-Range": f"bytes {st_b}-{en_b}/{fsize}", "Content-Length": str(len(data))})
                        except Exception: pass
                    return _R(content=open(j["p"], "rb").read(), status_code=200, media_type="audio/aac", headers={**_CORS, "Accept-Ranges": "bytes", "Content-Length": str(fsize)})
                async def _yg():
                    with open(j["p"], "rb") as f:
                        while True:
                            c = f.read(32768)
                            if c: yield c
                            elif j["d"]: break
                            else: await _A.sleep(0.05)
                return _SR(_yg(), status_code=200, media_type="audio/aac", headers=_CORS)
        return await call_next(req)

    if ish:
        hrt = await lp.run_in_executor(_POOL, _hdd_xml, p, qi)
        if hrt is not None: return _fmt(hrt, js, cb)

    rid = str(qd.get("id") or "")
    if rid.startswith("ar_") and "/getartist" in pl and "getartists" not in pl:
        rt = await lp.run_in_executor(_POOL, _build_artist_page, rid)
        return _fmt(rt, js, cb)

    vid = rid[3:] if rid.startswith("al_") else rid
    if any(x in pl for x in ("/getalbum","/getsong","/getmusicdirectory")):
        if _RE.match(r"^[A-Za-z0-9_-]{11}$", vid):
            s = await lp.run_in_executor(_POOL, _ensure_vid_meta, vid)
            if s:
                rt = _E.Element("subsonic-response", status="ok")
                if "song" in pl: _E.SubElement(rt, "song", **s)
                else:
                    tg = "directory" if "directory" in pl else "album"
                    al = _E.SubElement(rt, tg, id="al_"+vid, parent=s["artistId"], isDir="true", name=s["album"], title=s["album"], album=s["album"], artist=s["artist"], artistId=s["artistId"], coverArt=vid, songCount="1", duration=s["duration"], playCount="1", created=s["created"], year="2026", genre=s.get("genre","YouTube Music"))
                    _E.SubElement(al, "child" if tg=="directory" else "song", **s)
                return _fmt(rt, js, cb)

    if "search" in pl:
        global _S_SEQ, _LAST_Q
        q_str = (qd.get("query") or qd.get("any") or qd.get("q") or "").strip('" ')
        st = "searchResult2" if "2" in pl else "searchResult3"
        if len(q_str) == 1:
            rt = _E.Element("subsonic-response", status="ok"); _E.SubElement(rt, st); return _fmt(rt, js, cb)
        if q_str:
            qn, now = _no_acc(q_str), _TM.time()
            if qn not in _SCACHE or _SCACHE[qn][1] < now:
                enter = (qn == _LAST_Q); _LAST_Q = qn; _S_SEQ += 1; ms = _S_SEQ
                if not enter:
                    await _A.sleep(0.45)
                    if ms != _S_SEQ and qn not in _SCACHE:
                        rt = _E.Element("subsonic-response", status="ok"); _E.SubElement(rt, st); return _fmt(rt, js, cb)
                f1 = lp.run_in_executor(_POOL, _yt_search, q_str, "searchResult3")
                f2 = lp.run_in_executor(_POOL, _hdd_search, q_str, qi)
                mrt, hrt = await f1, await f2
                if hrt is not None and len(hrt) > 0: _ilv(mrt[0], hrt[0])
                _SCACHE[qn] = (_E.tostring(mrt), now + 300)
            mrt = _E.fromstring(_SCACHE[qn][0]); mrt[0].tag = st
            return _fmt(mrt, js, cb)

    fut = lp.run_in_executor(_POOL, _hdd_xml, p, qi)
    if "getalbumlist" in pl or "getrandomsongs" in pl:
        if not _YL: await lp.run_in_executor(_POOL, _yt_search, "Thùy Dung Lệ Quyên Bằng Kiều Bolero")
        mrt = _E.Element("subsonic-response", status="ok")
        off = int(qd.get("offset") or 0)
        if "albumlist" in pl:
            mc = _E.SubElement(mrt, "albumList2" if "2" in pl else "albumList")
            if off == 0 and qd.get("type","") != "starred":
                for s in [x for x in _YL.values() if not str(x.get("id","")).startswith("loc_")][:14]:
                    _E.SubElement(mc, "album", id="al_"+s["id"], parent=s["artistId"], isDir="true", name=s["album"], title=s["album"], album=s["album"], artist=s["artist"], artistId=s["artistId"], coverArt=s["id"], songCount="1", duration=s["duration"], playCount="1", created=s["created"], year="2026", genre=s.get("genre","YouTube Music"))
        else:
            mc = _E.SubElement(mrt, "randomSongs")
            for s in [x for x in _YL.values() if not str(x.get("id","")).startswith("loc_")][:15]:
                _E.SubElement(mc, "song", **s)
    else:
        resp = await call_next(req)
        body = b"".join([c async for c in resp.body_iterator])
        try: mrt = _cln(_E.fromstring(body))
        except Exception: mrt = None

    hrt = await fut
    if hrt is not None:
        if mrt is None or mrt.attrib.get("status") != "ok" or len(mrt) == 0: mrt = hrt
        elif len(hrt) > 0: _ilv(mrt[0], hrt[0])
    return _fmt(mrt, js, cb) if mrt is not None else _R(content=b"", status_code=200, headers=_CORS)

@app.get("/users", response_class=_HR)
async def _users_ui(admin_pass: str = "", act: str = "", u: str = "", p: str = ""):
    mp = _OS.environ.get("MIRASONIC_PASSWORD", _PASS)
    msg = ""
    if act and admin_pass != mp: msg = "<p style='color:#ff5555'>❌ Sai mật khẩu Admin!</p>"
    elif act == "add" and u and p:
        _USERS[u.strip()] = p.strip(); _save_u(); msg = f"<p style='color:#1db954'>✅ Đã lưu: <b>{u}</b></p>"
    elif act == "del" and u in _USERS:
        del _USERS[u]; _save_u(); msg = f"<p style='color:#ffb86c'>🗑️ Đã xóa: <b>{u}</b></p>"
    rows = "".join(f"<tr><td><b>{k}</b></td><td><code>{v}</code></td><td><a href='/users?act=del&u={k}&admin_pass={admin_pass}' style='color:#ff5555'>Xóa</a></td></tr>" for k, v in _USERS.items())
    return f"<!DOCTYPE html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Users</title><style>body{{background:#121212;color:#fff;font-family:sans-serif;padding:20px;max-width:480px;margin:auto}}input,button{{width:100%;padding:10px;margin:6px 0;border-radius:6px;border:1px solid #333;background:#222;color:#fff;box-sizing:border-box}}button{{background:#1db954;border:none;font-weight:bold}}table{{width:100%;margin-top:15px;border-collapse:collapse}}td,th{{padding:10px;border-bottom:1px solid #333;text-align:left}}</style></head><body><h2>👥 Quản Lý Tài Khoản</h2>{msg}<form method='get' action='/users'><input type='hidden' name='act' value='add'><input name='admin_pass' type='password' placeholder='Mật khẩu Admin' value='{admin_pass}' required><input name='u' placeholder='Tài khoản bạn bè' required><input name='p' placeholder='Mật khẩu' required><button type='submit'>➕ Lưu Tài Khoản</button></form><table><tr><th>User</th><th>Pass</th><th>Xóa</th></tr>{rows}</table></body></html>"

_SCACHE.clear()
_POOL.submit(_init_hdd)
_POOL.submit(_yt_search, "Thùy Dung Lệ Quyên Bằng Kiều Salvatore Adamo")







# --- K9: V1.3.2 CRISP 500x500 ARTWORK, YOUTUBE THUMBNAIL FIX & HDD->YTM->YT ORDER ---
import random as _RND
import unicodedata as _UD
import concurrent.futures as _CF

_REQ_POOL = _CF.ThreadPoolExecutor(max_workers=20)
_SUB_POOL = _CF.ThreadPoolExecutor(max_workers=24)
_ART_POOL = _CF.ThreadPoolExecutor(max_workers=12)

_ART_DIR = "/data/art_cache_v20" if _OS.path.isdir("/data") else "/tmp/art_cache_v20"
_STAR_F  = "/data/starred_v10.json" if _OS.path.isdir("/data") else "/tmp/starred_v10.json"
_OS.makedirs(_ART_DIR, exist_ok=True)
_ART_RAM.clear()
_CHAN_ID = {}
_ARTIST_THUMB = {}
_PL_THUMB = {}
_PL_META = {}
_PL_TRACKS_CACHE = {}
_ART_CACHE_SONGS = {}
_ART_CACHE_PLS = {}

try:
    _STARRED = _J.load(open(_STAR_F))
except Exception:
    _STARRED = {"artists": {}, "albums": {}, "songs": {}}

def _save_star():
    try:
        _J.dump(_STARRED, open(_STAR_F, "w"))
    except Exception:
        pass

def _nfc(s):
    return _UD.normalize("NFC", str(s or "")).lower().strip()

def _has_acc(s):
    return any(ord(c) > 127 for c in _nfc(s))

def _clean_artist_name(s):
    s = _UD.normalize("NFC", str(s or ""))
    s = _RE.sub(r"\[HDD\]|\[YT\]|\b(Official|Channel|Music|Topic|VEVO|Ca\s+s\u0129|Cs)\b", " ", s, flags=_RE.I)
    s = _RE.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split()).strip()

def _is_same_artist(target_art, cand_art, strict_acc=True):
    t_raw = _clean_artist_name(target_art)
    c_raw = _clean_artist_name(cand_art)
    t_nfc, c_nfc = _nfc(t_raw), _nfc(c_raw)
    if not t_nfc or not c_nfc:
        return False
    t_no, c_no = _no_acc(t_nfc), _no_acc(c_nfc)
    if not t_no or not c_no:
        return False
    if strict_acc and _has_acc(t_nfc):
        if _has_acc(c_nfc):
            if t_nfc == c_nfc:
                return True
            tw = [w for w in t_nfc.split() if len(w) > 1]
            cw = c_nfc.split()
            if len(tw) >= 2 and all(w in cw for w in tw) and t_nfc in c_nfc:
                return True
            if len(tw) == 1 and len(cw) == 1 and tw[0] == cw[0]:
                return True
            return False
        else:
            return t_no == c_no
    if t_no == c_no:
        return True
    tw = [w for w in t_no.split() if len(w) > 1]
    cw = c_no.split()
    if len(tw) >= 2 and all(w in cw for w in tw) and t_no in c_no:
        return True
    return False

def _match_query_text(q, text):
    q_nfc = _nfc(q)
    t_nfc = _nfc(text)
    if _has_acc(q_nfc):
        q_words = [w for w in _RE.sub(r"[^\w\s]", " ", q_nfc).split() if w]
        t_words = _RE.sub(r"[^\w\s]", " ", t_nfc).split()
        for qw in q_words:
            if _has_acc(qw):
                if qw not in t_words and qw not in t_nfc:
                    return False
            else:
                if qw not in _no_acc(t_nfc):
                    return False
        return True
    else:
        q_words = [w for w in _no_acc(q_nfc).split() if w]
        t_no = _no_acc(t_nfc)
        return all(w in t_no for w in q_words)

def _fix_url(u):
    u = str(u or "").strip()
    if u.startswith("//"):
        u = "https:" + u
    if "yt3.ggpht.com" in u or "googleusercontent.com" in u:
        u = _RE.sub(r"=s\d+[^/]*$", "=s500-c-k-c0x00ffffff-no-rj", u)
        u = _RE.sub(r"=w\d+-h\d+[^/]*$", "=w500-h500-l95-rj", u)
    return u

def _make_song(vid, tit, art, alb="", dur="240", is_vid=False, thumb_url=""):
    vid = str(vid or "").strip()
    if not _RE.match(r"^[A-Za-z0-9_-]{11}$", vid): return None
    tit = _UD.normalize("NFC", str(tit or "YouTube Music")).strip()
    art = _UD.normalize("NFC", str(art or "YouTube Music")).strip()
    if art.lower() in ("youtube", "various artists", ""): art = "YouTube Music"
    alb = _UD.normalize("NFC", str(alb or "")).strip() or tit
    if thumb_url and ("googleusercontent.com" in thumb_url or "ggpht.com" in thumb_url):
        _YTM_THUMB[vid] = _RE.sub(r"=w\d+-h\d+[^/]*$", "=w500-h500-l95-rj", thumb_url)
    arid = "ar_" + _H.md5(_nfc(art).encode("utf-8")).hexdigest()[:10]
    _ARTIST_NAMES[arid] = art
    d_int = int(dur) if str(dur).isdigit() and int(dur) > 0 else 240
    ap = _OS.path.join(_AUD_DIR, f"{vid}.aac")
    f_sz = str(_OS.path.getsize(ap)) if (_OS.path.exists(ap) and _OS.path.getsize(ap) > 32768) else str(d_int * 32000)
    s = {
        "id": vid, "parent": "al_" + vid, "isDir": "false",
        "title": tit, "album": alb, "artist": art, "track": "1",
        "year": "2026", "genre": "YouTube" if is_vid else "YouTube Music",
        "coverArt": vid, "size": f_sz, "contentType": "audio/aac",
        "suffix": "aac", "transcodedContentType": "audio/aac", "transcodedSuffix": "aac",
        "duration": str(d_int), "bitRate": "256",
        "samplingRate": "44100", "bitDepth": "16", "channelCount": "2",
        "path": f"{art}/{alb}/{tit}.aac", "isVideo": "false",
        "created": "2026-01-01T00:00:00.000Z",
        "albumId": "al_" + vid, "artistId": arid, "type": "music", "mediaType": "song"
    }
    _YL[vid] = s
    return s

def _hdd_first(mc, hc):
    bt = {}
    for x in hc:
        bt.setdefault(x.tag, []).append(x)
    for x in mc:
        bt.setdefault(x.tag, []).append(x)
    out = []
    for tg in ("artist", "album", "song"):
        out.extend(bt.pop(tg, []))
    for _, lst in bt.items():
        out.extend(lst)
    mc[:] = out

def _hdd_search(q, qi):
    has_ac = _has_acc(q)
    vq = _UD.normalize("NFC", q).strip() if has_ac else _vn_q(q)
    nqi = [(k, vq if k in ("query","any","q") else v) for k, v in qi]
    rt = _hdd_xml("/rest/search3.view", nqi)
    if not q:
        return rt
    if rt is None or len(rt) == 0:
        rt = _E.Element("subsonic-response", status="ok")
        _E.SubElement(rt, "searchResult3")
    sr = rt[0]
    if has_ac:
        kept = []
        for x in list(sr):
            a = x.attrib
            full_t = f"{a.get('title','')} {a.get('name','')} {a.get('artist','')} {a.get('album','')}"
            if _match_query_text(q, full_t):
                kept.append(x)
        sr[:] = kept
    seen = {x.attrib.get("id") for x in sr}
    for tg, at, txt in _H_IDX:
        if at.get("id") in seen:
            continue
        full_t = f"{at.get('title','')} {at.get('name','')} {at.get('artist','')} {at.get('album','')}"
        if _match_query_text(q, full_t):
            seen.add(at.get("id"))
            _E.SubElement(sr, tg, **at)
    return rt

def _ytm_search_items(q, params=None, is_vid=False):
    out = []
    try:
        pl = {"query": q}
        if params: pl["params"] = params
        data = _ytm_post("search", pl)
        for m in _find_k(data, "musicResponsiveListItemRenderer"):
            vid = (m.get("playlistItemData") or {}).get("videoId") or ""
            if not vid:
                for w in _find_k(m, "watchEndpoint"):
                    if w.get("videoId"): vid = w["videoId"]; break
            if not _RE.match(r"^[A-Za-z0-9_-]{11}$", str(vid)): continue
            turl = ""
            for ths in _find_k(m, "thumbnails"):
                if isinstance(ths, list) and ths and isinstance(ths[-1], dict):
                    turl = _fix_url(ths[-1].get("url") or "")
            cols = m.get("flexColumns") or []
            if not cols: continue
            tit = "".join(r.get("text","") for runs in _find_k(cols[0], "runs") for r in runs if isinstance(r, dict)).strip()
            meta_txt = "".join(r.get("text","") for runs in _find_k(cols[1], "runs") for r in runs if isinstance(r, dict)).strip() if len(cols) > 1 else ""
            parts = [p.strip() for p in meta_txt.split("\u2022") if p.strip()]
            dur = "240"
            for fcol in (m.get("fixedColumns") or []):
                ftxt = "".join(r.get("text","") for runs in _find_k(fcol, "runs") for r in runs if isinstance(r, dict)).strip()
                if _RE.match(r"^\d+:\d+(:\d+)?$", ftxt):
                    dur = _dur_sec(ftxt)
            clean_p = []
            for p in parts:
                if _RE.match(r"^\d+:\d+(:\d+)?$", p): dur = _dur_sec(p)
                elif p.lower() not in ("song","b\u00e0i h\u00e1t","video","ep","single","album") and not _RE.search(r"\b(views|l\u01b0\u1ee3t xem|subscribers|ng\u01b0\u1eddi \u0111\u0103ng k\u00fd)\b", p, _RE.I):
                    clean_p.append(p)
            art = clean_p[0] if clean_p else "YouTube Music"
            alb = clean_p[1] if (len(clean_p) > 1 and not is_vid) else tit
            s = _make_song(vid, tit, art, alb, dur, is_vid=is_vid, thumb_url=turl)
            if s: out.append(s)
    except Exception:
        pass
    if out: _save_meta()
    return out

def _ytm_search_artists_and_pls(q):
    pls_out = []
    try:
        d_ar = _ytm_post("search", {"query": q, "params": "EgWKAQIgAWoKEAkQBRAKEAMQBA=="})
        for m in _find_k(d_ar, "musicResponsiveListItemRenderer"):
            cols = m.get("flexColumns") or []
            if not cols: continue
            aname = _UD.normalize("NFC", "".join(r.get("text","") for runs in _find_k(cols[0], "runs") for r in runs if isinstance(r, dict))).strip()
            if not aname: continue
            arid = "ar_" + _H.md5(_nfc(aname).encode("utf-8")).hexdigest()[:10]
            _ARTIST_NAMES[arid] = aname
            for ths in _find_k(m, "thumbnails"):
                if isinstance(ths, list) and ths and isinstance(ths[-1], dict):
                    u = _fix_url(ths[-1].get("url") or "")
                    if u: _ARTIST_THUMB[arid] = u
            for b in _find_k(m, "browseEndpoint"):
                bid = str(b.get("browseId") or "")
                if bid.startswith("UC"):
                    _CHAN_ID[arid] = bid
                    _CHAN_ID[_nfc(aname)] = bid
    except Exception:
        pass
    try:
        d_pl = _ytm_post("search", {"query": q, "params": "EgWKAQIoAWoKEAkQBRAKEAMQBA=="})
        for m in _find_k(d_pl, "musicResponsiveListItemRenderer"):
            bid = ""
            for b in _find_k(m, "browseEndpoint"):
                b_str = str(b.get("browseId") or "")
                if b_str.startswith(("VL","PL","RD","OLAK")):
                    bid = b_str[2:] if b_str.startswith("VL") else b_str
                    break
            if not bid: continue
            cols = m.get("flexColumns") or []
            if not cols: continue
            ptit = _UD.normalize("NFC", "".join(r.get("text","") for runs in _find_k(cols[0], "runs") for r in runs if isinstance(r, dict))).strip()
            mtxt = _UD.normalize("NFC", "".join(r.get("text","") for runs in _find_k(cols[1], "runs") for r in runs if isinstance(r, dict))).strip() if len(cols) > 1 else ""
            parts = [p.strip() for p in mtxt.split("\u2022") if p.strip()]
            owner = ""
            cnt = "20"
            for p in parts:
                if _RE.search(r"\d+\s*(b\u00e0i|songs|tracks|video)", p, _RE.I):
                    m_c = _RE.search(r"(\d+)", p)
                    if m_c: cnt = m_c.group(1)
                elif p.lower() not in ("playlist","danh s\u00e1ch ph\u00e1t","album","ep") and not _RE.search(r"\b(views|l\u01b0\u1ee3t xem)\b", p, _RE.I):
                    if not owner: owner = p
            turl = ""
            for ths in _find_k(m, "thumbnails"):
                if isinstance(ths, list) and ths and isinstance(ths[-1], dict):
                    turl = _fix_url(ths[-1].get("url") or "")
            if turl: _PL_THUMB[bid] = turl
            p_info = {"id": "pl_" + bid, "raw_id": bid, "title": ptit, "artist": owner or q, "songCount": cnt, "coverArt": "pl_" + bid}
            _PL_META[bid] = p_info
            pls_out.append(p_info)
    except Exception:
        pass
    return pls_out

def _extract_v_len(v):
    l_obj = v.get("lengthText") or {}
    d_str = l_obj.get("simpleText") or "".join(x.get("text","") for x in l_obj.get("runs", [])).strip()
    if not d_str:
        for ov in _find_k(v, "thumbnailOverlayTimeStatusRenderer"):
            tx = ov.get("text") or {}
            d_str = tx.get("simpleText") or "".join(x.get("text","") for x in tx.get("runs", [])).strip()
            if d_str: break
    return d_str

def _yt_web_search(q, sp_param=""):
    out_songs, out_chans, out_pls = [], [], []
    try:
        body = {
            "context": {"client": {"clientName": "WEB", "clientVersion": "2.20250101.00.00", "hl": "vi", "gl": "VN"}},
            "query": q
        }
        if sp_param:
            body["params"] = sp_param
        req = _U.Request(
            "https://www.youtube.com/youtubei/v1/search?prettyPrint=false",
            data=_J.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"},
            method="POST"
        )
        with _U.urlopen(req, timeout=6, context=_CTX) as r:
            data = _J.loads(r.read().decode("utf-8", "ignore"))

        for ch in _find_k(data, "channelRenderer"):
            cname = _UD.normalize("NFC", (ch.get("title") or {}).get("simpleText") or "".join(x.get("text","") for x in (ch.get("title") or {}).get("runs", []))).strip()
            if not cname: continue
            cid = str(ch.get("channelId") or ((ch.get("navigationEndpoint") or {}).get("browseEndpoint") or {}).get("browseId") or "").strip()
            arid = "ar_" + _H.md5(_nfc(cname).encode("utf-8")).hexdigest()[:10]
            _ARTIST_NAMES[arid] = cname
            if cid.startswith("UC"):
                _CHAN_ID[arid] = cid
                _CHAN_ID[_nfc(cname)] = cid
            ths = (ch.get("thumbnail") or {}).get("thumbnails") or []
            if ths and isinstance(ths[-1], dict) and ths[-1].get("url"):
                _ARTIST_THUMB[arid] = _fix_url(ths[-1]["url"])
            out_chans.append((arid, cname))

        for pl in _find_k(data, "playlistRenderer"):
            pid = str(pl.get("playlistId") or "").strip()
            if not pid: continue
            ptit = _UD.normalize("NFC", (pl.get("title") or {}).get("simpleText") or "".join(x.get("text","") for x in (pl.get("title") or {}).get("runs", []))).strip()
            byl = pl.get("shortBylineText") or pl.get("longBylineText") or {}
            owner = _UD.normalize("NFC", "".join(x.get("text","") for x in byl.get("runs", []))).strip() or q
            cnt = str(pl.get("videoCount") or "20")
            for u_cand in _find_k(pl, "url"):
                if isinstance(u_cand, str) and ("ytimg.com" in u_cand or "ggpht.com" in u_cand or "googleusercontent.com" in u_cand):
                    _PL_THUMB[pid] = _fix_url(u_cand)
            p_info = {"id": "pl_" + pid, "raw_id": pid, "title": ptit, "artist": owner, "songCount": cnt, "coverArt": "pl_" + pid}
            _PL_META[pid] = p_info
            out_pls.append(p_info)

        for lk in _find_k(data, "lockupViewModel"):
            cid_lk = str(lk.get("contentId") or "").strip()
            ctype = str(lk.get("contentType") or "")
            if not cid_lk or ("PLAYLIST" not in ctype and not cid_lk.startswith(("PL","OLAK"))):
                continue
            md = ((lk.get("metadata") or {}).get("lockupMetadataViewModel") or {})
            ptit = _UD.normalize("NFC", str((md.get("title") or {}).get("content") or "")).strip()
            if not ptit: continue
            owner = q
            rows = (((md.get("metadata") or {}).get("contentMetadataViewModel") or {}).get("metadataRows") or [])
            if rows and isinstance(rows[0], dict):
                pts = rows[0].get("metadataParts") or []
                if pts and isinstance(pts[0], dict):
                    owner = _UD.normalize("NFC", str(((pts[0].get("text") or {}).get("content")) or q)).strip()
            for src in _find_k(lk, "sources"):
                if isinstance(src, list) and src and isinstance(src[-1], dict) and src[-1].get("url"):
                    _PL_THUMB[cid_lk] = _fix_url(src[-1]["url"])
            p_info = {"id": "pl_" + cid_lk, "raw_id": cid_lk, "title": ptit, "artist": owner, "songCount": "20", "coverArt": "pl_" + cid_lk}
            _PL_META[cid_lk] = p_info
            out_pls.append(p_info)

        for v in _find_k(data, "videoRenderer"):
            vid = str(v.get("videoId") or "").strip()
            if not _RE.match(r"^[A-Za-z0-9_-]{11}$", vid): continue
            tit = _UD.normalize("NFC", "".join(x.get("text","") for x in (v.get("title") or {}).get("runs", []))).strip()
            if not tit: continue
            byl_obj = v.get("ownerText") or v.get("longBylineText") or v.get("shortBylineText") or {}
            runs = byl_obj.get("runs") or []
            art = _UD.normalize("NFC", "".join(x.get("text","") for x in runs)).strip() or "YouTube"
            arid = "ar_" + _H.md5(_nfc(art).encode("utf-8")).hexdigest()[:10]
            _ARTIST_NAMES[arid] = art
            for rn in runs:
                bid = str(((rn.get("navigationEndpoint") or {}).get("browseEndpoint") or {}).get("browseId") or "").strip()
                if bid.startswith("UC"):
                    _CHAN_ID[arid] = bid
                    _CHAN_ID[_nfc(art)] = bid
            if arid not in _ARTIST_THUMB:
                for cht in _find_k(v.get("channelThumbnailSupportedRenderers") or {}, "thumbnails"):
                    if isinstance(cht, list) and cht and isinstance(cht[-1], dict) and cht[-1].get("url"):
                        _ARTIST_THUMB[arid] = _fix_url(cht[-1]["url"])
            d_str = _extract_v_len(v)
            if not d_str: continue
            s = _make_song(vid, tit, art, tit, _dur_sec(d_str), is_vid=True)
            if s: out_songs.append(s)
    except Exception:
        pass
    if out_songs or out_chans:
        _save_meta()
    return out_songs, out_chans, out_pls

def _yt_playlist_tracks(raw_pid, fallback_art=""):
    raw_pid = _RE.sub(r"^(VL|pl_|yt\d*_)+", "", str(raw_pid or "").strip())
    if not raw_pid: return []
    if raw_pid in _PL_TRACKS_CACHE and _PL_TRACKS_CACHE[raw_pid]:
        return _PL_TRACKS_CACHE[raw_pid]
    out, seen_v = [], set()
    try:
        d_nx = _ytm_post("next", {"playlistId": raw_pid, "isAudioOnly": True, "enablePersistentPlaylistPanel": True})
        for p in _find_k(d_nx, "playlistPanelVideoRenderer"):
            vid = str(p.get("videoId") or "").strip()
            if not _RE.match(r"^[A-Za-z0-9_-]{11}$", vid) or vid in seen_v: continue
            tit = _UD.normalize("NFC", "".join(r.get("text","") for r in (p.get("title") or {}).get("runs", []))).strip()
            byl = _UD.normalize("NFC", "".join(r.get("text","") for r in ((p.get("longBylineText") or p.get("shortBylineText") or {}).get("runs", [])))).strip()
            parts = [x.strip() for x in byl.split("\u2022") if x.strip()]
            art = parts[0] if parts else (fallback_art or "YouTube Music")
            dstr = "".join(r.get("text","") for r in (p.get("lengthText") or {}).get("runs", [])).strip()
            seen_v.add(vid)
            s = _make_song(vid, tit, art, tit, _dur_sec(dstr or "240"), is_vid=False)
            if s: out.append(s)
    except Exception:
        pass
    if len(out) < 5:
        bid = raw_pid if raw_pid.startswith("VL") else ("VL" + raw_pid)
        try:
            body = {
                "context": {"client": {"clientName": "WEB", "clientVersion": "2.20250101.00.00", "hl": "vi", "gl": "VN"}},
                "browseId": bid
            }
            req = _U.Request(
                "https://www.youtube.com/youtubei/v1/browse?prettyPrint=false",
                data=_J.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"},
                method="POST"
            )
            with _U.urlopen(req, timeout=6, context=_CTX) as r:
                data = _J.loads(r.read().decode("utf-8", "ignore"))
            for pv in _find_k(data, "playlistVideoRenderer"):
                vid = str(pv.get("videoId") or "").strip()
                if not _RE.match(r"^[A-Za-z0-9_-]{11}$", vid) or vid in seen_v: continue
                t_obj = pv.get("title") or {}
                tit = _UD.normalize("NFC", t_obj.get("simpleText") or "".join(x.get("text","") for x in t_obj.get("runs", []))).strip()
                if not tit: continue
                byl = pv.get("shortBylineText") or pv.get("longBylineText") or {}
                art = _UD.normalize("NFC", "".join(x.get("text","") for x in byl.get("runs", []))).strip() or fallback_art or "YouTube"
                d_str = _extract_v_len(pv)
                seen_v.add(vid)
                s = _make_song(vid, tit, art, tit, _dur_sec(d_str or "240"), is_vid=True)
                if s: out.append(s)
        except Exception:
            pass
    if out:
        _PL_TRACKS_CACHE[raw_pid] = out
        _save_meta()
    return out

def _yt_channel_videos(cid, cname):
    out, seen_v = [], set()
    if not cid or not str(cid).startswith("UC"):
        return out
    cname = _UD.normalize("NFC", str(cname or "YouTube")).strip()
    for prm in ("EgZ2aWRlb3PyBgQKAjoA", ""):
        try:
            body = {
                "context": {"client": {"clientName": "WEB", "clientVersion": "2.20250101.00.00", "hl": "vi", "gl": "VN"}},
                "browseId": cid
            }
            if prm: body["params"] = prm
            req = _U.Request(
                "https://www.youtube.com/youtubei/v1/browse?prettyPrint=false",
                data=_J.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"},
                method="POST"
            )
            with _U.urlopen(req, timeout=6, context=_CTX) as r:
                data = _J.loads(r.read().decode("utf-8", "ignore"))
            v_nodes = list(_find_k(data, "videoRenderer")) + list(_find_k(data, "gridVideoRenderer")) + list(_find_k(data, "playlistVideoRenderer"))
            for v in v_nodes:
                vid = str(v.get("videoId") or "").strip()
                if not _RE.match(r"^[A-Za-z0-9_-]{11}$", vid) or vid in seen_v: continue
                t_obj = v.get("title") or {}
                tit = _UD.normalize("NFC", t_obj.get("simpleText") or "".join(x.get("text","") for x in t_obj.get("runs", []))).strip()
                if not tit: continue
                d_str = _extract_v_len(v)
                if not d_str: continue
                seen_v.add(vid)
                s = _make_song(vid, tit, cname, tit, _dur_sec(d_str), is_vid=True)
                if s: out.append(s)
            if len(out) >= 25:
                break
        except Exception:
            pass
    if len(out) < 20:
        uu_tracks = _yt_playlist_tracks("UU" + str(cid)[2:], cname)
        for s in uu_tracks:
            if s["id"] not in seen_v:
                seen_v.add(s["id"])
                s["artist"] = cname
                s["artistId"] = "ar_" + _H.md5(_nfc(cname).encode("utf-8")).hexdigest()[:10]
                out.append(s)
    if out: _save_meta()
    return out

def _yt_search(q, st="searchResult3"):
    rt = _E.Element("subsonic-response", status="ok")
    sr = _E.SubElement(rt, st)
    q = _UD.normalize("NFC", str(q or "")).strip()
    if not q: return rt
    if _RE.match(r"^[A-Za-z0-9_-]{11}$", q) and " " not in q:
        s = _ensure_vid_meta(q)
        if s: _E.SubElement(sr, "song", **s)
        return rt

    has_ac = _has_acc(q)
    f_songs = _SUB_POOL.submit(_ytm_search_items, q, "EgWKAQIIAWoKEAkQBRAKEAMQBA==", False)
    f_vids  = _SUB_POOL.submit(_ytm_search_items, q, "EgWKAQJQAWoKEAkQBRAKEAMQBA==", True)
    f_web   = _SUB_POOL.submit(_yt_web_search, q, "")
    f_ar_pl = _SUB_POOL.submit(_ytm_search_artists_and_pls, q)

    s_list = f_songs.result() or []
    yv_list = f_vids.result() or []
    v_list, ch_list, w_pls = f_web.result() or ([], [], [])
    ytm_pls = f_ar_pl.result() or []

    if has_ac:
        s_list  = [s for s in s_list  if _match_query_text(q, f"{s.get('title','')} {s.get('artist','')} {s.get('album','')}") or _is_same_artist(q, s.get("artist",""), True)]
        yv_list = [s for s in yv_list if _match_query_text(q, f"{s.get('title','')} {s.get('artist','')}") or _is_same_artist(q, s.get("artist",""), True)]
        v_list  = [s for s in v_list  if _match_query_text(q, f"{s.get('title','')} {s.get('artist','')}") or _is_same_artist(q, s.get("artist",""), True)]
        ch_list = [(arid, cn) for arid, cn in ch_list if _match_query_text(q, cn) or _is_same_artist(q, cn, True)]

    all_vids = v_list + yv_list
    merged, seen = [], set()
    ch_match = [v for v in all_vids if _is_same_artist(q, v.get("artist",""), has_ac)]
    other_v  = [v for v in all_vids if v not in ch_match]
    v_ordered = ch_match + other_v

    # Sap xep: YouTube Music chinh thuc (s_list) len truoc, sau do den YouTube thuong (v_ordered)
    for s_it in s_list:
        if s_it["id"] not in seen:
            seen.add(s_it["id"]); merged.append(s_it)
    for v_it in v_ordered:
        if v_it["id"] not in seen:
            seen.add(v_it["id"]); merged.append(v_it)

    ar_candidates, seen_ar = [], set()
    for arid, cname in ch_list:
        if arid not in seen_ar:
            seen_ar.add(arid)
            ar_candidates.append((0 if _is_same_artist(q, cname, has_ac) else 1, arid, cname))
    for s in merged:
        arid = s["artistId"]
        cname = s["artist"]
        if arid not in seen_ar:
            if has_ac and not (_match_query_text(q, cname) or _is_same_artist(q, cname, True)):
                continue
            seen_ar.add(arid)
            ar_candidates.append((0 if _is_same_artist(q, cname, has_ac) else 2, arid, cname))

    ar_candidates.sort(key=lambda x: x[0])
    for _, arid, cname in ar_candidates[:20]:
        at_dict = {"id": arid, "name": cname, "coverArt": arid, "albumCount": "30"}
        if arid in _STARRED.get("artists", {}):
            at_dict["starred"] = "2026-11-15T00:00:00.000Z"
        _E.SubElement(sr, "artist", **at_dict)

    seen_pl = set()
    for p_info in (w_pls + ytm_pls)[:8]:
        pid = p_info["id"]
        if pid in seen_pl: continue
        if has_ac and not _match_query_text(q, f"{p_info.get('title','')} {p_info.get('artist','')}"):
            continue
        seen_pl.add(pid)
        arid = "ar_" + _H.md5(_nfc(p_info["artist"]).encode("utf-8")).hexdigest()[:10]
        p_name = f"\U0001f4d1 [DS Ph\u00e1t] {p_info['title']}"
        _E.SubElement(sr, "album", id=pid, parent=arid, isDir="true", name=p_name, title=p_name, album=p_name, artist=p_info["artist"], artistId=arid, coverArt=pid, songCount=p_info["songCount"], duration="7200", created="2026-01-01T00:00:00.000Z", year="2026", genre="Playlist")

    for s in merged:
        arid = s["artistId"]
        _E.SubElement(sr, "album", id="al_"+s["id"], parent=arid, isDir="true", name=s["album"], title=s["album"], album=s["album"], artist=s["artist"], artistId=arid, coverArt=s["id"], songCount="1", duration=s["duration"], created=s["created"], year=s["year"], genre=s["genre"])
        _E.SubElement(sr, "song", **s)
    return rt

def _collect_artist_data(artist_q):
    raw_name = _UD.normalize("NFC", _resolve_artist_name(artist_q)).strip()
    art_clean = _clean_artist_name(raw_name) or raw_name or "Th\u00f9y Dung"
    ckey = _nfc(art_clean)
    now = _TM.time()
    if ckey in _ART_CACHE_SONGS and _ART_CACHE_SONGS[ckey][1] > now:
        return raw_name, _ART_CACHE_SONGS[ckey][0], _ART_CACHE_PLS.get(ckey, [])

    arid = str(artist_q) if str(artist_q).startswith("ar_") else ("ar_" + _H.md5(ckey.encode("utf-8")).hexdigest()[:10])
    f_ytm_s = _SUB_POOL.submit(_ytm_search_items, art_clean, "EgWKAQIIAWoKEAkQBRAKEAMQBA==", False)
    f_ytm_v = _SUB_POOL.submit(_ytm_search_items, art_clean, "EgWKAQJQAWoKEAkQBRAKEAMQBA==", True)
    f_web_g = _SUB_POOL.submit(_yt_web_search, art_clean, "")
    f_web_v = _SUB_POOL.submit(_yt_web_search, art_clean, "EgIQAQ==")
    f_web_p = _SUB_POOL.submit(_yt_web_search, art_clean, "EgIQAw==")
    f_ytm_p = _SUB_POOL.submit(_ytm_search_artists_and_pls, art_clean)

    ytm_songs = f_ytm_s.result() or []
    ytm_vids  = f_ytm_v.result() or []
    web_g_s, ch_list, web_g_p = f_web_g.result() or ([], [], [])
    web_v_s, _, _ = f_web_v.result() or ([], [], [])
    _, _, web_p_p = f_web_p.result() or ([], [], [])
    ytm_pls = f_ytm_p.result() or []

    cid = _CHAN_ID.get(arid) or _CHAN_ID.get(_nfc(raw_name)) or _CHAN_ID.get(ckey)
    canon_name = raw_name
    if not cid:
        for ch_arid, cname in ch_list:
            if _is_same_artist(art_clean, cname, True):
                cid = _CHAN_ID.get(ch_arid) or _CHAN_ID.get(_nfc(cname))
                canon_name = cname
                if ch_arid in _ARTIST_THUMB: _ARTIST_THUMB[arid] = _ARTIST_THUMB[ch_arid]
                break
    if not cid:
        for s in (web_g_s + web_v_s):
            if _is_same_artist(art_clean, s.get("artist",""), True):
                cid = _CHAN_ID.get(s.get("artistId","")) or _CHAN_ID.get(_nfc(s.get("artist","")))
                canon_name = s.get("artist") or canon_name
                if s.get("artistId") in _ARTIST_THUMB and arid not in _ARTIST_THUMB:
                    _ARTIST_THUMB[arid] = _ARTIST_THUMB[s["artistId"]]
                if cid: break

    ch_uploads = _yt_channel_videos(cid, canon_name) if cid else []

    hdd_list, seen = [], set()
    if _H_IDX:
        for tg, at, _ in _H_IDX:
            if tg in ("song", "child", "entry") and at.get("isDir") != "true":
                if _is_same_artist(art_clean, at.get("artist",""), True):
                    sid = at.get("id","")
                    if sid not in seen:
                        seen.add(sid); hdd_list.append(dict(at))

    yt_official_songs = []
    for s in ytm_songs:
        if s["id"] not in seen and _is_same_artist(art_clean, s.get("artist",""), True):
            sc = dict(s); sc["artistId"] = arid; _YL[sc["id"]] = sc
            seen.add(sc["id"]); yt_official_songs.append(sc)

    yt_channel_vids = []
    for s in (ch_uploads + web_v_s + web_g_s + ytm_vids):
        if s["id"] not in seen and _is_same_artist(art_clean, s.get("artist",""), True):
            sc = dict(s); sc["artistId"] = arid; _YL[sc["id"]] = sc
            seen.add(sc["id"]); yt_channel_vids.append(sc)

    out_tracks = hdd_list + yt_official_songs + yt_channel_vids

    pls_filtered, seen_pl = [], set()
    for p_info in (web_g_p + web_p_p + ytm_pls):
        pid = p_info["id"]
        if pid in seen_pl: continue
        if _is_same_artist(art_clean, p_info.get("artist",""), True) or _match_query_text(art_clean, p_info.get("title","")):
            seen_pl.add(pid)
            p_copy = dict(p_info)
            p_copy["owner_arid"] = arid
            _PL_META[p_info["raw_id"]] = p_copy
            pls_filtered.append(p_copy)

    if out_tracks:
        _ART_CACHE_SONGS[ckey] = (out_tracks, now + 600)
        _ART_CACHE_PLS[ckey] = pls_filtered
        for p_inf in pls_filtered[:4]:
            _SUB_POOL.submit(_yt_playlist_tracks, p_inf["raw_id"], canon_name)
    return canon_name, out_tracks, pls_filtered

def _build_top_songs(artist_q, count=60):
    _, tracks, _ = _collect_artist_data(artist_q)
    rt = _E.Element("subsonic-response", status="ok")
    mc = _E.SubElement(rt, "topSongs")
    for s in tracks[:count]:
        _E.SubElement(mc, "song", **s)
    return rt

def _build_artist_page(rid):
    aname, tracks, pls = _collect_artist_data(rid)
    yt_tracks = [s for s in tracks if not str(s.get("id","")).startswith("hdd_")]
    if aname:
        _ARTIST_NAMES[rid] = aname
    rt = _E.Element("subsonic-response", status="ok")
    top_pls = pls[:8]
    top_vids = yt_tracks[:45]
    ar_attr = {"id": rid, "name": (aname or "YouTube Music"), "coverArt": rid, "albumCount": str(len(top_pls) + len(top_vids))}
    if rid in _STARRED.get("artists", {}):
        ar_attr["starred"] = "2026-11-15T00:00:00.000Z"
    ar = _E.SubElement(rt, "artist", **ar_attr)
    for s in top_vids:
        real_t = s.get("title") or s.get("album") or "YouTube Music"
        _E.SubElement(ar, "album", id="al_"+s["id"], parent=rid, isDir="true", name=real_t, title=real_t, album=real_t, artist=(aname or s["artist"]), artistId=rid, coverArt=s["id"], songCount="1", duration=s["duration"], created=s["created"], year="2026", genre=s.get("genre","YouTube Music"))
    for p_info in top_pls:
        p_name = f"\U0001f4d1 [DS Ph\u00e1t] {p_info['title']}"
        _E.SubElement(ar, "album", id=p_info["id"], parent=rid, isDir="true", name=p_name, title=p_name, album=p_name, artist=(aname or p_info["artist"]), artistId=rid, coverArt=p_info["id"], songCount=str(p_info.get("songCount","20")), duration="7200", created="2026-01-01T00:00:00.000Z", year="2026", genre="Playlist")
    return rt

def _build_playlist_or_all_album(aid, is_dir=False):
    rt = _E.Element("subsonic-response", status="ok")
    tg = "directory" if is_dir else "album"
    ch_tg = "child" if is_dir else "song"
    if aid.startswith(("al_all_", "ar_")):
        rid = aid[7:] if aid.startswith("al_all_") else aid
        aname, tracks, _ = _collect_artist_data(rid)
        all_title = f"\U0001f4bf T\u1ea5t C\u1ea3 B\u00e0i H\u00e1t - {aname}"
        dur_all = str(sum(int(x.get("duration", 240)) for x in tracks if str(x.get("duration","")).isdigit()) or 7200)
        cov = tracks[0]["id"] if tracks else rid
        al = _E.SubElement(rt, tg, id=aid, parent=rid, isDir="true", name=all_title, title=all_title, album=all_title, artist=aname, artistId=rid, coverArt=cov, songCount=str(len(tracks)), duration=dur_all, created="2026-01-01T00:00:00.000Z", year="2026", genre="YouTube Music")
        for idx, s in enumerate(tracks[:100], 1):
            sc = dict(s); sc["track"] = str(idx); sc["artistId"] = rid
            _E.SubElement(al, ch_tg, **sc)
        return rt
    raw_pid = _RE.sub(r"^(yt\d*_|pl_)+", "", aid)
    p_info = _PL_META.get(raw_pid, {})
    ptit = p_info.get("title") or "Danh S\u00e1ch Ph\u00e1t YouTube"
    part = p_info.get("artist") or "YouTube Music"
    arid = p_info.get("owner_arid") or ("ar_" + _H.md5(_nfc(part).encode("utf-8")).hexdigest()[:10])
    tracks = _yt_playlist_tracks(raw_pid, part)
    dur_all = str(sum(int(x.get("duration", 240)) for x in tracks if str(x.get("duration","")).isdigit()) or 7200)
    cov = "pl_" + raw_pid if raw_pid in _PL_THUMB else (tracks[0]["id"] if tracks else "")
    al = _E.SubElement(rt, tg, id="pl_"+raw_pid, parent=arid, isDir="true", name=ptit, title=ptit, album=ptit, artist=part, artistId=arid, coverArt=cov, songCount=str(len(tracks)), duration=dur_all, created="2026-01-01T00:00:00.000Z", year="2026", genre="Playlist")
    for idx, s in enumerate(tracks[:100], 1):
        sc = dict(s); sc["track"] = str(idx); sc["album"] = ptit; sc["albumId"] = "pl_" + raw_pid; sc["artistId"] = arid
        _E.SubElement(al, ch_tg, **sc)
    return rt

def _ens_fast(vid):
    ap = _OS.path.join(_AUD_DIR, f"{vid}.aac")
    if _OS.path.exists(ap) and _OS.path.getsize(ap) > 32768:
        if vid in _YL: _YL[vid]["size"] = str(_OS.path.getsize(ap))
        return {"p": ap, "d": True}
    if vid in _FAST_JOBS and not _FAST_JOBS[vid]["d"]:
        return _FAST_JOBS[vid]
    j = {"p": ap, "d": False}
    _FAST_JOBS[vid] = j
    def _run():
        try:
            c1 = ["yt-dlp", "--no-playlist", "--no-warnings", "-q", "--no-check-certificate", "--concurrent-fragments", "8", "-f", "140/251/bestaudio/best", "-o", "-", f"https://www.youtube.com/watch?v={vid}"]
            c2 = ["ffmpeg", "-y", "-loglevel", "error", "-threads", "4", "-i", "pipe:0", "-vn", "-ac", "2", "-ar", "44100", "-c:a", "aac", "-b:a", "256k", "-f", "adts", ap]
            p1 = _SP.Popen(c1, stdout=_SP.PIPE, stderr=_SP.DEVNULL)
            p2 = _SP.Popen(c2, stdin=p1.stdout, stdout=_SP.DEVNULL, stderr=_SP.DEVNULL)
            p1.stdout.close(); p2.communicate(timeout=3600)
            if _OS.path.exists(ap) and _OS.path.getsize(ap) > 32768 and vid in _YL:
                _YL[vid]["size"] = str(_OS.path.getsize(ap))
        except Exception:
            pass
        finally:
            j["d"] = True
    _TH.Thread(target=_run, daemon=True).start()
    return j

def _to_car_jpg(d, is_yt_16_9=False):
    if not d or len(d) < 400:
        return None
    tid = f"{_TH.get_ident()}_{_RND.randint(1000,9999)}"
    fin = f"/tmp/art_in_{tid}.jpg"
    fout = f"/tmp/art_out_{tid}.jpg"
    try:
        open(fin, "wb").write(d)
        vf = "crop=270:270:105:45,scale=480:480" if is_yt_16_9 else "scale=480:480:force_original_aspect_ratio=increase,crop=480:480"
        p = _SP.run(["ffmpeg","-y","-loglevel","error","-i",fin,"-frames:v","1","-vf",vf,"-strict","-2","-q:v","2",fout], stdout=_SP.PIPE, stderr=_SP.PIPE, timeout=5)
        if p.returncode == 0 and _OS.path.exists(fout) and _OS.path.getsize(fout) > 400:
            return open(fout, "rb").read()
    except Exception:
        pass
    finally:
        for fp in (fin, fout):
            try:
                if _OS.path.exists(fp): _OS.remove(fp)
            except Exception:
                pass
    return d if d.startswith(b"\xff\xd8") else None

def _get_art_bytes(cid):
    if not cid: return None
    s = str(cid or "").strip()
    is_hdd = bool(_RE.match(r"^hdd\d*_", s))
    clean_s = _RE.sub(r"^(hdd\d*_|yt\d*_)+", "", s)
    if clean_s.startswith("al_all_"):
        clean_s = clean_s[7:]

    if _RE.match(r"^ar_[0-9a-f]{10}$", clean_s) and not is_hdd:
        ckey = "ar_" + clean_s
        if ckey in _ART_RAM: return _ART_RAM[ckey]
        cf = _OS.path.join(_ART_DIR, _H.md5(ckey.encode()).hexdigest() + ".jpg")
        if _OS.path.exists(cf) and _OS.path.getsize(cf) > 400:
            d = open(cf, "rb").read()
            _ART_RAM[ckey] = d; return d
        if clean_s not in _ARTIST_THUMB:
            aname = _resolve_artist_name(clean_s)
            if aname:
                _yt_web_search(aname, "")
                if clean_s not in _ARTIST_THUMB:
                    _ytm_search_artists_and_pls(aname)
        tu = _ARTIST_THUMB.get(clean_s, "")
        if tu:
            try:
                with _U.urlopen(_U.Request(_fix_url(tu), headers={"User-Agent":"Mozilla/5.0"}), timeout=5, context=_CTX) as r:
                    d = r.read()
                    if r.status == 200 and len(d) > 400:
                        jpg = d if d.startswith(b"\xff\xd8") else (_to_car_jpg(d, False) or d)
                        open(cf, "wb").write(jpg); _ART_RAM[ckey] = jpg; return jpg
            except Exception:
                pass

    if clean_s.startswith("pl_") and len(clean_s) > 12 and not is_hdd:
        raw_pid = clean_s[3:]
        ckey = "pl_" + raw_pid
        if ckey in _ART_RAM: return _ART_RAM[ckey]
        cf = _OS.path.join(_ART_DIR, _H.md5(ckey.encode()).hexdigest() + ".jpg")
        if _OS.path.exists(cf) and _OS.path.getsize(cf) > 400:
            d = open(cf, "rb").read()
            _ART_RAM[ckey] = d; return d
        tu = _PL_THUMB.get(raw_pid, "")
        if tu:
            try:
                with _U.urlopen(_U.Request(_fix_url(tu), headers={"User-Agent":"Mozilla/5.0"}), timeout=5, context=_CTX) as r:
                    d = r.read()
                    if r.status == 200 and len(d) > 400:
                        jpg = _to_car_jpg(d, is_yt_16_9=("hqdefault" in tu)) or d
                        open(cf, "wb").write(jpg); _ART_RAM[ckey] = jpg; return jpg
            except Exception:
                pass
        tr_list = _PL_TRACKS_CACHE.get(raw_pid) or _yt_playlist_tracks(raw_pid)
        if tr_list and tr_list[0].get("id"):
            clean_s = tr_list[0]["id"]

    raw_id = clean_s[3:] if (clean_s.startswith("al_") and len(clean_s) == 14) else clean_s
    kind = "hdd" if is_hdd else ("yt" if (s.startswith("yt") or _RE.match(r"^[A-Za-z0-9_-]{11}$", raw_id)) else "hdd")
    k_id = (_ART_MAP.get(raw_id, "") or raw_id) if kind == "hdd" else raw_id
    ckey = f"{kind}_{k_id}"
    if ckey in _ART_RAM: return _ART_RAM[ckey]
    cf = _OS.path.join(_ART_DIR, _H.md5(ckey.encode()).hexdigest() + ".jpg")
    if _OS.path.exists(cf) and _OS.path.getsize(cf) > 400:
        d = open(cf, "rb").read()
        if len(_ART_RAM) < 800: _ART_RAM[ckey] = d
        return d

    if kind == "hdd" and _HDD:
        alid = _ART_MAP.get(raw_id, "")
        cands = []
        for x in ([raw_id, "mf-"+raw_id, "al-"+alid, alid, "al-"+raw_id, raw_id[3:] if raw_id.startswith(("al-","mf-")) else ""] if alid else [raw_id, "mf-"+raw_id, "al-"+raw_id, raw_id[3:] if raw_id.startswith(("al-","mf-")) else ""]):
            if x and x not in cands: cands.append(x)
        with _ART_SEM:
            for cand in cands:
                try:
                    cq = [x for x in _hdd_q([("id", cand), ("size", "500")]) if x[0] != "f"]
                    with _U.urlopen(_U.Request(f"{_HDD}/rest/getCoverArt.view?{_ue(cq)}", headers={"User-Agent":"Resonus"}), timeout=6, context=_CTX) as r:
                        d = r.read()
                        if r.status == 200 and len(d) > 400 and not d.lstrip().startswith(b"<"):
                            jpg = d if d.startswith(b"\xff\xd8") else (_to_car_jpg(d, False) or d)
                            open(cf, "wb").write(jpg); _ART_RAM[ckey] = jpg; return jpg
                except Exception:
                    pass
        qt = _HDD_META.get(raw_id) or _HDD_META.get(k_id)
        if qt:
            s_items = _ytm_search_items(qt, "EgWKAQIIAWoKEAkQBRAKEAMQBA==", False)
            if s_items: raw_id = s_items[0]["id"]

    m = _RE.search(r"([A-Za-z0-9_-]{11})$", raw_id)
    if m:
        vid = m.group(1)
        if vid in _YTM_THUMB:
            try:
                tu = _RE.sub(r"=w\d+-h\d+[^/]*$", "=w500-h500-l95-rj", _YTM_THUMB[vid])
                with _U.urlopen(_U.Request(tu, headers={"User-Agent":"Mozilla/5.0"}), timeout=4, context=_CTX) as r:
                    d = r.read()
                    if r.status == 200 and len(d) > 400:
                        jpg = d if d.startswith(b"\xff\xd8") else (_to_car_jpg(d, False) or d)
                        open(cf, "wb").write(jpg); _ART_RAM[ckey] = jpg; return jpg
            except Exception:
                pass
        for rs in ("hqdefault.jpg", "mqdefault.jpg"):
            try:
                with _U.urlopen(_U.Request(f"https://i.ytimg.com/vi/{vid}/{rs}", headers={"User-Agent":"Mozilla/5.0"}), timeout=4, context=_CTX) as r:
                    d = r.read()
                    if r.status == 200 and len(d) > 400:
                        jpg = _to_car_jpg(d, is_yt_16_9=(rs == "hqdefault.jpg")) or d
                        if jpg:
                            open(cf, "wb").write(jpg); _ART_RAM[ckey] = jpg; return jpg
            except Exception:
                pass
    return None

def _prewarm_list(ids):
    for cid in ids[:24]:
        try: _get_art_bytes(cid)
        except Exception: pass

def _fmt(rt, js, cb=None):
    _cln(rt)
    warm = []
    for x in rt.iter():
        a = x.attrib
        ca = a.get("coverArt")
        if ca:
            raw_ca = _RE.sub(r"^(hdd\d*_|yt\d*_)+", "", str(ca))
            if str(ca).startswith("hdd") or str(a.get("id","")).startswith("hdd_"):
                a["coverArt"] = "hdd20_" + raw_ca
            else:
                a["coverArt"] = "yt20_" + raw_ca
            warm.append(a["coverArt"])
        if x.tag in ("song","child","entry") and a.get("isDir") != "true" and not str(a.get("id","")).startswith("hdd_"):
            a.setdefault("bitRate", "256")
            a.setdefault("suffix", "aac")
            a.setdefault("contentType", "audio/aac")
            a.setdefault("samplingRate", "44100")
            a.setdefault("bitDepth", "16")
            a.setdefault("channelCount", "2")
            a.setdefault("type", "music")
    if warm:
        _ART_POOL.submit(_prewarm_list, list(dict.fromkeys(warm)))
    rt.attrib.update({"status":rt.attrib.get("status","ok"),"version":"1.16.1","type":"subsonic","serverVersion":"1.16.1"})
    if js:
        b = _J.dumps({"subsonic-response": _x2d(rt, 0)})
        return _R(content=f"{cb}({b});" if cb else b, media_type="application/json; charset=utf-8", headers=_CORS)
    rt.set("xmlns", "http://subsonic.org/restapi")
    return _R(content=b'<?xml version="1.0" encoding="UTF-8"?>\n' + _E.tostring(rt, encoding="utf-8"), media_type="application/xml; charset=utf-8", headers=_CORS)

def _hdd_songs():
    if not _H_IDX: _init_hdd()
    return [dict(a) for t, a, _ in _H_IDX if t in ("song","child","entry") and a.get("isDir") != "true"]

@app.middleware("http")
async def _k9_mw(req, call_next):
    if req.method == "OPTIONS": return await call_next(req)
    _load_addon_opts()
    p = _RE.sub(r"/+", "/", req.url.path); pl = p.lower()
    if not pl.startswith("/rest/"): return await call_next(req)
    qi = list(_pq(req.scope.get("query_string", b"").decode("utf-8","ignore"), keep_blank_values=True))
    if req.method == "POST":
        bf = await req.body()
        if bf:
            try: qi.extend([(str(k),str(v)) for k,v in _J.loads(bf).items()] if bf.strip().startswith(b"{") else _pq(bf.decode("utf-8","ignore"), keep_blank_values=True))
            except Exception: pass
        async def _rc(): return {"type": "http.request", "body": bf, "more_body": False}
        req._receive = _rc
    qd = dict(qi); fv = qd.get("f","").lower()
    js = fv in ("json","jsonp") or "application/json" in req.headers.get("accept","").lower()
    cb = qd.get("callback") if fv == "jsonp" else None
    lp = _A.get_running_loop()

    if "/getcoverart" in pl:
        cid = str(qd.get("id") or "").strip()
        d = await lp.run_in_executor(_ART_POOL, _get_art_bytes, cid) if cid else None
        if d:
            return _R(content=d, media_type="image/jpeg", headers={**_CORS, "Content-Length": str(len(d)), "Cache-Control": "public, max-age=604800"})
        return _R(content=_PNG_1X1, media_type="image/png", headers={**_CORS, "Content-Length": str(len(_PNG_1X1))})

    if not _check_auth(qd): return await call_next(req)

    if "getartistinfo" in pl:
        rid = str(qd.get("id") or "").strip()
        rt = _E.Element("subsonic-response", status="ok")
        tg = "artistInfo2" if "2" in pl else "artistInfo"
        ai = _E.SubElement(rt, tg)
        if rid.startswith("ar_"):
            auth_p = _ue([(k, v) for k, v in qi if k in ("u","p","t","s","v","c")])
            base_u = str(req.base_url).rstrip("/")
            if "nhacuakhoa.online" in base_u and base_u.startswith("http://"):
                base_u = "https://" + base_u[7:]
            img_u = f"{base_u}/rest/getCoverArt.view?id=yt20_{rid}&{auth_p}"
            for itg in ("smallImageUrl", "mediumImageUrl", "largeImageUrl"):
                el = _E.SubElement(ai, itg); el.text = img_u
        return _fmt(rt, js, cb)

    if "gettopsongs" in pl:
        art_q = qd.get("artist") or qd.get("query") or qd.get("id") or ""
        cnt = int(qd.get("count") or 60)
        rt = await lp.run_in_executor(_REQ_POOL, _build_top_songs, art_q, cnt)
        return _fmt(rt, js, cb)

    rid = str(qd.get("id") or "").strip()
    if rid.startswith("ar_") and "/getartist" in pl and "getartists" not in pl and "artistinfo" not in pl:
        rt = await lp.run_in_executor(_REQ_POOL, _build_artist_page, rid)
        return _fmt(rt, js, cb)

    if (rid.startswith("pl_") or rid.startswith("al_all_") or (rid.startswith("ar_") and any(x in pl for x in ("/getalbum","/getmusicdirectory")))) and any(x in pl for x in ("/getalbum","/getmusicdirectory")):
        rt = await lp.run_in_executor(_REQ_POOL, _build_playlist_or_all_album, rid, "directory" in pl)
        return _fmt(rt, js, cb)

    # Tim kiem voi thu tu uu tien: 1. HDD -> 2. YouTube Music -> 3. YouTube
    if "search" in pl:
        global _S_SEQ, _LAST_Q
        q_str = _UD.normalize("NFC", (qd.get("query") or qd.get("any") or qd.get("q") or "").strip('" '))
        st = "searchResult2" if "2" in pl else "searchResult3"
        if len(q_str) == 1:
            rt = _E.Element("subsonic-response", status="ok"); _E.SubElement(rt, st); return _fmt(rt, js, cb)
        if q_str:
            qn, now = _nfc(q_str), _TM.time()
            if qn not in _SCACHE or _SCACHE[qn][1] < now:
                enter = (qn == _LAST_Q); _LAST_Q = qn; _S_SEQ += 1; ms = _S_SEQ
                if not enter:
                    await _A.sleep(0.4)
                    if ms != _S_SEQ and qn not in _SCACHE:
                        rt = _E.Element("subsonic-response", status="ok"); _E.SubElement(rt, st); return _fmt(rt, js, cb)
                f1 = lp.run_in_executor(_REQ_POOL, _yt_search, q_str, "searchResult3")
                f2 = lp.run_in_executor(_REQ_POOL, _hdd_search, q_str, qi)
                mrt, hrt = await f1, await f2
                if hrt is not None and len(hrt) > 0:
                    _hdd_first(mrt[0], hrt[0])
                _SCACHE[qn] = (_E.tostring(mrt), now + 300)
            mrt = _E.fromstring(_SCACHE[qn][0]); mrt[0].tag = st
            return _fmt(mrt, js, cb)

    if "/star" in pl or "/unstar" in pl:
        is_un = "/unstar" in pl
        hdd_qi = []
        for k, v in qi:
            if k in ("id", "albumId", "artistId") and v:
                sv = str(v).strip()
                if sv.startswith("hdd_"):
                    hdd_qi.append((k, sv))
                elif k == "artistId" or sv.startswith("ar_"):
                    if is_un: _STARRED.get("artists", {}).pop(sv, None)
                    else: _STARRED.setdefault("artists", {})[sv] = _resolve_artist_name(sv)
                elif k == "albumId" or sv.startswith(("al_","pl_")):
                    vid = sv[3:] if sv.startswith("al_") else sv
                    if is_un: _STARRED.get("albums", {}).pop(vid, None)
                    else: _STARRED.setdefault("albums", {})[vid] = _YL.get(vid, {})
                else:
                    if is_un: _STARRED.get("songs", {}).pop(sv, None)
                    else: _STARRED.setdefault("songs", {})[sv] = _YL.get(sv, {})
        _save_star()
        if hdd_qi:
            await lp.run_in_executor(_REQ_POOL, _hdd_xml, p, qi)
        return _fmt(_E.Element("subsonic-response", status="ok"), js, cb)

    if "getstarred" in pl:
        hrt = await lp.run_in_executor(_REQ_POOL, _hdd_xml, p, qi)
        rt = _E.Element("subsonic-response", status="ok")
        tg = "starred2" if "2" in pl else "starred"
        sc = _E.SubElement(rt, tg)
        if hrt is not None and len(hrt) > 0:
            for x in hrt[0]: sc.append(x)
        for arid, aname in (_STARRED.get("artists") or {}).items():
            _E.SubElement(sc, "artist", id=arid, name=(aname or "YouTube Music"), coverArt=arid, albumCount="25", starred="2026-11-15T00:00:00.000Z")
        for vid, s in (_STARRED.get("albums") or {}).items():
            if isinstance(s, dict) and s.get("id"):
                _E.SubElement(sc, "album", id="al_"+s["id"], parent=s.get("artistId",""), isDir="true", name=s.get("album",""), title=s.get("album",""), album=s.get("album",""), artist=s.get("artist",""), artistId=s.get("artistId",""), coverArt=s["id"], songCount="1", duration=s.get("duration","240"), created="2026-01-01T00:00:00.000Z", year="2026", starred="2026-11-15T00:00:00.000Z")
        for vid, s in (_STARRED.get("songs") or {}).items():
            if not (isinstance(s, dict) and s.get("id")) and vid in _YL:
                s = _YL[vid]
            if isinstance(s, dict) and s.get("id"):
                _E.SubElement(sc, "song", **{**s, "starred": "2026-11-15T00:00:00.000Z"})
        return _fmt(rt, js, cb)

    if "/stream" in pl or "/download" in pl:
        sid = str(qd.get("id") or "").strip()
        if sid:
            try:
                await _A.wait_for(lp.run_in_executor(_ART_POOL, _get_art_bytes, sid), timeout=1.5)
            except Exception:
                pass
        if not sid.startswith("hdd_"):
            m = _RE.search(r"([A-Za-z0-9_-]{11})$", sid)
            if m:
                vid = m.group(1)
                dur_v = int((_YL.get(vid) or {}).get("duration") or 240)
                if dur_v > 600:
                    j = _ens_fast(vid)
                    for _ in range(60):
                        if (_OS.path.exists(j["p"]) and _OS.path.getsize(j["p"]) >= 32768) or j["d"]: break
                        await _A.sleep(0.1)
                    if _OS.path.exists(j["p"]) and _OS.path.getsize(j["p"]) > 0:
                        async def _long_g():
                            with open(j["p"], "rb") as f:
                                while True:
                                    c = f.read(32768)
                                    if c: yield c
                                    elif j["d"]: break
                                    else: await _A.sleep(0.05)
                        return _SR(_long_g(), status_code=200, media_type="audio/aac", headers=_CORS)
        return await call_next(req)

    if "getrandomsongs" in pl:
        cnt = int(qd.get("size") or qd.get("count") or 30)
        h_all = await lp.run_in_executor(_REQ_POOL, _hdd_songs)
        if not _YL:
            await lp.run_in_executor(_REQ_POOL, _yt_search, "Bolero Nh\u1ea1c V\u00e0ng L\u1ec7 Quy\u00ean B\u1eb1ng Ki\u1ec1u Th\u00f9y Dung")
        y_all = [dict(x) for x in _YL.values() if not str(x.get("id","")).startswith("loc_")]
        h_pick = _RND.sample(h_all, min(len(h_all), cnt // 2 + 5)) if h_all else []
        y_pick = _RND.sample(y_all, min(len(y_all), cnt // 2 + 5)) if y_all else []
        rt = _E.Element("subsonic-response", status="ok")
        mc = _E.SubElement(rt, "randomSongs")
        out = []
        for i in range(max(len(h_pick), len(y_pick))):
            if i < len(h_pick): out.append(h_pick[i])
            if i < len(y_pick): out.append(y_pick[i])
        for s in out[:cnt]:
            _E.SubElement(mc, "song", **s)
        return _fmt(rt, js, cb)

    if "getalbumlist" in pl:
        lt = qd.get("type","").lower()
        off = int(qd.get("offset") or 0)
        hrt = await lp.run_in_executor(_REQ_POOL, _hdd_xml, p, qi)
        mrt = _E.Element("subsonic-response", status="ok")
        mc = _E.SubElement(mrt, "albumList2" if "2" in pl else "albumList")
        h_als = list(hrt[0]) if (hrt is not None and len(hrt) > 0) else []
        if lt in ("alphabeticalbyname","alphabeticalbyartist","starred","byyear","bygenre") or off > 0:
            for x in h_als: mc.append(x)
            return _fmt(mrt, js, cb)
        seen_h = {x.attrib.get("id") for x in h_als}
        h_idx_als = [dict(at) for tg, at, _ in _H_IDX if tg == "album" and at.get("id") not in seen_h]
        if h_idx_als:
            _RND.shuffle(h_idx_als)
            for at in h_idx_als[:max(0, 10 - len(h_als))]:
                h_als.append(_E.Element("album", **at))
        if lt == "random" and h_als:
            _RND.shuffle(h_als)
        if not _YL:
            await lp.run_in_executor(_REQ_POOL, _yt_search, "Th\u00f9y Dung L\u1ec7 Quy\u00ean B\u1eb1ng Ki\u1ec1u Bolero")
        yt_all = [x for x in _YL.values() if not str(x.get("id","")).startswith("loc_")]
        if lt == "recent": yt_sl = yt_all[0:10]
        elif lt == "newest": yt_sl = yt_all[10:20] or yt_all[0:10]
        elif lt == "frequent": yt_sl = yt_all[20:30] or yt_all[:10]
        else: yt_sl = _RND.sample(yt_all, min(10, len(yt_all))) if yt_all else []
        y_als = [_E.Element("album", id="al_"+s["id"], parent=s["artistId"], isDir="true", name=s["album"], title=s["album"], album=s["album"], artist=s["artist"], artistId=s["artistId"], coverArt=s["id"], songCount="1", duration=s["duration"], playCount="1", created=s["created"], year="2026", genre=s.get("genre","YouTube Music")) for s in yt_sl]
        for i in range(max(len(h_als), len(y_als))):
            if i < len(h_als): mc.append(h_als[i])
            if i < len(y_als): mc.append(y_als[i])
        return _fmt(mrt, js, cb)

    return await call_next(req)

_SCACHE.clear()

# --- K10: V1.3.3 AUTO-CLEANUP CACHE, LIVE LYRICS & SCROBBLE HISTORY ---
import shutil as _SH

_LSTS.update({"structuredLyrics", "line"})
_INTS.update({"start", "offset"})

_HIST_F = "/data/yt_history_v1.json" if _OS.path.isdir("/data") else "/tmp/yt_history_v1.json"
_LYR_F  = "/data/lyrics_cache_v1.json" if _OS.path.isdir("/data") else "/tmp/lyrics_cache_v1.json"

try:
    _YT_HIST = _J.load(open(_HIST_F))
except Exception:
    _YT_HIST = {"recent": [], "counts": {}}

try:
    _LYR_CACHE = _J.load(open(_LYR_F))
except Exception:
    _LYR_CACHE = {}

def _save_hist():
    try:
        _J.dump(_YT_HIST, open(_HIST_F, "w"))
    except Exception:
        pass

def _save_lyr():
    try:
        if len(_LYR_CACHE) > 500:
            for k in list(_LYR_CACHE.keys())[:100]:
                _LYR_CACHE.pop(k, None)
        _J.dump(_LYR_CACHE, open(_LYR_F, "w"))
    except Exception:
        pass

def _record_yt_play(vid, is_scrobble=False):
    if not _RE.match(r"^[A-Za-z0-9_-]{11}$", str(vid or "")):
        return
    rec = _YT_HIST.setdefault("recent", [])
    if vid in rec:
        rec.remove(vid)
    rec.insert(0, vid)
    del rec[60:]
    cnts = _YT_HIST.setdefault("counts", {})
    cnts[vid] = int(cnts.get(vid, 0)) + 1
    if vid in _YL:
        _YL[vid]["playCount"] = str(cnts[vid])
    ap = _OS.path.join(_AUD_DIR, f"{vid}.aac")
    if _OS.path.exists(ap):
        try:
            _OS.utime(ap, None)
        except Exception:
            pass
    _save_hist()

def _trim_folder(folder, max_bytes, target_bytes):
    if not _OS.path.isdir(folder):
        return
    files = []
    total = 0
    for fn in _OS.listdir(folder):
        fp = _OS.path.join(folder, fn)
        if _OS.path.isfile(fp):
            try:
                st = _OS.stat(fp)
                files.append((st.st_mtime, st.st_size, fp))
                total += st.st_size
            except Exception:
                pass
    if total > max_bytes:
        files.sort(key=lambda x: x[0])
        for _, sz, fp in files:
            try:
                _OS.remove(fp)
                total -= sz
                if total <= target_bytes:
                    break
            except Exception:
                pass

def _cleanup_cache_loop():
    _TM.sleep(20)
    # Xoa cac thu muc cache anh cu tu v5 -> v19 de giai phong dung luong o cung
    if _OS.path.isdir("/data"):
        for dname in _OS.listdir("/data"):
            if dname.startswith("art_cache_") and dname != _OS.path.basename(_ART_DIR):
                try:
                    _SH.rmtree(_OS.path.join("/data", dname), ignore_errors=True)
                except Exception:
                    pass
    while True:
        try:
            # Gioi han cache nhac YouTube 2.0 GB (xoa bot file cu nhat xuong con 1.4 GB)
            _trim_folder(_AUD_DIR, 2000 * 1024 * 1024, 1400 * 1024 * 1024)
            # Gioi han cache anh bia 400 MB (xoa bot xuong con 280 MB)
            _trim_folder(_ART_DIR, 400 * 1024 * 1024, 280 * 1024 * 1024)
            for fn in _OS.listdir("/tmp"):
                if fn.startswith(("art_in_", "art_out_")):
                    try:
                        _OS.remove(_OS.path.join("/tmp", fn))
                    except Exception:
                        pass
        except Exception:
            pass
        _TM.sleep(1800)

_TH.Thread(target=_cleanup_cache_loop, daemon=True).start()

def _parse_lrc(lrc_txt):
    lines = []
    for raw_line in str(lrc_txt or "").splitlines():
        m = _RE.match(r"^\[(\d{1,2}):(\d{2})(?:[.:](\d{1,3}))?\](.*)$", raw_line.strip())
        if m:
            mm, ss, ms_s, txt = int(m.group(1)), int(m.group(2)), (m.group(3) or "0"), m.group(4).strip()
            ms = int(ms_s.ljust(3, "0")[:3])
            lines.append((mm * 60000 + ss * 1000 + ms, txt))
    return lines

def _fetch_lyrics_data(vid, art, tit):
    ckey = vid if vid else _nfc(f"{art}_{tit}")
    if ckey in _LYR_CACHE:
        return _LYR_CACHE[ckey]

    c_art = _clean_artist_name(art)
    c_tit = _RE.sub(r"\(.*?\)|\[.*?\]", " ", _clean_seed(tit)).strip() or _clean_seed(tit)

    # 1. Tim loi bai hat dong bo thoi gian thuc (Synced LRC) tren LRCLIB
    if c_tit:
        q_str = f"{c_art} {c_tit}".strip()
        try:
            req = _U.Request(
                f"https://lrclib.net/api/search?q={_U.quote(q_str)}",
                headers={"User-Agent": "Mirasonic/1.3.3 (https://nhacuakhoa.online)"}
            )
            with _U.urlopen(req, timeout=5, context=_CTX) as r:
                items = _J.loads(r.read().decode("utf-8", "ignore"))
                if isinstance(items, list) and items:
                    best = None
                    for it in items:
                        if it.get("syncedLyrics") and _is_same_artist(c_art, it.get("artistName", ""), False):
                            best = it; break
                    if not best:
                        best = next((it for it in items if it.get("syncedLyrics")), items[0])
                    if best.get("syncedLyrics"):
                        lrc_lines = _parse_lrc(best["syncedLyrics"])
                        if lrc_lines:
                            plain = "\n".join(x[1] for x in lrc_lines if x[1])
                            res = {"synced": True, "lines": lrc_lines, "plain": plain}
                            _LYR_CACHE[ckey] = res; _save_lyr()
                            return res
                    if best.get("plainLyrics"):
                        p_txt = str(best["plainLyrics"]).strip()
                        l_arr = [(idx * 4000, ln.strip()) for idx, ln in enumerate(p_txt.splitlines()) if ln.strip()]
                        res = {"synced": False, "lines": l_arr, "plain": p_txt}
                        _LYR_CACHE[ckey] = res; _save_lyr()
                        return res
        except Exception:
            pass

    # 2. Tim loi bai hat chinh thuc tren YouTube Music (InnerTube MPLYt)
    if not vid and c_tit:
        s_items = _ytm_search_items(f"{c_art} {c_tit}".strip(), "EgWKAQIIAWoKEAkQBRAKEAMQBA==", False)
        if s_items:
            vid = s_items[0]["id"]
    if vid:
        try:
            d_nx = _ytm_post("next", {"videoId": vid})
            mply = ""
            for b in _find_k(d_nx, "browseEndpoint"):
                bid = str(b.get("browseId") or "")
                if bid.startswith("MPLY"):
                    mply = bid; break
            if mply:
                d_br = _ytm_post("browse", {"browseId": mply})
                for sh in _find_k(d_br, "musicDescriptionShelfRenderer"):
                    runs = (sh.get("description") or {}).get("runs") or []
                    p_txt = "".join(x.get("text", "") for x in runs).strip()
                    if p_txt:
                        l_arr = [(idx * 4000, ln.strip()) for idx, ln in enumerate(p_txt.splitlines()) if ln.strip()]
                        res = {"synced": False, "lines": l_arr, "plain": p_txt}
                        _LYR_CACHE[ckey] = res; _save_lyr()
                        return res
        except Exception:
            pass

    return {"synced": False, "lines": [], "plain": ""}

def _build_lyrics_resp(qd, is_structured=False):
    rid = str(qd.get("id") or "").strip()
    art = _UD.normalize("NFC", str(qd.get("artist") or "")).strip()
    tit = _UD.normalize("NFC", str(qd.get("title") or "")).strip()
    vid = ""
    if rid:
        raw_id = _RE.sub(r"^(hdd\d*_|yt\d*_|al_|pl_)+", "", rid)
        if rid.startswith("hdd"):
            for tg, at, _ in _H_IDX:
                aid = _RE.sub(r"^(hdd\d*_|hdd_)+", "", str(at.get("id", "")))
                if aid in (raw_id, "mf-" + raw_id):
                    tit = tit or at.get("title") or at.get("name") or ""
                    art = art or at.get("artist") or ""
                    break
        elif _RE.match(r"^[A-Za-z0-9_-]{11}$", raw_id):
            vid = raw_id
            if vid in _YL:
                tit = tit or _YL[vid].get("title", "")
                art = art or _YL[vid].get("artist", "")
            else:
                s = _ensure_vid_meta(vid)
                if s:
                    tit = tit or s.get("title", "")
                    art = art or s.get("artist", "")

    ldata = _fetch_lyrics_data(vid, art, tit)
    rt = _E.Element("subsonic-response", status="ok")
    if is_structured:
        ll = _E.SubElement(rt, "lyricsList")
        if ldata.get("lines"):
            sl = _E.SubElement(ll, "structuredLyrics", displayArtist=(art or "Mirasonic"), displayTitle=(tit or "Lyrics"), lang="vi", offset="0", synced=("true" if ldata.get("synced") else "false"))
            for st_ms, l_txt in ldata["lines"]:
                _E.SubElement(sl, "line", start=str(st_ms), value=l_txt)
    else:
        ly = _E.SubElement(rt, "lyrics", artist=(art or "Mirasonic"), title=(tit or "Lyrics"))
        if ldata.get("plain"):
            ly.text = ldata["plain"]
    return rt

@app.middleware("http")
async def _k10_mw(req, call_next):
    if req.method == "OPTIONS":
        return await call_next(req)
    p = _RE.sub(r"/+", "/", req.url.path); pl = p.lower()
    if not pl.startswith("/rest/"):
        return await call_next(req)
    qi = list(_pq(req.scope.get("query_string", b"").decode("utf-8", "ignore"), keep_blank_values=True))
    if req.method == "POST":
        bf = await req.body()
        if bf:
            try:
                qi.extend([(str(k), str(v)) for k, v in _J.loads(bf).items()] if bf.strip().startswith(b"{") else _pq(bf.decode("utf-8", "ignore"), keep_blank_values=True))
            except Exception:
                pass
        async def _rc():
            return {"type": "http.request", "body": bf, "more_body": False}
        req._receive = _rc
    qd = dict(qi); fv = qd.get("f", "").lower()
    js = fv in ("json", "jsonp") or "application/json" in req.headers.get("accept", "").lower()
    cb = qd.get("callback") if fv == "jsonp" else None
    if not _check_auth(qd):
        return await call_next(req)
    lp = _A.get_running_loop()

    # 1. Xu ly Loi bai hat (getLyrics & getLyricsBySongId)
    if "getlyrics" in pl:
        rt = await lp.run_in_executor(_REQ_POOL, _build_lyrics_resp, qd, ("bysongid" in pl))
        return _fmt(rt, js, cb)

    # 2. Ghi nhan lich su nghe nhac (scrobble & stream)
    if "/scrobble" in pl:
        sid = str(qd.get("id") or "").strip()
        if sid.startswith("hdd"):
            await lp.run_in_executor(_REQ_POOL, _hdd_xml, p, qi)
        else:
            m = _RE.search(r"([A-Za-z0-9_-]{11})$", sid)
            if m:
                _record_yt_play(m.group(1), True)
        return _fmt(_E.Element("subsonic-response", status="ok"), js, cb)

    if "/stream" in pl:
        sid = str(qd.get("id") or "").strip()
        if not sid.startswith("hdd"):
            m = _RE.search(r"([A-Za-z0-9_-]{11})$", sid)
            if m:
                _record_yt_play(m.group(1), False)
        return await call_next(req)

    # 3. Dong bo lich su Recently Played & Most Played cua YouTube ra Trang chu Wavio (Tron 50-50 voi HDD)
    if "getalbumlist" in pl:
        lt = qd.get("type", "").lower()
        off = int(qd.get("offset") or 0)
        if off == 0 and lt in ("recent", "frequent"):
            hrt = await lp.run_in_executor(_REQ_POOL, _hdd_xml, p, qi)
            mrt = _E.Element("subsonic-response", status="ok")
            mc = _E.SubElement(mrt, "albumList2" if "2" in pl else "albumList")
            h_als = list(hrt[0]) if (hrt is not None and len(hrt) > 0) else []
            seen_h = {x.attrib.get("id") for x in h_als}
            h_idx_als = [dict(at) for tg, at, _ in _H_IDX if tg == "album" and at.get("id") not in seen_h]
            if h_idx_als:
                _RND.shuffle(h_idx_als)
                for at in h_idx_als[:max(0, 10 - len(h_als))]:
                    h_als.append(_E.Element("album", **at))
            if not _YL:
                await lp.run_in_executor(_REQ_POOL, _yt_search, "Th\u00f9y Dung L\u1ec7 Quy\u00ean B\u1eb1ng Ki\u1ec1u Bolero")

            yt_sl, seen_y = [], set()
            if lt == "recent":
                for vid in _YT_HIST.get("recent", []):
                    if vid in _YL and vid not in seen_y:
                        seen_y.add(vid); yt_sl.append(_YL[vid])
            else:
                top_v = sorted(_YT_HIST.get("counts", {}).items(), key=lambda x: int(x[1]), reverse=True)
                for vid, _ in top_v:
                    if vid in _YL and vid not in seen_y:
                        seen_y.add(vid); yt_sl.append(_YL[vid])

            for s in _YL.values():
                if len(yt_sl) >= 10:
                    break
                if s["id"] not in seen_y and not str(s["id"]).startswith("loc_"):
                    seen_y.add(s["id"]); yt_sl.append(s)

            y_als = [_E.Element("album", id="al_"+s["id"], parent=s["artistId"], isDir="true", name=s["album"], title=s["album"], album=s["album"], artist=s["artist"], artistId=s["artistId"], coverArt=s["id"], songCount="1", duration=s["duration"], playCount=str(_YT_HIST.get("counts",{}).get(s["id"], 1)), created=s["created"], year="2026", genre=s.get("genre","YouTube Music")) for s in yt_sl[:10]]
            for i in range(max(len(h_als), len(y_als))):
                if i < len(h_als): mc.append(h_als[i])
                if i < len(y_als): mc.append(y_als[i])
            return _fmt(mrt, js, cb)

    return await call_next(req)




# --- K11: V1.4.0 ZERO-WEBP BASELINE JPEG + INSTANT RAM PRECACHE FOR BYD WIDGET ---
_ART_DIR = "/data/art_cache_v26" if _OS.path.isdir("/data") else "/tmp/art_cache_v26"
_OS.makedirs(_ART_DIR, exist_ok=True)
_ART_RAM.clear()
_ART_LOCKS = {}
_ART_LOCKS_GUARD = _TH.Lock()
_DL_POOL = _CF.ThreadPoolExecutor(max_workers=32)

def _get_lock(ckey):
    with _ART_LOCKS_GUARD:
        if ckey not in _ART_LOCKS:
            _ART_LOCKS[ckey] = _TH.Lock()
        return _ART_LOCKS[ckey]

def _fix_url(u):
    u = str(u or "").strip()
    if u.startswith("//"):
        u = "https:" + u
    if "yt3.ggpht.com" in u or "googleusercontent.com" in u:
        u = _RE.sub(r"=s\d+[^/]*$", "=s500-c-k-c0x00ffffff-no-rj", u)
        u = _RE.sub(r"=w\d+-h\d+[^/]*$", "=w500-h500-l95-rj", u)
    return u

def _dl_fast(url, min_sz=400, timeout=2.0):
    try:
        req = _U.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with _U.urlopen(req, timeout=timeout, context=_CTX) as r:
            d = r.read()
            if r.status == 200 and len(d) >= min_sz and not d.lstrip().startswith(b"<"):
                return d
    except Exception:
        pass
    return None

def _to_baseline_jpg(d, mode="square", px=500):
    if not d or len(d) < 250:
        return None
    tid = f"{_TH.get_ident()}_{_RND.randint(1000,9999)}"
    fin = f"/tmp/art_in_{tid}.img"
    fout = f"/tmp/art_out_{tid}.jpg"
    try:
        open(fin, "wb").write(d)
        if mode == "maxres":
            w_sc = int(px * 16 / 9) + 2
            vf = f"scale={w_sc}:{px}:force_original_aspect_ratio=increase,crop={px}:{px}"
        elif mode == "hq43":
            w_sc = int(px * 16 / 9) + 2
            h_sc = int(px * 4 / 3) + 2
            vf = f"scale={w_sc}:{h_sc},crop={px}:{px}"
        else:
            vf = f"scale={px}:{px}:force_original_aspect_ratio=increase,crop={px}:{px}"
        qv = "3" if px <= 260 else "2"
        p = _SP.run([
            "ffmpeg", "-y", "-loglevel", "error", "-i", fin,
            "-frames:v", "1", "-vf", vf,
            "-pix_fmt", "yuvj420p", "-strict", "-2",
            "- metadata:s:v", "comment=", "-q:v", qv, fout
        ], stdout=_SP.PIPE, stderr=_SP.PIPE, timeout=4)
        if p.returncode != 0 or not _OS.path.exists(fout):
            p = _SP.run([
                "ffmpeg", "-y", "-loglevel", "error", "-i", fin,
                "-frames:v", "1", "-vf", vf,
                "-pix_fmt", "yuvj420p", "-strict", "-2", "-q:v", qv, fout
            ], stdout=_SP.PIPE, stderr=_SP.PIPE, timeout=4)
        if p.returncode == 0 and _OS.path.exists(fout) and _OS.path.getsize(fout) > 250:
            out_b = open(fout, "rb").read()
            if out_b.startswith(b"\xff\xd8"):
                return out_b
    except Exception:
        pass
    finally:
        for fp in (fin, fout):
            try:
                if _OS.path.exists(fp): _OS.remove(fp)
            except Exception:
                pass
    # Chi tra ve d neu d THUC SU la file JPEG (bat dau bang ff d8), TUYET DOI khong tra ve WebP/PNG gia danh JPEG!
    return d if d.startswith(b"\xff\xd8") else None

def _canon_key(cid):
    s = str(cid or "").strip()
    is_hdd = bool(_RE.match(r"^hdd\d*_", s))
    clean_s = _RE.sub(r"^(hdd\d*_|yt\d*_)+", "", s)
    if clean_s.startswith("al_all_"):
        clean_s = clean_s[7:]
    if _RE.match(r"^ar_[0-9a-f]{10}$", clean_s) and not is_hdd:
        return ("ar", clean_s, clean_s)
    if clean_s.startswith("pl_") and len(clean_s) > 12 and not is_hdd:
        return ("pl", clean_s[3:], clean_s[3:])
    raw_id = clean_s[3:] if (clean_s.startswith("al_") and len(clean_s) == 14) else clean_s
    if not is_hdd and (s.startswith("yt") or _RE.match(r"^[A-Za-z0-9_-]{11}$", raw_id)):
        m = _RE.search(r"([A-Za-z0-9_-]{11})$", raw_id)
        vid = m.group(1) if m else raw_id
        return ("yt", vid, vid)
    # Voi HDD: Quy dong ca song_id (mf-...) va album_id (al-...) ve cung 1 khoa duy nhat!
    raw_hdd = _RE.sub(r"^(al-|mf-)", "", raw_id)
    alid = _ART_MAP.get(raw_id) or _ART_MAP.get(raw_hdd) or _ART_MAP.get("mf-" + raw_hdd) or raw_hdd
    alid_clean = _RE.sub(r"^(al-|mf-)", "", str(alid))
    return ("hdd", alid_clean, raw_id)

def _fetch_raw_source(kind, k_id, raw_id):
    if kind == "ar":
        if k_id not in _ARTIST_THUMB:
            aname = _resolve_artist_name(k_id)
            if aname:
                _yt_web_search(aname, "")
                if k_id not in _ARTIST_THUMB:
                    _ytm_search_artists_and_pls(aname)
        tu = _ARTIST_THUMB.get(k_id, "")
        if tu:
            d = _dl_fast(_fix_url(tu), 400, 2.5)
            if d: return (d, "square")
        return (None, "square")

    if kind == "pl":
        tu = _PL_THUMB.get(k_id, "")
        if tu:
            m_vi = _RE.search(r"/vi/([A-Za-z0-9_-]{11})/", tu)
            if m_vi:
                return _fetch_raw_source("yt", m_vi.group(1), m_vi.group(1))
            clean_tu = tu.split("?")[0] if "ytimg.com" in tu else _fix_url(tu)
            d = _dl_fast(clean_tu, 400, 2.5)
            if d: return (d, "hq43" if "hqdefault" in clean_tu else "square")
        tr_list = _PL_TRACKS_CACHE.get(k_id) or _yt_playlist_tracks(k_id)
        if tr_list and tr_list[0].get("id"):
            vid0 = tr_list[0]["id"]
            return _fetch_raw_source("yt", vid0, vid0)
        return (None, "square")

    if kind == "hdd" and _HDD:
        cands = []
        for x in [raw_id, "al-"+k_id, k_id, "mf-"+raw_id, "al-"+raw_id]:
            if x and x not in cands: cands.append(x)
        with _ART_SEM:
            for cand in cands:
                try:
                    cq = [x for x in _hdd_q([("id", cand), ("size", "500")]) if x[0] != "f"]
                    with _U.urlopen(_U.Request(f"{_HDD}/rest/getCoverArt.view?{_ue(cq)}", headers={"User-Agent":"Resonus"}), timeout=4, context=_CTX) as r:
                        d = r.read()
                        if r.status == 200 and len(d) > 400 and not d.lstrip().startswith(b"<"):
                            return (d, "square")
                except Exception:
                    pass
        qt = _HDD_META.get(raw_id) or _HDD_META.get(k_id)
        if qt:
            s_items = _ytm_search_items(qt, "EgWKAQIIAWoKEAkQBRAKEAMQBA==", False)
            if s_items:
                return _fetch_raw_source("yt", s_items[0]["id"], s_items[0]["id"])
        return (None, "square")

    # kind == "yt": Uu tien hqdefault.jpg (luon ton tai 100% va cuc nhanh) + googleusercontent
    vid = k_id
    if vid in _YTM_THUMB:
        tu = _RE.sub(r"=w\d+-h\d+[^/]*$", "=w500-h500-l95-rj", _YTM_THUMB[vid])
        d = _dl_fast(tu, 400, 1.8)
        if d:
            return (d, "square")
    f_hq  = _DL_POOL.submit(_dl_fast, f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg", 600, 1.5)
    f_max = _DL_POOL.submit(_dl_fast, f"https://i.ytimg.com/vi/{vid}/maxresdefault.jpg", 1500, 0.85)
    try:
        d_max = f_max.result(timeout=0.9)
        if d_max:
            return (d_max, "maxres")
    except Exception:
        pass
    try:
        d_hq = f_hq.result(timeout=1.6)
        if d_hq:
            return (d_hq, "hq43")
    except Exception:
        pass
    d_mq = _dl_fast(f"https://i.ytimg.com/vi/{vid}/mqdefault.jpg", 400, 1.4)
    if d_mq:
        return (d_mq, "maxres")
    return (None, "square")

def _ensure_both_res(cid):
    if not cid:
        return (None, None)
    kind, k_id, raw_id = _canon_key(cid)
    k500 = f"500_{kind}_{k_id}"
    k240 = f"240_{kind}_{k_id}"
    if k500 in _ART_RAM and k240 in _ART_RAM:
        return (_ART_RAM[k500], _ART_RAM[k240])

    with _get_lock(f"{kind}_{k_id}"):
        if k500 in _ART_RAM and k240 in _ART_RAM:
            return (_ART_RAM[k500], _ART_RAM[k240])
        h_base = _H.md5(f"{kind}_{k_id}".encode()).hexdigest()
        cf500 = _OS.path.join(_ART_DIR, h_base + "_500.jpg")
        cf240 = _OS.path.join(_ART_DIR, h_base + "_240.jpg")

        if _OS.path.exists(cf500) and _OS.path.exists(cf240):
            d5 = open(cf500, "rb").read()
            d2 = open(cf240, "rb").read()
            if d5.startswith(b"\xff\xd8") and d2.startswith(b"\xff\xd8"):
                _ART_RAM[k500] = d5
                _ART_RAM[k240] = d2
                return (d5, d2)

        raw_d, mode = _fetch_raw_source(kind, k_id, raw_id)
        if not raw_d:
            return (None, None)

        d5 = _to_baseline_jpg(raw_d, mode, 500)
        if not d5:
            return (None, None)
        d2 = _to_baseline_jpg(d5, "square", 240) or d5

        try:
            open(cf500, "wb").write(d5)
            open(cf240, "wb").write(d2)
        except Exception:
            pass
        _ART_RAM[k500] = d5
        _ART_RAM[k240] = d2
        return (d5, d2)

def _get_art_bytes(cid, want_car_240=False):
    d5, d2 = _ensure_both_res(cid)
    return d2 if want_car_240 else d5

def _prewarm_list(ids):
    for cid in ids[:50]:
        _DL_POOL.submit(_ensure_both_res, cid)

def _fmt(rt, js, cb=None):
    _cln(rt)
    warm = []
    for x in rt.iter():
        a = x.attrib
        sid = str(a.get("id") or "")
        ca = str(a.get("coverArt") or "")
        if x.tag in ("song", "child", "entry") and a.get("isDir") != "true":
            if sid.startswith("hdd"):
                alid = _RE.sub(r"^(hdd\d*_|hdd_|al-|mf-)+", "", str(a.get("albumId") or a.get("parent") or ""))
                rid_clean = _RE.sub(r"^(hdd\d*_|hdd_|al-|mf-)+", "", sid)
                if alid and rid_clean:
                    _ART_MAP[rid_clean] = alid
                ca_use = ("al-" + alid) if alid else rid_clean
                a["coverArt"] = "hdd26_" + ca_use
            else:
                vid_clean = _RE.sub(r"^(yt\d*_|al_)+", "", sid)
                a["coverArt"] = "yt26_" + vid_clean
            warm.append(a["coverArt"])
            if not sid.startswith("hdd"):
                a.setdefault("bitRate", "256")
                a.setdefault("suffix", "aac")
                a.setdefault("contentType", "audio/aac")
                a.setdefault("samplingRate", "44100")
                a.setdefault("bitDepth", "16")
                a.setdefault("channelCount", "2")
                a.setdefault("type", "music")
        elif ca:
            raw_ca = _RE.sub(r"^(hdd\d*_|yt\d*_)+", "", ca)
            if ca.startswith("hdd") or sid.startswith("hdd"):
                a["coverArt"] = "hdd26_" + raw_ca
            else:
                a["coverArt"] = "yt26_" + raw_ca
            warm.append(a["coverArt"])
    if warm:
        _ART_POOL.submit(_prewarm_list, list(dict.fromkeys(warm)))
    rt.attrib.update({"status":rt.attrib.get("status","ok"),"version":"1.16.1","type":"subsonic","serverVersion":"1.16.1"})
    if js:
        b = _J.dumps({"subsonic-response": _x2d(rt, 0)})
        return _R(content=f"{cb}({b});" if cb else b, media_type="application/json; charset=utf-8", headers=_CORS)
    rt.set("xmlns", "http://subsonic.org/restapi")
    return _R(content=b'<?xml version="1.0" encoding="UTF-8"?>\n' + _E.tostring(rt, encoding="utf-8"), media_type="application/xml; charset=utf-8", headers=_CORS)

@app.middleware("http")
async def _k11_mw(req, call_next):
    p = _RE.sub(r"/+", "/", req.url.path); pl = p.lower()
    if not pl.startswith("/rest/"):
        return await call_next(req)
    qi = list(_pq(req.scope.get("query_string", b"").decode("utf-8", "ignore"), keep_blank_values=True))
    qd = dict(qi)
    lp = _A.get_running_loop()

    if "/getcoverart" in pl:
        cid = str(qd.get("id") or "").strip()
        sz_s = str(qd.get("size") or "").strip()
        ua = req.headers.get("user-agent", "").lower()
        # Neu la trinh phat ngam cua xe Android (Dalvik/ExoPlayer/MediaSession) hoac size <= 350 -> Cap anh 240x240 Baseline JPEG (~12KB)
        is_car = ("dart" not in ua) or (sz_s.isdigit() and int(sz_s) <= 350)
        d = await lp.run_in_executor(_ART_POOL, _get_art_bytes, cid, is_car) if cid else None
        if d:
            return _R(content=d, media_type="image/jpeg", headers={**_CORS, "Content-Length": str(len(d)), "Cache-Control": "public, max-age=604800"})
        return _R(content=_PNG_1X1, media_type="image/png", headers={**_CORS, "Content-Length": str(len(_PNG_1X1))})

    if "/stream" in pl or "/download" in pl or "/scrobble" in pl:
        sid = str(qd.get("id") or "").strip()
        if sid:
            try:
                await _A.wait_for(lp.run_in_executor(_ART_POOL, _ensure_both_res, sid), timeout=1.5)
            except Exception:
                pass
    return await call_next(req)

_SCACHE.clear()
