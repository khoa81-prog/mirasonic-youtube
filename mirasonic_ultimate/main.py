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
_HDD, _PASS = "https://music.nhacuakhoa.online", "Haidang06"
_CTX, _POOL, _TLS = _S._create_unverified_context(), _TP(max_workers=12), _TH.local()
_UC, _JOBS, _JL, _YL, _SCACHE, _H_IDX = {}, {}, _TH.Lock(), {}, {}, []
_S_SEQ, _LAST_Q = 0, ""
_IDS = ("id","parent","albumId","artistId","coverArt","playlistId")
_INTS = {"duration","size","bitRate","track","discNumber","year","songCount","albumCount","playCount","code","offset","totalHits"}
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
def _cln(el):
    for x in el.iter():
        if "}" in x.tag: x.tag = x.tag.split("}", 1)[1]
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
    r = {}
    for k, v in el.attrib.items():
        if k not in {"title","name","album","artist","id","parent","comment","path"} and v in ("true","false"): r[k] = (v == "true")
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
    if el.tag.endswith(("searchResult3","searchResult2")):
        r.setdefault("artist", []); r.setdefault("album", []); r.setdefault("song", [])
    elif el.tag.endswith(("albumList2","albumList")): r.setdefault("album", [])
    return r

def _pfx(el):
    _cln(el)
    for x in el.iter():
        a = x.attrib
        for k in _IDS:
            if k in a and a[k] and not str(a[k]).startswith("hdd_"): a[k] = "hdd_" + str(a[k])
        if x.tag.endswith(("song","child","entry","album")):
            for f in ("album","name","title"):
                if f in a and a[f] and (f == "album" or x.tag.endswith("album") or a.get("isDir") == "true") and not a[f].startswith("[HDD]"):
                    a[f] = "[HDD] " + a[f]
    return el

def _hdd_q(qi):
    q = [(k, v[4:] if isinstance(v, str) and v.startswith("hdd_") else v) for k, v in qi if k not in ("f","u","p","t","s","v","c","musicFolderId")]
    q.extend([("u","admin"), ("p",_OS.environ.get("MIRASONIC_PASSWORD",_PASS)), ("v","1.16.1"), ("c","Resonus"), ("f","xml")])
    return q

def _hdd_xml(path, qi):
    try:
        with _U.urlopen(_U.Request(f"{_HDD}{path}?{_ue(_hdd_q(qi))}", headers={"User-Agent":"Resonus"}), timeout=8, context=_CTX) as r:
            rt = _cln(_E.fromstring(r.read()))
            if rt.attrib.get("status") == "ok": return _pfx(rt)
    except Exception: pass
    return None

def _init_hdd():
    rt = _hdd_xml("/rest/search3.view", [("query",""),("artistCount","500"),("albumCount","1000"),("songCount","3000")])
    if rt is not None and len(rt) > 0:
        for x in rt[0]:
            a = x.attrib
            txt = _no_acc(f"{a.get('title','')} {a.get('name','')} {a.get('artist','')} {a.get('album','')}")
            _H_IDX.append((x.tag, dict(a), txt))

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

def _fmt(rt, js, cb=None):
    _cln(rt)
    rt.attrib.update({"status":rt.attrib.get("status","ok"),"version":"1.16.1","type":"navidrome","serverVersion":"0.52.5","openSubsonic":"true"})
    if js:
        b = _J.dumps({"subsonic-response": _x2d(rt, 0)})
        return _R(content=f"{cb}({b});" if cb else b, media_type="application/json; charset=utf-8", headers=_CORS)
    rt.set("xmlns", "http://subsonic.org/restapi")
    return _R(content=b'<?xml version="1.0" encoding="UTF-8"?>\n' + _E.tostring(rt, encoding="utf-8"), media_type="application/xml; charset=utf-8", headers=_CORS)

_POOL.submit(_init_hdd)
_POOL.submit(_yt_search, "Thùy Dung Lệ Quyên Bolero")

# --- K4 ---
@app.middleware("http")
async def _umw(req, call_next):
    if req.method == "OPTIONS": return _R(status_code=200, headers=_CORS)
    p = _RE.sub(r"/+", "/", req.url.path); pl = p.lower()
    if not pl.startswith("/rest/"): return await call_next(req)
    qi = list(_pq(req.scope.get("query_string", b"").decode("utf-8","ignore"), keep_blank_values=True))
    if req.method == "POST":
        bf = await req.body()
        if bf:
            try: qi.extend([(str(k),str(v)) for k,v in _J.loads(bf).items()] if bf.strip().startswith(b"{") else _pq(bf.decode("utf-8","ignore"), keep_blank_values=True))
            except Exception: pass
    qd = dict(qi); fv = qd.get("f","").lower()
    js = fv in ("json","jsonp") or "application/json" in req.headers.get("accept","").lower()
    cb = qd.get("callback") if fv == "jsonp" else None
    ish = any(isinstance(v, str) and v.startswith("hdd_") for k, v in qi if k in ("id","albumId","artistId","playlistId","songId"))
    lp = _A.get_running_loop()

    if "ping" in pl: return _fmt(_E.Element("subsonic-response", status="ok"), js, cb)
    if "getlicense" in pl:
        rt = _E.Element("subsonic-response", status="ok"); _E.SubElement(rt, "license", valid="true", email="a@b.c"); return _fmt(rt, js, cb)

    if any(x in pl for x in ("/stream","/download","/getcoverart","/avatar","/hls")):
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
        if "/getcoverart" in pl and m:
            for rs in ("maxresdefault.jpg","sddefault.jpg","hqdefault.jpg"):
                try:
                    with _U.urlopen(_U.Request(f"https://i.ytimg.com/vi/{m.group(1)}/{rs}", headers={"User-Agent":"Mozilla/5.0"}), timeout=5, context=_CTX) as r:
                        if r.status == 200: return _R(content=r.read(), media_type="image/jpeg", headers=_CORS)
                except Exception: pass
        if ("/stream" in pl or "/download" in pl) and m:
            j = _ens(m.group(1))
            for _ in range(80):
                if (_OS.path.exists(j["p"]) and _OS.path.getsize(j["p"]) >= 16384) or j["d"]: break
                await _A.sleep(0.1)
            if _OS.path.exists(j["p"]) and _OS.path.getsize(j["p"]) > 0:
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

    rid = qd.get("id", ""); vid = rid[3:] if rid.startswith("al_") else rid
    if vid in _YL and any(x in pl for x in ("/getalbum","/getsong","/getmusicdirectory")):
        s = _YL[vid]; rt = _E.Element("subsonic-response", status="ok")
        if "song" in pl: _E.SubElement(rt, "song", **s)
        else:
            tg = "directory" if "directory" in pl else "album"
            al = _E.SubElement(rt, tg, id="al_"+vid, parent=s["artistId"], isDir="true", name=s["album"], title=s["album"], album=s["album"], artist=s["artist"], artistId=s["artistId"], coverArt=vid, songCount="1", duration=s["duration"], playCount="1", created=s["created"], year="2026", genre="YouTube")
            _E.SubElement(al, "child" if tg=="directory" else "song", **s)
        return _fmt(rt, js, cb)
    if rid.startswith("ar_") and "/getartist" in pl and "getartists" not in pl:
        als = [s for s in _YL.values() if s.get("artistId") == rid]
        rt = _E.Element("subsonic-response", status="ok")
        ar = _E.SubElement(rt, "artist", id=rid, name=(als[0]["artist"] if als else "YouTube"), albumCount=str(len(als)))
        for s in als: _E.SubElement(ar, "album", id="al_"+s["id"], parent=rid, isDir="true", name=s["album"], title=s["album"], album=s["album"], artist=s["artist"], artistId=rid, coverArt=s["id"], songCount="1", duration=s["duration"], created=s["created"], year="2026")
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
                    await _A.sleep(0.55)
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
    if "getalbumlist" in pl or "getrandomsongs" in pl or "gettopsongs" in pl:
        if not _YL: await lp.run_in_executor(_POOL, _yt_search, "Thùy Dung Lệ Quyên Bolero")
        mrt = _E.Element("subsonic-response", status="ok")
        if "albumlist" in pl:
            mc = _E.SubElement(mrt, "albumList2" if "2" in pl else "albumList")
            for s in list(_YL.values())[:15]:
                _E.SubElement(mc, "album", id="al_"+s["id"], parent=s["artistId"], isDir="true", name=s["album"], title=s["album"], album=s["album"], artist=s["artist"], artistId=s["artistId"], coverArt=s["id"], songCount="1", duration=s["duration"], playCount="1", created=s["created"], year="2026", genre="YouTube")
        else:
            mc = _E.SubElement(mrt, "topSongs" if "top" in pl else "randomSongs")
            for s in list(_YL.values())[:15]: _E.SubElement(mc, "song", **s)
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

# --- K6 SAFE: MULTI-USER MANAGER & VOICE UI ---
from fastapi.responses import HTMLResponse as _HR

_USR_F = "/media/500GB/Mirasonic/users.json"
try: _USERS = _J.load(open(_USR_F))
except Exception: _USERS = {"banbe": "123456"}

def _save_u():
    try: _J.dump(_USERS, open(_USR_F, "w"))
    except Exception: pass
_save_u()

def _check_auth(qd):
    u = str(qd.get("u") or "").strip()
    if not u: return True
    all_u = {**_USERS, "admin": _OS.environ.get("MIRASONIC_PASSWORD", _PASS)}
    if u not in all_u: return False
    exp = str(all_u[u])
    t, s = str(qd.get("t") or ""), str(qd.get("s") or "")
    if t and s:
        return _H.md5((exp + s).encode()).hexdigest().lower() == t.lower()
    rp = str(qd.get("p") or "")
    if rp.startswith("enc:"):
        try: rp = bytes.fromhex(rp[4:]).decode("utf-8", "ignore")
        except Exception: pass
    return (not rp) or (rp == exp)

@app.middleware("http")
async def _auth_gate(req, call_next):
    pl = req.url.path.lower()
    if not pl.startswith("/rest/") or "/getcoverart" in pl or "/ping" in pl or req.method == "OPTIONS":
        return await call_next(req)
    qi = list(_pq(req.scope.get("query_string", b"").decode("utf-8","ignore"), keep_blank_values=True))
    qd = dict(qi)
    if not _check_auth(qd):
        js = qd.get("f","").lower() in ("json","jsonp")
        rt = _E.Element("subsonic-response", status="failed")
        _E.SubElement(rt, "error", code="40", message="Sai tai khoan hoac mat khau!")
        return _fmt(rt, js, qd.get("callback"))
    mp = _OS.environ.get("MIRASONIC_PASSWORD", _PASS)
    nqi = [(k, v) for k, v in qi if k not in ("u","p","t","s")] + [("u","admin"),("p",mp)]
    req.scope["query_string"] = _ue(nqi).encode()
    return await call_next(req)

@app.get("/users", response_class=_HR)
async def _users_ui(admin_pass: str = "", act: str = "", u: str = "", p: str = ""):
    mp = _OS.environ.get("MIRASONIC_PASSWORD", _PASS)
    msg = ""
    if act and admin_pass != mp: msg = "<p style='color:#ff5555'>❌ Sai mật khẩu Admin!</p>"
    elif act == "add" and u and p:
        _USERS[u.strip()] = p.strip(); _save_u(); msg = f"<p style='color:#1db954'>✅ Đã lưu tài khoản: <b>{u}</b></p>"
    elif act == "del" and u in _USERS:
        del _USERS[u]; _save_u(); msg = f"<p style='color:#ffb86c'>🗑️ Đã xóa: <b>{u}</b></p>"
    rows = "".join(f"<tr><td><b>{k}</b></td><td><code>{v}</code></td><td><a href='/users?act=del&u={k}&admin_pass={admin_pass}' style='color:#ff5555'>Xóa</a></td></tr>" for k, v in _USERS.items())
    return f"""<!DOCTYPE html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Quản Lý Tài Khoản</title>
    <style>body{{background:#121212;color:#fff;font-family:sans-serif;padding:20px;max-width:480px;margin:auto}}input,button{{width:100%;padding:10px;margin:6px 0;border-radius:6px;border:1px solid #333;background:#222;color:#fff;box-sizing:border-box}}button{{background:#1db954;border:none;font-weight:bold;cursor:pointer}}table{{width:100%;margin-top:15px;border-collapse:collapse}}td,th{{padding:10px;border-bottom:1px solid #333;text-align:left}}</style></head>
    <body><h2>👥 Quản Lý Tài Khoản Bạn Bè</h2>{msg}
    <form method='get' action='/users'><input type='hidden' name='act' value='add'>
    <input name='admin_pass' type='password' placeholder='Mật khẩu Admin gốc (bắt buộc)' value='{admin_pass}' required>
    <input name='u' placeholder='Tên tài khoản bạn bè (VD: linh)' required>
    <input name='p' placeholder='Mật khẩu cho bạn bè (VD: 123456)' required>
    <button type='submit'>➕ Thêm / Đổi Mật Khẩu</button></form>
    <table><tr><th>Tài khoản</th><th>Mật khẩu</th><th>Thao tác</th></tr>{rows}</table></body></html>"""
