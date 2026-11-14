import asyncio, xml.etree.ElementTree as ET
from urllib.parse import urlencode
from fastapi import Request
from fastapi.responses import Response, JSONResponse

INTS = {"duration","bitRate","size","track","year","discNumber","songCount","albumCount","playCount","userRating","averageRating","offset","totalTracks"}
BOOLS = {"isDir","isVideo"}
LISTS = {"song","album","artist","directory","child","entry","playlist","genre","index","musicFolder"}

def _cast(k, v):
    if k in BOOLS: return str(v).lower() in ("true","1","yes")
    if k in INTS:
        try: return int(float(v))
        except: return 0
    return v

def _enrich(s):
    sid = str(s.get("id", "yt"))
    s["id"] = sid
    s["title"] = str(s.get("title") or "Unknown Title")
    s["artist"] = str(s.get("artist") or "Unknown Artist")
    s["album"] = str(s.get("album") or s["artist"] or "YouTube Music")
    s["albumId"] = str(s.get("albumId") or f"al_{sid}")
    s["artistId"] = str(s.get("artistId") or f"ar_{sid}")
    s["isDir"], s["isVideo"], s["type"] = False, False, "music"
    s["suffix"] = str(s.get("suffix") or "m4a")
    s["contentType"] = str(s.get("contentType") or "audio/mp4")
    s["duration"] = int(s.get("duration") or 240)
    s["bitRate"] = int(s.get("bitRate") or 129)
    s["size"] = int(s.get("size") or s["duration"] * 16300)
    s.setdefault("created", "2026-01-01T00:00:00.000Z")
    return s

def _e2d(el):
    d = {k: _cast(k, v) for k, v in el.attrib.items()}
    ch = {}
    for c in el:
        t = c.tag.split("}")[-1]
        v = _e2d(c)
        if isinstance(v, dict) and (t == "song" or (t in ("child","entry") and not v.get("isDir"))):
            v = _enrich(v)
        ch.setdefault(t, []).append(v)
    for t, items in ch.items():
        d[t] = items if (t in LISTS or len(items) > 1) else items[0]
    if el.text and el.text.strip() and not d:
        return el.text.strip()
    return d

async def _call_internal(app, base_scope, path, params):
    scope = dict(base_scope)
    scope["method"], scope["path"], scope["raw_path"] = "GET", path, path.encode()
    scope["query_string"] = urlencode(params, doseq=True).encode()
    chunks = []
    async def rcv(): return {"type": "http.request", "body": b"", "more_body": False}
    async def snd(m):
        if m["type"] == "http.response.body": chunks.append(m.get("body", b""))
    await app(scope, rcv, snd)
    return b"".join(chunks).decode("utf-8", errors="ignore")

async def _auto_save_songs(app, base_scope, auth, sids, pl_name=None):
    for sid in sids:
        p = dict(auth); p["id"] = sid
        await _call_internal(app, base_scope, "/rest/star.view", p)
    if pl_name and sids:
        p = dict(auth); p["name"] = pl_name; p["songId"] = sids
        await _call_internal(app, base_scope, "/rest/createPlaylist.view", p)

def setup_patch(app):
    @app.middleware("http")
    async def symfonium_mw(req: Request, call_next):
        p = req.url.path.lower()
        if any(x in p for x in ("/stream","/download","/getcoverart","/avatar","/hls")):
            return await call_next(req)
        q = dict(req.query_params)
        if req.method == "POST":
            try:
                for k, v in (await req.form()).multi_items(): q.setdefault(k, v)
            except: pass
        auth = {k: q[k] for k in ("u","p","t","s","v","c") if k in q}

        if "createplaylist" in p:
            name = q.get("name", "")
            if name.lower().startswith(("tim:", "add:", "+")):
                kw = name.split(":", 1)[-1].lstrip("+").strip()
                if kw:
                    sp = dict(auth); sp["query"] = kw; sp["songCount"] = "25"
                    xml_s = await _call_internal(app, req.scope, "/rest/search3.view", sp)
                    try:
                        sids = [el.attrib["id"] for el in ET.fromstring(xml_s).iter() if el.tag.endswith("song") and "id" in el.attrib]
                        if sids:
                            await _auto_save_songs(app, req.scope, auth, sids, f"YT: {kw}")
                    except: pass

        resp = await call_next(req)
        ct = (resp.headers.get("content-type") or "").lower()
        if any(x in ct for x in ("audio/","video/","image/","octet-stream")):
            return resp
        body = b"".join([c async for c in resp.body_iterator])
        txt = body.decode("utf-8", errors="ignore").strip()
        if txt.startswith("<?xml") or txt.startswith("<subsonic-response"):
            try:
                root = ET.fromstring(txt)
                if "search" in p:
                    sids = [el.attrib["id"] for el in root.iter() if el.tag.endswith("song") and "id" in el.attrib]
                    if sids:
                        asyncio.create_task(_auto_save_songs(app, req.scope, auth, sids[:20]))
                sub = _e2d(root)
                if not isinstance(sub, dict): sub = {}
                sub.setdefault("status", "ok")
                sub.setdefault("version", "1.16.1")
                sub["type"], sub["openSubsonic"] = "mirasonic", True
                return JSONResponse({"subsonic-response": sub}, status_code=resp.status_code)
            except: pass
        hdrs = dict(resp.headers); hdrs.pop("content-length", None)
        return Response(content=body, status_code=resp.status_code, headers=hdrs, media_type=resp.media_type)
