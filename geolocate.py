#!/usr/bin/env python3
"""Search KartaView imagery around Valle Dorado and rank visual matches."""
from __future__ import annotations

import base64
import csv
import hashlib
import hmac
import urllib.parse
from datetime import datetime, timezone
import io
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
CACHE = ROOT / ".cache" / "images"
API_CACHE = ROOT / ".cache" / "api"
TARGET = DATA / "target.jpg"
CONFIG = ROOT / "config.json"
UA = "Mozilla/5.0 (compatible; ValleDoradoGeolocator/1.0; +https://github.com/Liuqiran/sample_website)"


@dataclass
class Candidate:
    photo_id: str
    sequence_id: str
    lat: float | None
    lng: float | None
    heading: float | None
    date: str
    image_url: str
    source_query: str
    score: float = -1.0
    orb_good: int = 0
    orb_ratio: float = 0.0
    inlier_ratio: float = 0.0
    phash_distance: int = 64
    hist_similarity: float = -1.0
    local_path: str = ""
    error: str = ""


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _douban_sign(url: str, ts: str) -> str:
    path = urllib.parse.urlparse(url).path
    raw = "&".join(["GET", urllib.parse.quote(path, safe=""), ts])
    return base64.b64encode(hmac.new(b"bf7dddc7c9cfe6f7", raw.encode(), hashlib.sha1).digest()).decode()


def _collect_image_options(obj: Any, context: str = "") -> list[tuple[str, int, int, str]]:
    out: list[tuple[str, int, int, str]] = []
    if isinstance(obj, dict):
        width = int(obj.get("width") or obj.get("w") or 0) if str(obj.get("width") or obj.get("w") or "0").isdigit() else 0
        height = int(obj.get("height") or obj.get("h") or 0) if str(obj.get("height") or obj.get("h") or "0").isdigit() else 0
        for k, v in obj.items():
            if isinstance(v, str) and v.startswith("http") and ("doubanio.com" in v or "douban.com" in v):
                if any(x in (k + context).lower() for x in ("image", "photo", "url", "large", "origin", "raw")):
                    out.append((v, width, height, f"{context}/{k}"))
            else:
                out.extend(_collect_image_options(v, f"{context}/{k}"))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.extend(_collect_image_options(v, f"{context}[{i}]"))
    return out


def fetch_target_from_douban(topic_id: str) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({
        "User-Agent": "api-client/1 com.douban.frodo/7.22.0.beta9(231) Android/23 product/Mate40 platform/mobile",
        "Accept": "application/json",
    })
    url = f"https://frodo.douban.com/api/v2/group/topic/{topic_id}"
    ts = datetime.now(timezone.utc).strftime("%Y%m%d")
    params = {"apiKey": "0dad551ec0f84ed02907ff5c42e8ec70", "_ts": ts, "_sig": _douban_sign(url, ts)}
    r = session.get(url, params=params, timeout=45)
    r.raise_for_status()
    payload = r.json()
    (DATA / "douban_topic.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    options = _collect_image_options(payload)
    target_ratio = 1266 / 1536
    scored = []
    for image_url, w, h, ctx in options:
        area = w * h
        ratio_bonus = 0.0
        if w and h:
            ratio_bonus = max(0.0, 1.0 - abs((w / h) - target_ratio)) * 10_000_000
        label_bonus = 5_000_000 if any(x in ctx.lower() for x in ("large", "origin", "raw")) else 0
        scored.append((area + ratio_bonus + label_bonus, image_url, ctx))
    errors = []
    for _, image_url, ctx in sorted(scored, reverse=True):
        try:
            ir = session.get(image_url, timeout=45)
            ir.raise_for_status()
            im = Image.open(io.BytesIO(ir.content))
            if im.width < 700 or im.height < 700:
                continue
            ratio = im.width / im.height
            if abs(ratio - target_ratio) > 0.15:
                continue
            TARGET.write_bytes(ir.content)
            print(f"Downloaded target from Douban: {image_url} ({im.width}x{im.height}, {ctx})")
            return
        except Exception as exc:
            errors.append(f"{image_url}: {exc}")
    raise RuntimeError("Could not retrieve target photo from Douban topic. " + "; ".join(errors[:3]))


def reconstruct_target(config: dict[str, Any]) -> None:
    if TARGET.exists() and TARGET.stat().st_size > 10000:
        return
    parts = sorted(DATA.glob("target.jpg.b64.*"))
    if parts:
        encoded = "".join(p.read_text(encoding="ascii").strip() for p in parts)
        TARGET.write_bytes(base64.b64decode(encoded))
        return
    fetch_target_from_douban(str(config.get("douban_topic_id", "312383214")))


def grid_points(bbox: dict[str, float], rows: int, cols: int) -> list[tuple[float, float]]:
    lats = np.linspace(bbox["south"], bbox["north"], rows)
    lngs = np.linspace(bbox["west"], bbox["east"], cols)
    return [(float(lat), float(lng)) for lat in lats for lng in lngs]


def request_json(session: requests.Session, url: str, params: dict[str, Any], cache_key: str) -> dict[str, Any]:
    API_CACHE.mkdir(parents=True, exist_ok=True)
    path = API_CACHE / f"{cache_key}.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            r = session.get(url, params=params, timeout=45)
            r.raise_for_status()
            data = r.json()
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            return data
        except Exception as exc:
            last_error = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"API request failed: {last_error}")


def walk_dicts(obj: Any) -> Iterable[dict[str, Any]]:
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from walk_dicts(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from walk_dicts(value)


def first(d: dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return None


def normalize_url(url: str | None) -> str:
    if not url:
        return ""
    url = str(url).replace("[[sizeprefix]]", "lth")
    if url.startswith("//"):
        return "https:" + url
    if url.startswith("http://"):
        return "https://" + url[7:]
    if not url.startswith("http"):
        return "https://" + url.lstrip("/")
    return url


def extract_candidates(payload: dict[str, Any], source_query: str) -> list[Candidate]:
    out: list[Candidate] = []
    seen: set[str] = set()
    url_keys = (
        "fileurlLTh", "fileurlProc", "fileurl", "lth_name", "lthName",
        "procUrl", "imageUrl", "photoUrl", "name", "th_name", "thName"
    )
    for d in walk_dicts(payload):
        url = normalize_url(first(d, *url_keys))
        pid = first(d, "id", "photoId", "photo_id")
        if not url or pid is None:
            continue
        if "photo" not in url.lower() and not any(k in d for k in ("sequenceIndex", "sequence_index", "heading")):
            continue
        sid = first(d, "sequenceId", "sequence_id", "sequence")
        key = str(pid)
        if key in seen:
            continue
        seen.add(key)
        try:
            lat = float(first(d, "lat", "latitude", "matchLat", "match_lat"))
        except Exception:
            lat = None
        try:
            lng = float(first(d, "lng", "lon", "longitude", "matchLng", "match_lng"))
        except Exception:
            lng = None
        try:
            heading = float(first(d, "heading", "gpsHeading", "direction"))
        except Exception:
            heading = None
        out.append(Candidate(
            photo_id=str(pid), sequence_id=str(sid or ""), lat=lat, lng=lng,
            heading=heading, date=str(first(d, "dateAdded", "date_added", "date", "createdAt") or ""),
            image_url=url, source_query=source_query,
        ))
    return out


def collect_candidates(config: dict[str, Any]) -> tuple[list[Candidate], list[str]]:
    session = requests.Session()
    session.headers.update({"User-Agent": UA, "Accept": "application/json"})
    api_url = "https://api.openstreetcam.org/2.0/photo/"
    errors: list[str] = []
    all_candidates: dict[str, Candidate] = {}
    points = grid_points(config["bbox"], config.get("grid_rows", 5), config.get("grid_cols", 5))
    for idx, (lat, lng) in enumerate(points, start=1):
        params = {
            "lat": f"{lat:.7f}", "lng": f"{lng:.7f}",
            "zoomLevel": config.get("zoom_level", 16),
            "join": "sequence", "orderBy": "id", "orderDirection": "desc",
        }
        source = f"{lat:.7f},{lng:.7f}"
        try:
            payload = request_json(session, api_url, params, f"grid_{idx:02d}")
            found = extract_candidates(payload, source)
            for c in found:
                all_candidates.setdefault(c.photo_id, c)
            print(f"API {idx}/{len(points)} {source}: {len(found)} parsed, {len(all_candidates)} unique")
        except Exception as exc:
            msg = f"{source}: {exc}"
            errors.append(msg)
            print("WARNING", msg, file=sys.stderr)
    return list(all_candidates.values()), errors


def image_phash(gray: np.ndarray) -> int:
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(resized)
    block = dct[:8, :8]
    med = np.median(block[1:, :])
    bits = (block > med).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming64(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def color_hist(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(cv2.resize(bgr, (256, 256)), cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    return cv2.normalize(hist, hist).flatten()


def target_features(path: Path):
    target = cv2.imread(str(path))
    if target is None:
        raise RuntimeError(f"Could not load target {path}")
    max_dim = 1400
    scale = min(1.0, max_dim / max(target.shape[:2]))
    if scale < 1:
        target = cv2.resize(target, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    orb = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.02, edgeThreshold=12)
    kp, des = orb.detectAndCompute(gray, None)
    return target, gray, kp, des, image_phash(gray), color_hist(target)


def download_one(session: requests.Session, c: Candidate) -> Candidate:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{c.photo_id}.jpg"
    c.local_path = str(path.relative_to(ROOT))
    if path.exists() and path.stat().st_size > 3000:
        return c
    try:
        r = session.get(c.image_url, timeout=45)
        r.raise_for_status()
        content = r.content
        if len(content) < 3000:
            raise ValueError(f"small response ({len(content)} bytes)")
        path.write_bytes(content)
    except Exception as exc:
        c.error = f"download: {exc}"
    return c


def score_one(c: Candidate, target_data) -> Candidate:
    if c.error:
        return c
    target, target_gray, target_kp, target_des, target_hash, target_hist = target_data
    img = cv2.imread(str(ROOT / c.local_path))
    if img is None:
        c.error = "decode failed"
        return c
    max_dim = 1400
    scale = min(1.0, max_dim / max(img.shape[:2]))
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    c.phash_distance = hamming64(target_hash, image_phash(gray))
    c.hist_similarity = float(cv2.compareHist(target_hist, color_hist(img), cv2.HISTCMP_CORREL))

    sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.02, edgeThreshold=12)
    kp, des = sift.detectAndCompute(gray, None)
    good = []
    inlier_ratio = 0.0
    if des is not None and target_des is not None and len(des) >= 2 and len(target_des) >= 2:
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        pairs = matcher.knnMatch(target_des, des, k=2)
        good = [m for m, n in pairs if m.distance < 0.72 * n.distance]
        if len(good) >= 8:
            src = np.float32([target_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 6.0)
            if mask is not None:
                inlier_ratio = float(mask.ravel().sum() / len(good))
    c.orb_good = len(good)
    c.orb_ratio = len(good) / max(1, min(len(target_kp), len(kp)))
    c.inlier_ratio = inlier_ratio

    phash_score = max(0.0, 1.0 - c.phash_distance / 64.0)
    hist_score = max(0.0, min(1.0, (c.hist_similarity + 1.0) / 2.0))
    feat_score = min(1.0, c.orb_good / 80.0)
    geom_score = min(1.0, inlier_ratio * 2.0)
    c.score = 0.45 * feat_score + 0.30 * geom_score + 0.15 * phash_score + 0.10 * hist_score
    return c


def make_contact_sheet(cands: list[Candidate], out: Path, limit: int = 24) -> None:
    cards = []
    for rank, c in enumerate(cands[:limit], start=1):
        try:
            im = Image.open(ROOT / c.local_path).convert("RGB")
            im.thumbnail((360, 240))
            card = Image.new("RGB", (380, 300), "white")
            card.paste(im, ((380 - im.width) // 2, 5))
            draw = ImageDraw.Draw(card)
            text = f"#{rank} score={c.score:.3f} id={c.photo_id}\n{c.lat},{c.lng}  SIFT={c.orb_good} inlier={c.inlier_ratio:.2f}"
            draw.multiline_text((8, 250), text, fill="black", spacing=2)
            cards.append(card)
        except Exception:
            continue
    if not cards:
        return
    cols = 3
    rows = math.ceil(len(cards) / cols)
    sheet = Image.new("RGB", (cols * 380, rows * 300), "white")
    for i, card in enumerate(cards):
        sheet.paste(card, ((i % cols) * 380, (i // cols) * 300))
    sheet.save(out, quality=88)


def write_results(cands: list[Candidate], api_errors: list[str], config: dict[str, Any]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    ranked = sorted((c for c in cands if c.score >= 0), key=lambda x: x.score, reverse=True)
    fields = list(asdict(Candidate("", "", None, None, None, "", "", "")).keys())
    with (RESULTS / "results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for c in ranked:
            w.writerow(asdict(c))
    make_contact_sheet(ranked, RESULTS / "top_matches.jpg")

    lines = [
        "# KartaView visual matching report", "",
        f"- Search bbox: `{json.dumps(config['bbox'])}`",
        f"- Parsed candidate photos: **{len(cands)}**",
        f"- Successfully scored: **{len(ranked)}**",
        f"- API errors: **{len(api_errors)}**", "",
        "## Top matches", "",
        "| Rank | Score | Photo ID | Sequence | Coordinates | SIFT good | Inlier ratio | pHash d | Image |",
        "|---:|---:|---:|---:|---|---:|---:|---:|---|",
    ]
    for i, c in enumerate(ranked[:30], start=1):
        coord = f"{c.lat:.7f}, {c.lng:.7f}" if c.lat is not None and c.lng is not None else "unknown"
        lines.append(
            f"| {i} | {c.score:.4f} | {c.photo_id} | {c.sequence_id} | {coord} | "
            f"{c.orb_good} | {c.inlier_ratio:.3f} | {c.phash_distance} | [open]({c.image_url}) |"
        )
    lines += ["", "## Interpretation", "",
              "A decisive same-scene result should normally show many SIFT matches plus a strong geometric inlier ratio. "
              "Weak scores only mean the open KartaView coverage did not contain an obvious match; they do not prove the neighborhood is wrong."]
    if api_errors:
        lines += ["", "## API errors", ""] + [f"- `{e}`" for e in api_errors]
    (RESULTS / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    config = load_config()
    reconstruct_target(config)
    candidates, api_errors = collect_candidates(config)
    if not candidates:
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / "report.md").write_text(
            "# No candidates parsed\n\nThe KartaView API returned no parseable photos for this bounding box.\n" +
            "\n".join(f"- {e}" for e in api_errors), encoding="utf-8")
        return 2

    session = requests.Session()
    session.headers.update({"User-Agent": UA})
    max_downloads = int(config.get("max_downloads", 2500))
    candidates = candidates[:max_downloads]
    with ThreadPoolExecutor(max_workers=int(config.get("download_workers", 12))) as pool:
        futures = [pool.submit(download_one, session, c) for c in candidates]
        downloaded = [f.result() for f in as_completed(futures)]
    target_data = target_features(TARGET)
    scored: list[Candidate] = []
    with ThreadPoolExecutor(max_workers=int(config.get("score_workers", 4))) as pool:
        futures = [pool.submit(score_one, c, target_data) for c in downloaded]
        for i, f in enumerate(as_completed(futures), start=1):
            scored.append(f.result())
            if i % 100 == 0:
                print(f"Scored {i}/{len(downloaded)}")
    write_results(scored, api_errors, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
