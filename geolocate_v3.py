#!/usr/bin/env python3
"""Exact-scene search over all KartaView frames in a configured bounding box."""
from __future__ import annotations

import csv
import io
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "v3"
CACHE = ROOT / ".cache" / "v3"
CFG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
BBOX = CFG["bbox"]
TARGET_URL = "https://img9.doubanio.com/view/group_topic/l/public/p735803186.webp"
UA = "Mozilla/5.0 (compatible; PineappleKnotGeolocator/3.0)"


@dataclass
class Photo:
    id: str
    sequence_id: str
    lat: float | None
    lng: float | None
    heading: float | None
    url: str
    date: str = ""
    score: float = -1.0
    good: int = 0
    inliers: int = 0
    inlier_ratio: float = 0.0
    coverage: float = 0.0
    crop: str = ""
    path: str = ""
    error: str = ""


def walk(x: Any) -> Iterable[dict[str, Any]]:
    if isinstance(x, dict):
        yield x
        for v in x.values():
            yield from walk(v)
    elif isinstance(x, list):
        for v in x:
            yield from walk(v)


def first(d: dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return None


def fnum(x: Any) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


def urlfix(u: Any) -> str:
    if not u:
        return ""
    u = str(u).replace("[[sizeprefix]]", "proc")
    if u.startswith("//"):
        u = "https:" + u
    if u.startswith("http://"):
        u = "https://" + u[7:]
    if not u.startswith("http"):
        u = "https://" + u.lstrip("/")
    return u


def parse(payload: Any, default_seq: str = "") -> list[Photo]:
    photos: dict[str, Photo] = {}
    url_keys = (
        "fileurlProc", "fileurlLTh", "fileurl", "fileUrl", "filepath",
        "procUrl", "lth_name", "lthName", "imageUrl", "photoUrl", "name",
    )
    for d in walk(payload):
        pid = first(d, "id", "photoId", "photo_id")
        u = urlfix(first(d, *url_keys))
        if pid is None or not u:
            continue
        # Avoid mistakenly treating sequence-level URLs as photos.
        if not any(x in u.lower() for x in ("photo", ".jpg", ".jpeg", ".webp")):
            continue
        seq = first(d, "sequenceId", "sequence_id", "sequence")
        if isinstance(seq, dict):
            seq = first(seq, "id", "sequenceId", "sequence_id")
        p = Photo(
            id=str(pid),
            sequence_id=str(seq or default_seq),
            lat=fnum(first(d, "lat", "latitude", "matchLat", "match_lat")),
            lng=fnum(first(d, "lng", "lon", "longitude", "matchLng", "match_lng")),
            heading=fnum(first(d, "heading", "gpsHeading", "direction")),
            url=u,
            date=str(first(d, "dateAdded", "date", "createdAt") or ""),
        )
        photos[p.id] = p
    return list(photos.values())


def get_json(session: requests.Session, url: str, params: dict[str, Any] | None = None) -> Any:
    err: Exception | None = None
    for n in range(4):
        try:
            r = session.get(url, params=params, timeout=90)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            err = exc
            time.sleep(2 ** n)
    raise RuntimeError(f"GET failed {url}: {err}")


def in_bbox(p: Photo, margin: float = 0.0015) -> bool:
    return (
        p.lat is not None and p.lng is not None
        and BBOX["south"] - margin <= p.lat <= BBOX["north"] + margin
        and BBOX["west"] - margin <= p.lng <= BBOX["east"] + margin
    )


def collect() -> tuple[list[Photo], dict[str, Any]]:
    session = requests.Session()
    session.headers.update({"User-Agent": UA, "Accept": "application/json"})
    nearby = "https://api.openstreetcam.org/2.0/photo/"
    lats = np.linspace(BBOX["south"], BBOX["north"], 9)
    lngs = np.linspace(BBOX["west"], BBOX["east"], 9)
    seed_photos: dict[str, Photo] = {}
    seqs: set[str] = set()
    errors: list[str] = []

    for index, (lat, lng) in enumerate((a, b) for a in lats for b in lngs, 1):
        try:
            payload = get_json(session, nearby, {
                "lat": f"{lat:.7f}", "lng": f"{lng:.7f}", "zoomLevel": 18,
                "join": "sequence", "orderBy": "id", "orderDirection": "desc",
                "radius": 500,
            })
            batch = parse(payload)
            for p in batch:
                seed_photos[p.id] = p
                if p.sequence_id:
                    seqs.add(p.sequence_id)
            if index % 10 == 0:
                print(f"nearby {index}/81: seeds={len(seed_photos)} sequences={len(seqs)}", flush=True)
        except Exception as exc:
            errors.append(f"nearby {lat},{lng}: {exc}")

    print(f"Seed summary: {len(seed_photos)} photos, {len(seqs)} sequences", flush=True)
    all_photos: dict[str, Photo] = {p.id: p for p in seed_photos.values() if in_bbox(p)}
    sequence_counts: dict[str, int] = {}

    for i, seq in enumerate(sorted(seqs), 1):
        batch: list[Photo] = []
        # Preferred official sequence endpoint.
        try:
            payload = get_json(session, f"https://api.openstreetcam.org/2.0/sequence/{seq}/photos")
            batch = parse(payload, seq)
        except Exception as exc:
            errors.append(f"sequence endpoint {seq}: {exc}")

        # Fallback to official paginated photo endpoint (max 150/page).
        if not batch:
            for page in range(1, 101):
                try:
                    payload = get_json(session, nearby, {
                        "sequenceId": seq, "page": page, "itemsPerPage": 150,
                    })
                    page_rows = parse(payload, seq)
                except Exception as exc:
                    errors.append(f"photo pages {seq}/{page}: {exc}")
                    break
                if not page_rows:
                    break
                batch.extend(page_rows)
                if len(page_rows) < 150:
                    break

        unique = {p.id: p for p in batch}
        inside = [p for p in unique.values() if in_bbox(p)]
        sequence_counts[seq] = len(inside)
        for p in inside:
            all_photos[p.id] = p
        print(
            f"sequence {i}/{len(seqs)} id={seq}: total={len(unique)} inside={len(inside)} cumulative={len(all_photos)}",
            flush=True,
        )

    diagnostics = {
        "bbox": BBOX,
        "seed_count": len(seed_photos),
        "sequence_ids": sorted(seqs),
        "sequence_counts_inside": sequence_counts,
        "candidate_count": len(all_photos),
        "errors": errors,
    }
    return list(all_photos.values()), diagnostics


def load_target() -> np.ndarray:
    OUT.mkdir(parents=True, exist_ok=True)
    r = requests.get(TARGET_URL, headers={"User-Agent": UA}, timeout=90)
    r.raise_for_status()
    raw = r.content
    (OUT / "target.webp").write_bytes(raw)
    arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise RuntimeError("Target image decode failed")
    print(f"Target: {arr.shape[1]}x{arr.shape[0]} bytes={len(raw)}", flush=True)
    return arr


def target_crops(img: np.ndarray) -> dict[str, np.ndarray]:
    h, w = img.shape[:2]
    return {
        "full": img,
        "left_houses": img[: int(h * 0.74), : int(w * 0.60)],
        "street_depth": img[int(h * 0.14): int(h * 0.82), int(w * 0.22): int(w * 0.82)],
        "right_side": img[: int(h * 0.80), int(w * 0.42):],
        "upper": img[: int(h * 0.62), :],
    }


def features(img: np.ndarray):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=7000, contrastThreshold=0.012, edgeThreshold=14)
    return sift.detectAndCompute(gray, None)


def prepare_target(img: np.ndarray):
    return {name: (crop, *features(crop)) for name, crop in target_crops(img).items()}


def download_one(p: Photo) -> Photo:
    CACHE.mkdir(parents=True, exist_ok=True)
    fp = CACHE / f"{p.id}.jpg"
    p.path = str(fp)
    if fp.exists() and fp.stat().st_size > 3000:
        return p
    try:
        r = requests.get(p.url, headers={"User-Agent": UA}, timeout=60)
        r.raise_for_status()
        if len(r.content) < 3000:
            raise ValueError(f"small image: {len(r.content)} bytes")
        fp.write_bytes(r.content)
    except Exception as exc:
        p.error = f"download: {exc}"
    return p


def match_one(p: Photo, target_data: dict[str, Any]) -> Photo:
    if p.error:
        return p
    candidate = cv2.imread(p.path)
    if candidate is None:
        p.error = "candidate decode failed"
        return p
    if max(candidate.shape[:2]) > 1800:
        scale = 1800 / max(candidate.shape[:2])
        candidate = cv2.resize(candidate, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    kp2, des2 = features(candidate)
    if des2 is None or len(kp2) < 4:
        p.error = "no candidate descriptors"
        return p

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    best = None
    for crop_name, (crop, kp1, des1) in target_data.items():
        if des1 is None or len(kp1) < 4:
            continue
        pairs = matcher.knnMatch(des1, des2, k=2)
        good = [m for m, n in pairs if m.distance < 0.70 * n.distance]
        inliers = 0
        ratio = 0.0
        coverage = 0.0
        if len(good) >= 6:
            a = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            b = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            _, mask = cv2.findHomography(a, b, cv2.RANSAC, 5.0)
            if mask is not None:
                valid = mask.ravel().astype(bool)
                inliers = int(valid.sum())
                ratio = inliers / len(good)
                if inliers >= 4:
                    pts = a[valid, 0, :]
                    x0, y0 = pts.min(axis=0)
                    x1, y1 = pts.max(axis=0)
                    coverage = float(max(0.0, (x1 - x0) * (y1 - y0)) / (crop.shape[0] * crop.shape[1]))
        # Heavily prioritize geometrically consistent correspondences.
        score = inliers * 10.0 + len(good) * 0.20 + min(coverage, 0.5) * 20.0 + ratio * 2.0
        row = (score, len(good), inliers, ratio, coverage, crop_name)
        if best is None or row[0] > best[0]:
            best = row

    if best:
        p.score, p.good, p.inliers, p.inlier_ratio, p.coverage, p.crop = best
    return p


def save_outputs(rows: list[Photo], diagnostics: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda p: p.score, reverse=True)
    (OUT / "diagnostics.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")

    fields = list(Photo.__dataclass_fields__.keys())
    with (OUT / "ranked.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for p in rows:
            writer.writerow(asdict(p))

    md = [
        "# KartaView full-sequence visual match v3", "",
        f"- Target: `{TARGET_URL}`", f"- Candidates scored: **{len(rows)}**", "",
        "|rank|score|photo|sequence|coordinates|good|inliers|ratio|coverage|crop|source|",
        "|---:|---:|---:|---:|---|---:|---:|---:|---:|---|---|",
    ]
    for i, p in enumerate(rows[:100], 1):
        md.append(
            f"|{i}|{p.score:.3f}|{p.id}|{p.sequence_id}|{p.lat},{p.lng}|{p.good}|{p.inliers}|"
            f"{p.inlier_ratio:.3f}|{p.coverage:.3f}|{p.crop}|[image]({p.url})|"
        )
    (OUT / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    cards = []
    for i, p in enumerate(rows[:30], 1):
        try:
            im = Image.open(p.path).convert("RGB")
            im.thumbnail((500, 335))
            card = Image.new("RGB", (520, 390), "white")
            card.paste(im, ((520 - im.width) // 2, 3))
            ImageDraw.Draw(card).multiline_text(
                (6, 344),
                f"#{i} score={p.score:.2f} id={p.id} seq={p.sequence_id}\n"
                f"{p.lat},{p.lng} good={p.good} inliers={p.inliers} cov={p.coverage:.3f}",
                fill="black",
            )
            card.save(OUT / f"top_{i:02d}_{p.id}.jpg", quality=66, optimize=True)
            cards.append(card)
        except Exception:
            pass

    if cards:
        cols = 3
        rows_count = math.ceil(len(cards) / cols)
        sheet = Image.new("RGB", (cols * 520, rows_count * 390), "white")
        for i, card in enumerate(cards):
            sheet.paste(card, ((i % cols) * 520, (i // cols) * 390))
        sheet.save(OUT / "contact_sheet.jpg", quality=72, optimize=True)


def main() -> int:
    target = load_target()
    target_data = prepare_target(target)
    photos, diagnostics = collect()
    print(f"Candidates inside bbox: {len(photos)}", flush=True)

    with ThreadPoolExecutor(max_workers=18) as pool:
        downloaded = [f.result() for f in as_completed([pool.submit(download_one, p) for p in photos])]

    scored: list[Photo] = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(match_one, p, target_data) for p in downloaded]
        for i, future in enumerate(as_completed(futures), 1):
            scored.append(future.result())
            if i % 100 == 0:
                print(f"Matched {i}/{len(downloaded)}", flush=True)

    save_outputs([p for p in scored if p.score >= 0], diagnostics)
    print("TOP RESULTS", flush=True)
    for p in sorted(scored, key=lambda x: x.score, reverse=True)[:20]:
        print(asdict(p), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
