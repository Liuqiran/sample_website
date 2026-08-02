#!/usr/bin/env python3
"""Visually search Google Street View coverage around Colonia Estrella.

The target is compared against perspective views extracted from every sampled
latest panorama. The highest-ranked locations are then expanded to historical
coverage and scanned again.
"""
from __future__ import annotations

import csv
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import py360convert
import requests
from PIL import Image, ImageDraw
from streetlevel import streetview

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "gsv_estrella"
CACHE = ROOT / ".cache" / "gsv_estrella"
TARGET_URL = "https://img9.doubanio.com/view/group_topic/l/public/p735803186.webp"

# Covers the jewel-street grid around the strongest prior coordinate while
# retaining a margin into the rest of Colonia Estrella.
BBOX = {
    "south": 19.4690,
    "north": 19.4805,
    "west": -99.1230,
    "east": -99.1090,
}
Z = 17
SPATIAL_SAMPLE_METERS = 9.0
DOWNLOAD_WORKERS = 5
TARGET_MAX_DIM = 1200
VIEW_HW = (720, 600)
VIEW_FOV = (82, 100)
YAW_STEP = 45


@dataclass
class PanoRecord:
    pano_id: str
    lat: float
    lon: float
    date: str
    source: str
    parent_id: str = ""
    is_historical: bool = False
    best_score: float = -1.0
    good_matches: int = 0
    inliers: int = 0
    inlier_ratio: float = 0.0
    coverage: float = 0.0
    yaw: int = 0
    pitch: int = 0
    target_crop: str = ""
    pano_path: str = ""
    view_path: str = ""
    error: str = ""


def tile_xy(lat: float, lon: float, z: int = Z) -> tuple[int, int]:
    n = 2 ** z
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def in_bbox(lat: float, lon: float) -> bool:
    return BBOX["south"] <= lat <= BBOX["north"] and BBOX["west"] <= lon <= BBOX["east"]


def capture_date(value: object) -> str:
    if value is None:
        return ""
    year = getattr(value, "year", None)
    month = getattr(value, "month", None)
    day = getattr(value, "day", None)
    if year and month:
        return f"{year:04d}-{month:02d}" + (f"-{day:02d}" if day else "")
    return str(value)


def enumerate_latest() -> tuple[list[object], dict]:
    x0, ys = tile_xy(BBOX["south"], BBOX["west"])
    x1, yn = tile_xy(BBOX["north"], BBOX["east"])
    xmin, xmax = sorted((x0, x1))
    ymin, ymax = sorted((yn, ys))
    panos: dict[str, object] = {}
    tile_stats: dict[str, int] = {}
    errors: list[str] = []
    for x in range(xmin, xmax + 1):
        for y in range(ymin, ymax + 1):
            try:
                rows = streetview.get_coverage_tile(x, y)
                tile_stats[f"{x},{y}"] = len(rows)
                for pano in rows:
                    if pano.lat is not None and pano.lon is not None and in_bbox(pano.lat, pano.lon):
                        panos[pano.id] = pano
                print(f"tile {x},{y}: {len(rows)} rows; {len(panos)} unique in bbox", flush=True)
            except Exception as exc:
                errors.append(f"{x},{y}: {exc!r}")
                print(f"tile ERROR {x},{y}: {exc!r}", flush=True)

    # Street View often stores panoramas every 2–5 m. Keep one per small
    # spatial cell to make the first pass tractable while preserving façades.
    lat_scale = 111_320.0
    lon_scale = 111_320.0 * math.cos(math.radians((BBOX["south"] + BBOX["north"]) / 2))
    cells: dict[tuple[int, int], object] = {}
    for pano in panos.values():
        gx = int((pano.lon - BBOX["west"]) * lon_scale / SPATIAL_SAMPLE_METERS)
        gy = int((pano.lat - BBOX["south"]) * lat_scale / SPATIAL_SAMPLE_METERS)
        cells.setdefault((gx, gy), pano)

    sampled = list(cells.values())
    diagnostics = {
        "bbox": BBOX,
        "tiles": tile_stats,
        "raw_latest_panos": len(panos),
        "sampled_latest_panos": len(sampled),
        "sample_meters": SPATIAL_SAMPLE_METERS,
        "errors": errors,
    }
    print(f"latest panorama inventory: raw={len(panos)} sampled={len(sampled)}", flush=True)
    return sampled, diagnostics


def load_target() -> tuple[np.ndarray, dict[str, tuple[np.ndarray, list, np.ndarray]]]:
    OUT.mkdir(parents=True, exist_ok=True)
    response = requests.get(TARGET_URL, timeout=90)
    response.raise_for_status()
    (OUT / "target.webp").write_bytes(response.content)
    target = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
    if target is None:
        raise RuntimeError("target decode failed")
    if max(target.shape[:2]) > TARGET_MAX_DIM:
        scale = TARGET_MAX_DIM / max(target.shape[:2])
        target = cv2.resize(target, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    h, w = target.shape[:2]
    crops = {
        "full": target,
        "upper_full": target[: int(h * 0.78), :],
        "left_houses": target[: int(h * 0.72), : int(w * 0.66)],
        "street_depth": target[int(h * 0.12): int(h * 0.82), int(w * 0.18): int(w * 0.82)],
        "right_depth": target[: int(h * 0.82), int(w * 0.35):],
    }
    sift = cv2.SIFT_create(nfeatures=6500, contrastThreshold=0.012, edgeThreshold=14)
    features: dict[str, tuple[np.ndarray, list, np.ndarray]] = {}
    for name, crop in crops.items():
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None)
        features[name] = (crop, kp, des)
    print(f"target loaded: {w}x{h}; crops={list(features)}", flush=True)
    return target, features


def pano_to_record(pano: object, parent_id: str = "", historical: bool = False) -> PanoRecord:
    return PanoRecord(
        pano_id=str(pano.id),
        lat=float(pano.lat),
        lon=float(pano.lon),
        date=capture_date(getattr(pano, "date", None)),
        source=str(getattr(pano, "source", "") or ""),
        parent_id=parent_id,
        is_historical=historical,
    )


def panorama_cache_path(record: PanoRecord) -> Path:
    safe_id = record.pano_id.replace("/", "_")
    return CACHE / "panoramas" / f"{safe_id}.jpg"


def download_panorama(record: PanoRecord) -> PanoRecord:
    path = panorama_cache_path(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    record.pano_path = str(path)
    if path.exists() and path.stat().st_size > 20_000:
        return record
    try:
        pano = streetview.find_panorama_by_id(record.pano_id)
        if pano is None:
            raise RuntimeError("metadata not found")
        image = streetview.get_panorama(pano, zoom=2)
        image.convert("RGB").save(path, format="JPEG", quality=88, optimize=True)
    except Exception as exc:
        record.error = f"download: {exc!r}"
    return record


def iter_views(pano_bgr: np.ndarray) -> Iterable[tuple[int, int, np.ndarray]]:
    rgb = cv2.cvtColor(pano_bgr, cv2.COLOR_BGR2RGB)
    for pitch in (-8, 4):
        for yaw in range(0, 360, YAW_STEP):
            view_rgb = py360convert.e2p(
                rgb,
                fov_deg=VIEW_FOV,
                u_deg=float(yaw),
                v_deg=float(pitch),
                out_hw=VIEW_HW,
                mode="bilinear",
            )
            yield yaw, pitch, cv2.cvtColor(view_rgb, cv2.COLOR_RGB2BGR)


def score_view(view: np.ndarray, target_features: dict) -> tuple:
    sift = cv2.SIFT_create(nfeatures=5200, contrastThreshold=0.012, edgeThreshold=14)
    gray = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
    kp2, des2 = sift.detectAndCompute(gray, None)
    if des2 is None or len(kp2) < 6:
        return (-1.0, 0, 0, 0.0, 0.0, "")
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    best = (-1.0, 0, 0, 0.0, 0.0, "")
    for crop_name, (crop, kp1, des1) in target_features.items():
        if des1 is None or len(kp1) < 6:
            continue
        pairs = matcher.knnMatch(des1, des2, k=2)
        good = [m for m, n in pairs if m.distance < 0.70 * n.distance]
        inliers = 0
        ratio = 0.0
        coverage = 0.0
        if len(good) >= 6:
            src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            if mask is not None:
                valid = mask.ravel().astype(bool)
                inliers = int(valid.sum())
                ratio = inliers / len(good)
                if inliers >= 4:
                    points = src[valid, 0, :]
                    x0, y0 = points.min(axis=0)
                    x1, y1 = points.max(axis=0)
                    coverage = float(max(0.0, (x1 - x0) * (y1 - y0)) / (crop.shape[0] * crop.shape[1]))
        # Genuine same-scene matches should have both many inliers and spatial
        # spread. Small repeated patterns are deliberately downweighted.
        score = (
            inliers * 12.0
            + min(len(good), 80) * 0.15
            + min(coverage, 0.45) * 28.0
            + ratio * 2.5
        )
        row = (score, len(good), inliers, ratio, coverage, crop_name)
        if row[0] > best[0]:
            best = row
    return best


def scan_record(record: PanoRecord, target_features: dict, phase: str) -> PanoRecord:
    if record.error:
        return record
    pano = cv2.imread(record.pano_path)
    if pano is None:
        record.error = "panorama decode failed"
        return record
    best_view: np.ndarray | None = None
    for yaw, pitch, view in iter_views(pano):
        score, good, inliers, ratio, coverage, crop_name = score_view(view, target_features)
        if score > record.best_score:
            record.best_score = score
            record.good_matches = good
            record.inliers = inliers
            record.inlier_ratio = ratio
            record.coverage = coverage
            record.yaw = yaw
            record.pitch = pitch
            record.target_crop = crop_name
            best_view = view.copy()
    if best_view is not None:
        views_dir = CACHE / "best_views" / phase
        views_dir.mkdir(parents=True, exist_ok=True)
        path = views_dir / f"{record.pano_id.replace('/', '_')}.jpg"
        cv2.imwrite(str(path), best_view, [cv2.IMWRITE_JPEG_QUALITY, 88])
        record.view_path = str(path)
    return record


def scan_records(records: list[PanoRecord], target_features: dict, phase: str) -> list[PanoRecord]:
    downloaded: list[PanoRecord] = []
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        futures = [pool.submit(download_panorama, record) for record in records]
        for index, future in enumerate(as_completed(futures), 1):
            downloaded.append(future.result())
            if index % 50 == 0:
                print(f"{phase}: downloaded {index}/{len(records)}", flush=True)

    # SIFT is CPU-heavy and internally threaded; process sequentially to avoid
    # oversubscribing GitHub's two-core runner.
    scored: list[PanoRecord] = []
    for index, record in enumerate(downloaded, 1):
        scored.append(scan_record(record, target_features, phase))
        if index % 25 == 0:
            top = max((r.best_score for r in scored), default=-1)
            print(f"{phase}: scored {index}/{len(downloaded)}; current top={top:.2f}", flush=True)
    return sorted(scored, key=lambda r: r.best_score, reverse=True)


def collect_historical(latest_ranked: list[PanoRecord], limit_latest: int = 80) -> list[PanoRecord]:
    historical: dict[str, PanoRecord] = {}
    errors: list[str] = []
    for index, latest in enumerate(latest_ranked[:limit_latest], 1):
        try:
            full = streetview.find_panorama_by_id(latest.pano_id)
            if full is None:
                continue
            candidates = list(getattr(full, "historical", []) or [])
            # Include full current panorama as metadata may improve date/source.
            for pano in candidates:
                if pano.lat is None or pano.lon is None:
                    continue
                record = pano_to_record(pano, parent_id=latest.pano_id, historical=True)
                historical[record.pano_id] = record
            print(
                f"history metadata {index}/{min(limit_latest, len(latest_ranked))}: "
                f"{latest.pano_id} -> {len(candidates)}; total={len(historical)}",
                flush=True,
            )
        except Exception as exc:
            errors.append(f"{latest.pano_id}: {exc!r}")
    (OUT / "historical_metadata_errors.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")
    return list(historical.values())


def write_csv(path: Path, rows: list[PanoRecord]) -> None:
    fields = list(PanoRecord.__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def build_contact_sheet(rows: list[PanoRecord], name: str, limit: int = 40) -> None:
    cards: list[Image.Image] = []
    thumbs_dir = OUT / name
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    for rank, row in enumerate(rows[:limit], 1):
        try:
            image = Image.open(row.view_path).convert("RGB")
            image.thumbnail((420, 500))
            card = Image.new("RGB", (440, 560), "white")
            card.paste(image, ((440 - image.width) // 2, 4))
            text = (
                f"#{rank} score={row.best_score:.2f} inliers={row.inliers} good={row.good_matches}\n"
                f"{row.lat:.7f}, {row.lon:.7f} date={row.date or '?'} yaw={row.yaw} pitch={row.pitch}\n"
                f"pano={row.pano_id} crop={row.target_crop} cov={row.coverage:.3f}"
            )
            ImageDraw.Draw(card).multiline_text((6, 510), text, fill="black", spacing=2)
            thumb_path = thumbs_dir / f"{rank:03d}_{row.pano_id.replace('/', '_')}.jpg"
            card.save(thumb_path, quality=72, optimize=True)
            cards.append(card)
        except Exception as exc:
            print(f"thumbnail error {row.pano_id}: {exc!r}", flush=True)
    if not cards:
        return
    cols = 3
    rows_n = math.ceil(len(cards) / cols)
    sheet = Image.new("RGB", (cols * 440, rows_n * 560), "white")
    for index, card in enumerate(cards):
        sheet.paste(card, ((index % cols) * 440, (index // cols) * 560))
    sheet.save(OUT / f"{name}_contact_sheet.jpg", quality=76, optimize=True)


def report(latest: list[PanoRecord], historical: list[PanoRecord], diagnostics: dict) -> None:
    combined = sorted(latest + historical, key=lambda r: r.best_score, reverse=True)
    write_csv(OUT / "latest_ranked.csv", latest)
    write_csv(OUT / "historical_ranked.csv", historical)
    write_csv(OUT / "combined_ranked.csv", combined)
    build_contact_sheet(latest, "latest_top", 45)
    build_contact_sheet(historical, "historical_top", 45)
    build_contact_sheet(combined, "combined_top", 60)

    lines = [
        "# Colonia Estrella Google Street View visual search", "",
        f"- Search bbox: `{json.dumps(BBOX)}`",
        f"- Raw latest panoramas: **{diagnostics.get('raw_latest_panos', 0)}**",
        f"- Sampled latest panoramas: **{diagnostics.get('sampled_latest_panos', 0)}**",
        f"- Historical panoramas scanned: **{len(historical)}**", "",
        "|rank|score|inliers|good|coverage|date|coordinates|yaw/pitch|pano|historical|",
        "|---:|---:|---:|---:|---:|---|---|---|---|---|",
    ]
    for rank, row in enumerate(combined[:100], 1):
        lines.append(
            f"|{rank}|{row.best_score:.2f}|{row.inliers}|{row.good_matches}|{row.coverage:.3f}|"
            f"{row.date}|{row.lat:.7f}, {row.lon:.7f}|{row.yaw}/{row.pitch}|`{row.pano_id}`|"
            f"{row.is_historical}|"
        )
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "diagnostics.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    _, target_features = load_target()
    latest_panos, diagnostics = enumerate_latest()
    latest_records = [pano_to_record(pano) for pano in latest_panos]
    latest_ranked = scan_records(latest_records, target_features, "latest")
    write_csv(OUT / "latest_checkpoint.csv", latest_ranked)
    build_contact_sheet(latest_ranked, "latest_checkpoint_top", 50)

    history_records = collect_historical(latest_ranked, limit_latest=80)
    historical_ranked = scan_records(history_records, target_features, "historical") if history_records else []
    diagnostics["historical_panos"] = len(historical_ranked)
    diagnostics["elapsed_seconds"] = time.time() - start
    report(latest_ranked, historical_ranked, diagnostics)

    print("FINAL TOP 30", flush=True)
    for row in sorted(latest_ranked + historical_ranked, key=lambda r: r.best_score, reverse=True)[:30]:
        print(json.dumps(asdict(row), ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
