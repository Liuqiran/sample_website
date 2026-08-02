#!/usr/bin/env python3
"""Coarse-to-fine Google Street View matcher for Colonia Estrella."""
from __future__ import annotations

import csv
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import py360convert
from PIL import Image, ImageDraw
from streetlevel import streetview

import gsv_scan_estrella as core

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "gsv_estrella_fast"
CACHE = ROOT / ".cache" / "gsv_estrella_fast"
TARGET_LOCAL = ROOT / "results" / "douban_candidates" / "21_full.jpg"
CENTER = (19.474720, -99.118565)
RADIUS_METERS = 600.0
COARSE_CELL_METERS = 25.0
COARSE_ZOOM = 1
FINE_ZOOM = 2
COARSE_PITCHES = (-4,)
FINE_PITCHES = (-10, 2)
COARSE_VIEW_HW = (520, 440)
FINE_VIEW_HW = (760, 620)
COARSE_FOV = (88, 104)
FINE_FOV = (82, 100)
COARSE_TOP_SEEDS = 55
REFINE_RADIUS_METERS = 48.0
HISTORY_PARENT_LIMIT = 45


def distance_m(lat: float, lon: float, center: tuple[float, float] = CENTER) -> float:
    dy = (lat - center[0]) * 111_320.0
    dx = (lon - center[1]) * 111_320.0 * math.cos(math.radians(center[0]))
    return math.hypot(dx, dy)


def distance_between(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
    dy = (a_lat - b_lat) * 111_320.0
    dx = (a_lon - b_lon) * 111_320.0 * math.cos(math.radians((a_lat + b_lat) / 2.0))
    return math.hypot(dx, dy)


def load_target_local():
    OUT.mkdir(parents=True, exist_ok=True)
    if not TARGET_LOCAL.exists():
        raise FileNotFoundError(f"Verified target not found: {TARGET_LOCAL}")
    raw = TARGET_LOCAL.read_bytes()
    (OUT / "target.webp").write_bytes(raw)
    target = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if target is None:
        raise RuntimeError("Verified target image could not be decoded")
    if max(target.shape[:2]) > 1250:
        scale = 1250 / max(target.shape[:2])
        target = cv2.resize(target, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    h, w = target.shape[:2]
    crops = {
        "full": target,
        "upper_full": target[: int(h * 0.80), :],
        "left_houses": target[: int(h * 0.74), : int(w * 0.68)],
        "street_depth": target[int(h * 0.10): int(h * 0.84), int(w * 0.16): int(w * 0.84)],
        "right_depth": target[: int(h * 0.84), int(w * 0.32):],
    }
    sift = cv2.SIFT_create(nfeatures=7000, contrastThreshold=0.011, edgeThreshold=14)
    features = {}
    for name, crop in crops.items():
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None)
        features[name] = (crop, kp, des)
    print(f"verified local target: {w}x{h}; bytes={len(raw)}", flush=True)
    return target, features


def enumerate_radius():
    # Use a bbox large enough to contain the circular scan radius.
    lat_delta = RADIUS_METERS / 111_320.0
    lon_delta = RADIUS_METERS / (111_320.0 * math.cos(math.radians(CENTER[0])))
    bbox = {
        "south": CENTER[0] - lat_delta,
        "north": CENTER[0] + lat_delta,
        "west": CENTER[1] - lon_delta,
        "east": CENTER[1] + lon_delta,
    }
    x0, ys = core.tile_xy(bbox["south"], bbox["west"])
    x1, yn = core.tile_xy(bbox["north"], bbox["east"])
    xmin, xmax = sorted((x0, x1))
    ymin, ymax = sorted((yn, ys))
    all_panos = {}
    errors = []
    tile_stats = {}
    for x in range(xmin, xmax + 1):
        for y in range(ymin, ymax + 1):
            try:
                rows = streetview.get_coverage_tile(x, y)
                tile_stats[f"{x},{y}"] = len(rows)
                for pano in rows:
                    if pano.lat is None or pano.lon is None:
                        continue
                    if distance_m(float(pano.lat), float(pano.lon)) <= RADIUS_METERS:
                        all_panos[pano.id] = pano
                print(f"tile {x},{y}: rows={len(rows)} radius_unique={len(all_panos)}", flush=True)
            except Exception as exc:
                errors.append(f"{x},{y}: {exc!r}")
                print(f"tile error {x},{y}: {exc!r}", flush=True)

    lat_scale = 111_320.0
    lon_scale = 111_320.0 * math.cos(math.radians(CENTER[0]))
    cells = {}
    for pano in all_panos.values():
        gx = int((pano.lon - bbox["west"]) * lon_scale / COARSE_CELL_METERS)
        gy = int((pano.lat - bbox["south"]) * lat_scale / COARSE_CELL_METERS)
        # Prefer the panorama nearest the center of the grid cell.
        cell_center_lon = bbox["west"] + (gx + 0.5) * COARSE_CELL_METERS / lon_scale
        cell_center_lat = bbox["south"] + (gy + 0.5) * COARSE_CELL_METERS / lat_scale
        d = distance_between(pano.lat, pano.lon, cell_center_lat, cell_center_lon)
        old = cells.get((gx, gy))
        if old is None or d < old[0]:
            cells[(gx, gy)] = (d, pano)
    sampled = [item[1] for item in cells.values()]
    diagnostics = {
        "center": CENTER,
        "radius_m": RADIUS_METERS,
        "coarse_cell_m": COARSE_CELL_METERS,
        "bbox": bbox,
        "tile_stats": tile_stats,
        "raw_radius_panos": len(all_panos),
        "coarse_sampled_panos": len(sampled),
        "errors": errors,
    }
    print(f"inventory: raw={len(all_panos)} coarse={len(sampled)}", flush=True)
    return list(all_panos.values()), sampled, diagnostics


def pano_cache(record: core.PanoRecord, phase: str, zoom: int) -> Path:
    safe = record.pano_id.replace("/", "_")
    return CACHE / "panoramas" / phase / f"z{zoom}_{safe}.jpg"


def download_record(record: core.PanoRecord, phase: str, zoom: int):
    path = pano_cache(record, phase, zoom)
    path.parent.mkdir(parents=True, exist_ok=True)
    record.pano_path = str(path)
    if path.exists() and path.stat().st_size > 10_000:
        return record
    try:
        pano = streetview.find_panorama_by_id(record.pano_id)
        if pano is None:
            raise RuntimeError("panorama metadata unavailable")
        image = streetview.get_panorama(pano, zoom=zoom)
        image.convert("RGB").save(path, "JPEG", quality=88, optimize=True)
    except Exception as exc:
        record.error = f"download: {exc!r}"
    return record


def iter_views(pano_bgr: np.ndarray, pitches, out_hw, fov):
    rgb = cv2.cvtColor(pano_bgr, cv2.COLOR_BGR2RGB)
    for pitch in pitches:
        for yaw in range(0, 360, 45):
            view_rgb = py360convert.e2p(
                rgb,
                fov_deg=fov,
                u_deg=float(yaw),
                v_deg=float(pitch),
                out_hw=out_hw,
                mode="bilinear",
            )
            yield yaw, pitch, cv2.cvtColor(view_rgb, cv2.COLOR_RGB2BGR)


def score_view_fast(view: np.ndarray, target_features):
    sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.011, edgeThreshold=14)
    kp2, des2 = sift.detectAndCompute(cv2.cvtColor(view, cv2.COLOR_BGR2GRAY), None)
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
            _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.5)
            if mask is not None:
                valid = mask.ravel().astype(bool)
                inliers = int(valid.sum())
                ratio = inliers / len(good)
                if inliers >= 4:
                    pts = src[valid, 0, :]
                    x0, y0 = pts.min(axis=0)
                    x1, y1 = pts.max(axis=0)
                    coverage = float(max(0.0, (x1 - x0) * (y1 - y0)) / (crop.shape[0] * crop.shape[1]))
        score = inliers * 12.0 + min(len(good), 90) * 0.15 + min(coverage, 0.50) * 30.0 + ratio * 2.5
        row = (score, len(good), inliers, ratio, coverage, crop_name)
        if row[0] > best[0]:
            best = row
    return best


def scan_one(record, target_features, phase, pitches, out_hw, fov):
    if record.error:
        return record
    pano = cv2.imread(record.pano_path)
    if pano is None:
        record.error = "panorama decode failed"
        return record
    best_image = None
    for yaw, pitch, view in iter_views(pano, pitches, out_hw, fov):
        score, good, inliers, ratio, coverage, crop_name = score_view_fast(view, target_features)
        if score > record.best_score:
            record.best_score = score
            record.good_matches = good
            record.inliers = inliers
            record.inlier_ratio = ratio
            record.coverage = coverage
            record.yaw = yaw
            record.pitch = pitch
            record.target_crop = crop_name
            best_image = view.copy()
    if best_image is not None:
        d = CACHE / "best_views" / phase
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{record.pano_id.replace('/', '_')}.jpg"
        cv2.imwrite(str(p), best_image, [cv2.IMWRITE_JPEG_QUALITY, 88])
        record.view_path = str(p)
    return record


def run_phase(records, target_features, phase, zoom, pitches, out_hw, fov):
    downloaded = []
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(download_record, r, phase, zoom) for r in records]
        for i, future in enumerate(as_completed(futures), 1):
            downloaded.append(future.result())
            if i % 50 == 0:
                print(f"{phase}: downloaded {i}/{len(records)}", flush=True)
    ranked = []
    for i, record in enumerate(downloaded, 1):
        ranked.append(scan_one(record, target_features, phase, pitches, out_hw, fov))
        if i % 20 == 0:
            top = max((r.best_score for r in ranked), default=-1)
            print(f"{phase}: scored {i}/{len(downloaded)} top={top:.2f}", flush=True)
    return sorted(ranked, key=lambda r: r.best_score, reverse=True)


def select_fine_raw(all_panos, coarse_ranked):
    selected = {}
    seeds = coarse_ranked[:COARSE_TOP_SEEDS]
    for pano in all_panos:
        for seed in seeds:
            if distance_between(pano.lat, pano.lon, seed.lat, seed.lon) <= REFINE_RADIUS_METERS:
                selected[pano.id] = pano
                break
    print(f"fine selection from {len(seeds)} seeds: {len(selected)} unique raw panos", flush=True)
    return list(selected.values())


def collect_history(fine_ranked):
    historical = {}
    errors = []
    for i, parent in enumerate(fine_ranked[:HISTORY_PARENT_LIMIT], 1):
        try:
            full = streetview.find_panorama_by_id(parent.pano_id)
            rows = list(getattr(full, "historical", []) or []) if full else []
            for pano in rows:
                if pano.lat is None or pano.lon is None:
                    continue
                record = core.pano_to_record(pano, parent_id=parent.pano_id, historical=True)
                historical[record.pano_id] = record
            print(f"history {i}/{HISTORY_PARENT_LIMIT}: {len(rows)}; cumulative={len(historical)}", flush=True)
        except Exception as exc:
            errors.append(f"{parent.pano_id}: {exc!r}")
    (OUT / "history_errors.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")
    return list(historical.values())


def write_csv(path, rows):
    fields = list(core.PanoRecord.__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def contact_sheet(rows, basename, limit=60):
    cards = []
    thumbs = OUT / basename
    thumbs.mkdir(parents=True, exist_ok=True)
    for rank, row in enumerate(rows[:limit], 1):
        try:
            im = Image.open(row.view_path).convert("RGB")
            im.thumbnail((440, 500))
            card = Image.new("RGB", (460, 570), "white")
            card.paste(im, ((460 - im.width) // 2, 4))
            text = (
                f"#{rank} score={row.best_score:.2f} inl={row.inliers} good={row.good_matches} cov={row.coverage:.3f}\n"
                f"{row.lat:.7f},{row.lon:.7f} date={row.date or '?'} yaw={row.yaw}/{row.pitch}\n"
                f"{row.pano_id} hist={row.is_historical} parent={row.parent_id}"
            )
            ImageDraw.Draw(card).multiline_text((5, 515), text, fill="black", spacing=2)
            card.save(thumbs / f"{rank:03d}_{row.pano_id.replace('/', '_')}.jpg", quality=74, optimize=True)
            cards.append(card)
        except Exception as exc:
            print(f"thumb error {row.pano_id}: {exc!r}", flush=True)
    if cards:
        cols = 3
        sheet = Image.new("RGB", (cols * 460, math.ceil(len(cards) / cols) * 570), "white")
        for i, card in enumerate(cards):
            sheet.paste(card, ((i % cols) * 460, (i // cols) * 570))
        sheet.save(OUT / f"{basename}_contact_sheet.jpg", quality=78, optimize=True)


def checkpoint(name, rows, diagnostics=None):
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / f"{name}.csv", rows)
    contact_sheet(rows, f"{name}_top", limit=60)
    if diagnostics is not None:
        (OUT / "diagnostics.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")


def final_report(coarse, fine, historical, diagnostics):
    combined = sorted(fine + historical, key=lambda r: r.best_score, reverse=True)
    write_csv(OUT / "coarse_ranked.csv", coarse)
    write_csv(OUT / "fine_ranked.csv", fine)
    write_csv(OUT / "historical_ranked.csv", historical)
    write_csv(OUT / "combined_ranked.csv", combined)
    contact_sheet(combined, "combined_top", limit=90)
    lines = [
        "# Colonia Estrella coarse-to-fine Street View match", "",
        f"- Center: `{CENTER[0]}, {CENTER[1]}`",
        f"- Radius: **{RADIUS_METERS:.0f} m**",
        f"- Raw latest panos: **{diagnostics['raw_radius_panos']}**",
        f"- Coarse panos: **{len(coarse)}**",
        f"- Fine panos: **{len(fine)}**",
        f"- Historical panos: **{len(historical)}**", "",
        "|rank|score|inliers|good|coverage|date|coordinates|yaw/pitch|pano|historical|parent|",
        "|---:|---:|---:|---:|---:|---|---|---|---|---|---|",
    ]
    for rank, row in enumerate(combined[:150], 1):
        lines.append(
            f"|{rank}|{row.best_score:.2f}|{row.inliers}|{row.good_matches}|{row.coverage:.3f}|{row.date}|"
            f"{row.lat:.7f}, {row.lon:.7f}|{row.yaw}/{row.pitch}|`{row.pano_id}`|"
            f"{row.is_historical}|`{row.parent_id}`|"
        )
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    diagnostics["coarse_count"] = len(coarse)
    diagnostics["fine_count"] = len(fine)
    diagnostics["historical_count"] = len(historical)
    (OUT / "diagnostics.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    _, target_features = load_target_local()
    all_panos, coarse_panos, diagnostics = enumerate_radius()

    coarse_records = [core.pano_to_record(pano) for pano in coarse_panos]
    coarse_ranked = run_phase(
        coarse_records, target_features, "coarse", COARSE_ZOOM,
        COARSE_PITCHES, COARSE_VIEW_HW, COARSE_FOV,
    )
    checkpoint("coarse_checkpoint", coarse_ranked, diagnostics)

    fine_panos = select_fine_raw(all_panos, coarse_ranked)
    fine_records = [core.pano_to_record(pano) for pano in fine_panos]
    fine_ranked = run_phase(
        fine_records, target_features, "fine", FINE_ZOOM,
        FINE_PITCHES, FINE_VIEW_HW, FINE_FOV,
    )
    checkpoint("fine_checkpoint", fine_ranked, diagnostics)

    history_records = collect_history(fine_ranked)
    history_ranked = run_phase(
        history_records, target_features, "history", FINE_ZOOM,
        FINE_PITCHES, FINE_VIEW_HW, FINE_FOV,
    ) if history_records else []

    diagnostics["elapsed_seconds"] = time.time() - start
    final_report(coarse_ranked, fine_ranked, history_ranked, diagnostics)
    print("FINAL TOP", flush=True)
    for row in sorted(fine_ranked + history_ranked, key=lambda r: r.best_score, reverse=True)[:40]:
        print(json.dumps(asdict(row), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
