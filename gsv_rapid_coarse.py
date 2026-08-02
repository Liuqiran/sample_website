#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

import gsv_scan_estrella as core
import gsv_scan_estrella_fast as fast

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results' / 'gsv_rapid'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # Reuse the verified target and GSV routines, but use a much coarser grid.
    fast.COARSE_CELL_METERS = 42.0
    start = time.time()
    _, target_features = fast.load_target_local()
    _, sampled, diagnostics = fast.enumerate_radius()
    records = [core.pano_to_record(p) for p in sampled]
    ranked = fast.run_phase(
        records,
        target_features,
        'rapid',
        1,
        (-4,),
        (440, 380),
        (90, 106),
    )
    fields = list(core.PanoRecord.__dataclass_fields__.keys())
    with (OUT / 'ranked.csv').open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in ranked:
            writer.writerow(asdict(row))
    diagnostics['elapsed_seconds'] = time.time() - start
    diagnostics['ranked_count'] = len(ranked)
    (OUT / 'diagnostics.json').write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding='utf-8')

    lines = [
        '# Rapid Colonia Estrella coarse scan', '',
        f"- panoramas scored: **{len(ranked)}**",
        f"- elapsed seconds: **{diagnostics['elapsed_seconds']:.1f}**", '',
        '|rank|score|inliers|good|coverage|date|coordinates|yaw/pitch|pano|',
        '|---:|---:|---:|---:|---:|---|---|---|---|',
    ]
    cards = []
    for rank, row in enumerate(ranked[:100], 1):
        lines.append(
            f'|{rank}|{row.best_score:.2f}|{row.inliers}|{row.good_matches}|{row.coverage:.3f}|'
            f'{row.date}|{row.lat:.7f}, {row.lon:.7f}|{row.yaw}/{row.pitch}|`{row.pano_id}`|'
        )
        if rank <= 60 and row.view_path:
            try:
                im = Image.open(row.view_path).convert('RGB')
                im.thumbnail((430, 420))
                card = Image.new('RGB', (450, 500), 'white')
                card.paste(im, ((450-im.width)//2, 3))
                ImageDraw.Draw(card).multiline_text(
                    (5, 440),
                    f'#{rank} score={row.best_score:.2f} inl={row.inliers} good={row.good_matches}\n'
                    f'{row.lat:.7f},{row.lon:.7f} yaw={row.yaw} date={row.date or "?"}\n{row.pano_id}',
                    fill='black', spacing=2,
                )
                thumb = OUT / f'top_{rank:03d}_{row.pano_id.replace("/", "_")}.jpg'
                card.save(thumb, quality=75, optimize=True)
                cards.append(card)
            except Exception:
                pass
    (OUT / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    if cards:
        cols = 3
        sheet = Image.new('RGB', (cols*450, math.ceil(len(cards)/cols)*500), 'white')
        for i, card in enumerate(cards):
            sheet.paste(card, ((i%cols)*450, (i//cols)*500))
        sheet.save(OUT / 'contact_sheet.jpg', quality=78, optimize=True)
    print('RAPID TOP 30', flush=True)
    for row in ranked[:30]:
        print(json.dumps(asdict(row), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
