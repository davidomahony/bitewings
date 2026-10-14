"""HTML gallery of auto-numbered DenPAR periapicals, to check the numbering by eye.

    python bitewings/denpar/preview.py --denpar-root <...>/Dataset --out-dir <dir> [--split train]

Draws each tooth's mask outline with its number (Universal and FDI). Anterior images come
first. Numbers come from bitewings/denpar/convert.py.
"""
import argparse
import html
import json
import random
from pathlib import Path

import cv2
import numpy as np
import pycocotools.mask as maskUtils

FDIS = [q * 10 + n for q in (1, 2, 3, 4) for n in range(1, 9)]
COLORS = [(66, 135, 245), (60, 200, 90), (245, 170, 40), (220, 60, 200), (40, 210, 220), (245, 90, 60),
          (150, 110, 245)]


def fdi_to_universal(fdi: int) -> int:
    q, n = divmod(fdi, 10)
    return {1: 9 - n, 2: 8 + n, 3: 25 - n, 4: 24 + n}[q]


def main(root: Path, out_dir: Path, split: str, n_anterior: int, n_side: int, seed: int):
    (out_dir / 'img').mkdir(parents=True, exist_ok=True)
    coco = json.loads((root / 'splits' / f'{split}_fdi.json').read_text())
    by_image = {}
    for ann in coco['annotations']:
        by_image.setdefault(ann['image_id'], []).append(ann)
    rng = random.Random(seed)
    anterior = [i for i in coco['images'] if i['site'] == 'Anterior']
    side = [i for i in coco['images'] if i['site'] != 'Anterior']
    picked = rng.sample(anterior, min(n_anterior, len(anterior))) + rng.sample(side, min(n_side, len(side)))

    cards = []
    for info in picked:
        img = cv2.imread(str(root / info['file_name']))
        out = img.copy()
        anns = sorted(by_image.get(info['id'], []), key=lambda a: a['bbox'][0])
        scale = max(img.shape[:2]) / 1000
        for k, ann in enumerate(anns):
            mask = maskUtils.decode(ann['segmentation']).astype(np.uint8)
            color = COLORS[k % len(COLORS)]
            tint = out.copy()
            tint[mask > 0] = color
            out = cv2.addWeighted(tint, 0.25, out, 0.75, 0)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, contours, -1, color, max(2, int(3 * scale)))
        for ann in anns:  # labels last, so no outline is drawn over them
            fdi = FDIS[ann['category_id'] - 1]
            x, y, w, h = ann['bbox']
            text = f'#{fdi_to_universal(fdi)} ({fdi})'
            org = (int(x + w / 2 - 75 * scale), int(y + h * 0.45))
            cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.85 * scale, (0, 0, 0), int(7 * scale))
            cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.85 * scale, (255, 255, 255), int(2 * scale))
        both = np.hstack([img, out])
        s = min(1.0, 1600 / both.shape[1])
        both = cv2.resize(both, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        name = f'{info["id"]}.jpg'
        cv2.imwrite(str(out_dir / 'img' / name), both, [cv2.IMWRITE_JPEG_QUALITY, 88])
        listed = ' '.join(str(FDIS[a['category_id'] - 1]) for a in anns)
        cards.append(
            f'<div class="card"><h3>DenPAR {info["id"]}: {html.escape(info["arch"])} {html.escape(info["site"])}'
            f' &middot; left to right (FDI): {listed}</h3><img src="img/{name}" loading="lazy"></div>')

    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>DenPAR numbering check</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{{--bg:#fff;--fg:#1a1a1a;--card:#f5f5f4;--muted:#666}}
@media (prefers-color-scheme: dark){{:root{{--bg:#151515;--fg:#eee;--card:#222;--muted:#aaa}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;max-width:1650px;margin:0 auto;padding:16px}}
.card{{background:var(--card);border-radius:10px;padding:10px 14px;margin:16px 0}}
h3{{font-size:14px;margin:2px 0 8px}} img{{width:100%;border-radius:6px;display:block}} p{{color:var(--muted)}}
</style></head><body>
<h1>DenPAR auto-numbering check ({len(cards)} images, {split} split)</h1>
<p>Real periapicals from DenPAR (CC BY 4.0). Left: X-ray. Right: DenPAR's own tooth masks with the numbers
assigned automatically (Universal, with FDI in brackets). Rule: masks left to right are paired with the listed
teeth ordered from the patient's right to left (standard frontal view). Anterior images first.
Check that each number sits on the right tooth.</p>
{''.join(cards)}</body></html>"""
    (out_dir / 'index.html').write_text(page, encoding='utf-8')
    print(f'wrote {out_dir / "index.html"}: {len(cards)} images')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--denpar-root', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--split', default='train')
    parser.add_argument('--n-anterior', type=int, default=16)
    parser.add_argument('--n-side', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    main(args.denpar_root, args.out_dir, args.split, args.n_anterior, args.n_side, args.seed)
