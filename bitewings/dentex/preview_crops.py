"""Write an HTML gallery of CropPeriapical training samples drawn from DENTEX.

Runs panoramics through the same loading + cropping transforms as training and
draws what the model receives: the crop, each tooth's mask outline and its
Universal number. Use it to check the crops before spending GPU time.

    python bitewings/dentex/preview_crops.py --dentex-root <quadrant_enumeration> --out-dir <dir>
"""
import argparse
import html
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))
sys.path.append(str(Path(__file__).resolve().parents[2] / 'onedl-mmdetection'))

import projects.DENTEX.datasets  # noqa: F401,E402  (registers the dataset and loaders)
import projects.DENTEX.datasets.transforms.loading  # noqa: F401,E402
import bitewings.data.periapical  # noqa: F401,E402  (registers CropPeriapical)
from projects.DENTEX.datasets.coco_multiclass import CocoMulticlassDataset  # noqa: E402
from mmengine.registry import init_default_scope  # noqa: E402

init_default_scope('mmdet')  # transforms are registered in mmdet's registry, as in training

FDIS = [q * 10 + n for q in (1, 2, 3, 4) for n in range(1, 9)]
ATTRIBUTES = ['implants', 'crowns', 'pontic', 'fillings', 'roots', 'caries', 'calculus']


def fdi_to_universal(fdi: int) -> int:
    q, n = divmod(fdi, 10)
    return {1: 9 - n, 2: 8 + n, 3: 25 - n, 4: 24 + n}[q]


def main(dentex_root: Path, out_dir: Path, split: str, n_images: int, crops_per_image: int, seed: int):
    np.random.seed(seed)
    (out_dir / 'img').mkdir(parents=True, exist_ok=True)

    # the dataset decodes every outline when it loads, so give it only the sampled images
    coco = json.loads((dentex_root / 'splits' / f'{split}_fdi.json').read_text())
    picked = np.random.choice(len(coco['images']), min(n_images, len(coco['images'])), replace=False)
    coco['images'] = [coco['images'][i] for i in sorted(picked)]
    ids = {img['id'] for img in coco['images']}
    coco['annotations'] = [ann for ann in coco['annotations'] if ann['image_id'] in ids]
    subset = out_dir / f'{split}_preview.json'
    subset.write_text(json.dumps(coco))

    dataset = CocoMulticlassDataset(
        ann_file=str(subset),
        data_root=str(dentex_root),
        data_prefix=dict(img=str(dentex_root / 'xrays')),
        metainfo=dict(classes=['tooth'], attributes=ATTRIBUTES),
        strict=True, decode_masks=False, merge_layers=True, num_workers=0,
        serialize_data=False,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadMulticlassAnnotations', merge_layers=True, with_bbox=True, with_mask=True),
            dict(type='CropPeriapical'),
        ],
    )

    cards, n_front = [], 0
    for i in range(len(dataset)):
        for k in range(crops_per_image):
            sample = dataset[i]
            img = sample['img'].copy()
            labels = sample['gt_bboxes_labels']
            front = bool(len(labels)) and bool(((labels % 8) <= 2).sum() * 2 >= len(labels))
            n_front += front
            for mask, label in zip(sample['gt_masks'].masks, labels):
                contours, _ = cv2.findContours((mask != 0).astype(np.uint8), cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                color = (0, 200, 255) if label % 8 <= 2 else (255, 255, 255)
                cv2.drawContours(img, contours, -1, color, 2)
                ys, xs = np.nonzero(mask)
                text = str(fdi_to_universal(FDIS[int(label)]))
                org = (int(xs.mean()) - 12, int(ys.mean()) + 8)
                cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5)
                cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            name = f'{Path(sample["img_path"]).stem}_{k}.jpg'
            cv2.imwrite(str(out_dir / 'img' / name), np.hstack([sample['img'], img]),
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
            h, w = img.shape[:2]
            kind = 'front' if front else 'back'
            cards.append(
                f'<div class="card"><h3>{html.escape(Path(sample["img_path"]).name)} — {kind} crop, '
                f'{w}×{h}px, {len(labels)} teeth</h3><img src="img/{name}" loading="lazy"></div>')

    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Periapical crop preview</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{{--bg:#fff;--fg:#1a1a1a;--card:#f5f5f4}}
@media (prefers-color-scheme: dark){{:root{{--bg:#151515;--fg:#eee;--card:#222}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;max-width:1500px;margin:0 auto;padding:16px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:14px}}
.card{{background:var(--card);border-radius:10px;padding:8px 12px}}
h3{{font-size:13px;margin:2px 0 6px}} img{{width:100%;border-radius:6px;display:block}}
</style></head><body>
<h1>CropPeriapical training samples from DENTEX ({len(cards)} crops, {n_front} front)</h1>
<p>Exactly what the model receives during training (left: crop, right: labels drawn on it).
Numbers are <b>Universal</b>; <b style="color:#d4a000">yellow</b> = incisors and canines, white = premolars and molars.
Teeth less than 30% inside the crop are not labelled.</p>
<div class="grid">{''.join(cards)}</div></body></html>"""
    (out_dir / 'index.html').write_text(page, encoding='utf-8')
    print(f'wrote {out_dir / "index.html"}: {len(cards)} crops ({n_front} front)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--dentex-root', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--split', default='train')
    parser.add_argument('--n-images', type=int, default=12)
    parser.add_argument('--crops-per-image', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    main(args.dentex_root, args.out_dir, args.split, args.n_images, args.crops_per_image, args.seed)
