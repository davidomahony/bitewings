"""Cut a fixed, reproducible evaluation set of periapical-style crops from DENTEX panoramics.

Training uses random CropPeriapical windows; evaluating on random windows would give a
different score every run. This script cuts a fixed number of front- and back-tooth
windows per held-out panoramic once (seeded) and saves them as images with ground truth:

    <out-dir>/images/*.png          the crops
    <out-dir>/images/coco.json      images only, for config_inference.py (BITEWINGS_IMAGES=<out-dir>/images)
    <out-dir>/annotations_gt.json   tooth_<FDI> masks (RLE) for bitewings/evaluation/numbering.py

    python bitewings/dentex/make_eval_crops.py --dentex-root <quadrant_enumeration> --out-dir <dir>
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pycocotools.mask as maskUtils

sys.path.append(str(Path(__file__).resolve().parents[2]))
sys.path.append(str(Path(__file__).resolve().parents[2] / 'onedl-mmdetection'))

from mmengine.registry import init_default_scope  # noqa: E402
import projects.DENTEX.datasets  # noqa: F401,E402
import projects.DENTEX.datasets.transforms.loading  # noqa: F401,E402
from projects.DENTEX.datasets.coco_multiclass import CocoMulticlassDataset  # noqa: E402
from bitewings.data.periapical import CropPeriapical  # noqa: E402

init_default_scope('mmdet')
FDIS = [q * 10 + n for q in (1, 2, 3, 4) for n in range(1, 9)]
ATTRIBUTES = ['implants', 'crowns', 'pontic', 'fillings', 'roots', 'caries', 'calculus']


def main(dentex_root: Path, out_dir: Path, split: str, front_crops: int, back_crops: int, seed: int):
    np.random.seed(seed)
    (out_dir / 'images').mkdir(parents=True, exist_ok=True)
    dataset = CocoMulticlassDataset(
        ann_file=str(dentex_root / 'splits' / f'{split}_fdi.json'),
        data_root=str(dentex_root),
        data_prefix=dict(img=str(dentex_root / 'xrays')),
        metainfo=dict(classes=['tooth'], attributes=ATTRIBUTES),
        strict=True, decode_masks=False, merge_layers=True, num_workers=0, serialize_data=False,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadMulticlassAnnotations', merge_layers=True, with_bbox=True, with_mask=True),
        ],
    )
    croppers = {'front': CropPeriapical(front_prob=1.0), 'back': CropPeriapical(front_prob=0.0)}
    plan = [('front', k) for k in range(front_crops)] + [('back', k) for k in range(back_crops)]

    images, annotations = [], []
    categories = [{'id': i, 'name': f'tooth_{fdi}'} for i, fdi in enumerate(FDIS, 1)]
    for idx in range(len(dataset)):
        sample = dataset[idx]
        stem = Path(sample['img_path']).stem
        for kind, k in plan:
            crop = croppers[kind].transform({
                key: (val.copy() if hasattr(val, 'copy') else val) for key, val in sample.items()
            })
            if crop['img'].shape == sample['img'].shape:  # no valid window for this panoramic
                continue
            # CropPeriapical falls back to the other region when a jaw has no front/back teeth
            is_front = bool(len(crop['gt_bboxes_labels'])) and \
                ((crop['gt_bboxes_labels'] % 8) <= 2).mean() >= 0.5
            name = f'{stem}_{kind}{k}.png'
            cv2.imwrite(str(out_dir / 'images' / name), crop['img'])
            h, w = crop['img'].shape[:2]
            image_id = len(images) + 1
            images.append({'id': image_id, 'file_name': name, 'height': h, 'width': w,
                           'crop_type': 'front' if is_front else 'back', 'source': Path(sample['img_path']).name})
            for mask, label in zip(crop['gt_masks'].masks, crop['gt_bboxes_labels']):
                rle = maskUtils.encode(np.asfortranarray((mask != 0).astype(np.uint8)))
                annotations.append({
                    'id': len(annotations) + 1, 'image_id': image_id, 'category_id': int(label) + 1,
                    'segmentation': {'size': rle['size'], 'counts': rle['counts'].decode()},
                    'area': float(maskUtils.area(rle)), 'bbox': [float(v) for v in maskUtils.toBbox(rle)],
                    'iscrowd': 0, 'extra': {},
                })

    gt = {'images': images, 'annotations': annotations, 'categories': categories}
    (out_dir / 'annotations_gt.json').write_text(json.dumps(gt))
    (out_dir / 'images' / 'coco.json').write_text(json.dumps({'images': images, 'annotations': [], 'categories': []}))
    teeth = np.array([a['category_id'] - 1 for a in annotations])
    n_front = sum(img['crop_type'] == 'front' for img in images)
    print(f'{len(images)} crops ({n_front} front, {len(images) - n_front} back) from {len(dataset)} panoramics; '
          f'{len(annotations)} teeth ({int(((teeth % 8) <= 2).sum())} front teeth)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--dentex-root', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--split', default='test')
    parser.add_argument('--front-crops', type=int, default=2)
    parser.add_argument('--back-crops', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    main(args.dentex_root, args.out_dir, args.split, args.front_crops, args.back_crops, args.seed)
