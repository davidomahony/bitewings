"""Convert the DENTEX quadrant-enumeration panoramics to the hierarchical tooth format.

DENTEX (MICCAI 2023, CC BY-NC-SA 4.0, https://huggingface.co/datasets/ibrahimhamamci/DENTEX)
labels every tooth of 634 panoramics with a quadrant (category_id_1) and a tooth
number within the quadrant (category_id_2). This script writes the same tooth
categories as bitewings/preprocess/preprocess.py (``tooth_<FDI>``), so the
panoramics can be loaded by CocoMulticlassDataset next to the Radboud bitewings.

DENTEX does not label fillings, crowns, root canals or caries in this set, so
these images only teach tooth segmentation and numbering.

    python bitewings/dentex/convert.py --dentex-root <training_data>/quadrant_enumeration

writes <dentex-root>/annotations_fdi.json and <dentex-root>/splits/{train,val,test}_fdi.json.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import pycocotools.mask as maskUtils

FDIS = [q * 10 + n for q in (1, 2, 3, 4) for n in range(1, 9)]


def convert(dentex_json: Path) -> dict:
    src = json.loads(dentex_json.read_text())
    quadrant = {c['id']: int(c['name']) for c in src['categories_1']}
    number = {c['id']: int(c['name']) for c in src['categories_2']}

    categories = [{'id': i, 'name': f'tooth_{fdi}'} for i, fdi in enumerate(FDIS, 1)]
    cat_ids = {fdi: i for i, fdi in enumerate(FDIS, 1)}
    images = {img['id']: img for img in src['images']}

    annotations, skipped = [], 0
    for ann in src['annotations']:
        img = images[ann['image_id']]
        polygons = [p for p in ann['segmentation'] if len(p) >= 6]
        if not polygons:
            skipped += 1
            continue
        rle = maskUtils.merge(maskUtils.frPyObjects(polygons, img['height'], img['width']))
        area = float(maskUtils.area(rle))
        if area <= 0:
            skipped += 1
            continue
        fdi = quadrant[ann['category_id_1']] * 10 + number[ann['category_id_2']]
        annotations.append({
            'id': len(annotations) + 1,
            'image_id': ann['image_id'],
            'category_id': cat_ids[fdi],
            'segmentation': polygons,
            'area': area,
            'bbox': [float(v) for v in maskUtils.toBbox(rle)],
            'iscrowd': 0,
            'extra': {},
        })

    print(f'{len(images)} images, {len(annotations)} teeth, {skipped} empty outlines skipped')
    return {
        'images': [
            {k: img[k] for k in ('id', 'file_name', 'height', 'width')} for img in src['images']
        ],
        'annotations': annotations,
        'categories': categories,
    }


def split(coco: dict, val_fraction: float, test_fraction: float, seed: int) -> dict:
    """Split by image. Every DENTEX panoramic is a different patient visit."""
    image_ids = sorted(img['id'] for img in coco['images'])
    random.Random(seed).shuffle(image_ids)
    n_test = round(len(image_ids) * test_fraction)
    n_val = round(len(image_ids) * val_fraction)
    parts = {
        'test': set(image_ids[:n_test]),
        'val': set(image_ids[n_test:n_test + n_val]),
        'train': set(image_ids[n_test + n_val:]),
    }
    return {
        name: {
            'images': [img for img in coco['images'] if img['id'] in ids],
            'annotations': [ann for ann in coco['annotations'] if ann['image_id'] in ids],
            'categories': coco['categories'],
        }
        for name, ids in parts.items()
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument(
        '--dentex-root', type=Path, required=True,
        help='DENTEX quadrant_enumeration folder with xrays/ and train_quadrant_enumeration.json.',
    )
    parser.add_argument('--val-fraction', type=float, default=0.1)
    parser.add_argument('--test-fraction', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    coco = convert(args.dentex_root / 'train_quadrant_enumeration.json')
    (args.dentex_root / 'annotations_fdi.json').write_text(json.dumps(coco))

    (args.dentex_root / 'splits').mkdir(exist_ok=True)
    for name, part in split(coco, args.val_fraction, args.test_fraction, args.seed).items():
        out = args.dentex_root / 'splits' / f'{name}_fdi.json'
        out.write_text(json.dumps(part))
        teeth = np.array([ann['category_id'] for ann in part['annotations']])
        front = int(np.isin(teeth, [i for i, fdi in enumerate(FDIS, 1) if fdi % 10 <= 3]).sum())
        print(f'{name}: {len(part["images"])} images, {len(part["annotations"])} teeth ({front} front)')
