"""Convert DenPAR periapicals to the hierarchical tooth format, with FDI numbers.

DenPAR (Scientific Data 2025, dataset CC BY 4.0, https://doi.org/10.5281/zenodo.16645076)
has 1,000 real intraoral periapicals with one binary mask per tooth ("Masks (Tooth-wise)")
and, in its metadata spreadsheet, the arch, site and FDI numbers of the visible teeth, but
not which mask is which tooth.

Masks are numbered here by sorting them left to right and pairing them with the listed FDI
numbers sorted from the patient's right to left (the standard frontal viewing orientation).
On 40 images this rule agreed with an independently trained model on 90% of side teeth,
against 40% for the spreadsheet's own order and 11% for the mirror image. Images whose mask
count differs from the number of listed teeth (partially visible or split teeth) cannot be
paired this way; they are left out and written to needs_review.csv for manual numbering.

DenPAR has no finding labels, so train on it with MarkPartialFindings (as with DENTEX).
DenPAR's own Training / Validation / Testing splits are kept, so its Testing images can serve
as a held-out set of real periapicals.

    python bitewings/denpar/convert.py --denpar-root <...>/Dataset

writes <denpar-root>/splits/{train,val,test}_fdi.json and <denpar-root>/needs_review.csv.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pycocotools.mask as maskUtils
from PIL import Image

FDIS = [q * 10 + n for q in (1, 2, 3, 4) for n in range(1, 9)]
SPLITS = {'Training': 'train', 'Validation': 'val', 'Testing': 'test'}
METADATA_XLSX = 'Characteristics of radiographs included.xlsx'
METADATA_CSV = 'characteristics.csv'


def anatomical_position(fdi: int) -> int:
    """Position from the patient's right to left within one jaw (1 = most patient-right)."""
    quadrant, number = divmod(fdi, 10)
    return 9 - number if quadrant in (1, 4) else 8 + number


def load_metadata(root: Path) -> dict:
    """{image id: (arch, site, [FDI, ...])}. Reads the CSV export if present, else the xlsx."""
    if (root / METADATA_CSV).exists():
        rows = list(csv.DictReader(open(root / METADATA_CSV, encoding='utf-8')))
    else:
        import pandas as pd  # needs openpyxl
        rows = pd.read_excel(root / METADATA_XLSX).astype(str).to_dict('records')
    meta = {}
    for row in rows:
        fdis = [int(t) for t in str(row['FDI notation of fully/partially visible teeth']).split(',')
                if t.strip().isdigit()]
        meta[int(float(row['id']))] = (row['Arch'].strip(), row['Site'].strip(), fdis)
    return meta


def tooth_masks(mask_dir: Path, height: int, width: int):
    """[(x centroid, RLE)] for every non-empty tooth mask, or None if a mask has the wrong size."""
    masks = []
    for path in sorted(mask_dir.glob('*.png')):
        mask = np.array(Image.open(path))
        if mask.ndim == 3:
            mask = mask[..., 0]
        if mask.shape != (height, width):
            return None
        mask = mask > 0
        if not mask.any():
            continue
        rle = maskUtils.encode(np.asfortranarray(mask.astype(np.uint8)))
        masks.append((float(np.nonzero(mask)[1].mean()), rle))
    return masks


def main(root: Path):
    meta = load_metadata(root)
    categories = [{'id': i, 'name': f'tooth_{fdi}'} for i, fdi in enumerate(FDIS, 1)]
    cat_ids = {fdi: i for i, fdi in enumerate(FDIS, 1)}
    review = []
    (root / 'splits').mkdir(exist_ok=True)

    for folder, split in SPLITS.items():
        images, annotations = [], []
        counts = {'Anterior': 0, 'Left': 0, 'Right': 0}
        for img_path in sorted((root / folder / 'Images').glob('*.jpg'), key=lambda p: int(p.stem)):
            image_id = int(img_path.stem)
            arch, site, fdis = meta[image_id]
            width, height = Image.open(img_path).size
            masks = tooth_masks(root / folder / 'Masks (Tooth-wise)' / str(image_id), height, width)
            reason = None
            if masks is None:
                reason = 'mask size differs from image'
            elif not fdis:
                reason = 'no FDI numbers listed'
            elif any(fdi not in cat_ids for fdi in fdis):
                # FDI 51-85 are primary (baby) teeth; the model only has the 32 permanent classes
                reason = 'primary teeth listed: ' + ' '.join(str(f) for f in fdis if f not in cat_ids)
            elif len(masks) != len(fdis):
                reason = f'{len(masks)} masks but {len(fdis)} listed teeth'
            elif len(set(fdis)) != len(fdis):
                reason = 'duplicate FDI numbers listed'
            if reason:
                review.append({'id': image_id, 'split': split, 'arch': arch, 'site': site,
                               'listed_fdi': ' '.join(map(str, fdis)), 'reason': reason})
                continue

            images.append({'id': image_id, 'file_name': f'{folder}/Images/{img_path.name}',
                           'height': height, 'width': width, 'arch': arch, 'site': site})
            counts[site] = counts.get(site, 0) + 1
            masks.sort(key=lambda m: m[0])
            for (_, rle), fdi in zip(masks, sorted(fdis, key=anatomical_position)):
                annotations.append({
                    'id': len(annotations) + 1, 'image_id': image_id, 'category_id': cat_ids[fdi],
                    'segmentation': {'size': rle['size'], 'counts': rle['counts'].decode()},
                    'area': float(maskUtils.area(rle)), 'bbox': [float(v) for v in maskUtils.toBbox(rle)],
                    'iscrowd': 0, 'extra': {},
                })
        out = root / 'splits' / f'{split}_fdi.json'
        out.write_text(json.dumps({'images': images, 'annotations': annotations, 'categories': categories}))
        front = sum(1 for a in annotations if FDIS[a['category_id'] - 1] % 10 <= 3)
        print(f'{split:5s}: {len(images)} images ({counts["Anterior"]} anterior, '
              f'{counts["Left"] + counts["Right"]} side), {len(annotations)} teeth ({front} front)')

    with open(root / 'needs_review.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'split', 'arch', 'site', 'listed_fdi', 'reason'])
        writer.writeheader()
        writer.writerows(review)
    anterior = sum(1 for r in review if r['site'] == 'Anterior')
    print(f'needs review: {len(review)} images ({anterior} anterior) -> {root / "needs_review.csv"}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--denpar-root', type=Path, required=True,
                        help='DenPAR "Dataset" folder with Training/, Validation/, Testing/ and the metadata file.')
    main(parser.parse_args().denpar_root)
