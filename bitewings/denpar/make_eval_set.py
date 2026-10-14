"""Build the held-out real-periapical evaluation set from DenPAR's validation and test splits.

Training uses only DenPAR's train split (bitewings/denpar/convert.py), so these images are
never trained on. The output has the same layout as the DENTEX evaluation crops, so it is
scored the same way:

    <out-dir>/images/*.jpg          the periapicals
    <out-dir>/images/coco.json      images only, for config_inference.py (BITEWINGS_IMAGES=<out-dir>/images)
    <out-dir>/annotations_gt.json   FDI-numbered tooth masks for bitewings/evaluation/numbering.py

Each image's crop_type is "front" (DenPAR site Anterior) or "back" (Left / Right), so the
numbering report splits results by image type as well as by tooth.

    python bitewings/denpar/make_eval_set.py --denpar-root <...>/Dataset --out-dir <dir>
"""
import argparse
import json
import shutil
from pathlib import Path


def main(root: Path, out_dir: Path, splits):
    (out_dir / 'images').mkdir(parents=True, exist_ok=True)
    images, annotations, categories = [], [], None
    for split in splits:
        coco = json.loads((root / 'splits' / f'{split}_fdi.json').read_text())
        categories = coco['categories']
        id_map = {}
        for img in coco['images']:
            name = Path(img['file_name']).name
            shutil.copy(root / img['file_name'], out_dir / 'images' / name)
            new_id = len(images) + 1
            id_map[img['id']] = new_id
            images.append({'id': new_id, 'file_name': name, 'height': img['height'], 'width': img['width'],
                           'crop_type': 'front' if img['site'] == 'Anterior' else 'back',
                           'source': f'DenPAR {split} {img["id"]}', 'arch': img['arch'], 'site': img['site']})
        for ann in coco['annotations']:
            annotations.append({**ann, 'id': len(annotations) + 1, 'image_id': id_map[ann['image_id']]})

    (out_dir / 'annotations_gt.json').write_text(
        json.dumps({'images': images, 'annotations': annotations, 'categories': categories}))
    (out_dir / 'images' / 'coco.json').write_text(
        json.dumps({'images': images, 'annotations': [], 'categories': []}))
    fdi_of = {c['id']: int(c['name'][-2:]) for c in categories}
    front_teeth = sum(1 for a in annotations if fdi_of[a['category_id']] % 10 <= 3)
    front_images = sum(1 for i in images if i['crop_type'] == 'front')
    print(f'{len(images)} real periapicals ({front_images} anterior) from {", ".join(splits)}; '
          f'{len(annotations)} teeth ({front_teeth} front)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--denpar-root', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--splits', nargs='+', default=['val', 'test'])
    args = parser.parse_args()
    main(args.denpar_root, args.out_dir, args.splits)
