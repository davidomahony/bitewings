import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
from typing import List

import numpy as np
from pycocotools.coco import COCO
from sklearn.model_selection import KFold
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[2]))

from bitewings import paths


def combine_train_val(
    root: Path,
    out_dir: Path,
):
    (out_dir / 'images').mkdir(parents=True, exist_ok=True)

    out_dict = {
        'images': [],
        'annotations': [],
        'categories': [],
    }
    for split in ['train', 'val']:
        split_dir = root / split
        coco = COCO(split_dir / f'{split}.json')

        out_dict['categories'] = coco.dataset['categories']
        for img_id, img_dict in enumerate(coco.imgs.values(), len(out_dict['images']) + 1):
            for ann_dict in coco.imgToAnns[img_dict['id']]:
                ann_dict['image_id'] = img_id

            rel_path = (split_dir / 'images' / img_dict['file_name']).relative_to(root)
            img_dict['file_name'] = rel_path.as_posix()
            img_dict['id'] = img_id

            out_dict['images'].append(img_dict)
            out_dict['annotations'].extend(coco.imgToAnns[img_dict['id']])

    for ann_id, ann_dict in enumerate(out_dict['annotations'], 1):
        ann_dict['id'] = ann_id    

    with open(out_dir / 'bitewings.json', 'w') as f:
        json.dump(out_dict, f, indent=2)

    return COCO(out_dir / 'bitewings.json')


def is_bitewing_suitable(
    coco: COCO,
    catid2name: dict,
    img_id: int,
):
    anns = coco.imgToAnns[img_id]
    fdis = np.array([int(catid2name[ann['category_id']][-2:]) for ann in anns])

    if np.any(fdis > 50):
        return False

    fdis = fdis[fdis > 10]
    fdis = fdis[(fdis % 10) >= 3]
    _, counts = np.unique(fdis // 10, return_counts=True)

    if counts.shape[0] != 4:
        return False

    return counts.min() >= 3


def select_bitewings(root: Path, coco: COCO, out_dir: Path):
    catid2name = {cat['id']: cat['name'] for cat in coco.cats.values()}

    coco_dict = {
        'images': [],
        'annotations': [],
        'categories': coco.dataset['categories'],
    }
    for img_id, img_dict in coco.imgs.items():
        if not is_bitewing_suitable(coco, catid2name, img_id):
            continue

        shutil.copy(
            root / img_dict['file_name'],
            out_dir / 'images' / img_dict['file_name'].split('/')[-1],
        )

        img_dict = copy.deepcopy(img_dict)
        img_dict['file_name'] = img_dict['file_name'].split('/')[-1]
        coco_dict['images'].append(img_dict)
        coco_dict['annotations'].extend(coco.imgToAnns[img_id])

    with open(out_dir / 'bitewings.json', 'w') as f:
        json.dump(coco_dict, f, indent=2)

    return COCO(out_dir / 'bitewings.json')


def copy_annotations(
    coco: COCO,
    files: List[Path],
    split,
):
    if split is None:
        train_idxs = []
        val_idxs = range(len(files))
    else:
        train_idxs, val_idxs = split
    train_stems = np.array([Path(files[idx]).stem for idx in train_idxs])
    val_stems = np.array([Path(files[idx]).stem for idx in val_idxs])

    id2stem = {img['id']: Path(img['file_name']).stem for img in coco.dataset['images']}
    img_stems = np.array(list(id2stem.values()))
    ann_stems = np.array([id2stem[ann['image_id']] for ann in coco.dataset['annotations']])

    train_coco_dict = {
        'categories': coco.dataset['categories'],
    }
    val_coco_dict = copy.deepcopy(train_coco_dict)

    mask = np.any(img_stems[np.newaxis] == train_stems[:, np.newaxis], axis=0)
    train_coco_dict['images'] = [img for img, b in zip(coco.dataset['images'], mask) if b]
    mask = np.any(img_stems[np.newaxis] == val_stems[:, np.newaxis], axis=0)
    val_coco_dict['images'] = [img for img, b in zip(coco.dataset['images'], mask) if b]

    mask = np.any(ann_stems[np.newaxis] == train_stems[:, np.newaxis], axis=0)
    train_coco_dict['annotations'] = [img for img, b in zip(coco.dataset['annotations'], mask) if b]
    mask = np.any(ann_stems[np.newaxis] == val_stems[:, np.newaxis], axis=0)
    val_coco_dict['annotations'] = [img for img, b in zip(coco.dataset['annotations'], mask) if b]

    return train_coco_dict, val_coco_dict


def test_split(
    ann_dir: Path,
    files: List[Path],
    test_size: float,
):
    splitter = KFold(
        n_splits=int(1 / test_size),
        shuffle=True,
        random_state=1234,
    )
    split = next(splitter.split(files, files))
    _, test_coco = copy_annotations(coco, files, split)

    with open(ann_dir / 'test.json', 'w') as f:
        json.dump(test_coco, f, indent=2)

    files = [files[idx] for idx in split[0]]

    return files


def kfold_split(
    root: Path,
    coco: COCO,
    n_splits: int,
    files: List[Path],
):
    splits = [None]
    if n_splits > 1:
        splitter = KFold(
            n_splits, shuffle=True, random_state=1234,
        )
        splits.extend(list(splitter.split(files, files)))

    out_dir = root / 'splits'
    out_dir.mkdir(exist_ok=True)

    for i, split in enumerate(tqdm(splits)):
        train_coco, val_coco = copy_annotations(coco, files, split)

        if split is None:
            with open(out_dir / f'trainval.json', 'w') as f:
                json.dump(val_coco, f, indent=2)
            continue

        with open(out_dir / f'train_{i}.json', 'w') as f:
            json.dump(train_coco, f, indent=2)

        with open(out_dir / f'val_{i}.json', 'w') as f:
            json.dump(val_coco, f, indent=2) 


def split(
    root: Path,
    coco: COCO,
    n_splits: int,
    test_size: float=0.0,
):
    files = sorted([img['file_name'] for img in coco.dataset['images']])

    if test_size > 0:
        files = test_split(root, files, test_size)

    kfold_split(root, coco, n_splits, files)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Crop OdontoAI panoramics into bitewing-like images.')
    parser.add_argument(
        '--odonto-root', type=Path, default=paths.odonto_root(),
        help='Folder with the OdontoAI data (default: $BITEWINGS_ODONTO_ROOT '
             f'or {paths.DEFAULT_ODONTO_ROOT}). Output goes to <odonto-root>/bitewings.',
    )
    root = parser.parse_args().odonto_root
    out_dir = root / 'bitewings'

    coco = combine_train_val(root, out_dir)
    coco = select_bitewings(root, coco, out_dir)
    
    split(
        out_dir,
        coco,
        n_splits=5,
        test_size=0.0,
    )
