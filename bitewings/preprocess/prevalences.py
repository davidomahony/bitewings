import argparse
from collections import defaultdict
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from bitewings import paths

import numpy as np
from pycocotools.coco import COCO
import pycocotools.mask as maskUtils


def coco_to_rle(ann, h, w):    
    if isinstance(ann, list):
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(ann, h, w)
        return maskUtils.merge(rles)
    
    if isinstance(ann['counts'], list):
        # uncompressed RLE
        return maskUtils.frPyObjects(ann, h, w)
    
    # rle
    return ann


def process_dataset(coco: COCO):
    cat_id2name = {cat['id']: cat['name'] for cat in coco.cats.values()}
    img_stats, tooth_stats = defaultdict(int), defaultdict(int)
    img_cats = []
    img_calculus = []
    img_overlaps = []
    for img_dict in coco.imgs.values():
        fdi2anns, fdi2cats = defaultdict(list), defaultdict(set)
        img_calculus.append(0)
        img_overlaps.append(np.zeros((img_dict['height'], img_dict['width'])))
        for ann in coco.imgToAnns[img_dict['id']]:
            cat_name = cat_id2name[ann['category_id']]
            if 'calculus' in cat_name:
                img_calculus[-1] = max(img_calculus[-1], ann['area'])
            fdi2cats[cat_name[-2:]].add(cat_name[:-3])
            fdi2anns[cat_name[-2:]].append(ann)
            rle = coco_to_rle(ann['segmentation'], img_dict['height'], img_dict['width'])
            img_overlaps[-1] += maskUtils.decode(rle)


        for fdi, cats in fdi2cats.items():
            for cat in cats:
                tooth_stats[cat] += 1
            
            rles = [
                coco_to_rle(ann['segmentation'], img_dict['height'], img_dict['width'])
                for ann in fdi2anns[fdi]
            ]
            ious = maskUtils.iou(rles, rles, [0]*len(rles))
            for idx1, idx2 in zip(*np.nonzero(ious > 0)):
                if idx1 == idx2:
                    continue

                cat_name1 = cat_id2name[fdi2anns[fdi][idx1]['category_id']]
                cat_name2 = cat_id2name[fdi2anns[fdi][idx2]['category_id']]

                tooth_stats[f'{cat_name1[:-3]}_{cat_name2[:-3]}'] += 1

        cats = set([cat for cats in fdi2cats.values() for cat in cats])
        img_cats.append(cats)
        for cat in cats:
            img_stats[cat] += 1

    keys = sorted(tooth_stats.keys())
    for key in keys:
        print(f'{key}: {tooth_stats[key]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Print tooth-finding prevalences.')
    parser.add_argument(
        '--data-root', type=Path, default=paths.data_root(),
        help='Folder with annotations_fdi.json (default: $BITEWINGS_DATA_ROOT '
             f'or {paths.DEFAULT_DATA_ROOT}).',
    )
    root = parser.parse_args().data_root

    coco = COCO(root / 'annotations_fdi.json')
    process_dataset(coco)
