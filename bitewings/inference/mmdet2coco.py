import argparse
import copy
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List

import cv2
import numpy as np
import pickle
import pycocotools.mask as maskUtils
from scipy import ndimage
import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[2]))

from bitewings import paths



def rle_to_coco(annotation: dict) -> list[dict]:
    """Transform the rle coco annotation (a single one) into coco style.
    In this case, one mask can contain several polygons, later leading to several `Annotation` objects.
    In case of not having a valid polygon (the mask is a single pixel) it will be an empty list.
    Parameters
    ----------
    annotation : dict
        rle coco style annotation
    Returns
    -------
    list[dict]
        list of coco style annotations (in dict format)
    """
    maskedArr = maskUtils.decode(annotation["segmentation"])
    maskedArr = ndimage.binary_closing(maskedArr, ndimage.generate_binary_structure(2, 2))
    contours, _ = cv2.findContours(maskedArr.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []

    for contour in contours:
        if contour.size >= 6:
            segmentation.append(contour)

    if len(segmentation) == 0:
        logging.debug(
            f"Annotation with id {annotation['id']} is not valid, it has no segmentations."
        )
        return []

    else:
        annotation = copy.deepcopy(annotation)
        annotation['segmentation'] = [
            seg.astype(float).flatten().tolist() for seg in segmentation
        ]
        return [annotation]


def mask_decode(
    mask: Dict[str, Any]
):
    return maskUtils.decode(mask)


def load_results(
    root: Path,
    name: str,        
):
    results_path = root / f'{name}.pkl'
    with open(results_path, 'rb') as f:
        results = pickle.load(f)

    return [results]


def aggregate_results(
    image_results,
):
    bboxes = torch.cat(
        [r['pred_instances']['bboxes'] for r in image_results],
    )
    labels = torch.cat(
        [r['pred_instances']['labels'] for r in image_results],
    )
    scores = torch.cat(
        [r['pred_instances']['scores'] for r in image_results],
    )

    masks = [
        mask for r in image_results
        for mask in r['pred_instances']['masks']
    ]

    image_result = {
        'img_id': image_results[0]['img_id'],
        'img_path': image_results[0]['img_path'],
        'height': image_results[0]['ori_shape'][0],
        'width': image_results[0]['ori_shape'][1],
        'bboxes': bboxes,
        'labels': labels,
        'scores': scores,
        'masks': masks,
    }

    return image_result


def convert_to_coco_maskdino(
    result,
    keep_idxs,
    coco_dict,
    classes,
):
    coco_dict['images'].append({
        'id': result['img_id'],
        'file_name': result['img_path'].name,
        'height': result['height'],
        'width': result['width'],
    })
    catname2id = {cat['name']: cat['id'] for cat in coco_dict['categories']}

    class_names = [
        'tooth', 'implants', 'crowns', 'pontic', 'fillings', 'roots', 'caries', 'calculus',
    ]
    for idx in keep_idxs:
        rles = result['masks'][idx]
        label = classes[result['labels'][idx].item()]

        for rle, attr in zip(rles, class_names):
            if attr == 'tooth':
                rle = maskUtils.merge(rles[:-1])
            
            area = maskUtils.area(rle)
            if area == 0.0:
                continue
            wolla = f'{attr}_{label[-2:]}'
            ann = {
                'id': len(coco_dict['annotations']) + 1,
                'image_id': result['img_id'],
                'score': result['scores'][idx].item(),
                'category_id': catname2id[wolla],
                'segmentation': {
                    'size': rle['size'],
                    'counts': rle['counts'].decode(),
                },
                'area': maskUtils.area(rle).item(),
                'bbox': maskUtils.toBbox(rle).tolist(),
                'iscrowd': 0,
            }
            coco_dict['annotations'].extend(rle_to_coco(ann))


def convert_to_coco_maskrcnn(
    result,
    keep_idxs,
    coco_dict,
    classes,
):
    coco_dict['images'].append({
        'id': result['img_id'],
        'file_name': Path(result['img_path']).name,
        'height': result['height'],
        'width': result['width'],
    })
    catname2id = {cat['name']: cat['id'] for cat in coco_dict['categories']}

    for idx in keep_idxs:
        rle = result['masks'][idx]
        label = classes[result['labels'][idx].item()]
        

        coco_dict['annotations'].append({
            'id': len(coco_dict['annotations']) + 1,
            'image_id': result['img_id'],
            'score': result['scores'][idx].item(),
            'category_id': catname2id[label],
            'segmentation': {
                'size': rle['size'],
                'counts': rle['counts'].decode(),
            },
            'area': maskUtils.area(rle).item(),
            'bbox': maskUtils.toBbox(rle).tolist(),
            'iscrowd': 0,
        })


def filter_results(
    image_results,
    score_thr: float,
):
    result = aggregate_results(image_results)
    keep = result['scores'] >= score_thr
    keep_idxs = torch.nonzero(keep)[:, 0]

    return result, keep_idxs


def ensemble_detections(
    root: Path,
    name: str,
    classes: List[str],
    score_thr: float=0.1,
):
    fold_results = load_results(root, name)

    coco_dict = {
        'images': [],
        'annotations': [],
        'categories': [
            {'id': i, 'name': label}
            for i, label in enumerate(classes)
        ],
    }
    t = tqdm(
        iterable=zip(*fold_results),
        total=len(fold_results[0]),
    )
    for image_results in t:

        result, keep_idxs = filter_results(
            image_results,
            score_thr=score_thr,
        )
        t.set_description(result['img_path'].name)

        convert_to_coco_maskdino(result, keep_idxs, coco_dict, classes)
        # convert_to_coco_maskrcnn(result, keep_idxs, coco_dict, classes)

    with open(root / 'pred.json', 'w') as f:
        json.dump(coco_dict, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'method', choices=['hierarchical', 'maskdino', 'maskrcnn', 'sparseinst'],
        help='Method that has done predictions to convert to COCO format.',
    )
    parser.add_argument(
        '--work-dirs', type=Path, default=paths.work_dirs(),
        help='Folder with the chart_filing_<method> run folders (default: $BITEWINGS_WORK_DIRS '
             f'or {paths.DEFAULT_WORK_DIRS}).',
    )
    args = parser.parse_args()

    root = paths.method_work_dir(args.method, args.work_dirs)
    classes = [
        'tooth', 'implants', 'crowns', 'pontic', 'fillings', 'roots', 'caries', 'calculus',
    ]
    fdis = [
        '11', '12', '13', '14', '15', '16', '17', '18',
        '21', '22', '23', '24', '25', '26', '27', '28',
        '31', '32', '33', '34', '35', '36', '37', '38',
        '41', '42', '43', '44', '45', '46', '47', '48',
    ]
    classes = [f'{label}_{fdi}' for label in classes for fdi in fdis]
    ensemble_detections(
        root, name='detections', classes=classes,
    )
