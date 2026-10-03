import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pycocotools.mask as maskUtils

from mmengine.structures import InstanceData
from mmdet.visualization import DetLocalVisualizer
from mmdet.datasets import CocoDataset

sys.path.append(str(Path(__file__).resolve().parents[2]))
sys.path.append(str(Path(__file__).resolve().parents[2] / 'onedl-mmdetection'))

from bitewings import paths
from projects.DENTEX.visualization.local_visualizer import MulticlassDetLocalVisualizer


classes = [
    'tooth_11', 'tooth_12', 'tooth_13', 'tooth_14', 'tooth_15', 'tooth_16', 'tooth_17', 'tooth_18',
    'tooth_21', 'tooth_22', 'tooth_23', 'tooth_24', 'tooth_25', 'tooth_26', 'tooth_27', 'tooth_28',
    'tooth_31', 'tooth_32', 'tooth_33', 'tooth_34', 'tooth_35', 'tooth_36', 'tooth_37', 'tooth_38',
    'tooth_41', 'tooth_42', 'tooth_43', 'tooth_44', 'tooth_45', 'tooth_46', 'tooth_47', 'tooth_48',
    'implants', 'crowns', 'pontic', 'fillings', 'roots', 'caries', 'calculus_medial', 'calculus_distal',
]
attributes = [
    'implants', 'crowns', 'pontic', 'fillings', 'roots', 'caries', 'calculus'
]

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


def visualize_flat(coco_dict: dict):
    cat_id2name =  {cat['id']: cat['name'] for cat in coco_dict['categories']}

    img_dict = coco_dict['images'][0]

    labels, bboxes, masks = [], [], []
    for ann in coco_dict['annotations']:
        cat_name = cat_id2name[ann['category_id']].lower()

        if cat_name not in classes:
            continue

        bbox = np.array(ann['bbox'])
        rle = coco_to_rle(ann['segmentation'], img_dict['height'], img_dict['width'])
        mask = maskUtils.decode(rle)

        labels.append(classes.index(cat_name))
        bboxes.append(np.concatenate((bbox[:2], bbox[:2] + bbox[2:])))
        masks.append(mask)

    labels = np.array(labels)
    idxs = np.argsort(labels)
    instances = InstanceData()
    instances['labels'] = np.array(labels)[idxs]
    instances['bboxes'] = np.stack(bboxes)[idxs]
    instances['masks'] = np.stack(masks)[idxs]

    image = cv2.imread(str(root / 'images' / img_dict['file_name']))
    visualizer = DetLocalVisualizer()
    out = visualizer._draw_instances(
        image,
        instances,
        [label[-2:] if 'tooth' in label else label[:10] for label in classes],
        CocoDataset.METAINFO['palette'],
    )
    
    return image, out


def visualize_hierarchy(coco_dict: dict):
    cat_id2name =  {cat['id']: cat['name'] for cat in coco_dict['categories']}

    img_dict = coco_dict['images'][0]

    fdi2anns = defaultdict(list)
    for ann in coco_dict['annotations']:
        cat_name = cat_id2name[ann['category_id']]

        if cat_name[:-3].lower() != 'tooth' and cat_name[:-3].lower() not in classes:
            continue

        fdi = cat_name[-2:]
        fdi2anns[fdi].append(ann)

    labels, bboxes, masks = [], [], []
    for fdi, anns in fdi2anns.items():
        mask = np.zeros((img_dict['height'], img_dict['width']))
        for ann in anns:
            rle = coco_to_rle(ann['segmentation'], img_dict['height'], img_dict['width'])
            binary_mask = maskUtils.decode(rle)

            cat_name = cat_id2name[ann['category_id']][:-3].lower()
            if cat_name == 'tooth':
                bbox = np.array(ann['bbox'])
                mask = np.maximum(mask, binary_mask)
            elif 'calculus' in cat_name:
                mask = np.maximum(mask, binary_mask * 8)
            else:            
                mask = np.maximum(mask, binary_mask * (2 + attributes.index(cat_name)))

        labels.append(classes.index(f'tooth_{fdi}'))
        bboxes.append(np.concatenate((bbox[:2], bbox[:2] + bbox[2:])))
        masks.append(mask)

    instances = InstanceData()
    instances['labels'] = np.array(labels)
    instances['bboxes'] = np.stack(bboxes)
    instances['masks'] = np.stack(masks).astype(np.int64)

    image = cv2.imread(str(root / 'images' / img_dict['file_name']))
    visualizer = MulticlassDetLocalVisualizer()
    out = visualizer._draw_instances(
        image,
        instances,
        classes,
        CocoDataset.METAINFO['palette'],
    )
    
    return image, out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Figure comparing flat and hierarchical annotations of one bitewing.')
    parser.add_argument(
        '--data-root', type=Path, default=paths.data_root(),
        help='Folder with annotations.json and annotations_fdi.json (default: $BITEWINGS_DATA_ROOT '
             f'or {paths.DEFAULT_DATA_ROOT}).',
    )
    parser.add_argument(
        '--image', default=None,
        help='File name of the bitewing to show (default: the first image in the dataset; the paper '
             'used 8CFA4C19-D474-4C5F-9570-017FC90F274A.jpg from the unreleased German data).',
    )
    args = parser.parse_args()
    root = args.data_root
    with open(root / 'annotations.json', 'r') as f:
        annotation_example = args.image or json.load(f)['images'][0]['file_name']

    with open(root / 'annotations.json', 'r') as f:
        coco_dict = json.load(f)
    img_dict = [img for img in coco_dict['images'] if img['file_name'] == annotation_example][0]
    coco_dict['images'] = [img_dict]
    coco_dict['annotations'] = [ann for ann in coco_dict['annotations'] if ann['image_id'] == img_dict['id']]

    image, out1 = visualize_flat(coco_dict)
    cv2.imwrite('empty.png', image)
    cv2.imwrite('traditional.png', out1[..., [2, 1, 0]])

    with open(root / 'annotations_fdi.json', 'r') as f:
        coco_dict = json.load(f)
    img_dict = [img for img in coco_dict['images'] if img['file_name'] == annotation_example][0]
    coco_dict['images'] = [img_dict]
    coco_dict['annotations'] = [ann for ann in coco_dict['annotations'] if ann['image_id'] == img_dict['id']]

    image, out2 = visualize_hierarchy(coco_dict)
    cv2.imwrite('hierarchical.png', out2[..., [2, 1, 0]])

    _, axs = plt.subplots(1, 3, figsize=(18, 5))
    axs[0].imshow(image)
    axs[0].axis('off')
    axs[1].imshow(out1)
    axs[1].axis('off')
    axs[2].imshow(out2)
    axs[2].axis('off')

    plt.tight_layout()
    plt.show()
