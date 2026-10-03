import argparse
from pathlib import Path
import pickle
import sys

import cv2
from mmengine.structures import InstanceData
import numpy as np
import pycocotools.mask as maskUtils
from scipy import ndimage
import torch
from tqdm import tqdm

from mmdet.visualization import DetLocalVisualizer

sys.path.append(str(Path(__file__).resolve().parents[2]))

from bitewings import paths


palette = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
         (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
         (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
         (165, 42, 42), (255, 77, 255), (0, 226, 252), (182, 182, 255),
         (0, 82, 0), (120, 166, 157), (110, 76, 0), (174, 57, 255),
         (199, 100, 0), (72, 0, 118), (255, 179, 240), (0, 125, 92),
         (209, 0, 151), (188, 208, 182), (0, 220, 176), (255, 99, 164),
         (92, 0, 73), (133, 129, 255), (78, 180, 255), (0, 228, 0),
         (134, 134, 103), (174, 255, 243), (45, 89, 255), (145, 148, 174),
         (255, 208, 186)]


def process_teeth(
    teeth,
    thresholds: list[float]=[0.0891, 0.3153, 0.4204, 0.1642, 0.1862, 0.0360, 0.0951],
):
    multi_masks = []
    for rles in teeth['masks']:
        multi_mask = 0
        for i, rle in enumerate(rles, 1):
            multi_mask += i * maskUtils.decode(rle)

        multi_masks.append(multi_mask)

    if 'multilogits' in teeth:
        multi_scores = teeth['multilogits'][:, 1:]
    else:
        multi_scores = torch.ones(teeth['labels'].shape[0], 7)

    if 'scores' in teeth:
        keep = teeth['scores'] >= 0.3
        teeth['labels'] = teeth['labels'][keep]
        multi_scores = multi_scores[keep]
        multi_masks = [multi_masks[i] for i in torch.nonzero(keep)[:, 0]]

    labels, bboxes, masks = [], [], []
    for tooth_label, multi_mask, multi_score in zip(teeth['labels'], multi_masks, multi_scores):
        tooth_mask = (multi_mask > 0) & (multi_mask < 8)
        tooth_mask = multi_mask == 1
        tooth_bbox = maskUtils.toBbox(maskUtils.encode(np.asfortranarray(tooth_mask)))

        for i, thr in enumerate(thresholds):
            mask = multi_mask == (i + 2)
            if mask.sum() == 0:
                continue

            if multi_score[i] < thr:
                if i != 6:
                    tooth_mask[mask] = True
                continue


            label_map, max_label = ndimage.label(mask)
            for label in range(1, max_label + 1):
                component_mask = label_map == label

                if component_mask.sum() < 10:
                    continue

                bbox = maskUtils.toBbox(maskUtils.encode(np.asfortranarray(component_mask)))

                labels.append(32 + i)
                bboxes.append([
                    bbox[0],
                    bbox[1],
                    bbox[0] + bbox[2],
                    bbox[1] + bbox[3],
                ])
                masks.append(component_mask)
            

        labels.append(tooth_label.item())
        bboxes.append([
            tooth_bbox[0],
            tooth_bbox[1],
            tooth_bbox[0] + tooth_bbox[2],
            tooth_bbox[1] + tooth_bbox[3],
        ])
        masks.append(tooth_mask)

    out = {
        'labels': torch.tensor(labels),
        'bboxes': torch.tensor(bboxes),
        'masks': [maskUtils.encode(np.asfortranarray(mask)) for mask in masks],
    }
    
    return out


def filter_teeth(instances: dict, thresholds: list[float]):
    labels = instances['labels']
    bboxes = instances['bboxes']
    pred_scores = instances['scores'] if 'scores' in instances else torch.ones_like(instances['labels'])
    rles = instances['masks']

    if torch.all(bboxes == 0):
        bboxes = torch.tensor([maskUtils.toBbox(rle).tolist() for rle in rles])
        bboxes[:, 2:] = bboxes[:, :2] + bboxes[:, 2:]


    tooth_labels = labels[labels < 32]
    tooth_bboxes = bboxes[labels < 32]
    tooth_rles = [rles[i] for i in torch.nonzero(labels < 32)[:, 0]]
    tooth_areas = np.array([maskUtils.area(rle) for rle in tooth_rles])
    
    finding_bboxes = bboxes[labels >= 32]
    finding_rles = [rles[i] for i in torch.nonzero(labels >= 32)[:, 0]]
    finding_areas = np.array([maskUtils.area(rle) for rle in finding_rles])
    finding_scores = pred_scores[labels >= 32]

    # make calculus intersect with corresponding tooth
    finding_labels = labels[labels >= 32]
    finding_masks = [maskUtils.decode(rle) for rle in finding_rles]
    finding_roll_masks = [
        np.roll(mask, -50 + 100 * (label == 38), axis=1) if label >= 38 else mask
        for mask, label in zip(finding_masks, finding_labels)
    ]
    finding_roll_rles = [maskUtils.encode(np.asfortranarray(mask)) for mask in finding_roll_masks]
    finding_labels = torch.clip(finding_labels - 32, 0, 6)

    # determine the matched teeth and tooth findings
    ious = maskUtils.iou(tooth_rles, finding_roll_rles, [0]*len(finding_rles))
    overlaps = ious / (finding_areas / tooth_areas[:, None])
    for i, overlap_scores in enumerate(overlaps):
        if not np.any(overlap_scores >= 0.5):
            continue

        tooth_mask = maskUtils.decode(tooth_rles[i])
        finding_idxs = np.nonzero(overlap_scores >= 0.5)[0]
        for idx in finding_idxs:
            if finding_scores[idx] < thresholds[finding_labels[idx]]:
                continue
            # if finding_scores[idx] < 0.3:
            #     continue

            tooth_mask[finding_masks[idx] == 1] = 0

        tooth_rles[i] = maskUtils.encode(np.asfortranarray(tooth_mask))

    tooth_keep = pred_scores[labels < 32] >= 0.3
    finding_keep = finding_scores >= thresholds[finding_labels]
    # finding_keep = finding_scores >= 0.3
    instances['labels'] = torch.cat((tooth_labels[tooth_keep], 32 + finding_labels[finding_keep]))
    instances['bboxes'] = torch.cat((tooth_bboxes[tooth_keep], finding_bboxes[finding_keep]))
    instances['masks'] = (
        [tooth_rles[i] for i in torch.nonzero(tooth_keep)[:, 0]]
        + [finding_rles[i] for i in torch.nonzero(finding_keep)[:, 0]]
    )

    return instances


def process_instances(visualizer, image, instances):
    masks = np.stack([maskUtils.decode(rle) for rle in instances['masks']])

    idxs = np.argsort(instances['labels'])
    out = InstanceData()
    out['labels'] = instances['labels'][idxs]
    out['bboxes'] = instances['bboxes'][idxs]
    out['masks'] = torch.from_numpy(masks)[idxs]

    image = visualizer._draw_instances(
        image=image,
        instances=out,
        classes=[
            '11', '12', '13', '14', '15', '16', '17', '18',
            '21', '22', '23', '24', '25', '26', '27', '28',
            '31', '32', '33', '34', '35', '36', '37', '38',
            '41', '42', '43', '44', '45', '46', '47', '48',
            'implants', 'crowns', 'pontic', 'fillings', 'roots', 'caries', 'calculus', 'calculus',
        ],
        palette=palette,
    )

    return image


def save_hierarchical(work_dir: Path, num_images: int):
    with open(work_dir / 'detections.pkl', 'rb') as f:
        results = pickle.load(f)

    file_names = [r['img_path'] for r in results]
    results = [results[idx] for idx in np.argsort(file_names)]

    visualizer = DetLocalVisualizer(alpha=0.75)

    img_paths, empties, anns, outs = [], [], [], []
    for result in tqdm(results[:num_images]):
        img_path = Path(result['img_path']).name

        image = cv2.imread(str(result['img_path']))
        gt_instances = process_teeth(result['gt_instances'])
        pred_instances = process_teeth(result['pred_instances'])

        gt_image = process_instances(visualizer, image, gt_instances)
        pred_image = process_instances(visualizer, image, pred_instances)

        img_paths.append(img_path)
        empties.append(image[..., ::-1])
        anns.append(gt_image[..., ::-1])
        outs.append(pred_image[..., ::-1])

    return img_paths, (empties, anns, outs)


def save_flat(
    work_dir: Path,
    thresholds: list[float],
    num_images: int,
):
    with open(work_dir / 'detections.pkl', 'rb') as f:
        results = pickle.load(f)
        
    file_names = [r['img_path'] for r in results]
    results = [results[idx] for idx in np.argsort(file_names)]

    visualizer = DetLocalVisualizer(alpha=0.75)

    outs = []
    for result in tqdm(results[:num_images]):
        img_path = Path(result['img_path'])
        image = cv2.imread(img_path.as_posix())
        instances = filter_teeth(result['pred_instances'], thresholds)
        pred_image = process_instances(visualizer, image, instances)

        outs.append(pred_image[..., ::-1])

    return outs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Side-by-side figures of annotations and predictions.')
    parser.add_argument(
        '--work-dirs', type=Path, default=paths.work_dirs(),
        help='Folder with the chart_filing_<method> run folders (default: $BITEWINGS_WORK_DIRS '
             f'or {paths.DEFAULT_WORK_DIRS}).',
    )
    parser.add_argument(
        '--out-dir', type=Path, default=Path('side_by_side'),
        help='Folder to write the figures to (default: side_by_side).',
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    num_images = 10

    img_paths, top_rows = save_hierarchical(
        paths.method_work_dir('hierarchical', args.work_dirs),
        num_images=num_images,
    )
    pred_images = []
    for work_dir, thresholds in zip(
        [
            paths.method_work_dir('maskrcnn', args.work_dirs),
            paths.method_work_dir('maskdino', args.work_dirs),
            paths.method_work_dir('sparseinst', args.work_dirs),
        ],
        [
            torch.tensor([0.0350, 0.9920, 0.0791, 0.4384, 0.5576, 0.0601, 0.2432]),
            torch.tensor([0.1411, 0.7918, 0.1251, 0.1151, 0.2673, 0.0761, 0.1261]),
            torch.tensor([0.4044, 0.5295, 0.5936, 0.3403, 0.3433, 0.2573, 0.2853]),
        ],
    ):
        pred_image = save_flat(
            work_dir,
            thresholds=thresholds,
            num_images=num_images,
        )
        pred_images.append(pred_image)

    for i, empty, ann, out, pred1, pred2, pred3 in zip(range(len(img_paths)), *top_rows, *pred_images):
        top_row = np.concatenate((empty, ann, out), axis=1)
        bottom_row = np.concatenate((pred1, pred2, pred3), axis=1)
        out_img = np.concatenate((top_row[:, :bottom_row.shape[1]], bottom_row[:, :top_row.shape[1]]), axis=0)

        print(img_paths[i])
        cv2.imwrite(str(out_dir / img_paths[i]), out_img)
