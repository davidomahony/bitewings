"""Random periapical-style crops from panoramic radiographs.

The authors' CropBitewing (bitewings/odonto/crop_bitewing.py) turns OdontoAI
panoramics into bitewing-like crops and deliberately skips the incisors and
canines. CropPeriapical does the opposite job: it cuts a window around a few
neighbouring teeth of one jaw, front teeth included, so that panoramics with
numbered front teeth (e.g. DENTEX) can teach the model periapical-style views.

Use it after LoadMulticlassAnnotations (needs gt_masks and gt_bboxes_labels),
before resizing.
"""
from typing import Optional, Tuple

import numpy as np
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS
from mmdet.structures.bbox import HorizontalBoxes

# per-instance arrays that must stay aligned when teeth are dropped
INSTANCE_KEYS = (
    'gt_bboxes_labels', 'gt_bboxes_multilabels', 'gt_ignore_flags', 'gt_instances_ids',
)


@TRANSFORMS.register_module()
class CropPeriapical(BaseTransform):
    """Crop a random window around 2-5 neighbouring teeth of one jaw.

    Args:
        front_prob: probability that the window is centred on an incisor or
            canine (label % 8 <= 2) instead of a premolar or molar.
        front_teeth: (min, max) number of neighbouring teeth in a front window.
        back_teeth: (min, max) number of neighbouring teeth in a back window.
        front_aspect: (min, max) height / width of a front window (portrait).
        back_aspect: (min, max) width / height of a back window (landscape).
        margin: (min, max) margin around the selected teeth, as a fraction of
            their height.
        min_visible: teeth with less than this fraction of their mask inside
            the window are dropped from the annotations.
    """

    def __init__(
        self,
        front_prob: float = 0.5,
        front_teeth: Tuple[int, int] = (2, 4),
        back_teeth: Tuple[int, int] = (3, 5),
        front_aspect: Tuple[float, float] = (1.1, 1.5),
        back_aspect: Tuple[float, float] = (1.1, 1.5),
        margin: Tuple[float, float] = (0.05, 0.2),
        min_visible: float = 0.3,
    ):
        self.front_prob = front_prob
        self.front_teeth = front_teeth
        self.back_teeth = back_teeth
        self.front_aspect = front_aspect
        self.back_aspect = back_aspect
        self.margin = margin
        self.min_visible = min_visible

    def _select_teeth(self, labels: np.ndarray, boxes: np.ndarray) -> Optional[np.ndarray]:
        upper = labels < 16
        jaws = [jaw for jaw in (upper, ~upper) if jaw.sum() >= 2]
        if not jaws:
            return None
        jaw = jaws[np.random.randint(len(jaws))]

        front = (labels % 8) <= 2
        want_front = np.random.rand() < self.front_prob
        anchors = np.nonzero(jaw & (front if want_front else ~front))[0]
        if not len(anchors):
            want_front = not want_front
            anchors = np.nonzero(jaw & (front if want_front else ~front))[0]
        if not len(anchors):
            return None
        anchor = np.random.choice(anchors)

        # neighbours along the jaw, ordered left to right by box centre
        jaw_idx = np.nonzero(jaw)[0]
        jaw_idx = jaw_idx[np.argsort((boxes[jaw_idx, 0] + boxes[jaw_idx, 2]) / 2)]
        pos = int(np.nonzero(jaw_idx == anchor)[0][0])
        lo, hi = self.front_teeth if want_front else self.back_teeth
        count = min(np.random.randint(lo, hi + 1), len(jaw_idx))
        start = int(np.clip(pos - np.random.randint(count), 0, len(jaw_idx) - count))
        self._front = want_front
        return jaw_idx[start:start + count]

    def _window(self, boxes: np.ndarray, upper: bool, img_h: int, img_w: int) -> Tuple[int, int, int, int]:
        x1, y1 = boxes[:, 0].min(), boxes[:, 1].min()
        x2, y2 = boxes[:, 2].max(), boxes[:, 3].max()
        tooth_h = float(np.max(boxes[:, 3] - boxes[:, 1]))
        m = tooth_h * np.random.uniform(*self.margin)
        x1, y1, x2, y2 = x1 - m, y1 - m, x2 + m, y2 + m

        # Reach a periapical-like aspect ratio without growing into the opposite jaw:
        # front windows (portrait) grow towards the roots, which a real periapical shows
        # with surrounding bone; back windows (landscape) grow sideways.
        w, h = x2 - x1, y2 - y1
        if self._front:
            target_h = w * np.random.uniform(*self.front_aspect)
            if target_h > h:
                if upper:
                    y1 -= target_h - h
                else:
                    y2 += target_h - h
        else:
            target_w = h * np.random.uniform(*self.back_aspect)
            if target_w > w:
                x1, x2 = x1 - (target_w - w) / 2, x2 + (target_w - w) / 2

        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(float(img_w), x2), min(float(img_h), y2)

        # front teeth sit near the image edge on panoramics, so growing towards the
        # roots can be cut short; narrow the window around its centre instead
        if self._front and (y2 - y1) < self.front_aspect[0] * (x2 - x1):
            centre, half = (x1 + x2) / 2, (y2 - y1) / np.random.uniform(*self.front_aspect) / 2
            x1, x2 = max(0.0, centre - half), min(float(img_w), centre + half)

        return int(np.floor(x1)), int(np.floor(y1)), int(np.ceil(x2)), int(np.ceil(y2))

    def transform(self, results: dict) -> Optional[dict]:
        labels = results.get('gt_bboxes_labels')
        if labels is None or len(labels) < 2:
            return results
        boxes = results['gt_bboxes'].numpy() if hasattr(results['gt_bboxes'], 'numpy') \
            else np.asarray(results['gt_bboxes'])
        labels = np.asarray(labels)
        selected = self._select_teeth(labels, boxes)
        if selected is None:
            return results

        img_h, img_w = results['img'].shape[:2]
        x1, y1, x2, y2 = self._window(boxes[selected], bool(labels[selected[0]] < 16), img_h, img_w)
        if x2 - x1 < 16 or y2 - y1 < 16:
            return results

        masks = results['gt_masks'].masks
        total = (masks != 0).sum(axis=(1, 2))
        inside = (masks[:, y1:y2, x1:x2] != 0).sum(axis=(1, 2))
        keep = (total > 0) & (inside >= self.min_visible * np.maximum(total, 1))
        if not keep.any():
            return results

        results['img'] = results['img'][y1:y2, x1:x2]
        results['img_shape'] = results['img'].shape[:2]
        cropped = results['gt_masks'][np.nonzero(keep)[0]].crop(np.array([x1, y1, x2, y2]))
        results['gt_masks'] = cropped

        # boxes from the cropped masks, so partially visible teeth get tight boxes
        new_boxes = []
        for mask in cropped.masks:
            ys, xs = np.nonzero(mask)
            new_boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
        results['gt_bboxes'] = HorizontalBoxes(np.array(new_boxes, dtype=np.float32))
        for key in INSTANCE_KEYS:
            if key in results:
                results[key] = results[key][keep]
        return results

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(front_prob={self.front_prob}, '
                f'front_teeth={self.front_teeth}, back_teeth={self.back_teeth}, '
                f'min_visible={self.min_visible})')
