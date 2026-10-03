"""Run the hierarchical chart-filing model on a folder of new bitewings.

    python bitewings/inference/init_images.py <images>
    BITEWINGS_IMAGES=<images> python onedl-mmdetection/tools/test.py \
        bitewings/configs/config_inference.py <checkpoint>.pth

Predictions are written to <images>/detections.pkl. With BITEWINGS_LOW_MEMORY=1
(the default) low-confidence candidates are dropped before full-resolution masks
are built and every tooth keeps a single FDI label, which keeps CPU inference
around 3 GB. Detections with score >= 0.1 are unchanged; set
BITEWINGS_LOW_MEMORY=0 to reproduce the original post-processing exactly.
"""
import os

_base_ = './config_hierarchical.py'

images_dir = '{{$BITEWINGS_IMAGES:test}}'
low_memory = '{{$BITEWINGS_LOW_MEMORY:1}}' == '1'

test_dataloader = dict(dataset=dict(
    data_root=images_dir,
    data_prefix=dict(img=images_dir),
    ann_file=os.path.join(images_dir, 'coco.json'),
))
test_evaluator = [
    dict(
        type='DumpMulticlassDetResults',
        score_thr=0.0,
        out_file_path=os.path.join(images_dir, 'detections.pkl'),
    ),
]

if low_memory:
    model = dict(test_cfg=dict(
        min_query_score=0.1,
        min_label_score=0.1,
        one_label_per_query=True,
    ))
