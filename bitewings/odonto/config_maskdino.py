import os
import sys

_base_ = '../../onedl-mmdetection/projects/DENTEX/configs/maskdino_r50_coco_multilabel.py'
# _base_ = './maskdino_swin-l_coco_multilabel.py'

# make the repository and the mmdetection submodule importable from any working directory
repo_dir = os.path.abspath(os.path.join('{{ fileDirname }}', '..', '..'))
sys.path.append(repo_dir)
sys.path.append(os.path.join(repo_dir, 'onedl-mmdetection'))

custom_imports = dict(
    imports=[
        'projects.DENTEX.datasets',
        'projects.DENTEX.datasets.dataset_wrappers',
        'projects.DENTEX.datasets.samplers',
        'projects.DENTEX.datasets.transforms.loading',
        'projects.DENTEX.datasets.transforms.formatting',
        'projects.DENTEX.datasets.transforms.transforms',
        'projects.DENTEX.evaluation',
        'projects.DENTEX.maskdino',
        'projects.DENTEX.visualization',
        'projects.DENTEX.hooks',
        'bitewings.odonto.crop_bitewing',
    ],
    allow_failed_imports=False,
)

# paths come from BITEWINGS_ODONTO_ROOT / BITEWINGS_WORK_DIRS / BITEWINGS_CHECKPOINTS (see bitewings/paths.py)
odonto_root = os.path.join('{{$BITEWINGS_ODONTO_ROOT:../data/odontoai}}', 'bitewings', '')
checkpoints_dir = '{{$BITEWINGS_CHECKPOINTS:../checkpoints}}'
work_dirs = '{{$BITEWINGS_WORK_DIRS:work_dirs}}'
fold = 1
work_dir = os.path.join(work_dirs, 'odonto_bitewings_maskdino', '')

classes = [
   'tooth-11','tooth-12','tooth-13','tooth-14','tooth-15','tooth-16','tooth-17','tooth-18',
   'tooth-21','tooth-22','tooth-23','tooth-24','tooth-25','tooth-26','tooth-27','tooth-28',
   'tooth-31','tooth-32','tooth-33','tooth-34','tooth-35','tooth-36','tooth-37','tooth-38',
   'tooth-41','tooth-42','tooth-43','tooth-44','tooth-45','tooth-46','tooth-47','tooth-48',
]

train_dataloader = dict(
    num_workers=0,
    persistent_workers=False,
    dataset=dict(
        key='bbox_label',
        dataset=dict(
            dataset=dict(
                type='CocoDataset',
                ann_file='splits/trainval.json',
                data_prefix=dict(img='images'),
                data_root=odonto_root,
                metainfo=dict(classes=classes),
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
                    dict(type='CropBitewing'),
                ],
            ),
            pipeline=[
                _base_.train_dataloader.dataset.dataset.pipeline[0],
                *_base_.train_dataloader.dataset.dataset.pipeline[2:-1],
                dict(type='PackDetInputs'),
            ],
        ),
    ),
)

val_dataloader = dict(
    num_workers=0,
    persistent_workers=False,
    dataset=dict(
        type='CocoDataset',
        ann_file=f'splits/val_{fold}.json',
        data_prefix=dict(img='images'),
        data_root=odonto_root,
        metainfo=dict(classes=classes),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='CropBitewing'),
            *_base_.val_dataloader.dataset.pipeline[2:-1],
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(type='PackDetInputs', meta_keys=(
                'img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'
            )),
        ],
    ),
)
val_evaluator = dict(
    _delete_=True,
    type='CocoMetric',
    ann_file=odonto_root + f'splits/val_{fold}.json',
    # ann_file=odonto_root + 'annotations/instances_val2017_onesample_139.json',  # TODO: delete before merging
    metric=['bbox', 'segm'],
)

custom_hooks = []
model = dict(
    type='MaskDINO',
    train_cfg=dict(num_classes=len(classes)),
    test_cfg=dict(
        instance_postprocess_cfg=dict(max_per_image=100),
        max_per_image=100,
    ),
    panoptic_head=dict(
        type='MaskDINOHead',
        num_things_classes=len(classes),
        num_stuff_classes=0,
        decoder=dict(
            num_classes=len(classes),
        ),
    ),
    panoptic_fusion_head=dict(
        type='MaskDINOFusionHead',
        num_things_classes=len(classes),
        num_stuff_classes=0,
    ),
)
load_from = os.path.join(checkpoints_dir, 'maskdino_coco.pth')

default_hooks = dict(
    checkpoint=dict(save_best='coco/segm_mAP'),
    visualization=dict(draw=True),
)
