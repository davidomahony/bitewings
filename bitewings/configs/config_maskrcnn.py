import os
import sys

_base_ = 'mmdet::swin/mask-rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco.py'

# make the repository and the mmdetection submodule importable from any working directory
repo_dir = os.path.abspath(os.path.join('{{ fileDirname }}', '..', '..'))
sys.path.append(repo_dir)
sys.path.append(os.path.join(repo_dir, 'onedl-mmdetection'))

custom_imports = dict(
    imports=[
        'bitewings.evaluation.metrics',
        'bitewings.models',
        'projects.DENTEX.datasets',
        'projects.DENTEX.evaluation',
    ],
    allow_failed_imports=False,
)

classes = [
    'tooth_11', 'tooth_12', 'tooth_13', 'tooth_14', 'tooth_15', 'tooth_16', 'tooth_17', 'tooth_18',
    'tooth_21', 'tooth_22', 'tooth_23', 'tooth_24', 'tooth_25', 'tooth_26', 'tooth_27', 'tooth_28',
    'tooth_31', 'tooth_32', 'tooth_33', 'tooth_34', 'tooth_35', 'tooth_36', 'tooth_37', 'tooth_38',
    'tooth_41', 'tooth_42', 'tooth_43', 'tooth_44', 'tooth_45', 'tooth_46', 'tooth_47', 'tooth_48',
    'implants', 'crowns', 'pontic', 'fillings', 'roots', 'caries', 'calculus_left', 'calculus_right',
]


# paths come from BITEWINGS_DATA_ROOT / BITEWINGS_WORK_DIRS / BITEWINGS_CHECKPOINTS (see bitewings/paths.py)
root = os.path.join('{{$BITEWINGS_DATA_ROOT:../data/Netherlands}}', '')
checkpoints_dir = '{{$BITEWINGS_CHECKPOINTS:../checkpoints}}'
work_dirs = '{{$BITEWINGS_WORK_DIRS:work_dirs}}'
fold = 1
work_dir = os.path.join(work_dirs, 'chart_filing_maskrcnn', '')
filter_empty = False

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomBitewingFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[[{
                'type':
                'RandomChoiceResize',
                'scales': [(480, 1333), (512, 1333), (544, 1333),
                            (576, 1333), (608, 1333), (640, 1333),
                            (672, 1333), (704, 1333), (736, 1333),
                            (768, 1333), (800, 1333)],
                'keep_ratio':
                True
            }],
            [{
                'type': 'RandomChoiceResize',
                'scales': [(400, 4200), (500, 4200), (600, 4200)],
                'keep_ratio': True
            }, {
                'type': 'RandomCrop',
                'crop_type': 'absolute_range',
                'crop_size': (384, 600),
                'allow_negative_crop': True
            }, {
                'type':
                'RandomChoiceResize',
                'scales':
                [(480, 1333), (512, 1333), (544, 1333),
                    (576, 1333), (608, 1333), (640, 1333),
                    (672, 1333), (704, 1333), (736, 1333),
                    (768, 1333), (800, 1333)],
                'keep_ratio':
                True
            }],
        ]),
    dict(type='PackDetInputs'),
]

workers = 0
train_dataloader = dict(
    batch_size=4,
    num_workers=workers,
    persistent_workers=False,
    dataset=dict(
        _delete_=True, 
        type='InstanceBalancedDataset',
        oversample_thr=0.1,
        key='bbox_label',
        dataset=dict(
            type='CocoDataset',
            filter_cfg=dict(filter_empty_gt=filter_empty),
            serialize_data=False,
            pipeline=train_pipeline,
            ann_file=f'splits/train_flat_{fold}.json',
            data_prefix=dict(img='images'),
            data_root=root,
            metainfo=dict(classes=classes),
        ),
    ),
)

val_pipeline=[
    dict(type='LoadImageFromFile'),
    dict(
        type='Resize',
        scale=(1333, 800),
        keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor'))
]

val_dataloader = dict(
    num_workers=workers,
    persistent_workers=False,
    dataset=dict(
        type='CocoDataset',
        pipeline=val_pipeline,
        metainfo=dict(classes=classes),
        ann_file=f'splits/val_flat_{fold}.json',
        data_prefix=dict(img='images'),
        data_root=root,
    ),
)
val_evaluator = [
    dict(
        type='CocoMetric',
        classwise=True,
        metric=['bbox', 'segm'],
        ann_file=root + f'splits/val_flat_{fold}.json',
    ),
    dict(
        type='AggregateLabelInstanceMetric',
        label_idxs=list(range(32, 39)),
        prefixes=classes[32:39],
    ),
]

test_dataloader = dict(
    num_workers=workers,
    persistent_workers=False,
    dataset=dict(
        type='CocoDataset',
        pipeline=val_pipeline,
        ann_file=f'splits/val_flat_{fold}.json',
        data_prefix=dict(img='images'),
        data_root=root,
        metainfo=dict(classes=classes),
    ),
)

test_evaluator = [
    dict(
        type='RelevantCocoMetric',
        classwise=True,
        metric=['bbox', 'segm'],
        ann_file=root + f'splits/val_flat_{fold}.json',
    ),
    dict(
        type='AggregateLabelInstanceMetric',
        label_idxs=list(range(32, 39)),
        prefixes=classes[32:39],
        remove_irrelevant=True,
    ),
    dict(
        type='DumpNumpyDetResults',
        score_thr=0.0,
        filter_wrong_arch=False,
        out_file_path=work_dir + 'detections.pkl',
    ),
]

model = dict(
    type='MaskRCNN',
    backbone=dict(
        _delete_=True,
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
    ),
    neck=dict(in_channels=[256, 512, 1024, 2048]),
    rpn_head=dict(type='RPNHead'),
    test_cfg=dict(rcnn=dict(score_thr=0.01, nms=dict(class_agnostic=False))),
    roi_head=dict(
        bbox_head=dict(num_classes=len(classes)),
        mask_head=dict(num_classes=len(classes)),
    ),
)
load_from = os.path.join(checkpoints_dir, 'maskrcnn_odonto.pth')

max_epochs = 36
train_cfg = dict(
    _delete_=True,
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=1,
)
param_scheduler = dict(
    _delete_=True,
    type='MultiStepLR',
    begin=0,
    end=max_epochs,
    by_epoch=True,
    milestones=[27, 33],
    gamma=0.1,
)

optim_wrapper = dict(    
    clip_grad=dict(max_norm=0.01, norm_type=2),
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.00005,
        betas=(0.9, 0.999),
        weight_decay=0.1,
    ),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)}),
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    checkpoint=dict(
        interval=12,
        by_epoch=True,
        max_keep_ckpts=2,
        save_best='aggregate/aggregate/f1-score',
        rule='greater',
    ),
    visualization=dict(
        draw=False,
        show=False,
        interval=50,
    ),
)

visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ],
)

tta_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TestTimeAug', transforms=[
        [
            {
                'type': 'Resize',
                'scale': scale,
                'keep_ratio': True,
            } for scale in [
                # (1333, 640), (1333, 672), (1333, 704),
                # (1333, 736), (1333, 768), (1333, 800),
                (1333, 800),
            ]
        ],
        [
            {'type': 'RandomFlip', 'prob': 0.0},
            # {'type': 'RandomFlip', 'prob': 1.0},
        ],
        [{
            'type': 'LoadAnnotations', 'with_bbox': True, 'with_mask': True,
        }],
        [{
            'type': 'PackDetInputs',
            'meta_keys': [
                'img_id', 'img_path', 'ori_shape', 'img_shape',
                'scale_factor', 'flip', 'flip_direction',
            ]
        }],
    ]),
]

tta_model = dict(
    type='LingyunTTAModel',
    tta_cfg=dict(
        nms=dict(type='nms', iou_threshold=0.75),
        max_per_img=100,
    ),
)
