"""Fine-tune on the Radboud bitewings plus periapical-style crops of DENTEX panoramics.

Adds the front teeth the released model barely knows: DENTEX numbers every tooth
of its panoramics, and CropPeriapical cuts them into periapical-like windows
(front teeth included). DENTEX does not annotate fillings, crowns, root canals or
caries, so its images are marked with MarkPartialFindings and the criterion only
supervises their tooth outlines and numbers; findings are still learned from the
Radboud bitewings. Validation and checkpoint selection stay on Radboud.

DENTEX is CC BY-NC-SA 4.0: models trained with this config are for
non-commercial use, and shared weights must carry the same license.

    BITEWINGS_DATA_ROOT=<Netherlands> BITEWINGS_DENTEX_ROOT=<DENTEX quadrant_enumeration> \
    python onedl-mmdetection/tools/train.py bitewings/configs/config_finetune_dentex.py

Prepare DENTEX with bitewings/dentex/convert.py. Besides the variables of
config_finetune.py:

    BITEWINGS_DENTEX_ROOT       DENTEX quadrant_enumeration folder (../data/dentex/quadrant_enumeration)
    BITEWINGS_DENTEX_TRAIN_ANN  DENTEX training annotations (splits/train_fdi.json)
    BITEWINGS_DENTEX_FRONT_PROB share of DENTEX crops centred on front teeth (0.5)
"""
import os

_base_ = './config_finetune.py'

dentex_root = os.path.join('{{$BITEWINGS_DENTEX_ROOT:../data/dentex/quadrant_enumeration}}', '')
dentex_train_ann = '{{$BITEWINGS_DENTEX_TRAIN_ANN:splits/train_fdi.json}}'
dentex_front_prob = float('{{$BITEWINGS_DENTEX_FRONT_PROB:0.5}}')

custom_imports = dict(
    imports=_base_.custom_imports.imports + ['bitewings.data.periapical'],
    allow_failed_imports=False,
)
run_name = '{{$BITEWINGS_RUN_NAME:finetune_dentex}}'
work_dir = os.path.join('{{$BITEWINGS_WORK_DIRS:work_dirs}}', run_name, '')

# the Radboud bitewings exactly as in config_finetune.py (augmentations, findings, oversampling)
radboud_train = _base_.train_dataloader.dataset.dataset

dentex_train = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type='CocoMulticlassDataset',
        strict=True,
        decode_masks=False,
        serialize_data=False,
        ann_file=dentex_train_ann,
        data_prefix=dict(img=os.path.join(dentex_root, 'xrays')),
        data_root=dentex_root,
        metainfo=dict(classes=_base_.classes, attributes=_base_.attributes),
        merge_layers=_base_.merge_layers,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadMulticlassAnnotations', merge_layers=_base_.merge_layers,
                 with_bbox=True, with_mask=True),
            dict(type='CropPeriapical', front_prob=dentex_front_prob),
            dict(type='MarkPartialFindings'),
        ],
    ),
    pipeline=[
        dict(type='RandomOPGFlip', prob=0.5),
        dict(
            type='RandomChoiceResize',
            scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333), (608, 1333),
                    (640, 1333), (672, 1333), (704, 1333), (736, 1333), (768, 1333),
                    (800, 1333)],
            keep_ratio=True,
        ),
        dict(
            type='PackMultilabelDetInputs',
            meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor',
                       'flip', 'flip_direction', 'partial_findings'),
        ),
    ],
)

train_dataloader = dict(dataset=dict(
    _delete_=True,
    type='InstanceBalancedDataset',
    oversample_thr=0.1,
    dataset=dict(type='ConcatDataset', datasets=[radboud_train, dentex_train]),
))
