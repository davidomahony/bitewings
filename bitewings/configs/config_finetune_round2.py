"""Round 2: Radboud bitewings + DENTEX periapical-style crops + real DenPAR periapicals.

Adds DenPAR's training split (bitewings/denpar/convert.py: real intraoral periapicals with
FDI-numbered tooth masks, 82 of them anterior) to config_finetune_dentex.py. DenPAR has no
finding labels, so its images are marked with MarkPartialFindings like DENTEX and only teach
tooth outlines and numbers. Score the result on the held-out real periapicals from
bitewings/denpar/make_eval_set.py (DenPAR val + test, never trained on).

DENTEX (CC BY-NC-SA 4.0) keeps this model non-commercial. DenPAR is CC BY 4.0.

    BITEWINGS_DATA_ROOT=<Netherlands> BITEWINGS_DENTEX_ROOT=<...>/quadrant_enumeration \
    BITEWINGS_DENPAR_ROOT=<DenPAR Dataset folder> \
    python onedl-mmdetection/tools/train.py bitewings/configs/config_finetune_round2.py

Besides the variables of config_finetune.py and config_finetune_dentex.py:

    BITEWINGS_DENPAR_ROOT       DenPAR "Dataset" folder with splits/ (../data/denpar/Dataset)
    BITEWINGS_DENPAR_TRAIN_ANN  DenPAR training annotations (splits/train_fdi.json)
    BITEWINGS_DENPAR_REPEAT     times each DenPAR image is repeated per epoch (2; real anterior
                                periapicals are scarce, so they are seen more often)
"""
import os

_base_ = './config_finetune_dentex.py'

denpar_root = os.path.join('{{$BITEWINGS_DENPAR_ROOT:../data/denpar/Dataset}}', '')
denpar_train_ann = '{{$BITEWINGS_DENPAR_TRAIN_ANN:splits/train_fdi.json}}'
denpar_repeat = int('{{$BITEWINGS_DENPAR_REPEAT:2}}')

run_name = '{{$BITEWINGS_RUN_NAME:finetune_round2}}'
work_dir = os.path.join('{{$BITEWINGS_WORK_DIRS:work_dirs}}', run_name, '')

denpar_train = dict(
    type='RepeatDataset',
    times=denpar_repeat,
    dataset=dict(
        type='MultiImageMixDataset',
        dataset=dict(
            type='CocoMulticlassDataset',
            strict=True,
            decode_masks=False,
            serialize_data=False,
            ann_file=denpar_train_ann,
            data_prefix=dict(img=denpar_root),
            data_root=denpar_root,
            metainfo=dict(classes=_base_.classes, attributes=_base_.attributes),
            merge_layers=_base_.merge_layers,
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadMulticlassAnnotations', merge_layers=_base_.merge_layers,
                     with_bbox=True, with_mask=True),
                dict(type='MarkPartialFindings'),
            ],
        ),
        # same augmentation as the DENTEX crops: flip (swapping left/right FDI) and multi-scale resize
        pipeline=_base_.dentex_train.pipeline,
    ),
)

train_dataloader = dict(dataset=dict(
    _delete_=True,
    type='InstanceBalancedDataset',
    oversample_thr=0.1,
    dataset=dict(type='ConcatDataset',
                 datasets=[_base_.radboud_train, _base_.dentex_train, denpar_train]),
))
