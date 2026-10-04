"""Fine-tune the released hierarchical chart-filing model on new data.

The original recipe (config_hierarchical.py) trains for 36 epochs starting from
the OdontoAI checkpoint. This config instead starts from the finished
chart-filing checkpoint and refines it with a shorter, lower learning-rate
schedule, which is what you want when adding data to an already-trained model.

    BITEWINGS_DATA_ROOT=/data BITEWINGS_CHECKPOINTS=/checkpoints \
    BITEWINGS_TRAIN_ANN=splits/train_fdi_1.json BITEWINGS_VAL_ANN=splits/val_fdi_1.json \
    python onedl-mmdetection/tools/train.py bitewings/configs/config_finetune.py

Annotation files are relative to BITEWINGS_DATA_ROOT unless absolute. Every
setting below can be changed with an environment variable:

    BITEWINGS_INIT_CHECKPOINT  starting checkpoint in BITEWINGS_CHECKPOINTS (hierarchical_chartfiling.pth)
    BITEWINGS_TRAIN_ANN        training annotations (splits/train_fdi_1.json)
    BITEWINGS_VAL_ANN          validation annotations (splits/val_fdi_1.json)
    BITEWINGS_RUN_NAME         output folder name in BITEWINGS_WORK_DIRS (finetune_hierarchical)
    BITEWINGS_EPOCHS           number of epochs (12)
    BITEWINGS_LR               peak learning rate (2e-5; the original recipe uses 1e-4 from scratch)
    BITEWINGS_BATCH_SIZE       images per GPU (2)
    BITEWINGS_NUM_WORKERS      data-loading processes (4; use 0 on Windows or for debugging)
"""
import os

_base_ = './config_hierarchical.py'

load_from = os.path.join(
    '{{$BITEWINGS_CHECKPOINTS:../checkpoints}}',
    '{{$BITEWINGS_INIT_CHECKPOINT:hierarchical_chartfiling.pth}}',
)
work_dir = os.path.join(
    '{{$BITEWINGS_WORK_DIRS:work_dirs}}',
    '{{$BITEWINGS_RUN_NAME:finetune_hierarchical}}',
    '',
)
# load_from replaces every weight, so skip downloading the ImageNet ResNet-50 the
# backbone would otherwise be initialised from (also lets training run offline)
model = dict(backbone=dict(init_cfg=None))

train_ann = '{{$BITEWINGS_TRAIN_ANN:splits/train_fdi_1.json}}'
val_ann = '{{$BITEWINGS_VAL_ANN:splits/val_fdi_1.json}}'
max_epochs = int('{{$BITEWINGS_EPOCHS:12}}')
lr = float('{{$BITEWINGS_LR:2e-5}}')
batch_size = int('{{$BITEWINGS_BATCH_SIZE:2}}')
num_workers = int('{{$BITEWINGS_NUM_WORKERS:4}}')

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=num_workers > 0,
    dataset=dict(dataset=dict(dataset=dict(ann_file=train_ann))),
)
val_dataloader = dict(
    num_workers=num_workers,
    persistent_workers=num_workers > 0,
    dataset=dict(ann_file=val_ann),
)
test_dataloader = dict(dataset=dict(ann_file=val_ann))

train_cfg = dict(max_epochs=max_epochs)
# constant learning rate, then a 10x drop for the last quarter of training
param_scheduler = [dict(
    type='MultiStepLR',
    begin=0,
    end=max_epochs,
    by_epoch=True,
    milestones=[max(1, round(max_epochs * 0.75))],
    gamma=0.1,
)]
optim_wrapper = dict(optimizer=dict(lr=lr))

# keep the best checkpoint (by the authors' aggregate F1) plus the latest one
default_hooks = dict(checkpoint=dict(max_keep_ckpts=1))
