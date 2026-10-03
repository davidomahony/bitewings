import os

_base_ = './config_maskrcnn.py'

custom_imports = dict(
    imports=[
        'bitewings.evaluation.metrics',
        'bitewings.models',
        'projects.DENTEX.datasets',
        'projects.DENTEX.evaluation',
        'projects.MaskDINO.maskdino',
    ],
    allow_failed_imports=False,
)

work_dir = os.path.join(_base_.work_dirs, 'chart_filing_maskdino', '')

model = dict(
    _delete_=True,
    type='MaskDINO',
    data_preprocessor=_base_.model.data_preprocessor,
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    panoptic_head=dict(
        type='MaskDINOHead',
        num_stuff_classes=0,
        num_things_classes=len(_base_.classes),
        encoder=dict(
            in_channels=[256, 512, 1024, 2048],
            in_strides=[4, 8, 16, 32],
            transformer_dropout=0.0,
            transformer_nheads=8,
            transformer_dim_feedforward=2048,
            transformer_enc_layers=6,
            conv_dim=256,
            mask_dim=256,
            norm_cfg=dict(type='GN', num_groups=32),
            transformer_in_features=['res3', 'res4', 'res5'],
            common_stride=4,
            num_feature_levels=3,
            total_num_feature_levels=4,
            feature_order='low2high'),
        decoder=dict(
            in_channels=256,
            num_classes=len(_base_.classes),
            hidden_dim=256,
            num_queries=300,
            nheads=8,
            dim_feedforward=2048,
            dec_layers=9,
            mask_dim=256,
            enforce_input_project=False,
            two_stage=True,
            dn='seg',
            noise_scale=0.4,
            dn_num=20,
            initialize_box_type='no',
            initial_pred=True,
            learn_tgt=False,
            total_num_feature_levels=4,
            dropout=0.0,
            activation='relu',
            nhead=8,
            dec_n_points=4,
            mask_classification=True,
            return_intermediate_dec=True,
            query_dim=4,
            dec_layer_share=False,
            semantic_ce_loss=False)),
    panoptic_fusion_head=dict(
        type='MaskDINOFusionHead',
        num_things_classes=len(_base_.classes),
        num_stuff_classes=0,
        loss_panoptic=None,  # MaskDINOFusionHead has no training loss
        init_cfg=None),  # MaskDINOFusionHead has no module
    train_cfg=dict(  # corresponds to SetCriterion
        num_classes=len(_base_.classes),
        matcher=dict(
            cost_class=4.0, cost_box=5.0, cost_giou=2.0,
            cost_mask=5.0, cost_dice=5.0, num_points=100),
        class_weight=4.0,
        box_weight=5.0,
        giou_weight=2.0,
        mask_weight=5.0,
        dice_weight=5.0,
        dn='seg',
        dec_layers=9,
        box_loss=True,
        two_stage=True,
        eos_coef=0.1,
        num_points=100,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        semantic_ce_loss=False,
        panoptic_on=False,  # TODO: Why?
        deep_supervision=True),
    test_cfg=dict(
        panoptic_on=False,
        instance_on=True,
        semantic_on=False,  # Not implemented yet
        panoptic_postprocess_cfg=dict(
            object_mask_thr=0.25,  # 0.8 for MaskFormer
            iou_thr=0.8,
            filter_low_score=True,  # it will filter mask area where score is less than 0.5.
            panoptic_temperature=0.06,
            transform_eval=True),
        instance_postprocess_cfg=dict(
            max_per_image=300,
            focus_on_box=False),            
        max_per_image=300),
    init_cfg=None)
load_from = os.path.join(_base_.checkpoints_dir, 'maskdino_odonto.pth')

embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
            'query_embed': embed_multi,
            'query_feat': embed_multi,
            'level_embed': embed_multi,
        },
        norm_decay_mult=0.0),
    clip_grad=dict(max_norm=0.01, norm_type=2))

_base_.test_evaluator[-1]['out_file_path'] = work_dir + 'detections.pkl'
