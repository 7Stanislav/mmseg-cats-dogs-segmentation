
_base_ = [
    'mmseg::_base_/default_runtime.py',
    'mmseg::_base_/schedules/schedule_40k.py',
    '/workspace/project/configs/dataset_catsdogs_256.py'
]

# model: DeepLabV3+ R50
model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        size=(256, 256),
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255
    ),
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        contract_dilation=True,
        init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://resnet50_v1c'),
    ),
    decode_head=dict(
        type='DepthwiseSeparableASPPHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        dilations=(1, 12, 24, 36),
        c1_in_channels=256,
        c1_channels=48,
        dropout_ratio=0.1,
        num_classes=3,
        align_corners=False,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            dict(type='DiceLoss', loss_weight=1.0),
        ],
    ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=3,
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4),
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

# optimizer (стабильно для deeplab)
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=5e-4),
)

# training
train_cfg = dict(type='IterBasedTrainLoop', max_iters=12000, val_interval=1000)
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=1000, max_keep_ckpts=2, save_best='mDice')
)

work_dir = '/workspace/project/work_dirs/hyp4_deeplabv3p_r50'
