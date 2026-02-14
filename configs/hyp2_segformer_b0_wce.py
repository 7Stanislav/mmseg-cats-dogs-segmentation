
_base_ = [
    'mmseg::_base_/default_runtime.py',
    'mmseg::_base_/schedules/schedule_40k.py',
    '/workspace/project/configs/dataset_catsdogs_256.py'
]

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
        type='MixVisionTransformer',
        in_channels=3,
        embed_dims=32,
        num_stages=4,
        num_layers=[2, 2, 2, 2],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
    ),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=3,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0,
                class_weight=[0.075, 1.37, 1.56],
            ),
            dict(type='DiceLoss', loss_weight=1.0),
        ],
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=3e-4, weight_decay=0.01),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)

train_cfg = dict(type='IterBasedTrainLoop', max_iters=12000, val_interval=1000)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=1000, max_keep_ckpts=2, save_best='mDice')
)

work_dir = '/workspace/project/work_dirs/hyp2_segformer_b0_wce'
