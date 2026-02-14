
dataset_type = 'BaseSegDataset'
data_root = '/workspace/project/clean_v1'

classes = ('background', 'cat', 'dog')
palette = [(0, 0, 0), (128, 0, 0), (0, 128, 0)]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]

train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='img/train', seg_map_path='labels/train'),
        pipeline=train_pipeline,
        metainfo=dict(classes=classes, palette=palette),
    )
)

val_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='img/val', seg_map_path='labels/val'),
        pipeline=test_pipeline,
        metainfo=dict(classes=classes, palette=palette),
    )
)

test_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='img/test', seg_map_path='labels/test'),
        pipeline=test_pipeline,
        metainfo=dict(classes=classes, palette=palette),
    )
)

val_evaluator = dict(type='IoUMetric', iou_metrics=['mDice'])
test_evaluator = dict(type='IoUMetric', iou_metrics=['mDice'])
