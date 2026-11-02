_base_ = ['./umpe_nocam_vector.py']

# Ablation: satellite + vector SD only (no rasterized SD).
# Start from the 12-epoch raster baseline and fine-tune 4 epochs so the
# comparison against the raster control (38.57) uses the same budget.
load_from = ('/home/lbb/Documents/space/umpe_hdmap/results_20261025/'
             '01_baseline_raster_12ep/best.pth')

total_epochs = 4
runner = dict(type='EpochBasedRunner', max_epochs=4)
checkpoint_config = dict(interval=1, max_keep_ckpts=2)
evaluation = dict(interval=2)

custom_hooks = [
    dict(type='AlphaScheduleHook', start_iter=0, end_iter=20000,
         start_alpha=0.54, target_alpha=0.6, schedule_type='cosine'),
    dict(type='FusionAlphaMonitorHook', log_interval=100),
]

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=6e-4,
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'map_encoder': dict(lr_mult=5.0),
            'pts_bbox_head.transformer.vector_fusion_layer': dict(lr_mult=5.0),
            'pts_bbox_head.transformer.raster_encoder': dict(lr_mult=1.0),
            'pts_bbox_head.transformer.fusion_module': dict(lr_mult=1.0),
            'pts_bbox_head.transformer.decoder.layers.0': dict(lr_mult=0.0),
            'pts_bbox_head.transformer.decoder.layers.1': dict(lr_mult=0.0),
            'pts_bbox_head.transformer.decoder.layers.2': dict(lr_mult=0.0),
            'pts_bbox_head.transformer.decoder.layers.3': dict(lr_mult=0.0),
            'pts_bbox_head.transformer.decoder.layers.4': dict(lr_mult=0.1),
            'pts_bbox_head.transformer.decoder.layers.5': dict(lr_mult=0.1),
        }),
)

model = dict(
    pts_bbox_head=dict(
        transformer=dict(
            vector_fusion_layer=dict(
                # keep the vector branch active from the first iteration;
                # the previous run learned to suppress it when raster SD was
                # also present.
                residual_scale=1.0,
                residual_scale_learnable=False,
            ),
        ),
    ),
)

train_pipeline = [
    dict(type='LoadSatelliteFromFile', to_float32=True, normalize=True),
    dict(type='DefaultFormatBundle3D', with_gt=False, with_label=False,
         class_names=['divider', 'ped_crossing', 'boundary']),
    dict(type='PackSDMap'),
    dict(type='CustomCollect3D', keys=['satellite_img', 'sdmap']),
]

test_pipeline = [
    dict(type='LoadSatelliteFromFile', to_float32=True, normalize=True),
    dict(type='DefaultFormatBundle3D', with_gt=False, with_label=False,
         class_names=['divider', 'ped_crossing', 'boundary']),
    dict(type='PackSDMap'),
    dict(type='CustomCollect3D', keys=['satellite_img', 'sdmap']),
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline),
)
