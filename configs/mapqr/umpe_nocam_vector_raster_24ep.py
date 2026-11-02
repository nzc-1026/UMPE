_base_ = ['./umpe_nocam_vector.py']

# VR24-F: vector SD + raster SD + satellite, 24 epochs from scratch.
# No camera, no HD prior.  Residual is fixed at 1.0 (cannot collapse).
load_from = None
find_unused_parameters = True
fp16 = None
total_epochs = 24
runner = dict(type='EpochBasedRunner', max_epochs=24)
checkpoint_config = dict(interval=2, max_keep_ckpts=3)
evaluation = dict(interval=4)

custom_hooks = [
    dict(type='AlphaScheduleHook', start_iter=0, end_iter=20000,
         start_alpha=0.2, target_alpha=0.6, schedule_type='cosine'),
    dict(type='FusionAlphaMonitorHook', log_interval=100),
]

# Standard from-scratch optimizer: all trainable parameters share lr.
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=6e-4,
    weight_decay=0.01,
    paramwise_cfg=None,
)

# 4 free GPUs x batch 5 = 20, same effective batch as B3 (5 x 4).
data = dict(samples_per_gpu=5, workers_per_gpu=4)

model = dict(
    pts_bbox_head=dict(
        transformer=dict(
            vector_fusion_layer=dict(
                # keep the vector branch active for the whole run
                use_residual=True,
                residual_scale=1.0,
                residual_scale_learnable=False,
            ),
        ),
    ),
)
