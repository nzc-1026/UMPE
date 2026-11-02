_base_ = ['./umpe_nocam_vector_only_sat.py']

# Paper-standard nuScenes schedule: 24 epochs, from scratch.
# Inputs: satellite + vector SD (no raster SD), no camera, no HD prior.
# The paper uses 8 GPUs x batch 1 (effective 8); we have 5 free GPUs and
# use 5 x 4 = 20 for a feasible wall-clock, keeping 24 epochs and lr 6e-4.
load_from = None
total_epochs = 24
runner = dict(type='EpochBasedRunner', max_epochs=24)
checkpoint_config = dict(interval=2, max_keep_ckpts=3)
evaluation = dict(interval=4)

custom_hooks = [
    dict(type='AlphaScheduleHook', start_iter=0, end_iter=20000,
         start_alpha=0.2, target_alpha=0.6, schedule_type='cosine'),
    dict(type='FusionAlphaMonitorHook', log_interval=100),
]

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=6e-4,
    weight_decay=0.01,
    paramwise_cfg=None,
)
