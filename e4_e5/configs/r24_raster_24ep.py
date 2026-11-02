_base_ = ['./umpe_nocam_raster.py']

# R24: raster-only baseline for the Z-as-BEV-query architecture.
# Satellite + raster SD -> StageFiLM -> LayerNorm -> + bev_pos -> decoder.
# No vector branch, no HD, no camera.
load_from = None
find_unused_parameters = True
fp16 = None
total_epochs = 24
runner = dict(type='EpochBasedRunner', max_epochs=24)
checkpoint_config = dict(interval=2, max_keep_ckpts=4)
evaluation = dict(interval=4, metric='chamfer',
                  save_best='NuscMap_chamfer/mAP', rule='greater')
custom_hooks = []

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=6e-4,
    weight_decay=0.01,
    paramwise_cfg=None,
)

data = dict(samples_per_gpu=5, workers_per_gpu=4)

model = dict(
    pts_bbox_head=dict(
        transformer=dict(
            raster_query=True,
            use_z_norm=True,
            vector_fusion_layer=None,
            fusion_module=None,
        ),
    ),
)
