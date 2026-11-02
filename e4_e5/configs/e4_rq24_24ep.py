_base_ = ['./umpe_nocam_vector.py']

# E4 / RQ24: raster prior Z is the BEV query; SD vector tokens are K/V.
# No camera, no HD, no final residual fusion.
load_from = None
find_unused_parameters = True
fp16 = None
total_epochs = 24
runner = dict(type='EpochBasedRunner', max_epochs=24)
checkpoint_config = dict(interval=2, max_keep_ckpts=3)
evaluation = dict(interval=4)

custom_hooks = []

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=6e-4,
    weight_decay=0.01,
    paramwise_cfg=None,
)

# 4 GPUs x batch 5 = 20; adjust if using a different GPU count.
data = dict(samples_per_gpu=5, workers_per_gpu=4)

model = dict(
    pts_bbox_head=dict(
        transformer=dict(
            raster_query=True,
            use_z_norm=True,
            fusion_module=None,
            vector_fusion_layer=dict(
                _delete_=True,
                type='RasterQueryFusion',
                embed_dims=256,
                num_heads=8,
                dropout=0.1,
                use_gate=False,
            ),
        ),
    ),
)
