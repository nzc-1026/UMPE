_base_ = ['./umpe_nocam_vector.py']

# E5 / RQ24-G: raster Z is the query, SD vector tokens are K/V, plus a
# per-token/channel gate between Z and the vector attention output.
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
                use_gate=True,
                gate_init_bias=-2.0,
            ),
        ),
    ),
)
