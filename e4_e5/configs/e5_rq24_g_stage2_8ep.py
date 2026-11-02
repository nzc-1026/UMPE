_base_ = ['./e5_rq24_g_stage1_4ep.py']

# E5-S stage 2: unfreeze the raster encoder at a low LR and let Z and the
# vector attention co-adapt for 8 more epochs.
load_from = ('/home/lbb/Documents/space/umpe_hdmap/results_20261029/'
             'E5_S/stage1/work_dirs/epoch_4.pth')
total_epochs = 8
runner = dict(type='EpochBasedRunner', max_epochs=8)
checkpoint_config = dict(interval=2, max_keep_ckpts=3)
evaluation = dict(interval=2)

model = dict(
    pts_bbox_head=dict(
        transformer=dict(
            raster_encoder=dict(freeze_layers=False),
        ),
    ),
)

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=3e-4,
    weight_decay=0.01,
    paramwise_cfg=dict(custom_keys={
        'pts_bbox_head.transformer.raster_encoder': dict(lr_mult=0.1),
        'pts_bbox_head.transformer.z_norm': dict(lr_mult=0.1),
        'pts_bbox_head.transformer.vector_fusion_layer': dict(lr_mult=5.0),
        'pts_bbox_head.transformer.decoder.layers.0': dict(lr_mult=0.1),
        'pts_bbox_head.transformer.decoder.layers.1': dict(lr_mult=0.1),
        'pts_bbox_head.transformer.decoder.layers.2': dict(lr_mult=0.1),
        'pts_bbox_head.transformer.decoder.layers.3': dict(lr_mult=0.1),
        'pts_bbox_head.transformer.decoder.layers.4': dict(lr_mult=0.1),
        'pts_bbox_head.transformer.decoder.layers.5': dict(lr_mult=0.1),
    }),
)
