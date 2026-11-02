_base_ = ['./e5_rq24_g_24ep.py']

# E5-S stage 1: initialize from the 24-epoch raster-only R24 model.
# Raster encoder and Z LayerNorm are frozen; the new vector cross-attention
# and the decoder adapt to the vector tokens for 4 epochs.
load_from = ('/home/lbb/Documents/space/umpe_hdmap/results_20261029/'
             'R24/work_dirs/epoch_24.pth')
total_epochs = 4
runner = dict(type='EpochBasedRunner', max_epochs=4)
checkpoint_config = dict(interval=1, max_keep_ckpts=2)
evaluation = dict(interval=2)
# The SD-map pickle is large; 1 worker keeps startup stable across ranks.
data = dict(samples_per_gpu=10, workers_per_gpu=1)

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=6e-4,
    weight_decay=0.01,
    paramwise_cfg=dict(custom_keys={
        'pts_bbox_head.transformer.raster_encoder': dict(lr_mult=0.0),
        'pts_bbox_head.transformer.z_norm': dict(lr_mult=0.0),
        'pts_bbox_head.transformer.vector_fusion_layer': dict(lr_mult=5.0),
        'pts_bbox_head.transformer.decoder.layers.0': dict(lr_mult=0.1),
        'pts_bbox_head.transformer.decoder.layers.1': dict(lr_mult=0.1),
        'pts_bbox_head.transformer.decoder.layers.2': dict(lr_mult=0.1),
        'pts_bbox_head.transformer.decoder.layers.3': dict(lr_mult=0.1),
        'pts_bbox_head.transformer.decoder.layers.4': dict(lr_mult=0.1),
        'pts_bbox_head.transformer.decoder.layers.5': dict(lr_mult=0.1),
    }),
)
