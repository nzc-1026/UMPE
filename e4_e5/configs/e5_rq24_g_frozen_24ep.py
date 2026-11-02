_base_ = ['./e5_rq24_g_24ep.py']

# E5-24F: 24 vector epochs on a fixed, mature R24 raster Z.
# Raster encoder and z_norm stay frozen for the whole run, so the only thing
# trained is the SD-vector cross-attention, the gate, and the decoder.
load_from = ('/home/lbb/Documents/space/umpe_hdmap/results_20261029/'
             'R24/work_dirs/epoch_24.pth')
total_epochs = 24
runner = dict(type='EpochBasedRunner', max_epochs=24)
checkpoint_config = dict(interval=2, max_keep_ckpts=4)
evaluation = dict(interval=4)
custom_hooks = []

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
