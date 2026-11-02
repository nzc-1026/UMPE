_base_ = ['./umpe_nocam_vector.py']

# Fine-tune the vector branch on top of the 12-epoch raster baseline
# (mAP 37.30).  The vector layer keeps a residual to the learnable BEV;
# the newly added prior modules use the official 5x learning rate.
load_from = ('/home/lbb/Documents/space/umpe_hdmap/UMPE-main/MapQR/'
             'work_dirs/nocam_raster_v2/epoch_12.pth')

total_epochs = 4
runner = dict(type='EpochBasedRunner', max_epochs=4)
checkpoint_config = dict(interval=1, max_keep_ckpts=2)
evaluation = dict(interval=2)

# keep the raster-fusion alpha learned by the baseline (~0.54) instead of
# restarting the schedule at 0.2
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
