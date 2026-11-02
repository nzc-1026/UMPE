_base_ = ['./mapqr_nusc_r50_24ep_new_Raster.py']

# ---------------------------------------------------------------
# Satellite + rasterized SD only, no camera, no HD prior.
# ---------------------------------------------------------------
_ROOT = '/home/lbb/Documents/space/umpe_hdmap/data/nuscenes/'

_AUX_OFF = dict(
    use_aux_seg=False,
    bev_seg=False,
    pv_seg=False,
    seg_classes=1,
    feat_down_sample=32,
    pv_thickness=1,
)

total_epochs = 12
find_unused_parameters = True
fp16 = None

# remove the fine-tuning parameter groups from the base config
optimizer = dict(
    type='AdamW',
    lr=6e-4,
    weight_decay=0.01,
    paramwise_cfg=None,
)

runner = dict(type='EpochBasedRunner', max_epochs=12)
checkpoint_config = dict(interval=1, max_keep_ckpts=3)

model = dict(
    use_grid_mask=False,
    img_backbone=None,
    img_neck=None,
    pretrained=None,
    pts_bbox_head=dict(
        aux_seg=dict(
            use_aux_seg=False,
            bev_seg=False,
            pv_seg=False,
            seg_classes=1,
            feat_down_sample=32,
        ),
    ),
)

train_pipeline = [
    dict(type='LoadSatelliteFromFile', to_float32=True, normalize=True),
    dict(type='LoadSDRasterFromFile', to_float32=True, normalize=True),
    dict(type='DefaultFormatBundle3D', with_gt=False, with_label=False,
         class_names=['divider', 'ped_crossing', 'boundary']),
    dict(type='CustomCollect3D', keys=['satellite_img', 'SDRaster_img']),
]

test_pipeline = [
    dict(type='LoadSatelliteFromFile', to_float32=True, normalize=True),
    dict(type='LoadSDRasterFromFile', to_float32=True, normalize=True),
    dict(type='DefaultFormatBundle3D', with_gt=False, with_label=False,
         class_names=['divider', 'ped_crossing', 'boundary']),
    dict(type='CustomCollect3D', keys=['satellite_img', 'SDRaster_img']),
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        data_root=_ROOT,
        ann_file=_ROOT + 'nuscenes_map_infos_temporal_train.pkl',
        pipeline=train_pipeline,
        aux_seg=_AUX_OFF,
    ),
    val=dict(
        data_root=_ROOT,
        ann_file=_ROOT + 'nuscenes_map_infos_temporal_val.pkl',
        map_ann_file=_ROOT + 'nuscenes_map_anns_val.json',
        pipeline=test_pipeline,
        aux_seg=_AUX_OFF,
    ),
    test=dict(
        data_root=_ROOT,
        ann_file=_ROOT + 'nuscenes_map_infos_temporal_val.pkl',
        map_ann_file=_ROOT + 'nuscenes_map_anns_val.json',
        pipeline=test_pipeline,
        aux_seg=_AUX_OFF,
    ),
)
