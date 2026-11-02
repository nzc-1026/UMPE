# -*- coding: utf-8 -*-
"""Scan nuScenes val frames that contain divider, ped_crossing and boundary GT."""
import copy
import sys

sys.path.insert(0, '/home/lbb/Documents/space/umpe_hdmap/UMPE-main/MapQR')
import mmcv
import numpy as np
from mmdet3d.datasets import build_dataset
import projects.mmdet3d_plugin  # noqa: F401

ROOT = '/home/lbb/Documents/space/umpe_hdmap/UMPE-main/MapQR'
CFG = ROOT + '/projects/configs/mapqr/umpe_nocam_vector_only_sat_24ep.py'

cfg = mmcv.Config.fromfile(CFG)
val_cfg = copy.deepcopy(cfg.data.val)
val_cfg.pop('samples_per_gpu', None)
ds = build_dataset(val_cfg)
print('VAL_LEN', len(ds), flush=True)


def prepare(idx):
    input_dict = ds.get_data_info(idx)
    ds.pre_pipeline(input_dict)
    ex = ds.pipeline(input_dict)
    ex = ds.vectormap_pipeline(ex, input_dict)
    if ex is None:
        return None
    gt = ex['gt_bboxes_3d'].data
    labels = np.asarray(gt.instance_labels)
    counts = [int((labels == i).sum()) for i in range(3)]
    token = input_dict.get('token', input_dict.get('scene_token', str(idx)))
    return token, counts, len(labels)


found = 0
for idx in range(0, len(ds), 5):
    r = prepare(idx)
    if r is None:
        continue
    token, counts, n = r
    if min(counts) >= 1 and 5 <= n <= 14 and counts[1] >= 2:
        print('CAND', idx, token, 'counts', counts, 'total', n, flush=True)
        found += 1
        if found >= 20:
            break
print('DONE', found, flush=True)
