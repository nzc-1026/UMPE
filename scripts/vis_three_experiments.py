# -*- coding: utf-8 -*-
"""Render the same val frames for the three final experiments.

E1: vector + raster + satellite, residual 0.1 learnable, 12+4 epochs
E2: vector + raster + satellite, residual 1.0 fixed, 24 epochs from scratch
E3: vector + satellite, residual 1.0 fixed, 24 epochs from scratch
"""
import copy
import os
import sys

sys.path.insert(0, '/home/lbb/Documents/space/umpe_hdmap/UMPE-main/MapQR')
import mmcv
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import projects.mmdet3d_plugin  # noqa: F401
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

ROOT = '/home/lbb/Documents/space/umpe_hdmap/UMPE-main/MapQR'
RES = '/home/lbb/Documents/space/umpe_hdmap/results_20261025'
OUT = RES + '/04_figures/three_exp'
os.makedirs(OUT, exist_ok=True)

FRAME_INDICES = [30, 835, 985]
CLS = ['divider', 'ped_crossing', 'boundary']
COLORS = {0: '#ff4d4d', 1: '#33cc66', 2: '#4d79ff'}

MODELS = [
    {
        'name': 'E1: vector+raster, 0.1 residual (12+4)',
        'short': 'E1 0.1 residual (12+4)',
        'cfg': ROOT + '/projects/configs/mapqr/umpe_nocam_vector_ft.py',
        'ckpt': RES + '/03_vector_ft_4ep/best.pth',
        'raster': True,
    },
    {
        'name': 'E2: vector+raster, 1.0 residual (24)',
        'short': 'E2 1.0 residual (24)',
        'cfg': ROOT + '/projects/configs/mapqr/umpe_nocam_vector_raster_24ep.py',
        'ckpt': RES + '/10_vector_raster_24ep/work_dirs/best_NuscMap_chamfer/mAP_epoch_24.pth',
        'raster': True,
    },
    {
        'name': 'E3: vector only, 1.0 residual (24)',
        'short': 'E3 pure vector (24)',
        'cfg': ROOT + '/projects/configs/mapqr/umpe_nocam_vector_only_sat_24ep.py',
        'ckpt': RES + '/09_vector_only_sat_24ep/best.pth',
        'raster': False,
    },
]


def build_ds(cfg_path):
    cfg = mmcv.Config.fromfile(cfg_path)
    val_cfg = copy.deepcopy(cfg.data.val)
    val_cfg.pop('samples_per_gpu', None)
    ds = build_dataset(val_cfg)
    token_to_idx = {}
    for i, info in enumerate(ds.data_infos):
        token_to_idx[info['token']] = i
    return cfg, ds, token_to_idx


def prepare(ds, idx):
    input_dict = ds.get_data_info(idx)
    ds.pre_pipeline(input_dict)
    ex = ds.pipeline(input_dict)
    ex = ds.vectormap_pipeline(ex, input_dict)
    if ex is None:
        raise RuntimeError('pipeline returned None at idx %s' % idx)
    return ex, input_dict


def to_px(p):
    u = (p[..., 0] + 15.0) / 30.0 * 100.0
    v = (30.0 - p[..., 1]) / 60.0 * 200.0
    return u, v


def draw_lines(ax, polys, labels, scores=None, linestyle='-', lw=1.5,
               alpha=0.95, gt_ghost=False):
    for k, p in enumerate(polys):
        p = np.asarray(p)
        if p.ndim == 1:
            p = p.reshape(-1, 2)
        u, v = to_px(p)
        cls = int(labels[k]) if labels is not None else 0
        color = COLORS.get(cls, 'white')
        ax.plot(u, v, color=color, linestyle=linestyle, lw=lw,
                alpha=(0.22 if gt_ghost else alpha))
        if scores is not None:
            ax.text(u[0], v[0], f'{scores[k]:.2f}', color=color, fontsize=5,
                    alpha=0.85)


def draw_osm(ax, sd):
    for cls, lines in sd.items():
        for line in lines:
            p = np.asarray(line)
            if p.ndim == 1:
                p = p.reshape(-1, 2)
            u, v = to_px(p)
            ax.plot(u, v, color='#00e5ff', lw=0.5, alpha=0.55)


def infer(model, ex, use_raster):
    meta = ex['img_metas'].data
    sat = torch.from_numpy(np.asarray(ex['satellite_img'])).float()[None].cuda()
    kwargs = dict(return_loss=False, rescale=True, img_metas=[meta],
                  satellite_img=sat, sdmap=ex['sdmap'])
    if use_raster:
        kwargs['SDRaster_img'] = torch.from_numpy(
            np.asarray(ex['SDRaster_img'])).float()[None].cuda()
    with torch.no_grad():
        result = model(**kwargs)
    pred = result[0]['pts_bbox'] if 'pts_bbox' in result[0] else result[0]
    pts = pred['pts_3d']
    if torch.is_tensor(pts):
        pts = pts.detach().cpu().numpy()
    if pts.ndim == 2 and pts.shape[1] == 40:
        pts = pts.reshape(-1, 20, 2)
    scores = pred['scores_3d'].detach().cpu().numpy()
    labels = pred['labels_3d'].detach().cpu().numpy()
    keep = scores > 0.50
    if keep.sum() < 3:
        keep = np.argsort(-scores)[:10]
    elif keep.sum() > 20:
        keep = np.argsort(-scores)[:20]
    return pts[keep], scores[keep], labels[keep]


def main():
    # data + models
    data = {}
    for m in MODELS:
        cfg, ds, tok2idx = build_ds(m['cfg'])
        data[m['short']] = (cfg, ds, tok2idx)
        print('DATASET', m['short'], len(ds), flush=True)

    models = {}
    for m in MODELS:
        cfg, _, _ = data[m['short']]
        model = build_model(cfg.model)
        load_checkpoint(model, m['ckpt'], map_location='cpu')
        model.cuda().eval()
        models[m['short']] = model
        print('MODEL_LOADED', m['short'], flush=True)

    for idx in FRAME_INDICES:
        rows = []
        all_counts = []
        for m in MODELS:
            _, ds, tok2idx = data[m['short']]
            if idx >= len(ds):
                raise IndexError('idx %s out of range for %s' % (idx, m['short']))
            ex, input_dict = prepare(ds, idx)
            pts, scores, labels = infer(models[m['short']], ex, m['raster'])
            gt = ex['gt_bboxes_3d'].data
            gt_polys = [np.asarray(p) for p in gt.instance_list]
            gt_labels = np.asarray(gt.instance_labels)
            counts = [int((gt_labels == i).sum()) for i in range(3)]
            all_counts.append(counts)
            rows.append((m, ex, input_dict, pts, scores, labels,
                         gt_polys, gt_labels, counts))
            print('INFER', idx, m['short'], 'pred', len(pts),
                  'gt', len(gt_polys), counts, flush=True)
        if any(c != all_counts[0] for c in all_counts[1:]):
            raise RuntimeError('GT counts differ across datasets at idx %s: %s' % (idx, all_counts))

        # one figure per frame, 5 columns
        fig, axes = plt.subplots(2, 3, figsize=(6.6, 7.4))
        ex0 = rows[0][1]
        counts = rows[0][8]
        sd = ex0['sdmap'].data if hasattr(ex0['sdmap'], 'data') else ex0['sdmap']

        ax = axes[0, 0]
        ax.imshow(np.clip(np.asarray(ex0['satellite_img']), 0, 1))
        draw_osm(ax, sd)
        ax.set_title(f'satellite + OSM\nGT: {counts[0]}D/{counts[1]}P/{counts[2]}B',
                     fontsize=8.5)
        ax.set_xlim(0, 100); ax.set_ylim(200, 0); ax.axis('off')

        panel_axes = [axes[0, 1], axes[0, 2], axes[1, 0]]
        for j, (m, ex, input_dict, pts, scores, labels, gt_polys,
                gt_labels, _) in enumerate(rows):
            ax = panel_axes[j]
            ax.set_facecolor('black')
            # faint GT for alignment, then solid prediction
            draw_lines(ax, gt_polys, gt_labels, linestyle='--', lw=0.9,
                       gt_ghost=True)
            draw_lines(ax, pts, labels, scores, lw=1.4)
            ax.set_title(f"{m['short']}\n{len(pts)} shown", fontsize=8)
            ax.set_xlim(0, 100); ax.set_ylim(200, 0); ax.axis('off')

        ax = axes[1, 1]
        ax.set_facecolor('black')
        draw_lines(ax, rows[0][6], rows[0][7], lw=1.4)
        ax.set_title(f'GT ({len(rows[0][6])} instances)', fontsize=8.5)
        ax.set_xlim(0, 100); ax.set_ylim(200, 0); ax.axis('off')

        # E3 + GT overlay (best model)
        ax = axes[1, 2]
        ax.set_facecolor('black')
        draw_lines(ax, rows[2][6], rows[2][7], linestyle='--', lw=1.0,
                   alpha=0.85)
        draw_lines(ax, rows[2][3], rows[2][5], rows[2][4], lw=1.4)
        ax.set_title('E3 + GT overlay\n(GT dashed / pred solid)', fontsize=8)
        ax.set_xlim(0, 100); ax.set_ylim(200, 0); ax.axis('off')

        handles = [Line2D([0], [0], color=COLORS[i], lw=2, label=CLS[i])
                   for i in range(3)]
        fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=8)
        token = str(rows[0][2].get('token', rows[0][2].get('scene_token', idx)))[:8]
        fig.suptitle(f'frame idx {idx} ({token})  |  threshold 0.50',
                     fontsize=9)
        plt.tight_layout(rect=[0, 0.04, 1, 0.94])
        out = os.path.join(OUT, f'three_exp_idx{idx}_2x3.png')
        plt.savefig(out, dpi=300)
        plt.close(fig)
        print('SAVED', out, flush=True)

    print('ALL_DONE', flush=True)


if __name__ == '__main__':
    main()
