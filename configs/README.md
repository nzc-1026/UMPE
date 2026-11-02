# Configs

The experiment configs are under `configs/mapqr/`; `configs/datasets/` and `configs/_base_/`
are the two small base files they inherit. This mirrors the original MapQR config layout so all
relative `_base_` paths resolve unchanged.

| File | Experiment | Notes |
|---|---|---|
| `mapqr_nusc_r50_24ep_new_Raster.py` | base | Original MapQR raster config; camera/HD/priors |
| `umpe_nocam_raster.py` | A0 baseline | Raster SD + satellite, 12 epochs, no camera |
| `umpe_nocam_vector.py` | base | Vector + raster + satellite, 12 epochs |
| `umpe_nocam_vector_ft.py` | **E1** | 12+4; residual 0.1 learnable; edit `load_from` to A0 checkpoint |
| `umpe_nocam_vector_raster_24ep.py` | **E2** | 24 from scratch; residual 1.0 fixed; 4×batch 5 |
| `umpe_nocam_vector_only_sat.py` | base | Vector + satellite; removes raster pipeline |
| `umpe_nocam_vector_only_sat_24ep.py` | **E3** | 24 from scratch; residual 1.0 fixed; 5×batch 4 |

All configs still contain the experiment server's absolute `/home/lbb/...` paths. Replace
`_ROOT`, `_SD_PKL`, and `load_from` with your own paths. The `_base_` references are relative and
work as long as these files stay in the same directory.

The three experiment configs are intentionally kept with their original filenames so the
`_base_` chain remains exactly the one used in training.
