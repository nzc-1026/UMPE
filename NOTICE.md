# Provenance and notice

This repository is a **research code snapshot** for three experiments. It is not a standalone
release of MapQR, UMPE, MMDetection3D, or nuScenes.

- MapQR / MapTRv2 code is from the original open-source MapQR repository.
- UMPE vector modules (`MapGraphTransformer`, `SineContinuousPositionalEncoding`,
  `CrossAttentionBEV`, `DualVecCrossFusion`) are from the original UMPE repository; they were
  ported into the MapQR plugin for this reproduction.
- The camera-free learnable-BEV path, the residual switch in `DualVecCrossFusion`, the
  `PackSDMap`/sdmap loading, the E1/E2/E3 configs, and the same-frame visualization scripts are
  adaptations made for this work.
- No nuScenes data, UMPE-Prior data, generated GT, model checkpoints, or training logs are included.

Keep and follow the original licenses of MapQR/UMPE and all third-party dependencies
(MMCV, MMDetection, MMDetection3D, nuScenes devkit, etc.). This snapshot does not relicense them.

The numbers in `EXPERIMENTS.md` are the results of the experiments described there. E1 is a
12+4 short fine-tune and is not a clean 24-epoch ablation; E2/E3 are the 24-epoch comparison.
