# Data

`exp/common/dataset.py` resolves the dataset root as `<repo>/data/processed`.
Place the downloaded `.npz` files there:

```
data/processed/
├── manifest.json                      # sizes + sha256 of every file (integrity check)
├── ieee39_windows.npz                 # IEEE39-Sim, 2,349 scenarios, 0.2 s window (~11 MB)
├── ieee39_windows_t31.npz             # 0.5 s variant (~16 MB)
├── ieee39_windows_t61.npz             # 1.0 s variant (~23 MB)
├── ieee39_windows_t121.npz            # 2.0 s variant (~40 MB)
├── ieee39_windows_t241.npz            # 4.0 s variant (~66 MB, 1,642 surviving scenarios)
├── ieee39_windows_t{13,31,61,121}_cohort.npz  # fixed-cohort variants (~5-24 MB)
├── ieee118_windows.npz                # IEEE118-Sim, 1,198 scenarios, 0.2 s (~30 MB)
├── ieee118_windows_t121.npz           # 2.0 s variant (~130 MB)
├── pf39.npz                           # PF-39 public benchmark features (~91 MB)
└── ne39_features.npz                  # NE39-Open public benchmark features (~1.5 MB)
```

## Download

Zenodo DOI: `10.5281/zenodo.TODO` (placeholder — update before release).

After unpacking, verify integrity:

```bash
python -c "import json,hashlib,pathlib; d=json.load(open('data/processed/manifest.json'))"
```

`exp/verify_data.py` performs shape/label sanity checks on every dataset.

## Regenerating from scratch

1. Install the ANDES simulator (`pip install andes`).
2. `python exp/data_pipeline.py` builds IEEE39-Sim / IEEE118-Sim scenario pools
   (fault set × operating points × contingencies, GENROU machines, 60 Hz),
   runs time-domain simulation, and exports the `*_windows*.npz` files above.
3. `python exp/gen_pretrain_windows.py` additionally exports the trajectory
   pre-training corpora (`pretrain_*_trainval.npz`, ~100-380 MB, needed only for
   the two-stage PIMT-Net pre-training ablation).
4. Public benchmarks: PF-39 and NE39-Open are downloaded from their original
   publications and converted by the same pipeline (see paper §4.1).

The raw simulation archives (ANDES checkpoints, COMTRADE waveforms; several GB)
are included in the Zenodo deposit but are not required for training or analysis.
