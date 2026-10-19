# PLSA: Physics-Informed LLM Synergy for Power-System Transient Stability Assessment

Code and result archive for the paper *"PLSA: A physics-informed neural network and large
language model synergy framework for power-system security deduction and transient stability
assessment"*.

The framework couples two engines:

- **PIMT-Net** — a physics-informed multi-task network that embeds swing-equation and
  kinematic residuals (evaluated on its own *predicted* trajectories) as differentiable
  losses in a joint stability-classification / rotor-angle-trajectory framework with
  per-machine learnable `M`, `D`, `Pm`.
- **PIVN + PhysCheck** — an LLM orchestration loop (intent understanding, scenario
  generation, tool calling, report narration) closed through a physics-verification layer
  so that every numeric claim in the final report is traceable to a registered tool.

All experiment JSONs produced for the paper are committed under
[`exp/results/`](exp/results), so every table and figure can be regenerated **without
retraining**.

## Repository layout

```
├── exp/                     # all experiment code
│   ├── common/              # dataset loaders, trainer, metrics
│   ├── models/              # PIMT-Net, deep baselines, physics-informed baselines, sklearn
│   ├── results/             # per-model/per-tag JSON results (committed)
│   ├── train.py             # unified training entry (protocol of paper §4.2)
│   ├── data_pipeline.py     # ANDES simulation -> npz dataset builder (IEEE39-Sim/IEEE118-Sim)
│   ├── aggregate.py         # results -> console tables
│   ├── significance_tests.py# exact Mann-Whitney U + Holm, Wilcoxon, rank-biserial
│   ├── ensemble_eval.py     # 10-seed ensemble, McNemar, bootstrap CIs
│   ├── param_ident.py       # M/D/Pm identifiability + s-conditioned control
│   ├── llm_direct_baseline.py # DeepSeek-V3 zero-/few-shot direct TSA judgment
│   ├── pivn_demo.py         # PIVN orchestration + PhysCheck case study
│   ├── train_fault_id.py    # fault-location identification heads
│   ├── build_cohort_datasets.py / run_window_cohort.py  # fixed-cohort window control
│   ├── window_ablation_report.py / window_cohort_report.py
│   ├── tune_pimtnet.py      # hyperparameter search (-> pimt_best.json)
│   ├── gen_pretrain_windows.py # trajectory pre-training corpora
│   ├── make_figures.py / make_figures_en.py # zh / en paper figures
│   └── pimt_best.json       # best PIMT-Net hyperparameters
├── data/processed/          # place downloaded npz here (see data/README.md)
└── figs_en/, figs_zh/, export/  # generated outputs (gitignored)
```

## Environment

```bash
conda create -n plsa python=3.10 -y
conda activate plsa
# PyTorch (CUDA 11.8 build used for the paper; pick the wheel for your CUDA)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

`andes` is only needed to re-generate raw simulation data from scratch
(`exp/data_pipeline.py`, `exp/param_ident.py` swing sweeps); training and all
analyses run from the shipped npz files alone.

## Data

`exp/common/dataset.py` expects `data/processed/*.npz` (dataset root is
`<repo>/data/processed`). Download the dataset archive from Zenodo
(DOI: `10.5281/zenodo.TODO`) and unpack it into `data/processed/`, then verify
against `data/processed/manifest.json`. The full file manifest and a
regeneration recipe are in [`data/README.md`](data/README.md).

## Reproducing the paper

All commands run from the repository root. `--seeds N` controls seed count; every
run writes JSONs to `exp/results/<model>/` and is skipped automatically if the file
already exists (safe to re-run).

| Paper artifact | Command |
|---|---|
| Main table (14 models × 4 dataset×window cells) | `python exp/train.py --models all --tag main --seeds 5` |
| PIMT-Net (tuned protocol, saves ensemble probs) | `python exp/train.py --models pimtnet --dataset ieee39 --tag main --seeds 10 --cosine --lr 3e-4 --epochs 500 --patience 50 --lam-t 1.0 --pimt-cfg @pimt_best.json --save-probs` |
| Long-window ablation (0.5/1.0/2.0/4.0 s) | `python exp/train.py --models pimtnet,svm,xgboost --dataset ieee39w121 --tag main --seeds 5` (repeat for `ieee39w31`, `ieee39w61`, `ieee39w241`) |
| Fixed-cohort control (1,642 scenarios) | `python exp/build_cohort_datasets.py && python exp/run_window_cohort.py --models pimtnet,svm,xgboost` |
| Noise robustness (σ = 1/2/5 %) | `python exp/train.py --models all_core --tag main --eval-tags noise1,noise2,noise5 --seeds 10` |
| Few-shot (25 % / 50 % training data) | `python exp/train.py --models pimtnet,svm,xgboost --tag sub25 --seeds 5` |
| Ensemble + McNemar + bootstrap (needs `--save-probs` runs) | `python exp/ensemble_eval.py --dataset ieee39` |
| Significance tests (Mann–Whitney + Holm) | `python exp/significance_tests.py` |
| M/D/Pm identifiability + s-conditioned model | `python exp/param_ident.py` |
| LLM direct-judgment baseline (full test set) | `set LLM_API_KEY=... && python exp/llm_direct_baseline.py --mode zeroshot --n 10000 --name llm_direct_zeroshot_full` |
| PIVN / PhysCheck case study | `set LLM_API_KEY=... && python exp/pivn_demo.py` |
| Fault-location identification | `python exp/train_fault_id.py` |
| Console summary of all results | `python exp/aggregate.py --filter tag=main` |
| All figures (EN) / (ZH) | `python exp/make_figures_en.py` / `python exp/make_figures.py` |

Smoke test with a tiny budget:

```bash
python exp/train.py --models mlp --dataset ieee39 --tag main --seeds 1 --epochs 3 --patience 3
python exp/window_cohort_report.py        # needs exp/results (committed) — no GPU required
```

## Results shipped in this repo

`exp/results/` contains the complete JSON record of every run used in the paper
(main / noise / few-shot / OOD / window / fixed-cohort / ablation / ensemble /
parameter-identification / LLM-baseline / case-study). `python exp/dump_paper_numbers.py`
prints the headline numbers and where they come from.

## Hardware

All experiments ran on a single 16 GB laptop GPU (RTX A5500); no multi-GPU or cluster
resources are required. SVM/XGBoost are CPU-only.

## Citation

```bibtex
@article{plsa2026,
  title   = {PLSA: A physics-informed neural network and large language model synergy
             framework for power-system security deduction and transient stability assessment},
  author  = {TODO},
  journal = {Expert Systems with Applications},
  year    = {2026},
  note    = {Code: https://github.com/TODO/plsa}
}
```

## License

Code is released under the [MIT License](LICENSE). The simulation datasets are
distributed via Zenodo (see `data/README.md`).
