# -*- coding: utf-8 -*-
"""P1-9 固定队列窗口消融：训练入口。

对队列受限数据集 ieee39c{13,31,61,121} 训练 pimtnet/svm/xgboost（5 种子），
协议与原窗口消融一致（单阶段调优配置：cosine 3e-4、patience 50、
epochs 500、λt=1.0、λp=0.1、pimt_best.json 结构超参）。
所有窗口共用同一 test split（_stratified_split(y,2026)，329+328 中的 328 例测试）。
结果写入 results/{model}_wc/ieee39c{t}_main_seed{0-4}.json。

用法：python run_window_cohort.py [--models pimtnet,svm,xgboost]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from common.dataset import LOADERS, _load_windows_arrays

for t in (13, 31, 61, 121):
    name = f"ieee39c{t}"
    LOADERS[name] = (lambda f: (lambda tag="main": _load_windows_arrays(f, tag)))(
        f"ieee39_windows_t{t}_cohort.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="pimtnet,svm,xgboost")
    args = ap.parse_args()

    for t in (13, 31, 61, 121):
        ds = f"ieee39c{t}"
        sys.argv = ["train.py", "--models", args.models, "--dataset", ds,
                    "--tag", "main", "--seeds", "5", "--sfx", "_wc",
                    "--cosine", "--lr", "3e-4", "--epochs", "500",
                    "--patience", "50", "--lam-t", "1.0",
                    "--pimt-cfg", "@pimt_best.json"]
        print(f"\n===== window t{t} ({ds}) =====", flush=True)
        t0 = time.time()
        import train
        train.main()
        torch.cuda.empty_cache()
        print(f"[window t{t} done in {time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
