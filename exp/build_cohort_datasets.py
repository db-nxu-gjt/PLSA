# -*- coding: utf-8 -*-
"""P1-9 固定队列（fixed-cohort）窗口消融：数据构建。

以 t241（4.0s 窗）存活的 1,642 个场景为固定队列，通过 cid 将其映射回
t13/t31/t61/t121 各窗口变体，生成队列受限 npz（ieee39_windows_t{t}_cohort.npz）。
train/val/test 用与 data_pipeline 同口径的 _stratified_split(y, seed=2026) 重算——
由于队列与标签相同，所有窗口得到完全一致的 985/329/328 划分（与 w241 现有
split 相同，故 4.0s 点可直接复用既有 ieee39w241 结果）。
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.dataset import PROCESSED, _stratified_split

WINDOWS = [13, 31, 61, 121]


def fname(t):
    return "ieee39_windows.npz" if t == 13 else f"ieee39_windows_t{t}.npz"


def main():
    d241 = np.load(PROCESSED / "ieee39_windows_t241.npz")
    cid_cohort = [int(c) for c in d241["cid"]]
    y_cohort = d241["y"]
    tr, va, te = _stratified_split(y_cohort, 2026)
    assert np.array_equal(te, d241["test_idx"]), "w241 split mismatch"
    print(f"cohort n={len(y_cohort)}  tr/va/te={len(tr)}/{len(va)}/{len(te)}  "
          f"unstable={y_cohort.mean()*100:.1f}%")

    for t in WINDOWS:
        d = np.load(PROCESSED / fname(t))
        cid2row = {int(c): i for i, c in enumerate(d["cid"])}
        rows = np.array([cid2row[c] for c in cid_cohort])
        assert np.array_equal(d["y"][rows], y_cohort)
        out = dict(
            X=d["X"][rows].astype(np.float32),
            desc=d["desc"][rows].astype(np.float32),
            y=d["y"][rows],
            cid=d["cid"][rows],
            bus=d["bus"][rows],
            ood_mask=d["ood_mask"][rows],
            traj_delta=d["traj_delta"][rows].astype(np.float32),
            traj_omega=d["traj_omega"][rows].astype(np.float32),
            traj_mask=d["traj_mask"][rows],
            train_idx=tr, val_idx=va, test_idx=te,
        )
        # subset_25/50：从队列训练集分层抽取（本实验不用，保持结构完整）
        rng = np.random.default_rng(2026)
        sub = []
        for frac in (0.25, 0.5):
            idx = []
            for cls in np.unique(y_cohort[tr]):
                c = np.where(y_cohort[tr] == cls)[0]
                rng.shuffle(c)
                idx += list(c[: int(round(frac * len(c)))])
            sub.append(np.sort(tr[np.array(idx, dtype=int)]))
        out["subset_25_idx"], out["subset_50_idx"] = sub
        dst = PROCESSED / f"ieee39_windows_t{t}_cohort.npz"
        np.savez_compressed(dst, **out)
        print(f"[ok] t{t}: X{out['X'].shape} -> {dst.name}")


if __name__ == "__main__":
    main()
