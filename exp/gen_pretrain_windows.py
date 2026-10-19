# -*- coding: utf-8 -*-
"""阶段一预训练数据：从原始波形滑窗生成 (13步观测 -> 24步 δ/ω 目标) 对。

每场景随机取 K=8 个窗口起点（覆盖暂态全程，含故障前/中/后），
窗口总长 37 步（13 观测 + 24 预测），1/60s 均匀插值。
输出 data/processed/pretrain_<dataset>_trainval.npz：
  ieee39:  X(N,13,30),  Yd/Yw(N,24,10), M(N,24)
  ieee118: X(N,13,162), Yd/Yw(N,24,54), M(N,24)

用法: python exp/gen_pretrain_windows.py [ieee39|ieee118]（默认 ieee39）
"""
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASETS = {
    "ieee39": dict(raw=ROOT / "data" / "IEEE39-Sim_生成脚本" / "IEEE39_Sim_2400.pkl",
                   windows="ieee39_windows.npz", n_mach=10,
                   out=ROOT / "data" / "processed" / "pretrain_ieee39_trainval.npz"),
    "ieee118": dict(raw=ROOT / "data" / "IEEE39-Sim_生成脚本" / "IEEE118_Sim_1200.pkl",
                    windows="ieee118_windows.npz", n_mach=54,
                    out=ROOT / "data" / "processed" / "pretrain_ieee118_trainval.npz"),
}

FS, T_PRED, K, SEED = 60, 24, 8, 2026


def uniform_grid(t, sig, t0, n_steps):
    grid = t0 + np.arange(n_steps) / FS
    out = np.empty((n_steps, sig.shape[1]), dtype=np.float32)
    mask = np.zeros(n_steps, dtype=np.float32)
    valid = grid <= (t[-1] + 1e-9)
    mask[valid] = 1.0
    for c in range(sig.shape[1]):
        out[:, c] = np.interp(grid, t, sig[:, c])
    out[~valid] = 0.0
    return out, mask


def main():
    ds = sys.argv[1] if len(sys.argv) > 1 else "ieee39"
    t_obs = 13
    if "w" in ds and ds.split("w")[0] in DATASETS:
        # 观测窗变体：ieee39w121 / ieee118w121 -> t{121} 窗口
        base, t_s = ds.split("w")
        cfg = dict(DATASETS[base])
        cfg["windows"] = f"{base}_windows_t{t_s}.npz"
        cfg["out"] = ROOT / "data" / "processed" / f"pretrain_{base}w{t_s}_trainval.npz"
        t_obs = int(t_s)
    else:
        cfg = dict(DATASETS[ds])
    nm = cfg["n_mach"]
    # 防泄漏：仅用 train∪val 且非 OOD 母线的场景（无标签轨迹预训练，
    # 但不得接触 test / OOD 场景的动态，否则 RQ2/RQ3 口径不干净）
    w = np.load(ROOT / "data" / "processed" / cfg["windows"])
    tv = np.concatenate([w["train_idx"], w["val_idx"]])
    allow = set(w["cid"][tv[w["ood_mask"][tv] == 0]].tolist())
    with open(cfg["raw"], "rb") as f:
        d = pickle.load(f)
    results = d["results"]
    results = [results[k] for k in sorted(results)] if isinstance(results, dict) else list(results)
    rng = np.random.default_rng(SEED)
    Xs, Yds, Yws, Ms = [], [], [], []
    n_skip = 0
    for i, res in enumerate(results):
        if i not in allow:
            n_skip += 1
            continue
        payload = res[2]
        if not isinstance(payload, tuple) or payload is None:
            n_skip += 1
            continue
        t, delta, omega, pe = payload
        if t is None or len(t) < 3:
            n_skip += 1
            continue
        sig = np.concatenate([delta, omega, pe], axis=1)
        dur = t[-1] - t[0]
        need = (t_obs + T_PRED) / FS
        if dur < need * 0.5:          # 波形过短，跳过
            n_skip += 1
            continue
        hi = max(t[0], t[-1] - need)
        starts = rng.uniform(t[0], hi, size=K) if hi > t[0] else [t[0]] * K
        for t0 in starts:
            w, m = uniform_grid(t, sig, t0, t_obs + T_PRED)
            Xs.append(w[:t_obs])
            Yds.append(w[t_obs:, :nm]); Yws.append(w[t_obs:, nm:2 * nm])
            Ms.append(m[t_obs:])
    X = np.stack(Xs).astype(np.float32)
    Yd = np.stack(Yds).astype(np.float32)
    Yw = np.stack(Yws).astype(np.float32)
    M = np.stack(Ms).astype(np.float32)
    np.savez_compressed(cfg["out"], X=X, Yd=Yd, Yw=Yw, M=M)
    print(f"pretrain windows: {len(X)} (skip {n_skip} scenarios), X{X.shape} -> {cfg['out']}")


if __name__ == "__main__":
    main()
