# -*- coding: utf-8 -*-
"""PLSA 论文数据管线：从原始仿真 pkl 构建训练用窗口数据集。

对应论文 4.1 节实验设计，产出（paper/data/processed/）：
  ieee39_windows.npz   观测窗 (N,13,30) + 场景量 (N,5) + 标签 + 未来24步轨迹目标
  ieee118_windows.npz  同协议，54 台机组 -> (N,13,162)
  pf39.npz             PF-39 (12852,60,50)，标签已翻转为 1=失稳
  ne39_features.npz    NE39-Open 特征版 (3120,354) + 标签
  manifest.json        全部统计信息

关键口径（与论文一致）：
  - 观测窗: 故障后 0.2s，60Hz -> 13 步；通道 = delta/omega/Pe × N 机
  - 场景量 5 维: [load_scale, M_scale, fault_bus/max_bus, dur, has_trip]
  - 标签: diverged 或全程最大功角差>180° -> 1(失稳)，复用生成脚本 label_case 规则
  - 划分: 全量 60/20/20 分层随机(seed=2026)；另选 8 个母线整体留出作 OOD 测试
  - 小样本: 训练集 25%/50% 分层子集索引
  - 噪声: 仅对测试集注入 σ∈{1%,2%,5%} 高斯噪声（σ×各通道训练集std，固定种子）

用法: python exp/data_pipeline.py
"""
import json
import pickle
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "IEEE39-Sim_生成脚本"
OUT = ROOT / "data" / "processed"

FS = 60            # 采样率 Hz
T_OBS = 13         # 观测窗步数（0.2s 含故障时刻）
T_PRED = 24        # 轨迹预测步数
TF = 1.0           # 故障发生时刻（生成脚本固定 tf=1.0）
OOD_N_BUS = 8      # 未见故障位置留出母线数（ieee39，8/39≈20% 场景）
OOD_N_BUS_118 = 24  # ieee118 同比例（24/118≈20% 母线）
SEED = 2026
NOISE_SIGMAS = (0.01, 0.02, 0.05)
SUBSET_FRACS = (0.25, 0.50)


# ---------------------------------------------------------------- 基础工具
def load_sim_pkl(path: Path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    plans = list(d["plans"].values()) if isinstance(d["plans"], dict) else list(d["plans"])
    results = d["results"]
    results = [results[k] for k in sorted(results)] if isinstance(results, dict) else list(results)
    return plans, results


def label_case(status: str, payload):
    """复用生成脚本 label_case: diverged 或最大功角差>180° -> 1(失稳)。"""
    if not isinstance(payload, tuple) or payload is None:
        return None, None
    _, delta, _, _ = payload
    spread = float(np.rad2deg(delta.max(axis=1) - delta.min(axis=1)).max())
    return (1 if (status == "diverged" or spread > 180.0) else 0), spread


def uniform_grid(t: np.ndarray, sig: np.ndarray, t0: float, n_steps: int, fs: int = FS):
    """把非均匀时间序列插值到 t0 起始的均匀 1/fs 网格，返回 (n_steps, n_ch) 与有效掩码。"""
    grid = t0 + np.arange(n_steps) / fs
    out = np.empty((n_steps, sig.shape[1]), dtype=np.float32)
    mask = np.zeros(n_steps, dtype=np.float32)
    valid = grid <= (t[-1] + 1e-9)
    mask[valid] = 1.0
    for c in range(sig.shape[1]):
        out[:, c] = np.interp(grid, t, sig[:, c])
    out[~valid] = 0.0
    return out, mask


# ---------------------------------------------------------------- 窗口构建
def build_windows(plans, results, case_name: str, t_obs: int = T_OBS) -> dict:
    """提取观测窗/场景量/标签/轨迹目标。跳过 pflow_fail/error 及波形不足观测窗的场景。"""
    n_skip_bad, n_skip_short = 0, 0
    xs, descs, ys, td, tw, tp, tm, buses, cids = [], [], [], [], [], [], [], [], []
    for plan, res in zip(plans, results):
        cid, status, payload = res[0], res[1], res[2]
        if not isinstance(payload, tuple) or payload is None:
            n_skip_bad += 1
            continue
        t, delta, omega, pe = payload
        t_end_obs = TF + (t_obs - 1) / FS
        if t is None or len(t) < 3 or t[-1] < t_end_obs - 1e-9:
            n_skip_short += 1
            continue
        y, _ = label_case(status, payload)
        # 观测窗 [tf, tf+(t_obs-1)/fs]
        x_obs, _ = uniform_grid(t, np.concatenate([delta, omega, pe], axis=1), TF, t_obs)
        # 轨迹目标：窗口结束后 24 步
        g_td, m_td = uniform_grid(t, delta, TF + t_obs / FS, T_PRED)
        g_tw, _ = uniform_grid(t, omega, TF + t_obs / FS, T_PRED)
        g_tp, _ = uniform_grid(t, pe, TF + t_obs / FS, T_PRED)
        # 场景量 5 维: load, M, bus_norm, dur, has_trip
        _, _, load_s, m_s, bus, _, dur, trip = plan[:8]
        desc = [load_s, m_s, bus / 39.0 if case_name == "ieee39" else bus / 118.0,
                dur, 0.0 if trip is None else 1.0]
        xs.append(x_obs); descs.append(desc); ys.append(y)
        td.append(g_td); tw.append(g_tw); tp.append(g_tp); tm.append(m_td)
        buses.append(bus); cids.append(cid)
    print(f"  [{case_name}] kept {len(ys)}, skipped bad={n_skip_bad}, short={n_skip_short}")
    return dict(
        X=np.stack(xs).astype(np.float32),
        desc=np.asarray(descs, dtype=np.float32),
        y=np.asarray(ys, dtype=np.int64),
        traj_d=np.stack(td).astype(np.float32),
        traj_w=np.stack(tw).astype(np.float32),
        traj_p=np.stack(tp).astype(np.float32),
        traj_m=np.stack(tm).astype(np.float32),
        bus=np.asarray(buses, dtype=np.int64),
        cid=np.asarray(cids, dtype=np.int64),
    )


# ---------------------------------------------------------------- 划分与增强
def stratified_split(y: np.ndarray, seed=SEED):
    """60/20/20 分层随机划分，返回 (train_idx, val_idx, test_idx)。"""
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_tr, n_va = int(round(0.6 * n)), int(round(0.2 * n))
        tr += list(idx[:n_tr]); va += list(idx[n_tr:n_tr + n_va]); te += list(idx[n_tr + n_va:])
    return (np.sort(np.array(tr)), np.sort(np.array(va)), np.sort(np.array(te)))


def stratified_subset(y, train_idx, frac, seed=SEED):
    """从训练集中按标签分层抽取 frac 比例的子集索引。"""
    rng = np.random.default_rng(seed + int(frac * 100))
    keep = []
    for cls in np.unique(y[train_idx]):
        idx = train_idx[y[train_idx] == cls]
        rng.shuffle(idx)
        keep += list(idx[:max(1, int(round(frac * len(idx))))])
    return np.sort(np.array(keep))


def add_noise(X_ref: np.ndarray, X_test: np.ndarray, sigma: float, seed=SEED):
    """注入高斯量测噪声: N(0, (sigma*robust_std_channel)^2)，固定种子可复现。

    robust_std 用 MAD(1.4826*median|x-med|) 估计——故障期 Pe 尖峰（std 高达 ~850 pu）
    会使普通 std 比例噪声失真，MAD 抗离群，量测噪声应对应正常运行量级。
    """
    rng = np.random.default_rng(seed)
    ref = X_ref.reshape(-1, X_ref.shape[2])
    med = np.median(ref, axis=0)
    mad = np.median(np.abs(ref - med), axis=0) * 1.4826 + 1e-12
    noise = rng.normal(0.0, 1.0, X_test.shape).astype(np.float32)
    return (X_test + noise * sigma * mad).astype(np.float32)


# ---------------------------------------------------------------- 公开数据集
def export_pf39():
    import os
    ds = ROOT / "data" / "PF-39" / "tsa_dataset"
    old = os.getcwd(); os.chdir(ds)
    try:
        import sys
        sys.path.insert(0, str(ds))
        import dataset_loader
        b = dataset_loader.load_tsa_data()
    finally:
        os.chdir(old)
    X = b.data.astype(np.float32)
    y = (b.target == 0).astype(np.int64)   # 官方 1=stable/0=unstable -> 翻转为 1=失稳
    return X, y, list(b.feature_names)


def export_ne39():
    import pandas as pd
    df = pd.read_csv(ROOT / "data" / "NE39-Open" / "PowerGridFeatures" / "GridDictionary2.csv")
    y = df["Stability"].to_numpy().astype(np.int64)   # 官方 1=unstable
    X = df.drop(columns=["Stability"]).to_numpy().astype(np.float32)
    return X, y, [c for c in df.columns if c != "Stability"]


# ---------------------------------------------------------------- 主流程
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["ieee39", "ieee118", "pf39", "ne39"], default=None,
                    help="只重建指定数据集（默认全部）；ieee118 现含完整字段")
    ap.add_argument("--tobs", type=int, default=T_OBS,
                    help="观测窗步数（默认 13=0.2s；窗口消融用 31/61/121/241）")
    ap.add_argument("--suffix", type=str, default="",
                    help="输出文件后缀（如 _t121）；仅与 --only ieee39/ieee118 组合使用")
    args = ap.parse_args()
    if args.suffix and args.only not in ("ieee39", "ieee118"):
        raise SystemExit("--suffix 仅支持 --only ieee39/ieee118")
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"fs": FS, "T_obs": T_OBS, "T_pred": T_PRED, "tf": TF, "seed": SEED,
                "noise_sigmas": NOISE_SIGMAS, "subset_fracs": SUBSET_FRACS}
    if args.only:
        mpath = OUT / "manifest.json"
        if mpath.exists():
            manifest.update(json.loads(mpath.read_text(encoding="utf-8")))

    # ---- IEEE39-Sim ----
    if args.only in (None, "ieee39"):
        print(f"== IEEE39-Sim (tobs={args.tobs}, suffix='{args.suffix}') ==")
        plans, results = load_sim_pkl(RAW / "IEEE39_Sim_2400.pkl")
        d39 = build_windows(plans, results, "ieee39", t_obs=args.tobs)
        n = len(d39["y"])
        tr, va, te = stratified_split(d39["y"])
        split = np.empty(n, dtype=np.int64); split[tr] = 0; split[va] = 1; split[te] = 2
        # OOD: 留出 8 个母线（从出现过的故障母线中选）
        rng = np.random.default_rng(SEED)
        all_buses = np.unique(d39["bus"])
        ood_buses = np.sort(rng.choice(all_buses, size=OOD_N_BUS, replace=False))
        ood_mask = np.isin(d39["bus"], ood_buses).astype(np.int64)
        sub25 = stratified_subset(d39["y"], tr, 0.25)
        sub50 = stratified_subset(d39["y"], tr, 0.50)
        X_te = d39["X"][te]
        noisy = {f"X_test_noisy_{int(s*100)}pct": add_noise(d39["X"][tr], X_te, s)
                 for s in NOISE_SIGMAS}
        np.savez_compressed(
            OUT / f"ieee39_windows{args.suffix}.npz",
            X=d39["X"], desc=d39["desc"], y=d39["y"], split=split, ood_mask=ood_mask,
            bus=d39["bus"], cid=d39["cid"], ood_buses=ood_buses,
            train_idx=tr, val_idx=va, test_idx=te, subset_25_idx=sub25, subset_50_idx=sub50,
            traj_delta=d39["traj_d"], traj_omega=d39["traj_w"], traj_pe=d39["traj_p"],
            traj_mask=d39["traj_m"], **noisy)
        manifest[f"ieee39{args.suffix or ''}"] = dict(
                n=int(n), unstable_pct=round(d39["y"].mean() * 100, 1),
                n_train=len(tr), n_val=len(va), n_test=len(te),
                ood_buses=ood_buses.tolist(), n_ood=int(ood_mask.sum()),
                n_sub25=len(sub25), n_sub50=len(sub50),
                channels=int(d39["X"].shape[2]), t_obs=args.tobs)

    # ---- IEEE118-Sim ----
    if args.only in (None, "ieee118"):
        print(f"== IEEE118-Sim (tobs={args.tobs}, suffix='{args.suffix}') ==")
        plans, results = load_sim_pkl(RAW / "IEEE118_Sim_1200.pkl")
        d118 = build_windows(plans, results, "ieee118", t_obs=args.tobs)
        n = len(d118["y"])
        tr, va, te = stratified_split(d118["y"])
        split = np.empty(n, dtype=np.int64); split[tr] = 0; split[va] = 1; split[te] = 2
        # OOD: 与 ieee39 同比例（8/39≈20%）-> 留出 24 个母线
        rng = np.random.default_rng(SEED)
        all_buses = np.unique(d118["bus"])
        ood_buses = np.sort(rng.choice(all_buses, size=OOD_N_BUS_118, replace=False))
        ood_mask = np.isin(d118["bus"], ood_buses).astype(np.int64)
        sub25 = stratified_subset(d118["y"], tr, 0.25)
        sub50 = stratified_subset(d118["y"], tr, 0.50)
        X_te = d118["X"][te]
        noisy = {f"X_test_noisy_{int(s*100)}pct": add_noise(d118["X"][tr], X_te, s)
                 for s in NOISE_SIGMAS}
        np.savez_compressed(
            OUT / f"ieee118_windows{args.suffix}.npz",
            X=d118["X"], desc=d118["desc"], y=d118["y"], split=split, ood_mask=ood_mask,
            bus=d118["bus"], cid=d118["cid"], ood_buses=ood_buses,
            train_idx=tr, val_idx=va, test_idx=te, subset_25_idx=sub25, subset_50_idx=sub50,
            traj_delta=d118["traj_d"], traj_omega=d118["traj_w"], traj_pe=d118["traj_p"],
            traj_mask=d118["traj_m"], **noisy)
        manifest[f"ieee118{args.suffix or ''}"] = dict(
                n=int(n), unstable_pct=round(d118["y"].mean() * 100, 1),
                n_train=len(tr), n_val=len(va), n_test=len(te),
                ood_buses=ood_buses.tolist(), n_ood=int(ood_mask.sum()),
                n_sub25=len(sub25), n_sub50=len(sub50),
                channels=int(d118["X"].shape[2]), t_obs=args.tobs)

    # ---- 公开数据集 ----
    if args.only in (None, "pf39"):
        print("== PF-39 ==")
        X, y, feats = export_pf39()
        np.savez_compressed(OUT / "pf39.npz", X=X, y=y, feature_names=np.array(feats))
        manifest["pf39"] = dict(n=int(len(y)), unstable_pct=round(y.mean() * 100, 1),
                                shape=list(X.shape), label_flipped=True)
    if args.only in (None, "ne39"):
        print("== NE39-Open ==")
        X, y, feats = export_ne39()
        np.savez_compressed(OUT / "ne39_features.npz", X=X, y=y, feature_names=np.array(feats))
        manifest["ne39"] = dict(n=int(len(y)), unstable_pct=round(y.mean() * 100, 1),
                                shape=list(X.shape))

    with open(OUT / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"\nDONE in {time.time()-t0:.1f}s -> {OUT}")


if __name__ == "__main__":
    main()
