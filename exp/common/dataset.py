# -*- coding: utf-8 -*-
"""数据集封装：读取 processed npz，构建 train/val/test DataLoader。

支持实验模式(tag)：
  main     标准测试集（默认）
  ood      未见故障位置：train/val 剔除 OOD 母线场景，test = 全部 OOD 场景
  sub25/sub50  小样本训练子集（val/test 标准）
  noise1/noise2/noise5  测试集注入 σ=1%/2%/5% 噪声变体（train/val 标准）
特征归一化：逐通道 z-score（统计量仅来自当前 train split）。
"""
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from pathlib import Path
PROCESSED = Path(__file__).resolve().parent.parent.parent / "data" / "processed"

NOISE_KEY = {"noise1": "X_test_noisy_1pct", "noise2": "X_test_noisy_2pct",
             "noise5": "X_test_noisy_5pct"}


def _load_ieee39():
    d = np.load(PROCESSED / "ieee39_windows.npz")
    return {k: d[k] for k in d.files}


def _load_windows_arrays(fname: str, tag="main"):
    d = np.load(PROCESSED / fname)
    X, y, desc = d["X"], d["y"], d["desc"]
    tr, va, te = d["train_idx"], d["val_idx"], d["test_idx"]
    ood = d["ood_mask"]
    X_te = d[NOISE_KEY[tag]] if tag in NOISE_KEY else X[te]
    if tag == "ood":
        tr, va = tr[ood[tr] == 0], va[ood[va] == 0]
        te = np.where(ood == 1)[0]
        X_te = X[te]
    elif tag == "sub25":
        tr = d["subset_25_idx"]
    elif tag == "sub50":
        tr = d["subset_50_idx"]
    return dict(X=X, desc=desc, y=y, tr=tr, va=va, te=te, X_te=X_te,
                traj_d=d["traj_delta"], traj_w=d["traj_omega"],
                traj_m=d["traj_mask"], bus=d["bus"])


def load_ieee39_arrays(tag="main"):
    return _load_windows_arrays("ieee39_windows.npz", tag)


def load_ieee118_arrays(tag="main"):
    d = np.load(PROCESSED / "ieee118_windows.npz")
    X, y, desc = d["X"], d["y"], d["desc"]
    tr, va, te = d["train_idx"], d["val_idx"], d["test_idx"]
    ood = d["ood_mask"]
    X_te = d[NOISE_KEY[tag]] if tag in NOISE_KEY else X[te]
    if tag == "ood":
        tr, va = tr[ood[tr] == 0], va[ood[va] == 0]
        te = np.where(ood == 1)[0]
        X_te = X[te]
    elif tag == "sub25":
        tr = d["subset_25_idx"]
    elif tag == "sub50":
        tr = d["subset_50_idx"]
    return dict(X=X, desc=desc, y=y, tr=tr, va=va, te=te, X_te=X_te,
                traj_d=d["traj_delta"], traj_w=d["traj_omega"],
                traj_m=d["traj_mask"], bus=d["bus"])


def _stratified_split(y, seed=2026):
    """与 data_pipeline.stratified_split 同口径：60/20/20 分层划分。"""
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]; rng.shuffle(idx)
        n_tr, n_va = int(round(0.6 * len(idx))), int(round(0.2 * len(idx)))
        tr += list(idx[:n_tr]); va += list(idx[n_tr:n_tr + n_va]); te += list(idx[n_tr + n_va:])
    return map(np.sort, (np.array(tr), np.array(va), np.array(te)))


def _zero_traj(n):
    """外部特征数据集无轨迹目标：返回全零 traj + 全零掩码（traj 损失恒为 0）。"""
    return (np.zeros((n, 24, 1), np.float32), np.zeros((n, 24, 1), np.float32),
            np.zeros((n, 24), np.float32))


def load_pf39_arrays(tag="main"):
    assert tag == "main", "外部数据集仅支持 main 标签"
    d = np.load(PROCESSED / "pf39.npz")
    X, y = d["X"].astype(np.float32), d["y"]
    tr, va, te = _stratified_split(y)
    td, tw, tm = _zero_traj(len(y))
    return dict(X=X, desc=np.zeros((len(y), 5), np.float32), y=y, tr=tr, va=va, te=te,
                X_te=X[te], traj_d=td, traj_w=tw, traj_m=tm,
                bus=np.zeros(len(y), np.int64))


def load_ne39_arrays(tag="main"):
    assert tag == "main", "外部数据集仅支持 main 标签"
    d = np.load(PROCESSED / "ne39_features.npz")
    X, y = d["X"].astype(np.float32), d["y"]
    X = X[:, :, None]                      # (N,354,1)：特征维作单通道序列
    tr, va, te = _stratified_split(y)
    td, tw, tm = _zero_traj(len(y))
    return dict(X=X, desc=np.zeros((len(y), 5), np.float32), y=y, tr=tr, va=va, te=te,
                X_te=X[te], traj_d=td, traj_w=tw, traj_m=tm,
                bus=np.zeros(len(y), np.int64))


def _make_win_loader(fname):
    """观测窗消融数据集别名：ieee39w31/61/121/241 -> ieee39_windows_t*.npz。"""
    def _load(tag="main"):
        return _load_windows_arrays(fname, tag)
    return _load


LOADERS = {"ieee39": load_ieee39_arrays, "ieee118": load_ieee118_arrays,
           "pf39": load_pf39_arrays, "ne39": load_ne39_arrays,
           **{f"ieee39w{t}": _make_win_loader(f"ieee39_windows_t{t}.npz")
              for t in (31, 61, 121, 241)},
           "ieee118w121": _make_win_loader("ieee118_windows_t121.npz")}


def _clip_traj(dataset, td, tw):
    """118 系（t13/t121，重生成数据）发散爆炸轨迹目标钳制（±10 次同步摆 / ±5 pu）：
    54 机系统数值发散目标可达 1e7-1e9 度，平方损失下摧毁轨迹头
    （118w121 曾致 pimtnet .777、118 t13 曾致 .6686，见 F11）。
    仅作用于 ieee118 系；39 系目标幅值小（t121 max 2.6e4 且训练健康），口径不变。"""
    if dataset in ("ieee118w121", "ieee118"):
        return (np.clip(td, -20 * np.pi, 20 * np.pi).astype(np.float32),
                np.clip(tw, -5.0, 5.0).astype(np.float32))
    return np.asarray(td, np.float32), np.asarray(tw, np.float32)


def make_loaders(dataset="ieee39", tag="main", batch_size=128, standardize=True):
    """返回 (loaders, meta)。loaders 含 train/val/test；meta 含 n_ch、归一化统计等。"""
    a = LOADERS[dataset](tag)
    X, desc = a["X"].astype(np.float32), a["desc"].astype(np.float32)
    traj_d, traj_w = _clip_traj(dataset, a["traj_d"], a["traj_w"])
    mu = sig = None
    if standardize:
        Xtr = X[a["tr"]]
        # ne39 的"时间维"其实是特征维，需逐特征归一化；其余逐通道（跨时间）
        axes = (0,) if dataset == "ne39" else (0, 1)
        mu, sig = Xtr.mean(axis=axes), Xtr.std(axis=axes) + 1e-8
        X = ((X - mu) / sig).astype(np.float32)
        X_te = ((a["X_te"] - mu) / sig).astype(np.float32)
    else:
        X_te = a["X_te"].astype(np.float32)

    def ds(idx, x):
        return TensorDataset(torch.from_numpy(x), torch.from_numpy(desc[idx]),
                             torch.from_numpy(a["y"][idx]),
                             torch.from_numpy(traj_d[idx]),
                             torch.from_numpy(traj_w[idx]),
                             torch.from_numpy(a["traj_m"][idx]))
    g = torch.Generator().manual_seed(2026)
    loaders = dict(
        train=DataLoader(ds(a["tr"], X[a["tr"]]), batch_size, shuffle=True,
                         num_workers=0, generator=g, drop_last=False),
        val=DataLoader(ds(a["va"], X[a["va"]]), 256, shuffle=False, num_workers=0),
        test=DataLoader(ds(a["te"], X_te), 256, shuffle=False, num_workers=0))
    meta = dict(n_ch=int(X.shape[2]), n_desc=int(desc.shape[1]), n_mach=int(a["traj_d"].shape[2]),
                T=int(X.shape[1]),
                mu=mu, sig=sig, te=a["te"], tr=a["tr"], va=a["va"],
                y=a["y"], bus=a["bus"], raw_X_te=a["X_te"])
    return loaders, meta
