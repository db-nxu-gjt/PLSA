# -*- coding: utf-8 -*-
"""统一训练入口（论文 4.2 节统一协议）。

用法示例：
  python exp/train.py --models mlp,pimtnet --dataset ieee39 --tag main --seeds 5
  python exp/train.py --models svm,rf,xgboost --tag sub25 --seeds 5
  python exp/train.py --models all_core --tag noise2 --seeds 5
tag ∈ main | ood | sub25 | sub50 | noise1 | noise2 | noise5
每组结果写入 exp/results/<model>/，用 exp/aggregate.py 汇总。
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from common.dataset import make_loaders, LOADERS, _clip_traj
from common.trainer import Trainer, save_result
from common.metrics import cls_metrics, traj_rmse
from models.core import BUILDERS
from models.pimtnet import PIMTNet
from models.physics_baselines import PHYS_BUILDERS, pretrain_tsa_llm
from models import sklearn_baselines as sk

TORCH_MODELS = list(BUILDERS) + ["pimtnet"] + list(PHYS_BUILDERS)
SK_MODELS = ["svm", "rf", "xgboost"]


def build_model(name, n_ch, n_desc, n_mach, pimt_cfg=None, T=13):
    if name == "pimtnet":
        return PIMTNet(n_ch, n_desc, n_mach, **(pimt_cfg or {}))
    if name in PHYS_BUILDERS:
        return PHYS_BUILDERS[name](n_ch, n_desc, n_mach, T=T)
    return BUILDERS[name](n_ch, n_desc, T=T)


def build_extra_sets(args, meta, eval_tags):
    """构建额外语义测试集（噪声变体等），标准化沿用训练 tag 的 mu/sig。

    返回 [(etag, {X, desc, y, td, tw, tm})]；若训练 tag 未标准化则用原始值。
    """
    out = []
    for etag in eval_tags:
        a = LOADERS[args.dataset](etag)
        X_te = a["X_te"].astype(np.float32)
        if meta["mu"] is not None:
            X_te = ((X_te - meta["mu"]) / meta["sig"]).astype(np.float32)
        te = a["te"]
        out.append((etag, dict(X=X_te, desc=a["desc"].astype(np.float32)[te],
                               y=a["y"][te],
                               td=_clip_traj(args.dataset, a["traj_d"][te], a["traj_w"][te])[0],
                               tw=_clip_traj(args.dataset, a["traj_d"][te], a["traj_w"][te])[1],
                               tm=a["traj_m"][te])))
    return out


def pretrain_traj(model, path, device, seed=0, epochs=30, lr=1e-3, batch=256, mu=None, sig=None):
    """阶段一：轨迹预训练（纯数据 MSE，无标签无分类头），返回训练信息。

    输入必须用与阶段二相同的 (mu, sig) 逐通道标准化，否则表征尺度错位
    （原始 Pe 尖峰上千、δ 极小），微调阶段分类头无法恢复。
    """
    import torch
    import torch.nn.functional as F
    from torch.utils.data import TensorDataset, DataLoader
    X = d = np.load(path)["X"]
    if mu is not None and sig is not None:
        X = ((X - mu) / sig).astype(np.float32)
    d = np.load(path)
    Yd, Yw, M = d["Yd"], d["Yw"], d["M"]
    # 清洗：发散场景数值爆炸区（标准化后 |x|、|y| > 100 或非有限）整窗丢弃，
    # 否则注意力前向/反向产生 NaN，梯度裁剪系数本身也会变 NaN 污染权重
    bad = (~np.isfinite(X)).any(axis=(1, 2)) | (np.abs(X) > 100).any(axis=(1, 2)) \
        | (~np.isfinite(Yd)).any(axis=(1, 2)) | (np.abs(Yd) > 100).any(axis=(1, 2)) \
        | (~np.isfinite(Yw)).any(axis=(1, 2)) | (np.abs(Yw) > 100).any(axis=(1, 2))
    X, Yd, Yw, M = X[~bad], Yd[~bad], Yw[~bad], M[~bad]
    print(f"    [pretrain] windows={len(X)} (dropped {int(bad.sum())} blow-up windows)")
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(Yd),
                       torch.from_numpy(Yw), torch.from_numpy(M))
    dl = DataLoader(ds, batch, shuffle=True, num_workers=0,
                    generator=torch.Generator().manual_seed(seed))
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.to(device); model.train()
    t0 = time.time()
    final, n_skip = 0.0, 0
    for ep in range(epochs):
        for x, yd, yw, m in dl:
            x, yd, yw, m = x.to(device), yd.to(device), yw.to(device), m.to(device)
            out = model(x, torch.zeros(len(x), 5, device=device))
            mm = m.unsqueeze(-1)
            loss = ((((out["traj_delta"] - yd) ** 2) * mm).sum() / (mm.sum() * yd.shape[-1] + 1e-9)
                    + (((out["traj_omega"] - yw) ** 2) * mm).sum() / (mm.sum() * yw.shape[-1] + 1e-9))
            if not torch.isfinite(loss):        # 发散场景梯度爆炸保护
                n_skip += 1
                opt.zero_grad()
                continue
            opt.zero_grad(); loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gn):      # NaN/Inf 梯度：裁剪系数本身会变 NaN，必须跳过
                n_skip += 1
                opt.zero_grad()
                continue
            opt.step()
            final = loss.item()
    return dict(pre_epochs=epochs, pre_loss=round(final, 5), pre_s=round(time.time() - t0, 1),
                pre_skipped=int(n_skip))


def run_torch(name, args, loaders, meta, seed, extra_sets=None, base=None):
    base = base or name
    tr = Trainer(max_epochs=args.epochs, patience=args.patience, lr=args.lr,
                 cosine=args.cosine, lam_t=args.lam_t, lam_p=args.lam_p,
                 verbose=args.verbose)
    model = build_model(base, meta["n_ch"], meta["n_desc"], meta["n_mach"],
                        pimt_cfg=args.pimt_cfg, T=meta["T"])
    pre_info = {}
    if args.pretrain and base == "pimtnet":
        pre_info = pretrain_traj(model, args.pretrain, tr.device, seed=seed,
                                 epochs=args.pre_epochs, lr=args.lr,
                                 mu=meta["mu"], sig=meta["sig"])
    elif args.pretrain and base == "tsa_llm":
        pre_info = pretrain_tsa_llm(model, args.pretrain, tr.device, seed=seed,
                                    epochs=args.pre_epochs, lr=args.lr,
                                    mu=meta["mu"], sig=meta["sig"])
    n_par = sum(p.numel() for p in model.parameters())
    fit = tr.fit(model, loaders["train"], loaders["val"], seed=seed)
    y, p, traj = tr.predict(model, loaders["test"])
    m = cls_metrics(y, p)
    if traj is not None:
        m.update(traj_rmse(traj[0], traj[1], traj[2], traj[3], traj[4]))
    m.update(tr.benchmark(model, meta["raw_X_te"]))
    if getattr(args, "save_probs", False):
        np.savez(Path(__file__).resolve().parent / "results" / name /
                 f"{args.dataset}_{args.tag}_seed{seed}_probs.npz", y=y, p=p)
    results = [dict(model=name, dataset=args.dataset, tag=args.tag, seed=seed,
                    params=int(n_par), **m, **fit, **pre_info)]
    for etag, es in (extra_sets or []):
        ds = TensorDataset(torch.from_numpy(es["X"]), torch.from_numpy(es["desc"]),
                           torch.from_numpy(es["y"]), torch.from_numpy(es["td"]),
                           torch.from_numpy(es["tw"]), torch.from_numpy(es["tm"]))
        dl = DataLoader(ds, 256, shuffle=False, num_workers=0)
        y2, p2, traj2 = tr.predict(model, dl)
        m2 = cls_metrics(y2, p2)
        if traj2 is not None:
            m2.update(traj_rmse(traj2[0], traj2[1], traj2[2], traj2[3], traj2[4]))
        if getattr(args, "save_probs", False):
            np.savez(Path(__file__).resolve().parent / "results" / name /
                     f"{args.dataset}_{etag}_seed{seed}_probs.npz", y=y2, p=p2)
        results.append(dict(model=name, dataset=args.dataset, tag=etag, seed=seed,
                            params=int(n_par), **m2))
    return results


def run_sklearn(name, args, loaders, meta, seed, extra_sets=None, base=None):
    base = base or name
    def to_xy(loader):
        Xs, Ds, ys = [], [], []
        for x, d, y, *_ in loader:
            Xs.append(x.numpy()); Ds.append(d.numpy()); ys.append(y.numpy())
        return (np.concatenate(Xs), np.concatenate(Ds), np.concatenate(ys))
    X_tr, D_tr, y_tr = to_xy(loaders["train"])
    X_te, D_te, y_te = to_xy(loaders["test"])
    t0 = time.time()
    m, _ = sk.train_sklearn(base, X_tr, D_tr, y_tr, X_te, D_te, y_te, seed=seed)
    results = [dict(model=name, dataset=args.dataset, tag=args.tag, seed=seed,
                    params=None, fit_s=round(time.time() - t0, 1), **m)]
    for etag, es in (extra_sets or []):
        # sklearn 重训成本低且同 seed 同训练集 → 等价于同一模型换测试集
        m2, _ = sk.train_sklearn(base, X_tr, D_tr, y_tr, es["X"], es["desc"], es["y"],
                                 seed=seed)
        results.append(dict(model=name, dataset=args.dataset, tag=etag, seed=seed,
                            params=None, fit_s=0.0, **m2))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True,
                    help="逗号分隔；all_core=5个纯torch基线+pimtnet；all_phys=5个物理信息基线；all=含sklearn")
    ap.add_argument("--dataset", default="ieee39", choices=sorted(LOADERS))
    ap.add_argument("--tag", default="main",
                    choices=["main", "ood", "sub25", "sub50", "noise1", "noise2", "noise5"])
    ap.add_argument("--eval-tags", default="",
                    help="训练一次、额外语义测试集逗号分隔（如 noise1,noise2,noise5）；"
                         "须与 --tag 组合：--tag main --eval-tags noise1,noise2,noise5")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--cosine", action="store_true")
    ap.add_argument("--lam-t", type=float, default=0.5)
    ap.add_argument("--lam-p", type=float, default=0.1)
    ap.add_argument("--pimt-cfg", type=str, default=None, help="PIMTNet kwargs JSON")
    ap.add_argument("--pretrain", type=str, default=None, help="预训练 npz 路径（两阶段）")
    ap.add_argument("--pre-epochs", type=int, default=30)
    ap.add_argument("--sfx", type=str, default="",
                    help="结果命名后缀（消融变体）：模型目录/JSON model 字段变为 <model><sfx>，构建仍按原模型")
    ap.add_argument("--save-probs", action="store_true",
                    help="保存测试集 (y, p) 到 results/<model>/<ds>_<tag>_seedN_probs.npz（集成评估用）")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    if args.pimt_cfg:
        if args.pimt_cfg.startswith("@"):   # @file 形式，规避 PowerShell 引号问题
            args.pimt_cfg = json.loads(Path(args.pimt_cfg[1:]).read_text(encoding="utf-8"))
        else:
            args.pimt_cfg = json.loads(args.pimt_cfg)
    else:
        args.pimt_cfg = None

    names = (list(BUILDERS) + ["pimtnet"]) if args.models == "all_core" else \
            (list(PHYS_BUILDERS)) if args.models == "all_phys" else \
            (TORCH_MODELS + SK_MODELS) if args.models == "all" else args.models.split(",")
    loaders, meta = make_loaders(args.dataset, args.tag)
    eval_tags = [t for t in args.eval_tags.split(",") if t and t != args.tag]
    extra_sets = build_extra_sets(args, meta, eval_tags)
    print(f"dataset={args.dataset} tag={args.tag} "
          f"train/val/test={len(meta['tr'])}/{len(meta['va'])}/{len(meta['te'])} "
          f"n_ch={meta['n_ch']} device={'cuda' if torch.cuda.is_available() else 'cpu'}"
          + (f" extra_evals={eval_tags}" if eval_tags else ""))

    for base in names:
        name = base + args.sfx if args.sfx else base
        out = Path(__file__).resolve().parent / "results" / name
        out.mkdir(parents=True, exist_ok=True)
        for seed in range(args.seeds):
            def _f(t):
                return out / f"{args.dataset}_{t}_seed{seed}.json"
            if all(_f(t).exists() for t in [args.tag] + eval_tags):
                print(f"[skip] {name} seed{seed} 已存在"); continue
            t0 = time.time()
            try:
                res_list = (run_torch if base in TORCH_MODELS else run_sklearn)(
                    name, args, loaders, meta, seed, extra_sets, base=base)
                for r in res_list:
                    save_result(r, name)
                r0 = res_list[0]
                extra_s = f" (+{len(res_list)-1} extra evals)" if len(res_list) > 1 else ""
                print(f"[done] {name} seed{seed} acc={r0['acc']:.4f} f1={r0['f1']:.4f} "
                      f"auc={r0['auc']:.4f} ({time.time()-t0:.0f}s){extra_s}")
            except Exception as e:
                print(f"[FAIL] {name} seed{seed}: {type(e).__name__}: {e}")
                if args.verbose:
                    raise


if __name__ == "__main__":
    main()
