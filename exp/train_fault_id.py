# -*- coding: utf-8 -*-
"""5.7 节故障位置识别（AlarmText 替代实验）。

任务：同 IEEE39-Sim 2349 场景、同 train/val/test 划分，标签由"是否失稳"改为
"故障发生在哪个母线"（39 类多分类），观测窗仍为 13 步。
防泄漏：desc 中的 bus_norm（即标签本身）整列剔除，n_desc 5→4。

用法：
  python exp/train_fault_id.py --models mlp,cnn1d --seeds 5
  python exp/train_fault_id.py --models svm,rf,xgboost --seeds 5
结果存 exp/results/<model>/ieee39_faultid_seed<seed>.json，并打印汇总表。
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from common.dataset import PROCESSED          # noqa: E402
from common.trainer import Trainer            # noqa: E402
from models.core import BUILDERS              # noqa: E402
from models.sklearn_baselines import handcraft_features  # noqa: E402

TORCH_MODELS = ["mlp", "cnn1d", "lstm", "transformer", "densenet"]
SKLEARN_MODELS = ["svm", "rf", "xgboost"]
RESULTS = ROOT / "results"


def multiclass_metrics(y_true, y_pred, n_cls):
    from sklearn.metrics import accuracy_score, f1_score
    return dict(acc=float(accuracy_score(y_true, y_pred)),
                macro_f1=float(f1_score(y_true, y_pred, average="macro",
                                        labels=range(n_cls), zero_division=0)),
                weighted_f1=float(f1_score(y_true, y_pred, average="weighted",
                                           labels=range(n_cls), zero_division=0)))


def load_faultid(ds_name="ieee39", windows_file="ieee39_windows.npz"):
    d = np.load(PROCESSED / windows_file)
    X = d["X"].astype(np.float32)
    T = X.shape[1]
    desc = d["desc"].astype(np.float32)
    bus = d["bus"]
    tr, va, te = d["train_idx"], d["val_idx"], d["test_idx"]
    # 防泄漏：剔除 desc 第 2 列 bus_norm（= 标签），其余场景量保留
    desc = np.delete(desc, 2, axis=1)
    classes = np.sort(np.unique(bus))
    y = np.searchsorted(classes, bus).astype(np.int64)   # 0..38
    n_cls = len(classes)
    # 稳健标准化：长观测窗（t121）含发散爆炸值（|x|~1e5，污染约 20% 样本），
    # 均值/std 甚至 p1/p99 都会被压垮；用中位数 + MAD（1.4826·median|x-med|，50% 崩溃点），
    # 标准化后截断 ±10（爆炸样本饱和，正常信号保留分辨率）
    xt = X[tr]
    med = np.median(xt, axis=(0, 1))
    mad = np.median(np.abs(xt - med), axis=(0, 1))
    sig = np.clip(1.4826 * mad, 1e-6, None)
    X = np.clip((X - med) / sig, -10.0, 10.0).astype(np.float32)
    return dict(X=X, desc=desc, y=y, tr=tr, va=va, te=te, n_cls=n_cls, T=T, ds_name=ds_name)


def run_torch(name, fid, args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out_rows = []
    for seed in args.seeds:
        f_out = RESULTS / name / f"{fid['ds_name']}_faultid_seed{seed}.json"
        if f_out.exists():
            print(f"[skip] {f_out.name}")
            out_rows.append(json.loads(f_out.read_text(encoding="utf-8")))
            continue
        torch.manual_seed(seed)
        np.random.seed(seed)
        N = len(fid["y"])
        # Trainer.batch_loss 按 6 元组解包；核心模型不返回 traj，全零占位即可
        zeros3 = torch.zeros(N, 24, 1)
        zeros2 = torch.zeros(N, 24)
        ds = TensorDataset(torch.tensor(fid["X"]), torch.tensor(fid["desc"]),
                           torch.tensor(fid["y"]), zeros3, zeros3, zeros2)
        g = torch.Generator().manual_seed(seed)
        tr_ld = DataLoader(torch.utils.data.Subset(ds, fid["tr"]), batch_size=128,
                           shuffle=True, generator=g)
        va_ld = DataLoader(torch.utils.data.Subset(ds, fid["va"]), batch_size=256)
        te_ld = DataLoader(torch.utils.data.Subset(ds, fid["te"]), batch_size=256)

        model = BUILDERS[name](30, fid["desc"].shape[1], T=fid["T"], n_cls=fid["n_cls"])
        tr_obj = Trainer(max_epochs=args.epochs, patience=args.patience,
                         lr=args.lr, device=dev, verbose=args.verbose)
        t0 = time.perf_counter()
        tr_obj.fit(model, tr_ld, va_ld, seed=seed)
        dt = time.perf_counter() - t0

        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for x, dc, y, _, _, _ in te_ld:
                ps.append(model(x.to(dev), dc.to(dev))["logits"].argmax(1).cpu())
                ys.append(y)
        m = multiclass_metrics(torch.cat(ys).numpy(), torch.cat(ps).numpy(), fid["n_cls"])
        m.update(model=name, dataset=fid["ds_name"], tag="faultid", seed=seed,
                 n_classes=fid["n_cls"], train_time=round(dt, 2))
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / name).mkdir(exist_ok=True)
        f_out.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[done] {name} seed{seed}: acc={m['acc']:.4f} macro_f1={m['macro_f1']:.4f} ({dt:.1f}s)")
        out_rows.append(m)
    return out_rows


def run_sklearn(name, fid, args):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    out_rows = []
    for seed in args.seeds:
        f_out = RESULTS / name / f"{fid['ds_name']}_faultid_seed{seed}.json"
        if f_out.exists():
            print(f"[skip] {f_out.name}")
            out_rows.append(json.loads(f_out.read_text(encoding="utf-8")))
            continue
        X_tr, d_tr, y_tr = fid["X"][fid["tr"]], fid["desc"][fid["tr"]], fid["y"][fid["tr"]]
        X_te, d_te, y_te = fid["X"][fid["te"]], fid["desc"][fid["te"]], fid["y"][fid["te"]]
        F_tr = handcraft_features(X_tr, d_tr)
        F_te = handcraft_features(X_te, d_te)
        if name == "svm":
            clf = make_pipeline(StandardScaler(),
                                SVC(C=10.0, gamma="scale", random_state=seed))
        elif name == "rf":
            clf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
        elif name == "xgboost":
            from xgboost import XGBClassifier
            clf = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1,
                                random_state=seed, eval_metric="mlogloss", n_jobs=-1)
        else:
            raise ValueError(name)
        t0 = time.perf_counter()
        clf.fit(F_tr, y_tr)
        dt = time.perf_counter() - t0
        m = multiclass_metrics(y_te, clf.predict(F_te), fid["n_cls"])
        m.update(model=name, dataset=fid["ds_name"], tag="faultid", seed=seed,
                 n_classes=fid["n_cls"], train_time=round(dt, 2))
        (RESULTS / name).mkdir(parents=True, exist_ok=True)
        f_out.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[done] {name} seed{seed}: acc={m['acc']:.4f} macro_f1={m['macro_f1']:.4f} ({dt:.1f}s)")
        out_rows.append(m)
    return out_rows


def summarize(rows):
    by = {}
    for r in rows:
        by.setdefault(r["model"], []).append(r)
    print("\n===== 故障位置识别汇总（mean±std, 39 类）=====")
    print(f"{'model':<12}{'acc':>18}{'macro_f1':>18}")
    for name, rs in sorted(by.items()):
        acc = [r["acc"] for r in rs]
        f1 = [r["macro_f1"] for r in rs]
        print(f"{name:<12}{np.mean(acc):>10.4f}±{np.std(acc):.4f}"
              f"{np.mean(f1):>10.4f}±{np.std(f1):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(TORCH_MODELS + SKLEARN_MODELS))
    ap.add_argument("--seeds", default="0,1,2,3,4", type=lambda s: [int(x) for x in s.split(",")])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--dataset", default="ieee39",
                    choices=["ieee39", "ieee39w121"],
                    help="ieee39=13 步窗（原口径）；ieee39w121=2.0s 观测窗（新操作点）")
    args = ap.parse_args()
    names = args.models.split(",")
    wf = "ieee39_windows.npz" if args.dataset == "ieee39" else "ieee39_windows_t121.npz"
    fid = load_faultid(ds_name=args.dataset, windows_file=wf)
    print(f"faultid[{args.dataset}]: {len(fid['y'])} scenarios, {fid['n_cls']} classes, "
          f"T={fid['T']}, n_desc={fid['desc'].shape[1]} (bus_norm 剔除)")
    rows = []
    for n in names:
        if n in TORCH_MODELS:
            rows += run_torch(n, fid, args)
        elif n in SKLEARN_MODELS:
            rows += run_sklearn(n, fid, args)
        else:
            raise ValueError(n)
    summarize(rows)


if __name__ == "__main__":
    main()
