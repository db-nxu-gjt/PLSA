# -*- coding: utf-8 -*-
"""集成统计检验（P1-6）：PIMT-Net 10 种子集成 vs 基线集成的逐样本配对检验。

内容：
  1) 复算 pimtnet_ens10（10 种子概率平均），与盘上 pimtnet_ens10 JSON 对账
  2) 基线集成：lstm/transformer 用 --save-probs 重训产出的 5 种子概率平均；
     svm/xgboost 用相同种子重训取概率后 5 种子平均（确定性，可复现原指标）
  3) McNemar 精确检验（逐样本配对二项）+ bootstrap 1000 次准确率差值 95% CI
输出：results/ensemble_eval/ieee39_main.json 与 ens10 概率 npz（pivn_demo 复用）。

用法：
  python exp/ensemble_eval.py --dataset ieee39 --tag main
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.dataset import make_loaders
from common.metrics import cls_metrics
from models import sklearn_baselines as sk

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def load_probs(model, ds, tag, seeds):
    ps, ys = [], None
    for s in seeds:
        f = RESULTS / model / f"{ds}_{tag}_seed{s}_probs.npz"
        if not f.exists():
            return None, None
        d = np.load(f)
        ps.append(d["p"])
        ys = d["y"]
    return ys, np.stack(ps)          # (S, N)


def mcnemar_exact(y, p_a, p_b):
    """逐样本配对 McNemar 精确检验（二项，双侧）。返回 (b, c, p)。"""
    from math import comb
    a = (p_a >= 0.5).astype(int)
    b_ = (p_b >= 0.5).astype(int)
    correct_a, correct_b = (a == y), (b_ == y)
    n01 = int((correct_a & ~correct_b).sum())   # A 对 B 错
    n10 = int((~correct_a & correct_b).sum())   # A 错 B 对
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    p = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n * 2
    return n01, n10, min(p, 1.0)


def wilson_ci(k, n, z=1.959963985):
    p = k / n
    den = 1 + z ** 2 / n
    ctr = (p + z ** 2 / (2 * n)) / den
    hw = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return ctr - hw, ctr + hw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ieee39")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    loaders, meta = make_loaders(args.dataset, args.tag)
    y = np.asarray(loaders["test"].dataset.tensors[2]).astype(int) \
        if hasattr(loaders["test"], "dataset") else None
    if y is None:
        raise SystemExit("cannot read test labels")

    # ---- PIMT-Net ens10 ----
    ys10, P10 = load_probs("pimtnet", args.dataset, args.tag, range(10))
    assert ys10 is not None, "pimtnet probs missing"
    assert (ys10.astype(int) == y).all(), "pimtnet probs label mismatch"
    p_ens10 = P10.mean(0)
    m_ens10 = cls_metrics(y, p_ens10)
    out_dir = RESULTS / "ensemble_eval"
    out_dir.mkdir(exist_ok=True)
    np.savez(out_dir / f"{args.dataset}_{args.tag}_ens10_probs.npz", y=y, p=p_ens10)

    # ---- 基线集成 ----
    comparators = {}
    for mdl, seeds in [("lstm", range(5)), ("transformer", range(5))]:
        ys, P = load_probs(mdl, args.dataset, args.tag, seeds)
        if ys is None:
            print(f"[warn] {mdl} probs missing (重训未完成?)，跳过")
            continue
        assert (ys.astype(int) == y).all(), f"{mdl} label mismatch"
        comparators[f"{mdl}_ens5"] = P.mean(0)
    for mdl in ["svm", "xgboost"]:
        # 与 train.py 相同协议逐种子重训（确定性）
        t = loaders["test"].dataset.tensors
        X_te, D_te = t[0].numpy(), t[1].numpy()
        trl = loaders["train"].dataset.tensors
        X_tr, D_tr, y_tr = trl[0].numpy(), trl[1].numpy(), trl[2].numpy()
        P, accs = [], []
        for s in range(5):
            m, prob = sk.train_sklearn(mdl, X_tr, D_tr, y_tr, X_te, D_te, y, seed=s)
            P.append(prob)
            accs.append(m["acc"])
        print(f"[info] {mdl} 5-seed retrain acc mean={np.mean(accs):.4f}")
        comparators[f"{mdl}_ens5"] = np.stack(P).mean(0)

    # ---- 检验 ----
    rows = {}
    correct10 = (p_ens10 >= 0.5).astype(int) == y
    k10 = int(correct10.sum())
    lo, hi = wilson_ci(k10, len(y))
    rows["pimtnet_ens10"] = dict(metrics=m_ens10, acc_wilson95=[round(lo, 4), round(hi, 4)],
                                 n_correct=k10, n=len(y))
    rng = np.random.default_rng(args.seed)
    for name, p_cmp in comparators.items():
        m_cmp = cls_metrics(y, p_cmp)
        b_, c_, pv = mcnemar_exact(y, p_ens10, p_cmp)
        # bootstrap acc 差值 CI
        n = len(y)
        diffs = []
        pred10 = (p_ens10 >= 0.5).astype(int)
        predc = (p_cmp >= 0.5).astype(int)
        acc10 = (pred10 == y).astype(float)
        accc = (predc == y).astype(float)
        for _ in range(args.boot):
            idx = rng.integers(0, n, n)
            diffs.append(acc10[idx].mean() - accc[idx].mean())
        diffs = np.sort(diffs)
        rows[name] = dict(metrics=m_cmp,
                          mcnemar=dict(n01_ens_right=b_, n10_ens_wrong=c_, p=round(pv, 4)),
                          boot_acc_diff=[round(float(diffs[int(0.025 * args.boot)]), 4),
                                         round(float(diffs[int(0.975 * args.boot)]), 4)])
        print(f"[test] ens10 vs {name}: acc {m_ens10['acc']:.4f} vs {m_cmp['acc']:.4f} "
              f"McNemar b={b_} c={c_} p={pv:.4g} bootCI=[{rows[name]['boot_acc_diff'][0]},"
              f"{rows[name]['boot_acc_diff'][1]}]")

    (out_dir / f"{args.dataset}_{args.tag}.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[ok] {out_dir / (args.dataset + '_' + args.tag + '.json')}")


if __name__ == "__main__":
    main()
