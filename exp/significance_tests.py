# -*- coding: utf-8 -*-
"""统计显著性检验（Wilcoxon 符号秩 + Mann-Whitney U + Holm 校正）。

检验组：
  A) 模型对比：pimtnet vs 13 基线 × 各数据集 main，metric=acc/f1/auc
     - Wilcoxon 符号秩 exact（按种子号配对共同种子）
     - pimtnet 有 10 种子时另做 Mann-Whitney U exact（10 vs 5，独立运行不配对）
     - Holm 校正范围：每个 (数据集, metric) 内 13 个对比
  B) 噪声退化：pimtnet main vs noise1/2/5 同种子配对（n=10）
     - Holm 范围：每个 (数据集, metric) 内 3 档噪声
  C) 消融：pimtnet vs 5 变体 × 4 个 (ds,tag)，acc/f1
     - Holm 范围：每个 (ds,tag,metric) 内 5 个变体
功效说明：n=5 配对 Wilcoxon 双侧最小 p=0.0625（全部同号时），无法达 .05；
10v5 MWU 双侧最小 p≈6.7e-4，可分辨。

用法：
  python exp/significance_tests.py                       # 全部 4 数据集
  python exp/significance_tests.py --datasets ieee39     # 子集
输出：submit/results/09_significance.xlsx
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUT = ROOT.parent / "export"

BASELINES = ["mlp", "cnn1d", "lstm", "transformer", "densenet",
             "pfnn", "pi_seq2seq", "pi_gcn", "es_pinn", "tsa_llm",
             "svm", "rf", "xgboost"]
ABL_VARIANTS = ["pimtnet_ab_nolt", "pimtnet_ab_nolp", "pimtnet_ab_noltlp",
                "pimtnet_ab_noaug", "pimtnet_ab_nohead"]
ABL_COMBOS = [("ieee39", "main"), ("ieee39", "sub25"),
              ("ieee39w121", "main"), ("ieee39w121", "sub25")]
METRICS = ["acc", "f1", "auc"]


def load(model, ds, tag, metric):
    """results/<model>/<ds>_<tag>_seed<N>.json -> {seed: value}"""
    out = {}
    for f in sorted((RESULTS / model).glob(f"{ds}_{tag}_seed*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        v = r.get(metric)
        if v is not None and np.isfinite(v):
            out[int(r["seed"])] = float(v)
    return out


def paired(a, b):
    ks = sorted(set(a) & set(b))
    return (np.array([a[k] for k in ks], float),
            np.array([b[k] for k in ks], float))


def wilcoxon_exact(x, y):
    """返回 (p, n_pairs, n_ties, rank_biserial)。"""
    d = x - y
    nz = d[~np.isclose(d, 0)]
    if len(nz) == 0:
        return 1.0, len(d), len(d), 0.0
    try:
        p = float(stats.wilcoxon(x, y, zero_method="wilcox",
                                 alternative="two-sided", method="exact").pvalue)
    except Exception:
        return np.nan, len(d), len(d) - len(nz), np.nan
    r = stats.rankdata(np.abs(nz))
    rb = float((r[nz > 0].sum() - r[nz < 0].sum()) / (r[nz > 0].sum() + r[nz < 0].sum()))
    return p, len(d), len(d) - len(nz), rb


def mwu_exact(x, y):
    """10 vs 5 独立样本；样本数相同（无增益）时返回 nan。"""
    if len(x) <= len(y):
        return np.nan
    try:
        return float(stats.mannwhitneyu(x, y, alternative="two-sided",
                                        method="exact").pvalue)
    except Exception:
        return np.nan


def holm(ps):
    """Holm-Bonferroni 逐步校正（忽略 nan，对应位置返回 nan）。"""
    ps = np.asarray(ps, float)
    out = np.full(len(ps), np.nan)
    ok = np.where(~np.isnan(ps))[0]
    if len(ok) == 0:
        return out
    p = ps[ok]
    order = np.argsort(p)
    run, m = 0.0, len(p)
    for i, idx in enumerate(order):
        run = max(run, (m - i) * p[idx])
        out[ok[idx]] = min(run, 1.0)
    return out


def fmt_p(p):
    return "—" if p is None or (isinstance(p, float) and np.isnan(p)) else f"{p:.4g}"


def ms(v):
    return f"{np.mean(v):.4f}±{np.std(v, ddof=1):.4f}" if len(v) > 1 else f"{v[0]:.4f}"


# ---------------- A) 模型对比 ----------------
def model_comparison(datasets):
    sheets = {}
    for ds in datasets:
        rows = []
        per_metric = {}
        for met in METRICS:
            pt = load("pimtnet", ds, "main", met)
            recs = []
            for b in BASELINES:
                bl = load(b, ds, "main", met)
                x, y = paired(pt, bl)
                p, n, ties, rb = wilcoxon_exact(x, y)
                u = mwu_exact(list(pt.values()), list(bl.values()))
                recs.append(dict(b=b, x=x, y=y, p=p, n=n, ties=ties, rb=rb, u=u))
            adj_w = holm([r["p"] for r in recs])
            adj_u = holm([r["u"] for r in recs])
            for r, hw, hu in zip(recs, adj_w, adj_u):
                r["holm_w"], r["holm_u"] = hw, hu
            per_metric[met] = {r["b"]: r for r in recs}
        for b in BASELINES:
            a = per_metric["acc"][b]
            rows.append({
                "baseline": b, "n_pairs": a["n"],
                "acc_pimt": ms(a["x"]), "acc_base": ms(a["y"]),
                "acc_diff_pp": round((a["x"].mean() - a["y"].mean()) * 100, 2),
                "W/L/T": f"{int((a['x']>a['y']).sum())}/{int((a['x']<a['y']).sum())}/{a['ties']}",
                "acc_wilcoxon_p": fmt_p(a["p"]), "acc_holm_p": fmt_p(a["holm_w"]),
                "acc_rank_biserial": round(a["rb"], 2),
                "acc_MWU_p(10v5)": fmt_p(a["u"]), "acc_MWU_holm_p": fmt_p(a["holm_u"]),
                "f1_wilcoxon_p": fmt_p(per_metric["f1"][b]["p"]),
                "f1_holm_p": fmt_p(per_metric["f1"][b]["holm_w"]),
                "auc_wilcoxon_p": fmt_p(per_metric["auc"][b]["p"]),
                "auc_holm_p": fmt_p(per_metric["auc"][b]["holm_w"]),
            })
        df = pd.DataFrame(rows)
        sheets[f"{ds}_main"] = df
        sig = sum(1 for r in rows if r["acc_holm_p"] != "—" and float(r["acc_holm_p"]) < .05)
        print(f"[A] {ds}: pimtnet 优于基线 acc Holm<0.05 的对比数 = {sig}/13")
    return sheets


# ---------------- B) 噪声退化（pimtnet 自比） ----------------
def noise_degradation(datasets):
    rows_all = {}
    for met in ["acc", "f1"]:
        rows = []
        for ds in datasets:
            main = load("pimtnet", ds, "main", met)
            recs = []
            for tag in ["noise1", "noise2", "noise5"]:
                nz = load("pimtnet", ds, tag, met)
                if not nz:
                    continue
                x, y = paired(main, nz)
                p, n, ties, rb = wilcoxon_exact(x, y)
                recs.append(dict(ds=ds, tag=tag, x=x, y=y, p=p, n=n, ties=ties, rb=rb))
            for r, h in zip(recs, holm([r["p"] for r in recs])):
                r["holm"] = h
            rows += [dict(dataset=r["ds"], noise_tag=r["tag"], n_pairs=r["n"],
                          main=ms(r["x"]), noisy=ms(r["y"]),
                          diff_pp=round((r["x"].mean() - r["y"].mean()) * 100, 2),
                          W_D_T=f"{int((r['x']>r['y']).sum())}/{int((r['x']<r['y']).sum())}/{r['ties']}",
                          wilcoxon_p=fmt_p(r["p"]), holm_p=fmt_p(r["holm"]),
                          rank_biserial=round(r["rb"], 2)) for r in recs]
        rows_all[met] = rows
    df = pd.DataFrame(rows_all["acc"])
    df["f1_wilcoxon_p"] = [r["wilcoxon_p"] for r in rows_all["f1"]]
    df["f1_holm_p"] = [r["holm_p"] for r in rows_all["f1"]]
    print(f"[B] 噪声退化配对检验 {len(df)} 组")
    return df


# ---------------- C) 消融对比 ----------------
def ablation_comparison():
    rows = []
    for ds, tag in ABL_COMBOS:
        for met in ["acc", "f1"]:
            pt = load("pimtnet", ds, tag, met)
            recs = []
            for v in ABL_VARIANTS:
                vv = load(v, ds, tag, met)
                if not vv or not pt:
                    continue
                x, y = paired(pt, vv)
                p, n, ties, rb = wilcoxon_exact(x, y)
                recs.append(dict(v=v, x=x, y=y, p=p, n=n, ties=ties, rb=rb))
            for r, h in zip(recs, holm([r["p"] for r in recs])):
                r["holm"] = h
            rows += [dict(dataset=ds, tag=tag, metric=met, variant=r["v"],
                          n_pairs=r["n"], pimtnet=ms(r["x"]), variant_mean=ms(r["y"]),
                          diff_pp=round((r["x"].mean() - r["y"].mean()) * 100, 2),
                          **{"W/L/T": f"{int((r['x']>r['y']).sum())}/{int((r['x']<r['y']).sum())}/{r['ties']}"},
                          wilcoxon_p=fmt_p(r["p"]), holm_p=fmt_p(r["holm"]),
                          rank_biserial=round(r["rb"], 2)) for r in recs]
    df = pd.DataFrame(rows)
    print(f"[C] 消融配对检验 {len(df)} 组")
    return df


NOTES = [
    ["检验目的", "为论文主对比、噪声鲁棒性主张（F4）与消融主张（F14/R8）提供统计支持"],
    ["检验 A（模型对比）", "pimtnet vs 13 基线，main 标签；Wilcoxon 符号秩 exact（双侧），按种子号配对共同种子（通常 5 对）；pimtnet 有 10 种子的数据集另做 Mann-Whitney U exact（10 vs 5 独立样本，不配对）"],
    ["检验 B（噪声退化）", "pimtnet 自比：main vs noise1/2/5 同种子配对（n=10 对），检验'零退化'主张"],
    ["检验 C（消融）", "pimtnet vs 5 个消融变体，4 个 (数据集,标签) 组合，acc/f1；支持 F14 regime 依赖结论"],
    ["多重比较校正", "Holm-Bonferroni 逐步法；校正族：A=每个(数据集,metric)内13个对比；B=每个(数据集,metric)内3档噪声；C=每个(数据集,标签,metric)内5个变体"],
    ["效应量", "配对 rank-biserial 相关系数 r（+1=全部配对中 pimtnet 更优，0=对称，-1=全劣）"],
    ["功效限制（重要）", "n=5 配对 Wilcoxon 双侧最小 p=0.0625（全部同号时），故 5 种子配对检验在 α=.05 下不可能显著——显著与否请以 MWU(10v5) 列与效应量为准；n=10 配对（检验B）最小 p≈0.002，可分辨"],
    ["配对均值口径", "表中 acc_pimt/acc_base 为共同种子的配对均值±std（与检验一致）；pimtnet 10 种子全量均值见 01 主表"],
    ["ens10 不作检验", "10 种子集成为单点结果（无分布），无法作假设检验；论文中以'超过全部单模型'的描述性主张呈现（F13）"],
    ["ieee118 t13 口径", "重生成数据轨迹目标爆炸曾致 6 个轨迹头模型失效（pimtnet .6686），已扩展钳制并重跑；若本表由重跑前生成，该数据集行无效"],
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="ieee39,ieee118,ieee39w121,ieee118w121")
    args = ap.parse_args()
    datasets = [d for d in args.datasets.split(",") if d]
    OUT.mkdir(parents=True, exist_ok=True)

    sheets = model_comparison(datasets)
    noise = noise_degradation(datasets)
    abl = ablation_comparison()

    path = OUT / "09_significance.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        pd.DataFrame(NOTES, columns=["项目", "说明"]).to_excel(w, sheet_name="说明", index=False)
        for name, df in sheets.items():
            df.to_excel(w, sheet_name=name[:31], index=False)
        noise.to_excel(w, sheet_name="噪声退化_pimtnet", index=False)
        abl.to_excel(w, sheet_name="消融对比", index=False)
    print(f"[ok] {path.relative_to(ROOT.parent.parent)}")


if __name__ == "__main__":
    main()
