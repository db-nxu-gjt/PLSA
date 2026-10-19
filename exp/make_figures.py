# -*- coding: utf-8 -*-
"""论文图表套件：从 exp/results/ JSON 生成投稿级图（300 dpi）→ submit/figures/。

风格：统一配色（PIMT-Net 强调色）、Microsoft YaHei 中文、无顶右边框、浅网格。
图目录：
  fig1  主对比热力图（14 模型 × 4 数据集，acc）
  fig2  噪声鲁棒性折线（4 数据集，σ∈{0,1,2,5}%）
  fig3  小样本折线（2×2，训练集 25%/50%/100%）
  fig4  消融 regime 依赖发散点图（Δacc = 变体 − PIMT-Net）
  fig5  10 种子集成森林图（单种子 vs Ens10 vs 最佳基线）
  fig6  显著性矩阵（MWU/Wilcoxon+Holm 方向）
  fig7  故障位置识别（t13 主口径 + t121 MAD 补充）
  fig8  观测窗消融曲线（0.2-4.0 s）
  fig9  外部数据集（PF-39 / NE39）
  fig10 通用 LLM 直接判稳基线（判别指标 + 漏报率双面板）
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUT = ROOT.parent / "figs_zh"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold",
})

C = {  # 统一模型配色
    "pimtnet": "#D62728", "pimtnet_ens10": "#8C1515",
    "mlp": "#1F77B4", "cnn1d": "#FF7F0E", "lstm": "#2CA02C",
    "transformer": "#9467BD", "densenet": "#8C564B",
    "pfnn": "#E377C2", "pi_seq2seq": "#17BECF", "pi_gcn": "#BCBD22",
    "es_pinn": "#7F7F7F", "tsa_llm": "#AEC7E8",
    "svm": "#4C72B0", "rf": "#55A868", "xgboost": "#8C6D31",
    "ab_nolt": "#4C72B0", "ab_nolp": "#55A868", "ab_noltlp": "#C44E52",
    "ab_noaug": "#8172B2", "ab_nohead": "#937860",
}
NAME = {"pimtnet": "PIMT-Net", "pimtnet_ens10": "PIMT-Net-Ens10",
        "mlp": "MLP", "cnn1d": "CNN-1D", "lstm": "LSTM",
        "transformer": "Transformer", "densenet": "DenseNet", "pfnn": "PFNN",
        "pi_seq2seq": "PI-seq2seq", "pi_gcn": "PI-GCN", "es_pinn": "ES-PINN",
        "tsa_llm": "TSA-LLM", "svm": "SVM", "rf": "RF", "xgboost": "XGBoost"}
DS_NAME = {"ieee39": "IEEE39 (t13)", "ieee118": "IEEE118 (t13)",
           "ieee39w121": "IEEE39 (t121)", "ieee118w121": "IEEE118 (t121)"}
MODELS = ["pimtnet", "mlp", "cnn1d", "lstm", "transformer", "densenet",
          "pfnn", "pi_seq2seq", "pi_gcn", "es_pinn", "tsa_llm",
          "svm", "rf", "xgboost"]
DATASETS = ["ieee39", "ieee118", "ieee39w121", "ieee118w121"]


def load(model, ds, tag="main", metric="acc"):
    vals = []
    for f in sorted((RESULTS / model).glob(f"{ds}_{tag}_seed*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        if r.get(metric) is not None:
            vals.append(float(r[metric]))
    return np.array(vals)


def ms(model, ds, tag="main", metric="acc"):
    v = load(model, ds, tag, metric)
    return (float(np.mean(v)), float(np.std(v, ddof=1)), len(v)) if len(v) else (np.nan, np.nan, 0)


def style_ax(ax):
    ax.grid(axis="x" if ax.get_xlabel() in ("", "none") else "y", alpha=0.25, linewidth=0.5)


# ---------------- fig1 主对比热力图 ----------------
def fig1():
    mat = np.array([[ms(m, d)[0] for d in DATASETS] for m in MODELS])
    best = np.nanargmax(mat, axis=0)
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")
    ax.set_xticks(range(4), [DS_NAME[d] for d in DATASETS])
    ax.set_yticks(range(len(MODELS)), [NAME[m] for m in MODELS])
    for i in range(len(MODELS)):
        for j in range(4):
            star = best[j] == i
            ax.text(j, i, f"{mat[i, j]:.4f}", ha="center", va="center",
                    fontsize=8, fontweight="bold" if star else "normal",
                    color="white" if mat[i, j] < 0.62 or mat[i, j] > 0.97 else "black")
    ax.add_patch(plt.Rectangle((-0.5, MODELS.index("pimtnet") - 0.5), 4, 1,
                               fill=False, edgecolor="#D62728", lw=2.2))
    ax.grid(False)
    ax.set_title("主对比：暂态稳定判别准确率（main 标签，5/10 种子均值）")
    cb = fig.colorbar(im, ax=ax, shrink=0.85); cb.set_label("Accuracy")
    fig.savefig(OUT / "fig1_main_heatmap.png"); plt.close(fig)


# ---------------- fig2 噪声鲁棒性 ----------------
def fig2():
    show = ["pimtnet", "svm", "xgboost", "mlp", "transformer", "lstm"]
    sig = [0, 1, 2, 5]
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 2.9), sharey=True)
    for ax, ds in zip(axes, DATASETS):
        for m in show:
            mu = [ms(m, ds, "main")[0]] + [ms(m, ds, f"noise{s}")[0] for s in sig[1:]]
            ax.plot(sig, mu, "-o", ms=3.5, color=C[m],
                    label=NAME[m], zorder=3 if m == "pimtnet" else 2,
                    lw=2.4 if m == "pimtnet" else 1.4)
        ax.set_title(DS_NAME[ds]); ax.set_xticks(sig, ["0", "1%", "2%", "5%"])
        ax.set_xlabel("噪声幅度 σ"); ax.set_ylim(0.62, 1.01)
    axes[0].set_ylabel("Accuracy")
    axes[-1].legend(fontsize=7, frameon=False, loc="lower left")
    fig.suptitle("抗噪鲁棒性：PIMT-Net（红线加粗）在全部数据集与噪声档位近零退化", y=1.04, fontsize=11, fontweight="bold")
    fig.savefig(OUT / "fig2_noise_robustness.png"); plt.close(fig)


# ---------------- fig3 小样本 ----------------
def fig3():
    show = ["pimtnet", "svm", "xgboost", "rf", "transformer", "lstm"]
    fr = [("sub25", 25), ("sub50", 50), ("main", 100)]
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.2), sharex=True)
    for ax, ds in zip(axes.ravel(), DATASETS):
        for m in show:
            mu = [ms(m, ds, t)[0] for t, _ in fr]
            ax.plot([f for _, f in fr], mu, "-o", ms=3.5,
                    lw=2.4 if m == "pimtnet" else 1.4, color=C[m], label=NAME[m])
        ax.set_title(DS_NAME[ds]); ax.set_xscale("log")
        ax.set_xticks([25, 50, 100], ["25%", "50%", "100%"])
        ax.set_ylim(0.60, 1.01)
    for ax in axes[1]: ax.set_xlabel("训练集比例")
    for ax in axes[:, 0]: ax.set_ylabel("Accuracy")
    axes[0, 0].legend(fontsize=7, frameon=False, ncol=2)
    fig.suptitle("小样本数据效率：训练集缩减至 25% / 50% 时的性能保持", y=1.0, fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_few_shot.png"); plt.close(fig)


# ---------------- fig4 消融 regime 依赖 ----------------
def fig4():
    variants = [("pimtnet_ab_nolt", "ab_nolt", "去除轨迹损失 $\\lambda_t$=0"),
                ("pimtnet_ab_nolp", "ab_nolp", "去除物理残差 $\\lambda_p$=0"),
                ("pimtnet_ab_noltlp", "ab_noltlp", "去除轨迹+物理损失"),
                ("pimtnet_ab_noaug", "ab_noaug", "去除输入增广"),
                ("pimtnet_ab_nohead", "ab_nohead", "去除轨迹预测头")]
    combos = [("ieee39", "main", "t13 全样本"), ("ieee39", "sub25", "t13 25%样本"),
              ("ieee39w121", "main", "t121 全样本"), ("ieee39w121", "sub25", "t121 25%样本")]
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    n_v, n_c = len(variants), len(combos)
    w = 0.16
    for k, (vd, key, lab) in enumerate(variants):
        off = (k - (n_v - 1) / 2) * w
        for j, (ds, tag, clab) in enumerate(combos):
            base = ms("pimtnet", ds, tag)[0]
            d = (ms(vd, ds, tag)[0] - base) * 100
            ax.scatter(j + off, d, s=42, color=C[key], zorder=3,
                       edgecolor="white", linewidth=0.6)
            ax.plot([j + off, j + off], [0, d], color=C[key], lw=1.1, alpha=0.75, zorder=2)
    ax.axhline(0, color="black", lw=1.2, zorder=1)
    ax.set_xticks(range(n_c), [c[2] for c in combos])
    ax.set_ylabel("ΔAccuracy 相对完整 PIMT-Net（百分点）")
    ax.set_title("消融的 regime 依赖：短窗去除多任务组件反有益（≥0），长窗物理项/增广有正贡献（<0）")
    handles = [plt.Line2D([], [], marker="o", ls="", color=C[key], label=lab)
               for _, key, lab in variants]
    ax.legend(handles=handles, fontsize=7.5, frameon=False, ncol=3, loc="lower left")
    fig.savefig(OUT / "fig4_ablation_regime.png"); plt.close(fig)


# ---------------- fig5 集成森林图 ----------------
def fig5():
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    rows = []
    for j, ds in enumerate(DATASETS):
        pm, ps, _ = ms("pimtnet", ds)
        ens = ms("pimtnet_ens10", ds)[0]
        base_accs = {m: ms(m, ds)[0] for m in MODELS if m != "pimtnet"}
        bm = max(base_accs, key=base_accs.get)
        rows.append((j, pm, ps, ens, bm, base_accs[bm]))
        ax.errorbar(j - 0.18, pm, yerr=ps, fmt="o", ms=6, color=C["pimtnet"],
                    capsize=3, label="PIMT-Net 单种子" if j == 0 else "")
        ax.scatter(j, ens, marker="*", s=210, color="#FFB000",
                   edgecolor="#8C1515", linewidth=0.8, zorder=4,
                   label="PIMT-Net-Ens10（10 种子概率平均）" if j == 0 else "")
        ax.scatter(j + 0.18, base_accs[bm], marker="s", s=42, color="#4C72B0",
                   label=f"最佳单模型基线" if j == 0 else "")
        ax.text(j + 0.30, base_accs[bm], NAME[bm], fontsize=7, va="center", color="#4C72B0")
    for j, pm, ps, ens, bm, ba in rows:
        ax.text(j, ens + 0.006, f"{ens:.4f}", ha="center", fontsize=8, fontweight="bold", color="#8C1515")
    ax.set_xticks(range(4), [DS_NAME[d] for d in DATASETS])
    ax.set_ylabel("Accuracy"); ax.set_ylim(0.84, 0.99)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.set_title("10 种子集成：三个操作点全面超越全部单模型（含最佳基线）")
    fig.savefig(OUT / "fig5_ensemble_forest.png"); plt.close(fig)


# ---------------- fig6 显著性矩阵 ----------------
def fig6():
    xl = pd.ExcelFile(ROOT.parent / "export" / "09_significance.xlsx")
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    grid = np.full((len(MODELS) - 1, 4), np.nan)
    anno = [["" for _ in range(4)] for _ in range(len(MODELS) - 1)]
    for j, ds in enumerate(DATASETS):
        df = xl.parse(f"{ds}_main")
        for i, m in enumerate([m for m in MODELS if m != "pimtnet"]):
            row = df[df["baseline"] == m]
            if row.empty:
                continue
            row = row.iloc[0]
            diff = row["acc_diff_pp"]
            def pv(col):
                v = row.get(col, "—")
                try: return float(v)
                except (TypeError, ValueError): return np.nan
            p = pv("acc_MWU_holm_p")
            if np.isnan(p):
                p = pv("acc_holm_p")
            if not np.isnan(p) and p < 0.05:
                grid[i, j] = 1.0 if diff > 0 else -1.0
            else:
                grid[i, j] = 0.0
            anno[i][j] = f"{diff:+.2f}"
    from matplotlib.colors import ListedColormap
    im = ax.imshow(grid, cmap=ListedColormap(["#C44E52", "#E8E8E8", "#55A868"]),
                   vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(4), [DS_NAME[d] for d in DATASETS])
    ax.set_yticks(range(len(MODELS) - 1), [NAME[m] for m in MODELS if m != "pimtnet"])
    for i in range(grid.shape[0]):
        for j in range(4):
            ax.text(j, i, anno[i][j], ha="center", va="center", fontsize=8)
    ax.grid(False)
    ax.set_title("PIMT-Net vs 各基线的显著性方向（acc 差异 pp；绿=显著更优，红=显著更差，灰=不可分辨）\n"
                 "MWU(10v5)+Holm，α=.05；IEEE118(t13) 受 5 种子功效限制（见 09 说明页）")
    fig.savefig(OUT / "fig6_significance_matrix.png"); plt.close(fig)


# ---------------- fig7 故障位置识别 ----------------
def fig7():
    models8 = ["lstm", "transformer", "xgboost", "cnn1d", "rf", "mlp", "densenet", "svm"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.6))
    for ax, ds, ttl in [(axes[0], "ieee39", "t13 主口径（0.2 s 窗）"),
                        (axes[1], "ieee39w121", "t121 补充口径（2.0 s 窗，MAD 定标）")]:
        accs = [ms(m, ds, "faultid")[0] for m in models8]
        order = np.argsort(accs)
        names = [NAME[models8[i]] for i in order]
        vals = [accs[i] for i in order]
        colors = ["#D62728" if models8[i] == "pimtnet" else C[models8[i]] for i in order]
        ax.barh(names, vals, color=colors, height=0.62)
        for y, v in enumerate(vals):
            ax.text(v + 0.004, y, f"{v:.4f}", va="center", fontsize=7.5)
        ax.set_xlim(0.60, 1.03); ax.set_title(ttl); ax.set_xlabel("Accuracy（39 类）")
    fig.suptitle("故障位置识别：序列模型全面领先，SVM 垫底——排名与稳定性判别任务明显不同",
                 y=1.05, fontsize=11, fontweight="bold")
    fig.savefig(OUT / "fig7_fault_location.png"); plt.close(fig)


# ---------------- fig8 观测窗消融 ----------------
def fig8():
    wins = [("ieee39", 0.2), ("ieee39w31", 0.5), ("ieee39w61", 1.0),
            ("ieee39w121", 2.0), ("ieee39w241", 4.0)]
    show = ["pimtnet", "svm", "xgboost", "transformer", "mlp", "cnn1d", "lstm"]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for m in show:
        mu = [ms(m, ds)[0] for ds, _ in wins]
        ax.plot([t for _, t in wins], mu, "-o", ms=4,
                lw=2.4 if m == "pimtnet" else 1.4, color=C[m], label=NAME[m])
    ax.axvspan(0.9, 2.1, color="#FFB000", alpha=0.12, lw=0)
    ax.text(1.45, 0.665, "推荐操作点\n1.0–2.0 s", ha="center", fontsize=8, color="#8C1515")
    ax.set_xscale("log"); ax.set_xticks([t for _, t in wins], ["0.2", "0.5", "1.0", "2.0", "4.0"])
    ax.set_xlabel("观测窗长度（s）"); ax.set_ylabel("Accuracy")
    ax.set_ylim(0.65, 1.01)
    ax.legend(fontsize=7.5, frameon=False, ncol=2)
    ax.set_title("观测窗消融：短窗段 SVM 领先、长窗段深度模型反超——复现差距之谜（F1）")
    fig.savefig(OUT / "fig8_window_ablation.png"); plt.close(fig)


# ---------------- fig9 外部数据集 ----------------
def fig9():
    models8 = MODELS[6:] + MODELS[1:6]  # 全部 13 基线即可
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.4))
    for ax, ds, ttl in [(axes[0], "pf39", "PF-39（PowerFactory，12,852 样本）"),
                        (axes[1], "ne39", "NE39-Open（官方 3,120 样本）")]:
        accs = [(m, ms(m, ds)[0]) for m in MODELS if m != "pimtnet"]
        accs.sort(key=lambda kv: kv[1])
        ax.barh([NAME[m] for m, _ in accs], [v for _, v in accs],
                color=[C[m] for m, _ in accs], height=0.62)
        for y, (m, v) in enumerate(accs):
            ax.text(v + 0.002, y, f"{v:.4f}", va="center", fontsize=6.8)
        ax.set_xlim(0.9, 1.015); ax.set_title(ttl); ax.set_xlabel("Accuracy")
    fig.suptitle("外部公开数据集：PF-39 近饱和（仅作通用性验证）；NE39 上深度模型反超 sklearn",
                 y=1.06, fontsize=11, fontweight="bold")
    fig.savefig(OUT / "fig9_external_datasets.png"); plt.close(fig)


# ---------------- fig10 通用 LLM 直接判稳基线 ----------------
def fig10():
    rows = [("llm_direct_zeroshot", "通用LLM\n零样本", "#B0B0B0"),
            ("llm_direct_fewshot", "通用LLM\n4-shot", "#6E6E6E"),
            ("pimtnet", "PIMT-Net\n(单种子)", C["pimtnet"]),
            ("pimtnet_ens10", "PIMT-Net\nEns10", "#FFB000"),
            ("svm", "SVM\n(最佳基线)", C["svm"])]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0),
                             gridspec_kw={"width_ratios": [3, 2]})
    # 左：判别指标 acc/F1/AUC
    ax = axes[0]
    w = 0.26
    for k, (metric, mlab) in enumerate([("acc", "Accuracy"), ("f1", "F1"), ("auc", "AUC")]):
        off = (k - 1) * w
        for j, (m, _, col) in enumerate(rows):
            mu, sd, n = ms(m, "ieee39", "main", metric)
            ax.bar(j + off, mu, w, color=col, alpha=1 - 0.16 * k, zorder=3,
                   yerr=(sd if n > 1 else 0), error_kw=dict(capsize=2, lw=0.8),
                   label=mlab if j == 0 else "")
            ax.text(j + off, mu + 0.02, f"{mu:.3f}", ha="center", fontsize=6.4)
    ax.set_xticks(range(len(rows)), [r[1] for r in rows])
    ax.set_ylim(0.55, 1.0); ax.set_ylabel("得分")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax.set_title("(a) 判别指标：通用 LLM 落后专用模型 20pp+")
    # 右：错误率（FNR 是关键）
    ax = axes[1]
    x = np.arange(len(rows)); w2 = 0.36
    fnr = [ms(m, "ieee39", "main", "fnr")[0] for m, _, _ in rows]
    fpr = [ms(m, "ieee39", "main", "fpr")[0] for m, _, _ in rows]
    ax.bar(x - w2 / 2, fnr, w2, color="#C44E52", zorder=3, label="FNR（漏报失稳）")
    ax.bar(x + w2 / 2, fpr, w2, color="#8C6D31", zorder=3, label="FPR（误报）")
    for xi, v in zip(x - w2 / 2, fnr):
        ax.text(xi, v + 0.008, f"{v:.3f}", ha="center", fontsize=6.6, color="#C44E52",
                fontweight="bold")
    for xi, v in zip(x + w2 / 2, fpr):
        ax.text(xi, v + 0.008, f"{v:.3f}", ha="center", fontsize=6.6, color="#8C6D31")
    ax.set_xticks(x, [r[1] for r in rows]); ax.set_ylim(0, 0.62)
    ax.axhline(0.5, color="#C44E52", lw=0.9, ls="--", alpha=0.6)
    ax.text(len(rows) - 0.55, 0.508, "漏报 50%", fontsize=6.8, color="#C44E52")
    ax.set_ylabel("错误率"); ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    ax.set_title("(b) 错误率：通用 LLM 失稳漏报近半")
    fig.suptitle("通用 LLM 直接判稳不可行（DeepSeek-V3，温度 0，100 例分层子集，ieee39 t13）"
                 "——数值判别须交给专用物理信息模型",
                 y=1.03, fontsize=11, fontweight="bold")
    fig.savefig(OUT / "fig10_llm_direct.png"); plt.close(fig)


if __name__ == "__main__":
    for f in [fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10]:
        f()
        print(f"[ok] {f.__name__}")
    print("输出目录:", OUT)
