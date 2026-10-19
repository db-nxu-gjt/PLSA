# -*- coding: utf-8 -*-
"""English figures for the ESWA manuscript -> submit/main/figs_en/ (PDF, 300dpi).

Reuses data loaders from make_figures.py (exp/results JSONs).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_figures as mf   # noqa: E402  (data loaders: ms/load/C)

OUT = Path(__file__).resolve().parent.parent / "figs_en"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold",
})
C, ms = mf.C, mf.ms
MODELS, DATASETS = mf.MODELS, mf.DATASETS
DS_NAME = {"ieee39": "IEEE39 (0.2 s)", "ieee118": "IEEE118 (0.2 s)",
           "ieee39w121": "IEEE39 (2.0 s)", "ieee118w121": "IEEE118 (2.0 s)"}
NAME = dict(mf.NAME, pimtnet_ens10="PIMT-Net-Ens10")


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print(f"[ok] {name}")


# ---------------- architecture ----------------
def fig_arch():
    fig, ax = plt.subplots(figsize=(10.2, 7.4))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 10)

    def box(x, y, w, h, title, lines, fc, tc="black", fs=8.2, tfs=9.2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                    fc=fc, ec="#333333", lw=1.0))
        ax.text(x + w / 2, y + h - 0.30, title, ha="center", va="top",
                fontsize=tfs, fontweight="bold", color=tc)
        for k, ln in enumerate(lines):
            ax.text(x + w / 2, y + h - 0.72 - 0.34 * k, ln, ha="center",
                    va="top", fontsize=fs, color=tc)

    def arrow(x1, y1, x2, y2, style="-|>", lw=1.4, color="#444444", ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                     mutation_scale=16, lw=lw, color=color,
                                     linestyle=ls, shrinkA=2, shrinkB=2))

    # top: three resource layers
    box(0.3, 8.0, 3.0, 1.7, "Data layer",
        ["PMU / WAMS · SCADA · alarms", "Fault records ·", "simulation sample library"],
        "#DCE6F1")
    box(3.6, 8.0, 2.8, 1.7, "Knowledge layer",
        ["Grid codes · fault cases ·", "network topology · stability",
         "theory · knowledge graph"], "#E4DFEC")
    box(6.7, 8.0, 3.0, 1.7, "Physics model layer",
        ["DAE / swing equations ·", "KCL / KVL · stability",
         "criteria · energy functions"], "#F2DCDB")
    # engines
    box(0.3, 4.9, 4.2, 2.4, "PINN physics computing engine",
        ["trajectory prediction · parameter ID", "stability assessment · simulation accel.",
         "(physics-constrained, sample-efficient)"], "#1F4E79", tc="white")
    box(5.5, 4.9, 4.2, 2.4, "LLM cognitive reasoning engine",
        ["intent parsing · scenario generation", "tool calling · report narration",
         "(RAG + agent orchestration)"], "#C55A11", tc="white")
    arrow(2.4, 8.0, 2.4, 7.3); arrow(4.9, 8.0, 3.6, 7.3)
    arrow(8.2, 8.0, 7.6, 7.3)
    arrow(2.4, 7.3, 2.4, 8.0, style="<|-")  # noqa
    # bidirectional coupling
    arrow(4.5, 6.1, 5.5, 6.1, style="<|-|>", lw=1.8, color="#8C1515")
    ax.text(5.0, 6.32, "engine\ncoupling", ha="center", fontsize=7.6, color="#8C1515")
    # orchestration band (PIVN)
    box(0.3, 3.1, 9.4, 1.35, "LLM orchestration loop (PIVN)",
        ["Plan  \u2192  Invoke registered tools  \u2192  Verify (PhysCheck)  \u2192  Narrate",
         "all numeric claims originate from registered tools: PIMT-Net · TDS · power flow · RAG"],
        "#E2EFDA", fs=8.6, tfs=10.0)
    arrow(7.6, 4.9, 7.6, 4.45); arrow(2.4, 4.9, 2.4, 4.45)
    # PhysCheck callout
    box(0.3, 1.55, 9.4, 1.15, "PhysCheck: physics-grounded verification of LLM outputs",
        ["(i) provenance check  ·  (ii) physics consistency (residuals, limits, power balance)",
         "(iii) confidence calibration — LLM acts as advisor, not decision maker"],
        "#FFF2CC", fs=8.6, tfs=10.0)
    arrow(9.9, 4.2, 9.9, 2.7, style="-|>", lw=1.4, color="#8C1515", ls="--")
    ax.text(9.55, 3.35, "verify", rotation=90, fontsize=7.6, color="#8C1515")
    # applications
    apps = [("Security deduction", "contingency set generation ·\nstrategy simulation"),
            ("Fault identification", "alarm classification ·\nfault location"),
            ("Stability prediction", "classification · trajectories ·\ncritical clearing time"),
            ("Decision support", "preventive control ·\nreport generation")]
    for k, (t, s) in enumerate(apps):
        x = 0.3 + k * 2.45
        box(x, 0.0, 2.25, 1.15, t, s.split("\n"), "#FFFFFF", fs=7.6, tfs=8.8)
        arrow(x + 1.12, 1.55, x + 1.12, 1.15)
    save(fig, "fig_arch.pdf")


# ---------------- main heatmap ----------------
def fig_heatmap():
    mat = np.array([[ms(m, d)[0] for d in DATASETS] for m in MODELS])
    best = np.nanargmax(mat, axis=0)
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0.55, vmax=1.0, aspect="auto")
    ax.set_xticks(range(4), [DS_NAME[d] for d in DATASETS])
    ax.set_yticks(range(len(MODELS)), [NAME[m] for m in MODELS])
    for i in range(len(MODELS)):
        for j in range(4):
            star = best[j] == i
            ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center",
                    fontsize=8, fontweight="bold" if star else "normal",
                    color="white" if mat[i, j] < 0.68 or mat[i, j] > 0.96 else "black")
    ax.add_patch(plt.Rectangle((-0.5, MODELS.index("pimtnet") - 0.5), 4, 1,
                               fill=False, edgecolor="#D62728", lw=2.2))
    ax.grid(False)
    ax.set_title("Stability classification accuracy (main split; mean over 5/10 seeds)")
    cb = fig.colorbar(im, ax=ax, shrink=0.85); cb.set_label("Accuracy")
    save(fig, "fig_heatmap.pdf")


# ---------------- window ablation ----------------
def fig_window():
    wins = [("ieee39", 0.2), ("ieee39w31", 0.5), ("ieee39w61", 1.0),
            ("ieee39w121", 2.0), ("ieee39w241", 4.0)]
    # fixed-cohort control: same 1,642 scenarios tracked across all windows
    cwins = [("ieee39c13", 0.2), ("ieee39c31", 0.5), ("ieee39c61", 1.0),
             ("ieee39c121", 2.0)]
    show = ["pimtnet", "svm", "xgboost", "transformer", "mlp", "lstm"]
    cohort = ["pimtnet", "svm", "xgboost"]
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    for m in show:
        mu = [ms(m, ds)[0] for ds, _ in wins]
        ax.plot([t for _, t in wins], mu, "-o", ms=4,
                lw=2.4 if m == "pimtnet" else 1.4, color=C[m], label=NAME[m])
    for m in cohort:
        mu = [ms(f"{m}_wc", ds)[0] for ds, _ in cwins] + [ms(m, "ieee39w241")[0]]
        ax.plot([t for _, t in cwins] + [4.0], mu, "--s", ms=3.5,
                lw=1.6, color=C[m], zorder=4)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color="#555555", ls="--", marker="s",
                          ms=3.5, lw=1.6))
    labels.append("fixed 1,642-scenario cohort")
    ax.axvspan(0.9, 2.1, color="#FFB000", alpha=0.12, lw=0)
    ax.text(1.5, 1.006, "recommended 1.0–2.0 s", ha="center", va="top",
            fontsize=7.8, color="#8C1515")
    ax.set_xscale("log"); ax.set_xticks([t for _, t in wins],
                                        ["0.2", "0.5", "1.0", "2.0", "4.0"])
    ax.set_xlabel("Post-fault observation window (s)"); ax.set_ylabel("Accuracy")
    ax.set_ylim(0.80, 1.01)
    ax.legend(handles, labels, fontsize=7.5, frameon=False, ncol=2,
              loc="lower right")
    ax.set_title("Accuracy vs. observation window length (IEEE39)")
    save(fig, "fig_window.pdf")


# ---------------- noise robustness ----------------
def fig_noise():
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
        ax.set_xlabel("Noise level $\\sigma$"); ax.set_ylim(0.60, 1.01)
    axes[0].set_ylabel("Accuracy")
    axes[-1].legend(fontsize=7, frameon=False, loc="lower left")
    fig.suptitle("Noise robustness: PIMT-Net (bold red) shows near-zero degradation"
                 " on all datasets and noise levels", y=1.05, fontsize=11,
                 fontweight="bold")
    save(fig, "fig_noise.pdf")


# ---------------- few-shot ----------------
def fig_fewshot():
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
    for ax in axes[1]:
        ax.set_xlabel("Training-set fraction")
    for ax in axes[:, 0]:
        ax.set_ylabel("Accuracy")
    axes[0, 0].legend(fontsize=7, frameon=False, ncol=2)
    fig.suptitle("Data efficiency under reduced training sets (25% / 50% / 100%)",
                 y=1.0, fontsize=11, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig_fewshot.pdf")


# ---------------- ablation ----------------
def fig_ablation():
    variants = [("pimtnet_ab_nolt", "ab_nolt", "w/o trajectory loss ($\\lambda_t$=0)"),
                ("pimtnet_ab_nolp", "ab_nolp", "w/o physics residual ($\\lambda_p$=0)"),
                ("pimtnet_ab_noltlp", "ab_noltlp", "w/o trajectory + physics"),
                ("pimtnet_ab_noaug", "ab_noaug", "w/o input augmentation"),
                ("pimtnet_ab_nohead", "ab_nohead", "w/o trajectory head")]
    combos = [("ieee39", "main", "0.2 s / full"), ("ieee39", "sub25", "0.2 s / 25%"),
              ("ieee39w121", "main", "2.0 s / full"), ("ieee39w121", "sub25", "2.0 s / 25%")]
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    n_v, w = len(variants), 0.16
    for k, (vd, key, lab) in enumerate(variants):
        off = (k - (n_v - 1) / 2) * w
        for j, (ds, tag, clab) in enumerate(combos):
            base = ms("pimtnet", ds, tag)[0]
            d = (ms(vd, ds, tag)[0] - base) * 100
            ax.scatter(j + off, d, s=42, color=C[key], zorder=3,
                       edgecolor="white", linewidth=0.6)
            ax.plot([j + off, j + off], [0, d], color=C[key], lw=1.1, alpha=0.75, zorder=2)
    ax.axhline(0, color="black", lw=1.2, zorder=1)
    ax.set_xticks(range(len(combos)), [c[2] for c in combos])
    ax.set_ylabel("$\\Delta$Accuracy vs. full PIMT-Net (pp)")
    ax.set_title("Regime-dependent ablation: multi-task terms can hurt in the "
                 "short-window/small-sample regime;\nphysics residuals and "
                 "augmentation contribute positively at 2.0 s")
    handles = [plt.Line2D([], [], marker="o", ls="", color=C[key], label=lab)
               for _, key, lab in variants]
    ax.legend(handles=handles, fontsize=7.5, frameon=False, ncol=3, loc="lower left")
    save(fig, "fig_ablation.pdf")


# ---------------- ensemble ----------------
def fig_ensemble():
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    for j, ds in enumerate(DATASETS):
        pm, ps, _ = ms("pimtnet", ds)
        ens = ms("pimtnet_ens10", ds)[0]
        base_accs = {m: ms(m, ds)[0] for m in MODELS if m != "pimtnet"}
        bm = max(base_accs, key=base_accs.get)
        ax.errorbar(j - 0.18, pm, yerr=ps, fmt="o", ms=6, color=C["pimtnet"],
                    capsize=3, label="PIMT-Net (single seed)" if j == 0 else "")
        ax.scatter(j, ens, marker="*", s=210, color="#FFB000",
                   edgecolor="#8C1515", linewidth=0.8, zorder=4,
                   label="PIMT-Net-Ens10 (10-seed prob. avg.)" if j == 0 else "")
        ax.scatter(j + 0.18, base_accs[bm], marker="s", s=42, color="#4C72B0",
                   label="best single baseline" if j == 0 else "")
        ax.text(j + 0.30, base_accs[bm], NAME[bm], fontsize=7, va="center", color="#4C72B0")
        ax.text(j, ens + 0.006, f"{ens:.4f}", ha="center", fontsize=8,
                fontweight="bold", color="#8C1515")
    ax.set_xticks(range(4), [DS_NAME[d] for d in DATASETS])
    ax.set_ylabel("Accuracy"); ax.set_ylim(0.84, 0.99)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.set_title("10-seed ensembles: best overall accuracy at three of four operating points")
    save(fig, "fig_ensemble.pdf")


# ---------------- LLM direct ----------------
def fig_llm():
    rows = [("llm_direct_zeroshot_full", "Generic LLM\nzero-shot", "#B0B0B0"),
            ("llm_direct_fewshot_full", "Generic LLM\n4-shot", "#6E6E6E"),
            ("pimtnet", "PIMT-Net\n(single seed)", C["pimtnet"]),
            ("pimtnet_ens10", "PIMT-Net\nEns10", "#FFB000"),
            ("svm", "SVM\n(best baseline)", C["svm"])]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0),
                             gridspec_kw={"width_ratios": [3, 2]})
    ax = axes[0]; w = 0.26
    for k, (metric, mlab) in enumerate([("acc", "Accuracy"), ("f1", "F1"), ("auc", "AUC")]):
        off = (k - 1) * w
        for j, (m, _, col) in enumerate(rows):
            mu, sd, n = ms(m, "ieee39", "main", metric)
            ax.bar(j + off, mu, w, color=col, alpha=1 - 0.16 * k, zorder=3,
                   yerr=(sd if n > 1 else 0), error_kw=dict(capsize=2, lw=0.8),
                   label=mlab if j == 0 else "")
            ax.text(j + off, mu + 0.02, f"{mu:.3f}", ha="center", fontsize=6.4)
    ax.set_xticks(range(len(rows)), [r[1] for r in rows])
    ax.set_ylim(0.55, 1.0); ax.set_ylabel("Score")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax.set_title("(a) Discrimination metrics")
    ax = axes[1]
    x = np.arange(len(rows)); w2 = 0.36
    fnr = [ms(m, "ieee39", "main", "fnr")[0] for m, _, _ in rows]
    fpr = [ms(m, "ieee39", "main", "fpr")[0] for m, _, _ in rows]
    ax.bar(x - w2 / 2, fnr, w2, color="#C44E52", zorder=3, label="FNR (missed instability)")
    ax.bar(x + w2 / 2, fpr, w2, color="#8C6D31", zorder=3, label="FPR (false alarm)")
    for xi, v in zip(x - w2 / 2, fnr):
        ax.text(xi, v + 0.008, f"{v:.3f}", ha="center", fontsize=6.6,
                color="#C44E52", fontweight="bold")
    for xi, v in zip(x + w2 / 2, fpr):
        ax.text(xi, v + 0.008, f"{v:.3f}", ha="center", fontsize=6.6, color="#8C6D31")
    ax.set_xticks(x, [r[1] for r in rows]); ax.set_ylim(0, 0.62)
    ax.axhline(0.5, color="#C44E52", lw=0.9, ls="--", alpha=0.6)
    ax.text(len(rows) - 0.55, 0.508, "50% missed", fontsize=6.8, color="#C44E52")
    ax.set_ylabel("Error rate"); ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    ax.set_title("(b) Error rates")
    fig.suptitle("Generic LLMs cannot perform TSA directly (DeepSeek-V3, temperature 0, "
                 "full 469-case test set, IEEE39 0.2 s):\naccuracy ~70% lags specialized "
                 "models by >16 pp and 33--43% of instabilities are missed",
                 y=1.06, fontsize=10.5, fontweight="bold")
    save(fig, "fig_llm.pdf")


# ---------------- fault location ----------------
def fig_fault():
    models8 = ["lstm", "transformer", "xgboost", "cnn1d", "rf", "mlp", "densenet", "svm"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.6))
    for ax, ds, ttl in [(axes[0], "ieee39", "0.2 s window (primary protocol)"),
                        (axes[1], "ieee39w121", "2.0 s window (MAD scaling)")]:
        accs = [ms(m, ds, "faultid")[0] for m in models8]
        order = np.argsort(accs)
        names = [NAME[models8[i]] for i in order]
        vals = [accs[i] for i in order]
        ax.barh(names, vals, color=[C[models8[i]] for i in order], height=0.62)
        for y, v in enumerate(vals):
            ax.text(v + 0.004, y, f"{v:.4f}", va="center", fontsize=7.5)
        ax.set_xlim(0.60, 1.03); ax.set_title(ttl); ax.set_xlabel("Accuracy (39 classes)")
    fig.suptitle("Fault-location identification (replacement task for non-public alarm "
                 "data): sequence models dominate", y=1.05, fontsize=11, fontweight="bold")
    save(fig, "fig_fault.pdf")


if __name__ == "__main__":
    for f in [fig_arch, fig_heatmap, fig_window, fig_noise, fig_fewshot,
              fig_ablation, fig_ensemble, fig_llm, fig_fault]:
        f()
    print("Output:", OUT)
