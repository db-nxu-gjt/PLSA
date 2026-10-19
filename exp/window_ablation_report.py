# -*- coding: utf-8 -*-
"""观测窗消融汇总：窗口-精度权衡曲线 + 07_window_ablation.xlsx。

窗口 = 故障后观测时长（tf=1.0s 起）：0.2/0.5/1.0/2.0/4.0s -> 13/31/61/121/241 步@60Hz。
t13 复用主表结果（ieee39 main），其余为 ieee39w{31,61,121,241} 数据集别名。
口径说明：长窗口排除早发散场景（N 2349->1642，失稳占比 55.4%->36.2%），各窗口独立划分。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUT = ROOT.parent / "export"

WINDOWS = [(13, 0.2, "ieee39"), (31, 0.5, "ieee39w31"), (61, 1.0, "ieee39w61"),
           (121, 2.0, "ieee39w121"), (241, 4.0, "ieee39w241")]
MODELS = ["pimtnet", "mlp", "lstm", "transformer", "svm", "xgboost"]
COLS = ["acc", "f1", "auc"]


def main():
    # 数据集口径信息（N / 失稳占比，来自 manifest）
    mf = json.loads((ROOT.parent / "data" / "processed" / "manifest.json").read_text(encoding="utf-8"))
    info = {}
    for steps, secs, ds in WINDOWS:
        k = "ieee39" if ds == "ieee39" else f"ieee39_t{steps}"
        m = mf[k]
        info[steps] = dict(window_s=secs, steps=steps, N=m["n"],
                           unstable_pct=m["unstable_pct"], n_test=m["n_test"])

    rows, raw = [], []
    for model in MODELS:
        for steps, secs, ds in WINDOWS:
            files = sorted((RESULTS / model).glob(f"{ds}_main_seed*.json"))
            recs = [json.loads(f.read_text(encoding="utf-8")) for f in files]
            r = dict(model=model, **info[steps], n_seeds=len(recs))
            for c in COLS:
                vals = [x[c] for x in recs]
                r[c] = round(float(np.mean(vals)), 4)
                r[c + "_std"] = round(float(np.std(vals)), 4)
            rows.append(r)
            for x in recs:
                raw.append(dict(model=model, steps=steps, window_s=secs, seed=x["seed"],
                                **{c: x[c] for c in COLS}))
    df = pd.DataFrame(rows)

    # ---- 打印 acc 矩阵 ----
    piv = df.pivot(index="model", columns="window_s", values="acc")
    piv = piv.reindex(MODELS)
    print("\n===== 观测窗-精度曲线（acc，mean over 5 seeds）=====")
    print(piv.round(4).to_string())
    print("\n各窗口样本：", {r["window_s"]: (r["N"], r["unstable_pct"]) for r in df.to_dict("records")
          if r["model"] == "pimtnet"})

    # ---- 曲线图 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.6), dpi=150)
    colors = {"pimtnet": "#d62728", "svm": "#1f77b4", "xgboost": "#9467bd",
              "lstm": "#2ca02c", "transformer": "#ff7f0e", "mlp": "#7f7f7f"}
    for model in MODELS:
        g = df[df["model"] == model].sort_values("window_s")
        lw, z = 2.6, 5 if model == "pimtnet" else 2
        ax.plot(g["window_s"], g["acc"], "-o", color=colors[model], lw=lw, zorder=z,
                ms=5, label="PIMT-Net (ours)" if model == "pimtnet" else model)
        ax.fill_between(g["window_s"], g["acc"] - g["acc_std"], g["acc"] + g["acc_std"],
                        color=colors[model], alpha=0.12, zorder=1)
    ax.set_xscale("log")
    ax.set_xticks([0.2, 0.5, 1.0, 2.0, 4.0])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Post-fault observation window (s, log scale)")
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0.82, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.set_title("Accuracy vs. observation window length (IEEE39-Sim)")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    png = OUT / "07_window_ablation_curve.png"
    fig.savefig(png)
    print(f"[ok] {png}")

    # ---- Excel ----
    xlsx = OUT / "07_window_ablation.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        df[[c for c in df.columns if not c.endswith("_std")]] \
            .to_excel(w, sheet_name="summary", index=False)
        for c in COLS:
            p = df.pivot(index="model", columns="window_s", values=c).reindex(MODELS)
            p.to_excel(w, sheet_name=f"{c}_matrix")
        pd.DataFrame(raw).to_excel(w, sheet_name="raw_per_seed", index=False)
        note = pd.DataFrame({
            "项目": ["窗口定义", "排除口径", "t13 来源", "PIMT-Net 协议", "划分", "核心发现"],
            "说明": [
                "观测窗 [tf=1.0s, 1.0+L s]，60Hz 均匀网格插值；L∈{0.2,0.5,1.0,2.0,4.0}",
                "波形早于窗口末结束（已发散）的场景被排除：N 2349→1642，失稳占比 55.4%→36.2%（早期预警任务固有口径）",
                "t13 复用主表 ieee39 main 结果（协议一致）",
                "单阶段最优配置（pimt_best.json + cosine lr3e-4 500ep patience50 λt=1.0）；主表为两阶段（≈等价）",
                "各窗口独立 60/20/20 分层划分（seed=2026）",
                "观测窗是原主表性能瓶颈：0.2s→1.0s 全模型 +8~10pp；PIMT-Net 在短窗口（0.2-0.5s）相对优势最大，长窗口全员趋饱和",
            ]})
        note.to_excel(w, sheet_name="口径说明", index=False)
    print(f"[ok] {xlsx}")


if __name__ == "__main__":
    main()
