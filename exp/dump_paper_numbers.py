# -*- coding: utf-8 -*-
"""为 ESWA 论文提取全部精确数字 → dump_numbers.txt"""
import json
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parent / "results"
OUT = Path(__file__).resolve().parent / "dump_numbers.txt"

MODELS = ["pimtnet", "mlp", "cnn1d", "lstm", "transformer", "densenet",
          "pfnn", "pi_seq2seq", "pi_gcn", "es_pinn", "tsa_llm",
          "svm", "rf", "xgboost"]
DS = ["ieee39", "ieee118", "ieee39w121", "ieee118w121"]
COLS = ["acc", "f1", "auc", "fnr", "fpr", "params", "latency_ms"]

lines = []

def stat(model, ds, tag, col):
    vals = []
    for f in sorted((R / model).glob(f"{ds}_{tag}_seed*.json")):
        v = json.loads(f.read_text(encoding="utf-8")).get(col)
        if v is not None:
            vals.append(float(v))
    if not vals:
        return None
    a = np.array(vals)
    return f"{a.mean():.4f}±{a.std(ddof=1):.4f}(n={len(a)})" if len(a) > 1 else f"{a.mean():.4f}(n=1)"

def sec(t):
    lines.append("\n" + "=" * 20 + " " + t + " " + "=" * 20)

sec("MAIN per dataset (acc/f1/auc/fnr)")
for ds in DS:
    lines.append(f"\n--- {ds} ---")
    for m in MODELS:
        row = [f"{m:12s}"] + [stat(m, ds, "main", c) or "-" for c in COLS]
        lines.append("  ".join(row))

sec("ENS10 (pimtnet_ens10)")
for ds in DS:
    for tag in ["main", "noise1", "noise2", "noise5"]:
        s = stat("pimtnet_ens10", ds, tag, "acc")
        if s:
            lines.append(f"{ds:12s} {tag:7s} acc={s}  f1={stat('pimtnet_ens10', ds, tag, 'f1')}  auc={stat('pimtnet_ens10', ds, tag, 'auc')}  fnr={stat('pimtnet_ens10', ds, tag, 'fnr')}")

sec("VARIANTS pimtnet vs best (ood/sub25/sub50/noise)")
for ds in DS:
    for tag in ["ood", "sub25", "sub50", "noise1", "noise2", "noise5"]:
        row = [f"{ds:12s} {tag:7s}"]
        for m in ["pimtnet", "svm", "xgboost", "transformer", "mlp", "lstm"]:
            row.append(f"{m}={stat(m, ds, tag, 'acc') or '-'}")
        lines.append("  ".join(row))

sec("ABLATION (acc)")
for m in ["pimtnet", "pimtnet_ab_nolt", "pimtnet_ab_nolp", "pimtnet_ab_noltlp",
          "pimtnet_ab_noaug", "pimtnet_ab_nohead"]:
    for ds in ["ieee39", "ieee39w121"]:
        for tag in ["main", "sub25"]:
            s = stat(m, ds, tag, "acc")
            if s:
                lines.append(f"{m:18s} {ds:12s} {tag:5s} acc={s}")

sec("FAULTID (acc)")
for ds in ["ieee39", "ieee39w121"]:
    for m in ["lstm", "transformer", "xgboost", "cnn1d", "rf", "mlp", "densenet", "svm"]:
        s = stat(m, ds, "faultid", "acc")
        if s:
            lines.append(f"{m:12s} {ds:12s} acc={s}")

sec("EXTERNAL (acc)")
for ds in ["pf39", "ne39"]:
    for m in MODELS:
        if m == "pimtnet":
            continue
        s = stat(m, ds, "main", "acc")
        if s:
            lines.append(f"{m:12s} {ds:6s} acc={s}  auc={stat(m, ds, 'main', 'auc')}")

sec("LLM DIRECT")
for m in ["llm_direct_zeroshot", "llm_direct_fewshot"]:
    f = R / m / "ieee39_main_seed2026.json"
    if f.exists():
        r = json.loads(f.read_text(encoding="utf-8"))
        lines.append(f"{m}: " + ", ".join(f"{k}={r[k]}" for k in
                     ["acc", "f1", "auc", "fnr", "fpr", "precision", "recall", "llm_model"]))

sec("WINDOW ABLATION (acc)")
for ds in ["ieee39", "ieee39w31", "ieee39w61", "ieee39w121", "ieee39w241"]:
    row = [f"{ds:12s}"]
    for m in ["pimtnet", "svm", "xgboost", "transformer", "mlp", "cnn1d", "lstm"]:
        row.append(f"{m}={stat(m, ds, 'main', 'acc') or '-'}")
    lines.append("  ".join(row))

sec("TRAJECTORY RMSE (pimtnet, main)")
for ds in DS:
    for col in ["traj_rmse_delta", "traj_rmse_omega", "rmse_delta", "rmse_omega"]:
        s = stat("pimtnet", ds, "main", col)
        if s:
            lines.append(f"{ds:12s} {col}: {s}")

sec("LATENCY/FIT (main, ieee39)")
for m in MODELS:
    s = stat(m, "ieee39", "main", "latency_ms")
    lines.append(f"{m:12s} latency_ms={s or '-'}")

OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"[ok] {OUT} ({len(lines)} lines)")
