# -*- coding: utf-8 -*-
"""PIMT-Net 调参扫描：seed0 快速筛选配置（按 val acc 选型，报告 test 指标）。

用法: python exp/tune_pimtnet.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from common.dataset import make_loaders
from common.trainer import Trainer
from common.metrics import cls_metrics
from models.pimtnet import PIMTNet

CFG_BASE = dict(d=64, heads=4, n_enc=2, n_dec=2, ff=128, dropout=0.1, augment=False)

CONFIGS = [
    ("base",                        {},                                        {}),
    ("d128L3",                      dict(d=128, heads=8, n_enc=3, n_dec=3, ff=256), {}),
    ("d128L3+aug",                  dict(d=128, heads=8, n_enc=3, n_dec=3, ff=256, augment=True), {}),
    ("d128L3+aug+cos",              dict(d=128, heads=8, n_enc=3, n_dec=3, ff=256, augment=True),
     dict(cosine=True, lr=3e-4, max_epochs=500, patience=50)),
    ("d64+aug+cos",                 dict(augment=True),
     dict(cosine=True, lr=3e-4, max_epochs=500, patience=50)),
    ("d128L3+aug+cos+lt1",          dict(d=128, heads=8, n_enc=3, n_dec=3, ff=256, augment=True),
     dict(cosine=True, lr=3e-4, max_epochs=500, patience=50, lam_t=1.0)),
    ("d128L3+aug+cos+lp03",         dict(d=128, heads=8, n_enc=3, n_dec=3, ff=256, augment=True),
     dict(cosine=True, lr=3e-4, max_epochs=500, patience=50, lam_p=0.3)),
    ("d128L3+aug+cos+dp02",         dict(d=128, heads=8, n_enc=3, n_dec=3, ff=256, augment=True, dropout=0.2),
     dict(cosine=True, lr=3e-4, max_epochs=500, patience=50)),
]


def main():
    loaders, meta = make_loaders("ieee39", "main")
    out = []
    for name, cfg, tr in CONFIGS:
        t0 = time.time()
        model = PIMTNet(meta["n_ch"], meta["n_desc"], meta["n_mach"], **{**CFG_BASE, **cfg})
        trainer = Trainer(**tr)
        fit = trainer.fit(model, loaders["train"], loaders["val"], seed=0)
        for split, loader in [("val", loaders["val"]), ("test", loaders["test"])]:
            y, p, _ = trainer.predict(model, loader)
            m = cls_metrics(y, p)
            out.append(dict(config=name, split=split, acc=m["acc"], f1=m["f1"], auc=m["auc"],
                            fit_s=fit["fit_s"], best_epoch=fit["best_epoch"]))
            if split == "val":
                val_acc = m["acc"]
        print(f"{name:<22} val_acc={val_acc:.4f}  test_acc={out[-1]['acc']:.4f} "
              f"test_f1={out[-1]['f1']:.4f} test_auc={out[-1]['auc']:.4f} "
              f"ep={fit['best_epoch']} ({time.time()-t0:.0f}s)")
    with open(Path(__file__).parent / "results" / "tune_pimtnet.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
