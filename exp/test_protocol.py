# -*- coding: utf-8 -*-
"""临时：钉死原版 pimtnet 训练协议（lam_t 0.5 vs 1.0，seed0 复现 485.62/44/0.86354）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from common.dataset import make_loaders
from common.trainer import Trainer
from common.metrics import cls_metrics
from models.pimtnet import PIMTNet

loaders, meta = make_loaders("ieee39", "main")
for lt in (0.5, 1.0):
    torch.manual_seed(0)
    model = PIMTNet(meta["n_ch"], meta["n_desc"], meta["n_mach"], d=128, heads=8,
                    n_enc=3, n_dec=3, ff=256, augment=True)
    tr = Trainer(cosine=True, lr=3e-4, max_epochs=500, patience=50, lam_t=lt)
    fit = tr.fit(model, loaders["train"], loaders["val"], seed=0)
    y, p, _ = tr.predict(model, loaders["test"])
    m = cls_metrics(y, p)
    print(f"lam_t={lt} acc={m['acc']:.6f} best_val={fit['best_val']:.4f} "
          f"best_ep={fit['best_epoch']} fit_s={fit['fit_s']}", flush=True)
