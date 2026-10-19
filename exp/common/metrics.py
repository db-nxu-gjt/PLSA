# -*- coding: utf-8 -*-
"""评估指标：分类（Acc/Pre/Rec/F1/AUC/FPR/FNR）+ 轨迹 RMSE（功角 deg / 角速度 pu）。"""
import numpy as np
import torch


def cls_metrics(y_true, y_prob, thr=0.5):
    """y_true/y_prob: 1D ndarray，正类=失稳(1)。"""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= thr).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    acc = (tp + tn) / max(len(y_true), 1)
    pre = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * pre * rec / max(pre + rec, 1e-12)
    fpr = fp / max(fp + tn, 1)
    fnr = fn / max(fn + tp, 1)
    # AUC（无 sklearn 依赖，秩实现；并列取平均秩）
    order = np.argsort(y_prob, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_prob) + 1)
    # 处理并列：对每个唯一值取平均秩
    for v in np.unique(y_prob):
        m = y_prob == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    n_pos, n_neg = max(int(y_true.sum()), 1), max(int((y_true == 0).sum()), 1)
    auc = (ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    auc = min(max(auc, 0.0), 1.0)
    return dict(acc=acc, precision=pre, recall=rec, f1=f1, auc=auc, fpr=fpr, fnr=fnr)


def traj_rmse(pred_d, pred_w, true_d, true_w, mask):
    """pred/true: (B,T,N) tensor 或 ndarray；mask: (B,T)。返回功角 RMSE(deg) 与角速度 RMSE(pu)。"""
    to_np = lambda x: x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    pred_d, pred_w, true_d, true_w, mask = map(to_np, (pred_d, pred_w, true_d, true_w, mask))
    m = mask.astype(bool)
    if m.sum() == 0:
        return dict(rmse_delta_deg=float("nan"), rmse_omega_pu=float("nan"))
    ed = np.rad2deg(pred_d[m] - true_d[m])
    ew = pred_w[m] - true_w[m]
    return dict(rmse_delta_deg=float(np.sqrt((ed ** 2).mean())),
                rmse_omega_pu=float(np.sqrt((ew ** 2).mean())))
