# -*- coding: utf-8 -*-
"""传统 ML 基线：SVM / RF / XGBoost。

特征 = 窗内统计量（mean/std/min/max/last，逐通道）+ 5 维场景量。
"""
import numpy as np


def handcraft_features(X, desc):
    """X (N,T,C), desc (N,5) -> (N, C*5+5)。"""
    feats = [X.mean(1), X.std(1), X.min(1), X.max(1), X[:, -1, :]]
    if desc is not None:
        feats.append(desc)
    return np.concatenate(feats, axis=1).astype(np.float32)


def train_sklearn(name, X_tr, desc_tr, y_tr, X_te, desc_te, y_te, seed=0):
    """训练并评估一个 sklearn 系基线，返回 (metrics_dict, y_prob)。"""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from common.metrics import cls_metrics

    F_tr, F_te = handcraft_features(X_tr, desc_tr), handcraft_features(X_te, desc_te)
    if name == "svm":
        clf = make_pipeline(StandardScaler(),
                            SVC(C=10.0, gamma="scale", probability=True, random_state=seed))
    elif name == "rf":
        clf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    elif name == "xgboost":
        from xgboost import XGBClassifier
        clf = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1,
                            random_state=seed, eval_metric="logloss", n_jobs=-1)
    else:
        raise ValueError(f"unknown sklearn baseline: {name}")
    clf.fit(F_tr, y_tr)
    prob = clf.predict_proba(F_te)[:, 1]
    return cls_metrics(y_te, prob), prob
