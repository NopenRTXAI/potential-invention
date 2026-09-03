"""Evaluation metrics: accuracy, precision, recall, F1, log-loss.

No sklearn dependency — everything is implemented directly so the engine
runs on minimal / weak machines.
"""

import numpy as np


def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def _prf_binary(y_true, y_pred, positive=1):
    tp = float(np.sum((y_pred == positive) & (y_true == positive)))
    fp = float(np.sum((y_pred == positive) & (y_true != positive)))
    fn = float(np.sum((y_pred != positive) & (y_true == positive)))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def classification_report(y_true, y_pred, n_classes):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    acc = accuracy(y_true, y_pred)
    if n_classes <= 2:
        prec, rec, f1 = _prf_binary(y_true, y_pred, positive=int(y_true.max()))
        macro = {"precision": prec, "recall": rec, "f1": f1}
    else:
        per = []
        for c in range(n_classes):
            per.append(_prf_binary(y_true, y_pred, positive=c))
        macro = {
            "precision": float(np.mean([p[0] for p in per])),
            "recall": float(np.mean([p[1] for p in per])),
            "f1": float(np.mean([p[2] for p in per])),
        }
    return {"accuracy": acc, **macro}


def log_loss(probs, y_true, eps=1e-9):
    probs = np.clip(np.asarray(probs, dtype=np.float64), eps, 1.0 - eps)
    y = np.asarray(y_true)
    if probs.ndim == 2:
        p = probs[np.arange(len(y)), y]
    else:
        p = np.where(y == 1, probs, 1.0 - probs)
    return float(-np.mean(np.log(p)))


def confusion_matrix(y_true, y_pred, n_classes):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm.tolist()
