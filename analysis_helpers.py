"""Three small helpers reused by the modelling and monitoring notebooks."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix


def make_features(data):
    """Only amount and type enter the model. No balances or outcome columns."""
    if set(data.columns) != {'amount', 'type'}:
        raise ValueError('Pass only amount and type, without labels or balances.')
    if not data['type'].isin(['TRANSFER', 'CASH_OUT']).all():
        raise ValueError('This model covers TRANSFER and CASH_OUT only.')
    if not np.isfinite(data['amount']).all() or data['amount'].lt(0).any():
        raise ValueError('Amounts must be finite and non-negative.')
    return pd.DataFrame({
        'log_amount': np.log1p(data['amount']),
        'is_transfer': data['type'].eq('TRANSFER').astype(int),
    }, index=data.index)


def evaluate(y_true, scores, threshold):
    """Precision = TP / alerts; recall = TP / fraud; undefined rates stay missing."""
    y = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    if y.ndim != 1 or scores.shape != y.shape or not np.isin(y, [0, 1]).all():
        raise ValueError('Provide aligned scores and binary labels.')
    if not np.isfinite(scores).all() or not 0 <= threshold <= 1:
        raise ValueError('Provide finite scores and a threshold between 0 and 1.')
    tn, fp, fn, tp = confusion_matrix(y, scores >= threshold, labels=[0, 1]).ravel()
    alerts, fraud, n = int(tp + fp), int(tp + fn), len(y)
    return {
        'transactions': n, 'fraud': fraud, 'alerts': alerts,
        'fraud_rate': fraud / n if n else np.nan,
        'alert_rate': alerts / n if n else np.nan,
        'precision': tp / alerts if alerts else np.nan,
        'recall': tp / fraud if fraud else np.nan,
        'F1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan,
        'AP': average_precision_score(y, scores) if len(np.unique(y)) == 2 else np.nan,
        'TP': int(tp), 'FP': int(fp), 'FN': int(fn), 'TN': int(tn),
    }


def threshold_comparison(y_true, scores, thresholds):
    """Compare a small, predefined list; choose only with validation labels."""
    return pd.DataFrame([
        {'threshold': threshold, **evaluate(y_true, scores, threshold)}
        for threshold in thresholds
    ])
