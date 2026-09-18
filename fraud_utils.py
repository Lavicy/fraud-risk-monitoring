"""Small shared data, feature and evaluation helpers for the three notebooks."""
from pathlib import Path
import hashlib
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / 'data' / 'PS_20174392719_1491204439457_log.csv'
TABLES = ROOT / 'reports' / 'tables'
FIGURES = ROOT / 'reports' / 'figures'
ARTIFACTS = ROOT / 'artifacts'
SEED = 42
RISK_TYPES = ['CASH_OUT', 'TRANSFER']
RAW_FEATURES = ['type', 'amount', 'oldbalanceOrg']
FEATURE_SETS = {
    'full': ['type', 'log_amount', 'log_opening_balance', 'log_amount_to_balance',
             'origin_balance_zero', 'amount_exceeds_balance', 'requests_full_balance'],
    'no_full_balance_flag': ['type', 'log_amount', 'log_opening_balance', 'log_amount_to_balance',
                             'origin_balance_zero', 'amount_exceeds_balance'],
    'no_explicit_relationships': ['type', 'log_amount', 'log_opening_balance', 'origin_balance_zero'],
    'amount_and_type_only': ['type', 'log_amount'],
}


def prepare_outputs():
    for path in [TABLES, FIGURES, ARTIFACTS]:
        path.mkdir(parents=True, exist_ok=True)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_risk_data():
    """Read only the legacy policy allowlist, time and label; preserve CSV order."""
    columns = ['step', *RAW_FEATURES, 'isFraud']
    chunks = []
    for chunk in pd.read_csv(DATA_PATH, usecols=columns, chunksize=500_000,
                             dtype={'type': 'category', 'amount': 'float64',
                                    'oldbalanceOrg': 'float64'}):
        chunks.append(chunk.loc[chunk.type.isin(RISK_TYPES), columns].copy())
    result = pd.concat(chunks, ignore_index=True)
    if result.empty or not np.isfinite(result.step).all() or not result.step.eq(np.floor(result.step)).all():
        raise ValueError('Expected non-empty data with finite, integer steps.')
    if not result.isFraud.isin([0, 1]).all():
        raise ValueError('Expected non-missing binary fraud labels.')
    result['step'] = result.step.astype('int64')
    result['isFraud'] = result.isFraud.astype('int8')
    result.index.name = 'row_id'
    return result


def make_features(raw, feature_set='full'):
    """Explicit inputs only. The original balance features remain provenance-sensitive."""
    required = ['type', 'amount'] if feature_set == 'amount_and_type_only' else RAW_FEATURES
    if set(raw.columns) != set(required):
        raise ValueError(f'Pass exactly {required}; labels, IDs, time and closing balances are prohibited.')
    if not raw.type.isin(RISK_TYPES).all():
        raise ValueError('Only CASH_OUT and TRANSFER can be scored.')
    numeric = raw[[name for name in required if name != 'type']]
    if not np.isfinite(numeric.to_numpy()).all() or numeric.lt(0).any().any():
        raise ValueError('Numeric inputs must be finite, non-missing and non-negative.')
    amount = raw.amount
    features = pd.DataFrame({'type': raw.type.astype(str), 'log_amount': np.log1p(amount)}, index=raw.index)
    if feature_set != 'amount_and_type_only':
        balance = raw.oldbalanceOrg
        features['log_opening_balance'] = np.log1p(balance)
        features['log_amount_to_balance'] = np.log1p(amount / (balance + 1.0))
        features['origin_balance_zero'] = balance.eq(0).astype(float)
        features['amount_exceeds_balance'] = amount.gt(balance).astype(float)
        features['requests_full_balance'] = (
            balance.gt(0) & np.isclose(amount, balance, rtol=0, atol=0.01)
        ).astype(float)
    return features.loc[:, FEATURE_SETS[feature_set]]


def features_for(frame, feature_set='full'):
    columns = ['type', 'amount'] if feature_set == 'amount_and_type_only' else RAW_FEATURES
    return make_features(frame.loc[:, columns], feature_set)


def make_model(kind='logistic', feature_set='full'):
    numeric = [name for name in FEATURE_SETS[feature_set] if name != 'type']
    preprocessing = ColumnTransformer([
        ('numeric', StandardScaler(), numeric),
        ('type', OneHotEncoder(categories=[RISK_TYPES], drop='first',
                               handle_unknown='error', sparse_output=False), ['type']),
    ], remainder='drop', verbose_feature_names_out=False)
    if kind == 'logistic':
        estimator = LogisticRegression(C=1.0, class_weight='balanced', solver='lbfgs',
                                       max_iter=1_000, random_state=SEED)
    elif kind in ['hgb', 'hgb_balanced']:
        estimator = HistGradientBoostingClassifier(
            learning_rate=0.08, max_iter=150, max_leaf_nodes=15, min_samples_leaf=100,
            l2_regularization=1.0, early_stopping=False, random_state=SEED,
            class_weight='balanced' if kind == 'hgb_balanced' else None,
        )
    else:
        raise ValueError(f'Unknown model: {kind}')
    return Pipeline([('preprocess', preprocessing), ('model', estimator)])


def fit_model(model, features, target):
    if pd.Series(target).nunique() != 2:
        raise ValueError('Training requires both classes; do not repair with a random split.')
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(features, target)
    # Verify that the scaler saw this training population only.
    scaler = model.named_steps['preprocess'].named_transformers_['numeric']
    assert int(scaler.n_samples_seen_) == len(features)
    return model


def score_model(model, frame, feature_set='full'):
    return model.predict_proba(features_for(frame, feature_set))[:, 1]


def metrics(target, scores, threshold):
    """Undefined conditional rates are NaN, rather than misleading zeros."""
    y = np.asarray(target)
    scores = np.asarray(scores, dtype=float)
    if y.ndim != 1 or scores.shape != y.shape or not np.isin(y, [0, 1]).all():
        raise ValueError('Scores and binary labels must be aligned one-dimensional arrays.')
    if not np.isfinite(scores).all() or not 0 <= threshold <= 1:
        raise ValueError('Expected finite scores and a probability-scale threshold.')
    alerts = scores >= threshold
    if len(y):
        tn, fp, fn, tp = confusion_matrix(y, alerts, labels=[0, 1]).ravel()
    else:
        tn = fp = fn = tp = 0
    n, positives, flagged = len(y), int(tp + fn), int(tp + fp)
    return {
        'transactions': n, 'fraud_count': positives,
        'fraud_prevalence': positives / n if n else np.nan,
        'precision': tp / flagged if flagged else np.nan,
        'recall': tp / positives if positives else np.nan,
        'F1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan,
        'AP': average_precision_score(y, scores) if len(np.unique(y)) == 2 else np.nan,
        'threshold': threshold, 'alerts': flagged, 'alert_rate': flagged / n if n else np.nan,
        'TP': int(tp), 'FP': int(fp), 'FN': int(fn), 'TN': int(tn),
    }


def choose_threshold(target, scores):
    """Maximum F1 on a development threshold window; ties favour precision, then threshold."""
    if pd.Series(target).nunique() != 2:
        raise ValueError('Threshold selection requires both classes.')
    precision, recall, thresholds = precision_recall_curve(target, scores)
    f1 = np.divide(2 * precision[:-1] * recall[:-1], precision[:-1] + recall[:-1],
                   out=np.zeros_like(thresholds), where=(precision[:-1] + recall[:-1]) > 0)
    table = pd.DataFrame({'threshold': thresholds, 'precision': precision[:-1],
                          'recall': recall[:-1], 'F1': f1})
    best = table.sort_values(['F1', 'precision', 'threshold'], ascending=False, kind='stable').iloc[0]
    return float(best.threshold), table


def time_windows(data, boundaries):
    """Disjoint consecutive windows (lower, upper], in chronological order."""
    if len(boundaries) < 2 or any(b >= c for b, c in zip(boundaries, boundaries[1:])):
        raise ValueError('Boundaries must increase strictly.')
    windows = [data.loc[data.step.gt(a) & data.step.le(b)] for a, b in zip(boundaries, boundaries[1:])]
    if any(window.empty for window in windows):
        raise ValueError('Empty time window.')
    for left, right in zip(windows, windows[1:]):
        assert left.step.max() < right.step.min()
        assert left.index.intersection(right.index).empty
    return windows


def save_table(table, name, index=False):
    table.to_csv(TABLES / f'{name}.csv', index=index)


def save_figure(fig, name):
    fig.savefig(FIGURES / f'{name}.png', dpi=160, bbox_inches='tight', facecolor='white')


def period_summary(frame, threshold, step_width=24):
    """Include zero-volume periods between the first and last step."""
    work = frame.assign(period=((frame.step - 1) // step_width).astype(int))
    rows = []
    for period in range(int(work.period.min()), int(work.period.max()) + 1):
        group = work.loc[work.period.eq(period)]
        start = max(period * step_width + 1, int(frame.step.min()))
        end = min((period + 1) * step_width, int(frame.step.max()))
        rows.append({'period': period, 'first_step': start, 'last_step': end,
                     'steps_covered': end - start + 1,
                     **metrics(group.isFraud, group.score, threshold),
                     'median_amount': group.amount.median(),
                     'median_opening_balance': group.oldbalanceOrg.median(),
                     'median_score': group.score.median(), 'p99_score': group.score.quantile(0.99),
                     'transfer_share': group.type.eq('TRANSFER').mean()})
    return pd.DataFrame(rows)


def distribution_bins(reference, quantiles=10):
    """Development quantile bins with open tails; ties reduce the number of bins."""
    values = np.asarray(reference, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Finite non-empty reference distribution required.')
    internal = np.unique(np.quantile(values, np.linspace(0, 1, quantiles + 1)[1:-1]))
    return np.r_[-np.inf, internal, np.inf]


def population_stability(reference, current, bins):
    """Descriptive PSI with 0.5 pseudo-count smoothing; not a significance test."""
    if len(reference) == 0 or len(current) == 0:
        return np.nan
    ref = np.histogram(reference, bins=bins)[0].astype(float) + 0.5
    cur = np.histogram(current, bins=bins)[0].astype(float) + 0.5
    ref /= ref.sum()
    cur /= cur.sum()
    return float(np.sum((cur - ref) * np.log(cur / ref)))
