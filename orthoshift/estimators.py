from __future__ import annotations

from .data import Sample
from .linear import features, predict, ridge_fit


def naive_difference(rows: list[Sample]) -> float:
    treated = [row.outcome for row in rows if row.treatment == 1.0]
    control = [row.outcome for row in rows if row.treatment == 0.0]
    return sum(treated) / len(treated) - sum(control) / len(control)


def adjusted_ols(rows: list[Sample]) -> float:
    x = [[row.treatment] + features(row.x1, row.x2) for row in rows]
    y = [row.outcome for row in rows]
    return ridge_fit(x, y, lam=1e-3)[0]


def _clip_prop(value: float) -> float:
    return min(0.97, max(0.03, value))


def dml_ate(rows: list[Sample], folds: int = 3) -> float:
    y_res = []
    t_res = []
    for fold in range(folds):
        train = [row for idx, row in enumerate(rows) if idx % folds != fold]
        test = [row for idx, row in enumerate(rows) if idx % folds == fold]
        x_train = [features(row.x1, row.x2) for row in train]
        m_y = ridge_fit(x_train, [row.outcome for row in train], lam=1e-2)
        m_t = ridge_fit(x_train, [row.treatment for row in train], lam=1e-2)
        for row in test:
            phi = features(row.x1, row.x2)
            y_res.append(row.outcome - predict(phi, m_y))
            t_res.append(row.treatment - _clip_prop(predict(phi, m_t)))
    num = sum(t * y for t, y in zip(t_res, y_res))
    den = sum(t * t for t in t_res)
    return num / den


def true_ate(rows: list[Sample]) -> float:
    return sum(row.true_tau for row in rows) / len(rows)
