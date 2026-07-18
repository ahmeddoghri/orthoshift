from __future__ import annotations


def features(x1: float, x2: float) -> list[float]:
    return [1.0, x1, x2, x1 * x1, x2 * x2, x1 * x2]


def solve(a: list[list[float]], b: list[float]) -> list[float]:
    n = len(b)
    aug = [row[:] + [rhs] for row, rhs in zip(a, b)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(aug[row][col]))
        aug[col], aug[pivot] = aug[pivot], aug[col]
        div = aug[col][col] or 1e-12
        aug[col] = [value / div for value in aug[col]]
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            aug[row] = [value - factor * ref for value, ref in zip(aug[row], aug[col])]
    return [row[-1] for row in aug]


def ridge_fit(x: list[list[float]], y: list[float], lam: float = 1e-3) -> list[float]:
    p = len(x[0])
    xtx = [[0.0 for _ in range(p)] for _ in range(p)]
    xty = [0.0 for _ in range(p)]
    for row, target in zip(x, y):
        for i in range(p):
            xty[i] += row[i] * target
            for j in range(p):
                xtx[i][j] += row[i] * row[j]
    for i in range(p):
        xtx[i][i] += lam
    return solve(xtx, xty)


def predict(row: list[float], weights: list[float]) -> float:
    return sum(a * b for a, b in zip(row, weights))
