from __future__ import annotations

from .data import make_dataset
from .estimators import adjusted_ols, dml_ate, naive_difference, true_ate


def main() -> None:
    rows = make_dataset(shift=0.7)
    truth = true_ate(rows)
    estimates = [
        ("naive_difference", naive_difference(rows)),
        ("adjusted_ols", adjusted_ols(rows)),
        ("orthogonal_dml", dml_ate(rows)),
    ]
    print("orthoshift benchmark: treatment effect under covariate shift")
    print(f"true_ate          {truth:.3f}")
    print("estimator          estimate  abs_error")
    for label, estimate in estimates:
        print(f"{label:17s} {estimate:8.3f}  {abs(estimate - truth):8.3f}")


if __name__ == "__main__":
    main()
