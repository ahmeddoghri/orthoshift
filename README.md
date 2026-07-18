# orthoshift

An inspectable double machine learning benchmark for treatment effects under
covariate shift. The benchmark creates a confounded synthetic population with a
known average treatment effect, then compares naive difference in means,
adjusted OLS, and an orthogonal residual estimator.

![CI](https://github.com/ahmeddoghri/orthoshift/actions/workflows/ci.yml/badge.svg)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![deps](https://img.shields.io/badge/runtime%20deps-none-success)
![license](https://img.shields.io/badge/license-MIT-black)

## Run it

```bash
git clone https://github.com/ahmeddoghri/orthoshift
cd orthoshift
pip install -e ".[dev]"
python -m orthoshift.benchmark
```

## Verified benchmark

These numbers were generated locally with `python -m orthoshift.benchmark`:

```text
true_ate          1.986
estimator          estimate  abs_error
naive_difference     3.798     1.812
adjusted_ols         1.872     0.114
orthogonal_dml       1.864     0.122
```

## Research trail

- Estimating causal effects with double machine learning, 2024: https://arxiv.org/abs/2403.14385
- Applied causal inference powered by ML and AI, 2024: https://arxiv.org/abs/2403.02467
- Doubly robust learning tutorial, 2024: https://arxiv.org/html/2406.00853v1
- DoubleML Deep for multimodal causal effects, 2024: https://arxiv.org/html/2402.01785v1

## Tests

```bash
pytest -q
ruff check .
```

MIT © Ahmed Doghri
