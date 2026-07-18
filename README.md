# orthoshift

"Treated users did better, so the treatment worked" is how every bad growth experiment gets greenlit. orthoshift builds a population where that logic is provably wrong, then measures exactly how wrong.

![CI](https://github.com/ahmeddoghri/orthoshift/actions/workflows/ci.yml/badge.svg)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![deps](https://img.shields.io/badge/runtime%20deps-none-success)
![license](https://img.shields.io/badge/license-MIT-black)

Confounding wins by looking exactly like a win. orthoshift generates a
synthetic population where treatment depends on covariates, outcomes depend
on the same covariates, and the true average treatment effect is known ahead
of time, which is the only way to actually grade an estimator instead of just
trusting it. The benchmark compares a naive difference in means, adjusted
OLS, and an orthogonal double machine learning estimator, and none of the
linear algebra is hidden behind a scikit-learn import.

## Run it

```bash
git clone https://github.com/ahmeddoghri/orthoshift
cd orthoshift
pip install -e ".[dev]"
python -m orthoshift.benchmark
```

## Verified benchmark

Generated locally with `python -m orthoshift.benchmark`:

```text
true_ate          1.986
estimator          estimate  abs_error
naive_difference     3.798     1.812
adjusted_ols         1.872     0.114
orthogonal_dml       1.864     0.122
```

The true effect is 1.986. Naive difference in means reports 3.798, which is
confounding lying to your face with a straight expression. Orthogonal DML
reports 1.864, missing by 0.122 instead of 1.812. That gap is the entire
reason the "just compare the means" move keeps burning product teams.

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
