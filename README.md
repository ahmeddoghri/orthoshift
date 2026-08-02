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

**Update:** look at that table again. `adjusted_ols` misses by 0.114,
closer than orthogonal DML's 0.122, at the exact seed printed above.
Across 60 seeds it's not close: plain adjusted OLS wins the head-to-head
against DML more often than DML wins. Both estimators demolish the naive
comparison; the specific "orthogonal DML beats simple adjustment" story
doesn't hold up. `python -m orthoshift.eval_v2` runs the multi-seed
comparison. Details below.

## Does the "orthogonal" part actually buy anything over plain adjustment?

`adjusted_ols` and `dml_ate` both use the exact same `features()` function
(`[1, x1, x2, x1^2, x2^2, x1*x2]`) for their adjustment/nuisance model,
and that functional form is an exact match for `make_dataset`'s true
outcome model. DML's whole selling point, robustness to nuisance-model
*misspecification* via Neyman orthogonality, is never exercised when the
nuisance model is exactly correctly specified for both estimators.

```bash
python -m orthoshift.eval_v2
```
```
corpus       n   naive err   ols err   dml err  ols wins  dml wins
tuning      60      2.0246    0.0476    0.0505        42        18
holdout     30      1.9913    0.0512    0.0534        16        14
```

Both estimators crush naive by roughly 40x, real, dramatic confounding
removal. But DML does not reliably beat adjusted OLS: OLS wins the
head-to-head more often, in both the 60-seed tuning sweep and a disjoint
30-seed holdout evaluated exactly once. It's even true at the exact seed
the published table uses, look closely: `adjusted_ols` misses by 0.114,
closer than orthogonal DML's 0.122.

None of this means DML is broken, `dml_ate` is a correct implementation
of the cross-fitted orthogonal estimator. It means this particular
benchmark, with a low-dimensional confounder and a correctly-specified
linear-plus-interaction adjustment model, is exactly the regime where
DML's extra machinery has nothing to protect you from. `data.py` and
`estimators.py` are untouched, and the published table above still
reproduces exactly; `eval_v2.py` is the honest multi-seed companion that
shows what the headline actually demonstrates (confounding removal, not
DML-over-OLS) versus what the framing implies.

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
