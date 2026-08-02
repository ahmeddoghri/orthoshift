"""Does orthogonal DML actually beat adjusted OLS, or just naive difference?

The published benchmark runs on a single seed (13). At that seed,
``orthogonal_dml`` narrowly beats ``adjusted_ols`` (abs error 0.122 vs
0.114 -- actually OLS is *closer* even in the published numbers, look
again). Both estimators crush the naive unadjusted difference in means,
which is real and dramatic. Whether DML's orthogonalization specifically
buys anything over plain adjusted linear regression is a different, much
narrower question that one seed cannot answer.

It matters here for a structural reason: ``estimators.adjusted_ols`` and
``estimators.dml_ate`` both use the exact same ``features()`` function
(``[1, x1, x2, x1^2, x2^2, x1*x2]``) for their adjustment/nuisance models,
and that functional form is an exact match for ``data.make_dataset``'s
true outcome model (``base = 0.8 + 1.4*x1 - 1.1*x2 + 0.35*x1*x2 +
0.25*x1^2``). DML's real selling point, robustness to nuisance-model
misspecification via Neyman orthogonality, isn't exercised at all when
the nuisance model is exactly correctly specified for both estimators.
"""
from __future__ import annotations

# Seeds used while characterizing the finding.
ADVERSARIAL_SEEDS = list(range(60))

# A disjoint set of seeds, evaluated exactly once.
HOLDOUT_SEEDS = list(range(1000, 1030))
