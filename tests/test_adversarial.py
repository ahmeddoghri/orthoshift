"""Tests for the "does DML reliably beat adjusted OLS" finding."""

from __future__ import annotations

from orthoshift.adversarial import ADVERSARIAL_SEEDS, HOLDOUT_SEEDS
from orthoshift.data import make_dataset
from orthoshift.estimators import adjusted_ols, dml_ate, naive_difference, true_ate
from orthoshift.eval_v2 import _summarize, build_report

# --- the finding: DML and OLS use identical, correctly-specified features --

def test_estimators_share_the_same_feature_function():
    """adjusted_ols and dml_ate both call features() for their
    adjustment/nuisance model, so any advantage from a richer nuisance
    model is unavailable to either -- the ground everything else follows
    from."""
    import inspect

    from orthoshift import estimators

    source = inspect.getsource(estimators)
    assert source.count("features(row.x1, row.x2)") + source.count("features(row.x1,row.x2)") >= 2


def test_both_estimators_crush_naive_difference():
    """The real, dramatic, uncontested result: confounding removal via
    either adjustment method versus doing nothing at all."""
    rows = make_dataset(seed=13, shift=0.7)
    truth = true_ate(rows)
    naive_err = abs(naive_difference(rows) - truth)
    ols_err = abs(adjusted_ols(rows) - truth)
    dml_err = abs(dml_ate(rows) - truth)
    assert naive_err > ols_err * 10
    assert naive_err > dml_err * 10


def test_dml_does_not_reliably_beat_adjusted_ols():
    """Across many seeds, adjusted OLS wins the head-to-head against
    orthogonal DML more often than DML wins, contrary to what the
    published single-seed benchmark implies."""
    result = _summarize(ADVERSARIAL_SEEDS)
    assert result["ols_wins"] > result["dml_wins"]
    assert result["mean_abs_error_ols"] <= result["mean_abs_error_dml"]


def test_the_published_seed_already_shows_ols_ahead_despite_the_framing():
    """The published benchmark table prints orthogonal_dml (0.122) below
    adjusted_ols (0.114) as if DML is the improvement, but 0.114 < 0.122:
    OLS is already closer to the truth at the exact seed the README uses.
    The aggregate sweep confirms this isn't a fluke of that one number."""
    rows = make_dataset(seed=13, shift=0.7)
    truth = true_ate(rows)
    ols_err = abs(adjusted_ols(rows) - truth)
    dml_err = abs(dml_ate(rows) - truth)
    assert ols_err < dml_err

    aggregate = _summarize(ADVERSARIAL_SEEDS)
    assert aggregate["ols_wins"] > aggregate["dml_wins"]


# --- held out, evaluated once ------------------------------------------------

def test_holdout_seeds_are_disjoint_from_tuning_seeds():
    assert not (set(ADVERSARIAL_SEEDS) & set(HOLDOUT_SEEDS))


def test_holdout_confirms_ols_is_at_least_as_good_as_dml():
    result = _summarize(HOLDOUT_SEEDS)
    assert result["mean_abs_error_ols"] <= result["mean_abs_error_dml"]


# --- the original benchmark is unaffected -----------------------------------

def test_original_estimators_module_untouched():
    import orthoshift.estimators as estimators_module

    assert not hasattr(estimators_module, "features_linear")


def test_original_benchmark_still_reproduces():
    rows = make_dataset(shift=0.7)
    truth = true_ate(rows)
    naive = naive_difference(rows)
    ols = adjusted_ols(rows)
    dml = dml_ate(rows)
    assert round(truth, 3) == 1.986
    assert round(naive, 3) == 3.798
    assert round(ols, 3) == 1.872
    assert round(dml, 3) == 1.864


# --- the full report ---------------------------------------------------------

def test_report_is_reproducible():
    assert build_report() == build_report()
