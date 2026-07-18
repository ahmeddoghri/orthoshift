from orthoshift import adjusted_ols, dml_ate, make_dataset, naive_difference
from orthoshift.estimators import true_ate


def test_dataset_has_known_treatment_effect() -> None:
    rows = make_dataset(n=50, seed=1)
    assert 1.5 < true_ate(rows) < 2.3


def test_dml_is_closer_than_naive_on_shifted_fixture() -> None:
    rows = make_dataset(seed=13, shift=0.7)
    truth = true_ate(rows)
    assert abs(dml_ate(rows) - truth) < abs(naive_difference(rows) - truth)


def test_adjusted_ols_returns_reasonable_number() -> None:
    estimate = adjusted_ols(make_dataset(seed=2, shift=0.2))
    assert 1.0 < estimate < 2.8
