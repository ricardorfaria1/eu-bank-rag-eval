import math

from eurag.stats import bootstrap_ci, paired_diff_ci


def test_bootstrap_ci_contains_mean_and_is_deterministic():
    values = [0.0, 1.0, 1.0, 0.5, 0.75, 0.25, 1.0, 0.0]
    mean, low, high = bootstrap_ci(values)
    assert math.isclose(mean, sum(values) / len(values))
    assert low <= mean <= high
    assert bootstrap_ci(values) == (mean, low, high)


def test_paired_diff_zero_when_equal():
    assert paired_diff_ci([0.2, 0.4, 0.9], [0.2, 0.4, 0.9]) == (0.0, 0.0, 0.0)


def test_paired_diff_sign_and_ignores_missing():
    a = [1.0, 1.0, None, 1.0]
    b = [0.0, 0.0, 0.0, 0.0]
    mean, low, high = paired_diff_ci(a, b)
    assert mean == 1.0 and low == 1.0 and high == 1.0


def test_bootstrap_ignores_none():
    assert bootstrap_ci([1.0, None, 0.0])[0] == 0.5


def test_bootstrap_ignores_nan():
    assert bootstrap_ci([1.0, float("nan"), 0.0])[0] == 0.5
