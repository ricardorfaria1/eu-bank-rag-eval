import math

import pytest

from eurag.costs import BUDGET_USD, BudgetExceeded, BudgetLedger, confirm, estimate_cost


def test_estimate_run_cost():
    # 150 answers x (6000 in x $4/M + 600 out x $20/M) = 150 x 0.036
    assert math.isclose(estimate_cost(50, 3, "run"), 5.40)


def test_estimate_evaluate_cost():
    # 150 answers x 2 metrics x 4 calls x (3000 x $0.10/M + 400 x $0.50/M)
    assert math.isclose(estimate_cost(50, 3, "evaluate"), 0.60)


def test_estimate_testset_cost_grows_with_chunks():
    assert estimate_cost(80, 1, "testset", n_chunks=2000) > estimate_cost(80, 1, "testset", n_chunks=100)


def test_budget_refuses_over_ceiling(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.json")
    ledger.record("run", 20.0)
    ledger.check(4.0)
    with pytest.raises(BudgetExceeded):
        ledger.check(5.5)
    assert BUDGET_USD == 25.0
    assert BudgetLedger(tmp_path / "budget.json").spent == 20.0


def test_confirm_yes_flag_skips_prompt(capsys):
    assert confirm(1.234, yes=True) is True
    assert "Estimated cost: $1.23" in capsys.readouterr().out


def test_confirm_reads_answer(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "n")
    assert confirm(1.0, yes=False) is False
