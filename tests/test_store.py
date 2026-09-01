"""
Golden-vector parity between the ONLINE running-sum statistics in src/store.py and the
BATCH pandas computations the training notebooks use. No trained model needed.

WHY THIS FILE EXISTS: a code review caught a real train/serve skew — CoarseStats.std()
divided by n (population / ddof=0) while the notebooks build uid_ambiguity_std_prior with
`s.expanding().std()`, whose pandas default is ddof=1. Same history, different number
(e.g. [0, 2] -> 1.0 online vs 1.4142 in training), on a feature the shipped ensemble
consumes. These tests pin both std features to their notebook definitions so it can't
silently drift again.
"""
import json
import math
import os

import pandas as pd
import pytest

from src.store import ClientHistoryStore, ClientStats, CoarseStats


def _coarse_after(values):
    c = CoarseStats()
    for v in values:
        c.n += 1
        c.d15n_sum += v
        c.d15n_sumsq += v ** 2
    return c


def _fine_after(amounts):
    s = ClientStats()
    for a in amounts:
        s.n += 1
        s.amt_sum += a
        s.amt_sumsq += a ** 2
    return s


# --- CoarseStats.std  ==  pandas expanding().std()  (ddof=1, sample) -----------------

@pytest.mark.parametrize("history", [
    [0.0, 2.0], [1.0, 5.0, 9.0], [3.3, 3.3, 3.3, 7.1], [-2.0, 4.0, 4.0, 10.0, 1.5],
])
def test_coarse_std_matches_pandas_expanding_std_ddof1(history):
    """uid_ambiguity_std_prior is built as `s.expanding().std()` in notebooks 02 and 06 —
    pandas default ddof=1. CoarseStats.std() must produce the same value for the same
    prior history."""
    online = _coarse_after(history).std()
    batch = pd.Series(history).expanding().std().iloc[-1]  # std over the full prior history
    assert online == pytest.approx(batch)


def test_coarse_std_is_none_below_two_observations():
    """expanding().std() is NaN for a single point; the online store returns None for the
    same case so features.build() maps it to a missing value, matching training."""
    assert _coarse_after([]).std() is None
    assert _coarse_after([42.0]).std() is None
    assert _coarse_after([1.0, 2.0]).std() is not None


def test_coarse_std_specific_regression_value():
    """The exact case from the review: [0, 2] must be 1.4142 (ddof=1), not 1.0 (ddof=0)."""
    assert _coarse_after([0.0, 2.0]).std() == pytest.approx(math.sqrt(2.0))
    assert _coarse_after([0.0, 2.0]).std() != pytest.approx(1.0)


# --- ClientStats.std  ==  notebook's sqrt(sumsq/n - mean**2)  (ddof=0, population) ---

@pytest.mark.parametrize("amounts", [
    [10.0, 30.0], [5.0, 5.0, 20.0], [100.0, 250.0, 90.0, 400.0],
])
def test_fine_std_matches_notebook_population_formula_ddof0(amounts):
    """uid_amt_std_prior is built in notebooks 02/06 as
    `sqrt((uid_amt_sumsq_prior / n_prior) - uid_amt_mean_prior**2)` — population / ddof=0.
    ClientStats.std() deliberately matches THAT (not expanding().std())."""
    s = _fine_after(amounts)
    n = len(amounts)
    mean = sum(amounts) / n
    expected = math.sqrt(max(sum(a * a for a in amounts) / n - mean ** 2, 0.0))
    assert s.std() == pytest.approx(expected)
    # and it must NOT equal the ddof=1 value (guards a "fix" that breaks parity the other way)
    assert s.std() != pytest.approx(pd.Series(amounts).std())


def test_fine_std_is_none_with_no_history():
    assert _fine_after([]).std() is None


# --- atomic save: no truncated file, no leftover temp -------------------------------

def test_save_is_atomic_and_leaves_no_temp_file(tmp_path):
    """save() writes via a temp file + os.replace so a crash or a second writer can't
    leave a truncated, unparseable history file (the 'lose history' half of the disclosed
    concurrency gap). After a normal update the real path is valid JSON and the tmp is gone."""
    path = tmp_path / "client_history.json"
    store = ClientHistoryStore(str(path))
    store.update(uid="c1", coarse_uid="b1", amount=100.0, txn_dt=1000,
                 device="d", email="e", d15n=5.0)
    store.update(uid="c1", coarse_uid="b1", amount=200.0, txn_dt=2000,
                 device="d", email="e", d15n=7.0)

    assert path.exists()
    data = json.loads(path.read_text())          # parses cleanly = not truncated
    assert data["fine"]["c1"]["n"] == 2
    leftovers = [p for p in os.listdir(tmp_path) if ".tmp" in p]
    assert not leftovers, f"atomic save left a temp file behind: {leftovers}"

    # a fresh store reloads the same state
    reloaded = ClientHistoryStore(str(path))
    assert reloaded.get_fine("c1").n == 2
