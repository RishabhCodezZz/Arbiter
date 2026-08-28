"""
Client history store — the online (real-time) analog of the batch expanding-window
aggregates built in notebooks/02_causal_features.py.

WHY THIS EXISTS: the notebooks computed causal features for 590k rows at once, vectorized,
because the whole dataset was already on disk. A live gateway doesn't have that — it sees
one transaction at a time and has to know "what has THIS client done before" from
somewhere. In production that's a feature store (Redis, a client-profile table). This is
the same idea, minimal: a JSON-backed dict, updated after every decision.

CORRECTNESS RULE, inherited directly from the causal-features work: a transaction may
only see PRIOR history. `get()` is always called before scoring; `update()` is always
called after — never the other way round, or a transaction would see itself.
"""
import json
import math
import os
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class ClientStats:
    n: int = 0                    # count of prior transactions
    amt_sum: float = 0.0
    amt_sumsq: float = 0.0
    last_txn_dt: Optional[int] = None
    devices_seen: list = field(default_factory=list)
    emails_seen: list = field(default_factory=list)

    def mean(self) -> Optional[float]:
        return self.amt_sum / self.n if self.n > 0 else None

    def std(self) -> Optional[float]:
        """POPULATION standard deviation (ddof=0) — deliberate, do NOT change to match
        CoarseStats.std(). This feeds uid_amt_std_prior, which the notebooks build as
        `sqrt(sumsq/n - mean**2)` (ddof=0), so this matches training. The ambiguity feature
        is the opposite case (built ddof=1 in training) — see CoarseStats.std()."""
        if self.n == 0:
            return None
        var = (self.amt_sumsq / self.n) - self.mean() ** 2
        return math.sqrt(max(var, 0.0))


@dataclass
class CoarseStats:
    """Tracks D15n values seen so far within a coarse (card1+addr1+D1n) bucket — feeds
    uid_ambiguity_std_prior. Separate from ClientStats because it's keyed on the COARSER
    identity grouping on purpose (see notebook 02's markdown for why)."""
    n: int = 0
    d15n_sum: float = 0.0
    d15n_sumsq: float = 0.0

    def std(self) -> Optional[float]:
        """SAMPLE standard deviation (ddof=1) — must match training exactly. The notebooks
        build this feature with `s.expanding().std()`, and pandas' default is ddof=1, so
        this uses ddof=1 too. An earlier version divided by n (population/ddof=0), which
        produced a different value for the same history (e.g. [0, 2] -> 1.0 here vs 1.4142
        in training) — a real train/serve skew on a feature the shipped ensemble consumes,
        caught in review. Returns None for n<2, matching `expanding().std()` (NaN for a
        single observation). NOTE: `ClientStats.std()` below deliberately stays ddof=0
        because *its* feature (uid_amt_std_prior) is built ddof=0 in the notebooks too."""
        if self.n < 2:
            return None
        # numerically stabler than (sumsq/n - mean**2): sum of squared deviations / (n-1)
        sum_sq_dev = self.d15n_sumsq - (self.d15n_sum ** 2) / self.n
        var = max(sum_sq_dev, 0.0) / (self.n - 1)
        return math.sqrt(var)


class ClientHistoryStore:
    """File-backed, loaded fully into memory (this dataset's client count is small enough
    that this is fine for a hackathon demo — a production version would swap this class
    for a real feature-store client without touching features.py or engine.py, since they
    only depend on this class's get/update interface, not its storage mechanism)."""

    def __init__(self, path: str):
        self.path = path
        self._fine: dict[str, ClientStats] = {}
        self._coarse: dict[str, CoarseStats] = {}
        self._load()

    def _load(self):
        if not os.path.exists(self.path):
            return
        with open(self.path) as f:
            raw = json.load(f)
        for uid, d in raw.get("fine", {}).items():
            self._fine[uid] = ClientStats(**d)
        for cuid, d in raw.get("coarse", {}).items():
            self._coarse[cuid] = CoarseStats(**d)

    def save(self):
        raw = {
            "fine": {uid: asdict(s) for uid, s in self._fine.items()},
            "coarse": {cuid: asdict(s) for cuid, s in self._coarse.items()},
        }
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(raw, f)

    def get_fine(self, uid: str) -> ClientStats:
        return self._fine.get(uid, ClientStats())

    def get_coarse(self, coarse_uid: str) -> CoarseStats:
        return self._coarse.get(coarse_uid, CoarseStats())

    def update(self, uid: str, coarse_uid: str, amount: float, txn_dt: int,
               device: Optional[str], email: Optional[str], d15n: Optional[float]):
        """Called AFTER a decision is made, never before — enforces the causal ordering."""
        s = self._fine.get(uid, ClientStats())
        s.n += 1
        s.amt_sum += amount
        s.amt_sumsq += amount ** 2
        s.last_txn_dt = txn_dt
        if device is not None and device not in s.devices_seen:
            s.devices_seen.append(device)
        if email is not None and email not in s.emails_seen:
            s.emails_seen.append(email)
        self._fine[uid] = s

        if d15n is not None:
            c = self._coarse.get(coarse_uid, CoarseStats())
            c.n += 1
            c.d15n_sum += d15n
            c.d15n_sumsq += d15n ** 2
            self._coarse[coarse_uid] = c

        self.save()
