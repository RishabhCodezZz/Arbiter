"""
LLM-as-classifier benchmark. Run from the repo root:

    python scripts/llm_benchmark.py --test-connection   # sanity check first, 1 call
    python scripts/llm_benchmark.py                     # full run

THE QUESTION THIS ANSWERS: can a general-purpose LLM score fraud probability well enough,
fast enough, cheaply enough to replace XGBoost? Per CLAUDE.md sec 6, the expected answer is
no on all three axes — this script is what turns that expectation into a measured number,
which is the actual evidence for the rubric's "where you chose not to use one" line. Per
PLAN.md's G9, BOTH outcomes are usable: if the LLM is surprisingly competitive, report that
honestly and argue deployment on latency/cost grounds instead.

WHAT IT DOES, for each transaction in the sample:
  1. Scores it with the real trained XGBoost model (identical code path to src/engine.py —
     features.build() + FraudModel — so this is a fair, apples-to-apples comparison, not a
     re-implementation that could quietly diverge).
  2. Serializes it to a short natural-language description and asks an LLM, directly, for a
     fraud probability.
  3. Times every call on both sides and computes PR-AUC (average precision) for both.

FAIRNESS NOTE ON THE PROMPT: only genuinely nameable columns go into the LLM's prompt
(amount, product code, card network/type, address/email/device info, the C/D/M feature
families, plus hour-of-day and day-of-week derived from TransactionDT the same way
src/features.py derives them for the model). The ~339 V-columns and most id_ device-
fingerprint columns are excluded — they're Vesta's own opaque engineered numbers with no
public semantic meaning, which is *why this project chose IEEE-CIS in the first place* (see
CLAUDE.md sec 5): a human or an LLM can't reason about "V204: 0.113" any better than noise.
XGBoost still sees those columns; the LLM doesn't need to see gibberish to get a fair shot at
the columns that actually mean something.

CALLS OLLAMA CLOUD DIRECTLY OVER HTTPS by default — no local Ollama install needed. Create a
key at https://ollama.com/settings/keys, then either:
    setx OLLAMA_API_KEY "your-key-here"     (Windows, persists across terminals)
  or just set it for one session:
    $env:OLLAMA_API_KEY = "your-key-here"   (PowerShell)
Do NOT paste the raw key into chat — set it as an environment variable yourself and just
confirm it's set. With OLLAMA_API_KEY present, this script talks to https://ollama.com/api;
without it, it falls back to a local http://localhost:11434. Run --test-connection first,
always, before a full run.

REVISED: Ollama Cloud's free tier proved unreliable under sustained sequential calls
in practice — repeated "too many concurrent requests" even from one process with real
exponential backoff (see journal). Use `--kaggle-results` instead: run the LLM side entirely
on Kaggle's own GPU (see notebooks/04_cost_model.py's LLM benchmark addendum, part 2 —
`gpt-oss:20b`, no external API, no rate limit), download the resulting JSON, and pass it here
to combine with the local XGBoost side. The cloud path above is left in place and still
works for a quick `--test-connection` sanity check, just not recommended for the full run
anymore.

Requires artifacts/{model.json, calibrator.json, feature_manifest.json} (engine-build addendum) and
ideally artifacts/llm_benchmark_sample.json (LLM benchmark addendum, 500 rows — see
notebooks/04_cost_model.py). Falls back to artifacts/sample_transactions.json (25 rows, from the engine-build addendum)
with a loud warning if the 500-row file isn't there yet — enough to prove the plumbing works,
nowhere near enough for a PR-AUC anyone should trust.
"""
import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src import features as features_mod
from src.model import FraudModel, ModelUnavailableError
from src.store import ClientHistoryStore

ARTIFACTS_DIR = "artifacts"
PREFERRED_SAMPLE = os.path.join(ARTIFACTS_DIR, "llm_benchmark_sample.json")
FALLBACK_SAMPLE = os.path.join(ARTIFACTS_DIR, "sample_transactions.json")
BENCHMARK_STORE_PATH = "data/llm_benchmark_history.json"  # throwaway, never data/client_history.json
RESULTS_PATH = os.path.join("docs", "llm_benchmark_results.json")

OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY")
# Cloud API (https://ollama.com/api) when a key is set — no local install needed. Falls
# back to a local `ollama serve` instance (http://localhost:11434) otherwise, for anyone
# who installs Ollama locally instead of using the cloud key.
OLLAMA_HOST = os.environ.get(
    "OLLAMA_HOST", "https://ollama.com" if OLLAMA_API_KEY else "http://localhost:11434"
)
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "nemotron-3-ultra")
OLLAMA_TIMEOUT_S = 60

SECONDS_PER_DAY = 24 * 60 * 60

# Columns that are genuinely nameable/interpretable — see module docstring's fairness note.
# Excludes: isFraud (the label), TransactionID/TransactionDT (replaced with derived
# hour/dow below, same as the model sees), and every V*/id_* opaque engineered column.
NAMEABLE_PREFIXES = ("card", "addr", "dist", "C", "D", "M")
NAMEABLE_EXACT = {
    "TransactionAmt", "ProductCD", "P_emaildomain", "R_emaildomain",
    "DeviceType", "DeviceInfo",
}


def _is_nameable(col: str) -> bool:
    if col in NAMEABLE_EXACT:
        return True
    for p in NAMEABLE_PREFIXES:
        if col.startswith(p) and (len(col) == len(p) or col[len(p):].isdigit()):
            return True
    return False


def load_sample() -> tuple[list, bool]:
    """Returns (transactions, is_full_size). Prefers the 500-row LLM-benchmark export; falls back
    to the 25-row engine-build one with a loud warning if that hasn't been downloaded yet."""
    if os.path.exists(PREFERRED_SAMPLE):
        with open(PREFERRED_SAMPLE) as f:
            return json.load(f), True
    if os.path.exists(FALLBACK_SAMPLE):
        with open(FALLBACK_SAMPLE) as f:
            data = json.load(f)
        print(f"!! WARNING: {PREFERRED_SAMPLE} not found — falling back to the 25-row "
              f"{FALLBACK_SAMPLE}. This is a PLUMBING DRY RUN ONLY. At the dataset's ~3.5% "
              f"fraud base rate, 25 rows carries under 1 expected fraud case — any PR-AUC "
              f"computed from this run is not a real result. Run the LLM benchmark addendum at the "
              f"end of notebooks/04_cost_model.py on Kaggle and download "
              f"artifacts/llm_benchmark_sample.json before trusting any number this prints.")
        return data, False
    raise FileNotFoundError(
        f"Neither {PREFERRED_SAMPLE} nor {FALLBACK_SAMPLE} exists. Run notebooks/04's "
        f"export addendum on Kaggle first and download the artifacts."
    )


def serialize_transaction(txn: dict) -> str:
    """Plain-English feature dump of the nameable columns only — see module docstring."""
    dt = txn.get("TransactionDT")
    lines = []
    if dt is not None:
        hour = (dt // 3600) % 24
        dow = (dt // SECONDS_PER_DAY) % 7
        lines.append(f"hour_of_day: {hour}")
        lines.append(f"day_of_week: {dow}")
    for col in sorted(txn.keys()):
        if col in ("TransactionID", "TransactionDT", "isFraud"):
            continue
        if not _is_nameable(col):
            continue
        val = txn.get(col)
        if val is None:
            continue
        lines.append(f"{col}: {val}")
    return "\n".join(lines)


def build_prompt(txn_text: str) -> str:
    return (
        "You are a payment fraud analyst. Given the following e-commerce transaction "
        "facts, estimate the probability this transaction is fraudulent.\n\n"
        f"{txn_text}\n\n"
        'Respond with ONLY a JSON object of the exact form {"fraud_probability": 0.NN}, '
        "a number between 0 and 1. No other text."
    )


class LLMResponseError(Exception):
    pass


class RateLimitError(LLMResponseError):
    """Raised specifically for HTTP 429, so the caller can back off instead of just
    recording a failure and immediately hammering the next call — see journal: a run with
    no delay between calls turned one rate-limit hit into a ~90-call streak of 429s, because
    nothing ever paused to let the window reset."""
    pass


def call_llm(prompt: str) -> tuple[float, float]:
    """Returns (fraud_probability, latency_ms). Raises LLMResponseError on any failure —
    connection, auth, timeout, or a response that doesn't validate — mirroring the same
    validate-don't-trust discipline as src/narrative.py's LLM response handling.

    Uses /api/chat (messages array), not /api/generate — this is the endpoint confirmed to
    work against Ollama Cloud's hosted API with Bearer-token auth; /api/generate's cloud
    support wasn't confirmed, so there was no reason to risk it. Works identically against
    a local `ollama serve` instance too (same endpoint shape, just no auth header)."""
    body = json.dumps({
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "format": "json",
    }).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if OLLAMA_API_KEY:
        headers["Authorization"] = f"Bearer {OLLAMA_API_KEY}"
    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/chat", data=body, headers=headers, method="POST",
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT_S) as resp:
            outer = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:300]
        hint = ""
        if e.code == 401:
            hint = " (check OLLAMA_API_KEY is set and valid)"
        elif e.code == 404:
            hint = f" (check the model name {OLLAMA_MODEL!r} is right)"
        if e.code == 429:
            raise RateLimitError(f"HTTP 429 from {OLLAMA_HOST}/api/chat: {detail}") from e
        raise LLMResponseError(f"HTTP {e.code} from {OLLAMA_HOST}/api/chat{hint}: {detail}") from e
    except urllib.error.URLError as e:
        reachability_hint = (
            "Check OLLAMA_API_KEY is valid." if OLLAMA_API_KEY else
            "Is Ollama installed and running locally (`ollama serve`)? Or set "
            "OLLAMA_API_KEY to use the cloud API instead — no local install needed."
        )
        raise LLMResponseError(f"could not reach {OLLAMA_HOST}: {e}. {reachability_hint}") from e
    except TimeoutError as e:
        raise LLMResponseError(f"timed out after {OLLAMA_TIMEOUT_S}s") from e
    latency_ms = (time.monotonic() - t0) * 1000

    raw_text = outer.get("message", {}).get("content", "")
    try:
        inner = json.loads(raw_text)
        p = float(inner["fraud_probability"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        raise LLMResponseError(f"unparseable response: {raw_text[:200]!r}") from e
    if not (0.0 <= p <= 1.0):
        raise LLMResponseError(f"fraud_probability out of range: {p}")
    return p, latency_ms


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """PR-AUC, computed by hand rather than importing sklearn — requirements.txt
    deliberately excludes sklearn locally (see its comment: Platt is re-implemented as a
    plain sigmoid for the same reason). Matches sklearn.metrics.average_precision_score's
    definition exactly: AP = sum_n (R_n - R_{n-1}) * P_n over scores sorted descending."""
    n_pos = y_true.sum()
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-y_score, kind="mergesort")
    y_true_sorted = y_true[order]
    tp_cum = np.cumsum(y_true_sorted)
    fp_cum = np.cumsum(1 - y_true_sorted)
    precision = tp_cum / (tp_cum + fp_cum)
    recall = tp_cum / n_pos
    recall_prev = np.concatenate(([0.0], recall[:-1]))
    return float(np.sum((recall - recall_prev) * precision))


def score_with_xgboost(transactions: list) -> dict:
    """Returns {TransactionID: calibrated_probability}. Reuses the exact same
    features.build() + FraudModel code path engine.py uses — not a parallel
    reimplementation that could quietly drift from what's actually deployed."""
    model = FraudModel(ARTIFACTS_DIR)
    store = ClientHistoryStore(BENCHMARK_STORE_PATH)
    ordered = sorted(transactions, key=lambda t: t.get("TransactionDT", 0))
    out = {}
    for txn in ordered:
        fv = features_mod.build(txn, store, model.manifest)
        X = model.to_dataframe(fv)
        _, calibrated = model.score_df(X)
        out[txn["TransactionID"]] = calibrated
        if fv.get("_uid") is not None:
            store.update(
                uid=fv["_uid"], coarse_uid=fv["_coarse_uid"], amount=fv["_amount"] or 0.0,
                txn_dt=fv["_dt"], device=fv.get("_device"), email=fv.get("_email"),
                d15n=fv.get("_d15n"),
            )
    return out


def progress_path_for(is_full_size: bool) -> str:
    """Separate checkpoint files for the 25-row dry run vs. the real 500-row sample, so
    switching between them (as this project did today) never mixes cached results from one
    sample into a run against the other."""
    suffix = "full500" if is_full_size else "dryrun25"
    return os.path.join("docs", f"llm_benchmark_progress_{suffix}.jsonl")


def load_progress(path: str) -> dict:
    """{TransactionID: result_dict}. Missing file = empty dict, not an error — just means
    this is the first call of a fresh run."""
    if not os.path.exists(path):
        return {}
    done = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                done[rec["TransactionID"]] = rec
    return done


def append_progress(path: str, record: dict):
    """Appended and flushed after EVERY call, not batched — at ~29s/call (measured, see
    journal) a 500-call run can genuinely outlast the free tier's 3-hour session window, so
    losing zero completed work to an interruption matters more than the tiny I/O cost of
    writing one line at a time."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()


def finalize_and_report(transactions, xgb_probs, llm_probs, llm_latencies_ms, failures,
                         llm_model, is_full_size, llm_source="ollama-cloud",
                         results_path=RESULTS_PATH):
    """Shared by both the Ollama Cloud path and the --kaggle-results path — the comparison
    math and reporting are identical regardless of where the LLM numbers came from."""
    y_true = np.array([t["isFraud"] for t in transactions])
    xgb_scores = np.array([xgb_probs[t["TransactionID"]] for t in transactions])
    xgb_ap = average_precision(y_true, xgb_scores)

    llm_ids = list(llm_probs.keys())
    if llm_ids:
        llm_y_true = np.array([t["isFraud"] for t in transactions if t["TransactionID"] in llm_probs])
        llm_scores = np.array([llm_probs[tid] for tid in llm_ids])
        llm_ap = average_precision(llm_y_true, llm_scores)
    else:
        llm_ap = float("nan")

    summary = {
        "n_transactions": len(transactions),
        "is_full_size_sample": is_full_size,
        "n_fraud": int(y_true.sum()),
        "xgboost_pr_auc": xgb_ap,
        "llm_pr_auc": llm_ap,
        "llm_model": llm_model,
        "llm_source": llm_source,
        "n_llm_failures": len(failures),
        "llm_latency_ms": {
            "median": statistics.median(llm_latencies_ms) if llm_latencies_ms else None,
            "mean": statistics.mean(llm_latencies_ms) if llm_latencies_ms else None,
            "n_calls": len(llm_latencies_ms),
        },
        "failures": failures,
    }

    print("\n" + "=" * 60)
    print(f"XGBoost PR-AUC:  {xgb_ap:.4f}")
    print(f"LLM PR-AUC:      {llm_ap:.4f}  (model={llm_model}, source={llm_source}, "
          f"{len(failures)} failed calls)")
    if llm_latencies_ms:
        print(f"LLM latency:     median {summary['llm_latency_ms']['median']:.0f}ms, "
              f"mean {summary['llm_latency_ms']['mean']:.0f}ms")
    print("XGBoost latency: microseconds per call (not separately timed here — see "
          "notebooks/04's own training-time benchmarks for that number).")
    if not is_full_size:
        print("\n!! This ran on the 25-row DRY RUN sample. Re-run once "
              f"{PREFERRED_SAMPLE} exists before citing any number above.")
    print("=" * 60)

    os.makedirs("docs", exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFull results written to {results_path}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-connection", action="store_true",
                     help="Send one call to the LLM and report success/failure. Do this "
                          "before committing to a full run.")
    ap.add_argument("--limit", type=int, default=None,
                     help="Cap the number of transactions scored (for a quick partial run).")
    ap.add_argument("--reset", action="store_true",
                     help="Discard any checkpointed progress for this sample and start over.")
    ap.add_argument("--kaggle-results", type=str, default=None,
                     help="Path to artifacts/llm_benchmark_kaggle_results.json (from "
                          "notebooks/04_cost_model.py's LLM benchmark addendum, part 2). Skips the "
                          "Ollama Cloud path entirely and combines this with local XGBoost "
                          "scoring instead — recommended since the cloud free tier proved "
                          "unreliable under sustained load (see journal).")
    args = ap.parse_args()

    if args.kaggle_results:
        with open(args.kaggle_results) as f:
            kaggle_data = json.load(f)
        kaggle_model = kaggle_data.get("llm_model", "unknown")
        kaggle_results = kaggle_data["results"]
        kaggle_ids = {r["TransactionID"] for r in kaggle_results}

        all_transactions, is_full_size = load_sample()
        by_id = {t["TransactionID"]: t for t in all_transactions}
        missing = kaggle_ids - set(by_id)
        if missing:
            print(f"WARNING: {len(missing)} TransactionIDs in {args.kaggle_results} "
                  f"aren't in {PREFERRED_SAMPLE if is_full_size else FALLBACK_SAMPLE} — "
                  f"skipping them (probably ran against a different/older export).")
        transactions = [by_id[tid] for tid in kaggle_ids if tid in by_id]
        if not transactions:
            print("FATAL: no overlap between --kaggle-results and the local sample file.")
            sys.exit(1)
        print(f"Combining {len(transactions)} Kaggle-scored transactions "
              f"(model={kaggle_model}) with local XGBoost scoring...")

        try:
            xgb_probs = score_with_xgboost(transactions)
        except ModelUnavailableError as e:
            print(f"FATAL: XGBoost model unavailable — {e}")
            sys.exit(1)
        print(f"XGBoost side done: {len(xgb_probs)} scored.")

        llm_probs, llm_latencies_ms, failures = {}, [], []
        for r in kaggle_results:
            if r["TransactionID"] not in by_id:
                continue
            if "error" in r:
                failures.append(r)
            else:
                llm_probs[r["TransactionID"]] = r["fraud_probability"]
                llm_latencies_ms.append(r["latency_ms"])

        finalize_and_report(transactions, xgb_probs, llm_probs, llm_latencies_ms, failures,
                             kaggle_model, is_full_size, llm_source="kaggle-gpu")
        return

    if args.test_connection:
        print(f"Testing connection to {OLLAMA_HOST}, model={OLLAMA_MODEL}...")
        try:
            p, latency = call_llm(build_prompt(
                "TransactionAmt: 100.0\nProductCD: W\ncard4: visa\ncard6: debit"
            ))
        except LLMResponseError as e:
            print(f"FAILED: {e}")
            sys.exit(1)
        print(f"OK — fraud_probability={p:.4f}, latency={latency:.0f}ms")
        return

    transactions, is_full_size = load_sample()
    if args.limit:
        transactions = transactions[: args.limit]
    print(f"Scoring {len(transactions)} transactions "
          f"({'full 500-row' if is_full_size else '25-row DRY RUN'} sample)...")

    try:
        xgb_probs = score_with_xgboost(transactions)
    except ModelUnavailableError as e:
        print(f"FATAL: XGBoost model unavailable — {e}")
        sys.exit(1)
    print(f"XGBoost side done: {len(xgb_probs)} scored.")

    progress_file = progress_path_for(is_full_size)
    if args.reset and os.path.exists(progress_file):
        os.remove(progress_file)
        print(f"--reset: cleared {progress_file}")

    # Only a past SUCCESS counts as "done" and gets skipped. A past failure (timeout,
    # transient rate-limit, etc. — see journal: 1/25 timed out on the dry run, nothing
    # wrong, just a slow response) gets retried fresh on resume rather than being treated
    # as a permanent write-off, so a handful of transient blips don't silently shrink the
    # effective sample size of the real run.
    done = load_progress(progress_file)
    succeeded = {tid: rec for tid, rec in done.items() if "error" not in rec}
    llm_probs = {tid: rec["fraud_probability"] for tid, rec in succeeded.items()}
    llm_latencies_ms = [rec["latency_ms"] for rec in succeeded.values()]
    failures = []

    remaining = [t for t in transactions if t["TransactionID"] not in succeeded]
    n_retrying = len(done) - len(succeeded)
    if done:
        msg = f"Resuming {progress_file}: {len(succeeded)} succeeded already, {len(remaining)} left"
        if n_retrying:
            msg += f" ({n_retrying} of those are retries of a previous failure)"
        print(msg + f". At ~29s/call (measured) that's roughly {len(remaining) * 29 / 60:.0f} minutes.")

    # Small courtesy delay between every call, and real backoff specifically on a 429 — a
    # run with neither turned one rate-limit hit into a ~90-call streak of 429s (see
    # journal), because nothing ever paused long enough for the window to actually reset.
    BASE_DELAY_S = 2
    MAX_BACKOFF_S = 90
    MAX_RATE_LIMIT_RETRIES = 5

    for i, txn in enumerate(remaining):
        prompt = build_prompt(serialize_transaction(txn))
        backoff_s = 15
        for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
            try:
                p, latency_ms = call_llm(prompt)
                break
            except RateLimitError as e:
                if attempt == MAX_RATE_LIMIT_RETRIES:
                    rec = {"TransactionID": txn["TransactionID"], "error": str(e)}
                    failures.append(rec)
                    append_progress(progress_file, rec)
                    p = None
                    break
                print(f"  rate limited on transaction {txn['TransactionID']} "
                      f"(attempt {attempt + 1}/{MAX_RATE_LIMIT_RETRIES}) — "
                      f"backing off {backoff_s}s before retrying...")
                time.sleep(backoff_s)
                backoff_s = min(backoff_s * 2, MAX_BACKOFF_S)
            except LLMResponseError as e:
                rec = {"TransactionID": txn["TransactionID"], "error": str(e)}
                failures.append(rec)
                append_progress(progress_file, rec)
                p = None
                break
        if p is None:
            time.sleep(BASE_DELAY_S)
            continue
        rec = {"TransactionID": txn["TransactionID"], "fraud_probability": p, "latency_ms": latency_ms}
        llm_probs[txn["TransactionID"]] = p
        llm_latencies_ms.append(latency_ms)
        append_progress(progress_file, rec)
        time.sleep(BASE_DELAY_S)
        if (i + 1) % 10 == 0:
            print(f"  ...{i + 1}/{len(remaining)} new calls this run "
                  f"({len(done) + i + 1}/{len(transactions)} total, {len(failures)} failed)")

    finalize_and_report(transactions, xgb_probs, llm_probs, llm_latencies_ms, failures,
                         OLLAMA_MODEL, is_full_size, llm_source="ollama-cloud")


if __name__ == "__main__":
    main()
