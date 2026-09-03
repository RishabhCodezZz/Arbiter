# Submission

Panel-ready copy for the buildathon form's build-related fields.

---

## Track

**02 — AI Risk Manager**

## Project name

**Arbiter**

## What it solves

Arbiter is a fraud decision system, not a fraud classifier. For every payment it prices
three outcomes — allow, step-up verification, or block — in real rupees and takes whichever
loses the least money, per transaction, because a ₹500 and a ₹50,000 payment don't carry
the same risk at the same fraud probability.

Built solo for the Razorpay AI Buildathon 2026 — Track 02, AI Risk Manager.

Measured on one untouched 92,427-transaction held-out test month:

- **+₹1.68 crore** vs running no fraud system, and **+₹77 lakh** vs the industry-default
  0.5 cutoff — both holding at 95% confidence across 2,000 bootstrap resamples.
- **False-positive cost reported exactly**, not estimated: 223 genuine customers wrongly
  blocked, ₹13.2 lakh.
- **Deliberately gradient-boosted trees, not an LLM** — benchmarked head to head, the model
  beats a capable LLM 3.65x on accuracy and scores each transaction about 60x faster
  (~100 ms per transaction on plain CPU, 65–135 ms across runs, vs a ~6.7 s median). The
  LLM only writes the one-line analyst explanation, never the decision.

Runs on plain CPU from a saved artifact — no notebook, no GPU, no retraining. Append-only
replayable audit trail, 105 tests, and every designed failure path broken on purpose to
confirm it recovers.

Full mechanism: [`README.md`](README.md); full evidence: [`docs/eval_report.md`](docs/eval_report.md).

## Repo

**https://github.com/RishabhCodezZz/arbiter**

## What broke, and how you got out

Four stories below are the ones worth a panel's time; the full account — every bug found,
how each one surfaced, and 13 more not included here — is in
[`journal/build-log.md`](journal/build-log.md), written as things happened rather than
reconstructed for this write-up.

**A silent 23x error in the number the whole system depends on.** After moving the project
into a clean, properly-pinned virtual environment, an already-passing test was re-run as a
sanity check — expecting no change. Instead, the same transaction's fraud probability had
shifted. That discrepancy was chased rather than dismissed as noise, and it resolved to
something serious: the locally-installed version of the model library gave a **23x
different probability on byte-identical input** than the version the model was actually
trained on — silently, with no error or warning at all. Every number a fail-safe test had
reported earlier had been computed with the wrong engine version. Fixed two ways, not one:
a hard version pin in `requirements.txt`, and a runtime guard in the model loader that
refuses to score at all on a version mismatch rather than risk a silent wrong answer. The
guard itself was independently tested — a mismatch was deliberately faked to confirm it
actually fires, not just assumed to work because the code looked right.

**A target that couldn't be reproduced, cut on evidence instead of forced.** Part of the
original plan depended on reconstructing which transactions belonged to the same
customer, matching purity numbers published by the competition's winning solution. The
first attempt came in roughly 17x worse than target. Rather than declare success on a
weak number or burn unlimited time chasing it, eight different reconstruction strategies
were tried against a stop threshold that had been set **before** the investigation began —
specifically so the decision wouldn't have to be made under time pressure while already
invested in the problem. The best attempt still landed ~11x off target, traced to the
winning team relying on a separate, undocumented matching script rather than a simple
column combination. The component was cut. The rest of the system — which doesn't depend
on that identity resolution — was unaffected, and the decision, with its evidence, is
logged rather than hidden.

**A textbook assumption about calibration turned out to be wrong, measurably.** Standard
practice says calibrating a model's probabilities is monotonic and therefore shouldn't
meaningfully change ranking-based metrics — that assumption was even written directly into
the project's own notebook comments. Applying it anyway produced the single largest score
drop in the entire project. The assumption was imprecise, not the code: monotonic only
forbids *reversing* order, not *tying* it, and the calibration method used was a step
function that collapsed **91,271 distinct model scores down to 323** — a huge number of
transactions the model could originally tell apart became indistinguishable. Switched to a
smooth calibration curve that structurally cannot introduce ties; it matched the original
ranking exactly and calibrated the probabilities even better than the method that had just
failed, with no trade-off. The wrong prior expectation is stated plainly here rather than
quietly corrected and forgotten, because getting caught being wrong and fixing it is more
convincing than being right by luck.

**A benchmark that kept failing wasn't fixed by fixing it harder — it was moved.** Running
a real, large-scale comparison against a cloud-hosted LLM meant hundreds of sequential API
calls. The run started healthy, then degraded to failing more often than succeeding. The
first fix — a leftover background process was still quietly hammering the same API key —
helped, but the failures continued anyway. So did a second, more substantial fix: real
exponential backoff between retries. Neither solved it. At that point the right move
wasn't a third patch on the same approach — it was recognizing that a shared free-tier
cloud API under real load simply wasn't reliable enough for this job, and moving the
entire benchmark onto a different, structurally immune resource instead: a GPU with no
external API and no shared rate limit. That version ran clean, and it's the number
actually reported: XGBoost beats a real, capably-chosen LLM by **3.65x** on accuracy and scores each
transaction roughly 60x faster (~100ms on plain CPU, end to end, 65–135ms across runs, vs
a ~6.7s median) for this task — measured evidence for exactly the "where you chose not to use AI"
question the rubric asks.

**The tally.** 23 real defects across the project — 17 found by our own testing (running
things and trying to break them, never by reading code and calling it correct), plus 6
more from three rounds of automated AI code review run over the repo after the migration.
The clearest of the review defects: a train/serve `std` mismatch on one feature (ddof=0
online vs ddof=1 in training — `[0,2]` → `1.0` vs `1.4142`); a malformed-but-valid artifact
that crashed engine construction instead of failing closed; a NaN/Infinity calibrator
coefficient that `float(...)` accepted and that would have produced a silent `allow`; a
version guard that a manifest simply *omitting* the version field could silently bypass
(the same 23x wrong-probability hole, re-opened); and the allow/step-up/block rupee
breakdown shown as a decision's rationale not being covered by the tamper hash. All fixed,
with regression tests. The review's genuinely production-grade findings — cryptographically
keyed audit signatures, a multi-host transactional feature store, a published release
manifest — stay in the honest exception list; two of them (single-process concurrency,
per-artifact checksums) were partly hardened rather than only disclosed. Every defect
logged as it surfaced, every one fixed, none shipped.
