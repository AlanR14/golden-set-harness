# Golden-set regression harness

A ten-row golden set and a test harness for a policy-question assistant. The harness runs
every golden question through the assistant, scores it with deterministic checks, names the
layer that failed, flags out-of-date answer-key rows, and **exits non-zero on any
regression**, so it can gate every configuration change.

Python 3.9+, standard library only. No API keys.

```bash
python3 run_eval.py                       # baseline: 9/9 pass, 1 stale row flagged
python3 run_eval.py --include-superseded  # corpus failure
python3 run_eval.py --answer-from 1       # retrieval (ranking) failure
python3 run_eval.py --max-sentences 1     # generation failure
```

## What is in here

| File | Role |
|---|---|
| `golden_set.jsonl` | 10 rows: question, reference claims, **versioned** sources, expected behaviour (ANSWER / REFUSE / ESCALATE), risk tier, named owner, review date |
| `corpus/` | Six short **fictional** policy documents for "Example Bank", including one superseded version |
| `assistant.py` | The system under test: a deliberately simple stand-in (BM25 keyword retrieval, quoted answers with citations, a refusal threshold, an escalation list) |
| `run_eval.py` | The harness. Only calls `Assistant.ask()`, so a real LLM-backed assistant can be swapped in |

## Checks, in triage order

1. **Stale gold.** The row cites an old version, or was reviewed before the current version
   took effect. It is not scored, and the owner is told to re-review it.
2. **Behaviour.** Did it answer, refuse or escalate as expected? (scope boundaries)
3. **Corpus.** Did the answer cite a superseded document?
4. **Retrieval.** Non-LLM context recall (were the reference sections in the top-k?), then
   ranking (were they ranked high enough to be used?).
5. **Generation.** Is every reference claim present in the answer?

Each run writes a JSON evidence file to `reports/` with the golden-set hash, the config and
every row's result: an audit trail of what was tested, against what, and when.

## Honest limits

- The assistant is a stand-in. No LLM is called; retrieval is keyword-based.
- Claim checking is phrase matching. That works for quoted answers. With a generative model,
  it becomes an **LLM judge per failure mode**, validated against SME labels (TPR/TNR on
  a held-out set) before anyone relies on its scores.
- Ten rows show the shape of the suite. A real one grows from production error analysis to
  100+ rows.
