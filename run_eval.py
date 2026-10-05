"""Golden-set regression harness for a policy-question assistant.

Runs every golden row through the assistant, scores it with deterministic checks,
names the failing layer (scope, corpus, retrieval or generation), flags stale
answer-key rows, and exits non-zero on any regression, so it can gate every
config change.

    python3 run_eval.py                      # baseline
    python3 run_eval.py --include-superseded # break the corpus
    python3 run_eval.py --answer-from 1      # break retrieval (ranking)
    python3 run_eval.py --max-sentences 1    # break generation
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from assistant import Assistant, load_corpus

HERE = Path(__file__).parent


def current_versions(corpus_dir):
    """doc_id -> (version, effective) for the current version of each document."""
    return {c.doc_id: (c.version, c.effective) for c in load_corpus(corpus_dir)}


def stale_reason(row, current):
    for src in row["sources"]:
        version, effective = current[src["doc_id"]]
        if src["version"] != version or row["reviewed"] < effective:
            return (f"cites {src['doc_id']} v{src['version']}, current is v{version} "
                    f"(effective {effective}); reviewed {row['reviewed']}")
    return None


def context_recall(row, resp):
    """Non-LLM context recall: share of reference sections found in the top-k."""
    if not row["sources"]:
        return None
    got = {(c.doc_id, c.section) for _, c in resp.retrieved}
    hits = sum((s["doc_id"], s["section"]) in got for s in row["sources"])
    return hits / len(row["sources"])


def score(row, resp, current):
    """Return (passed, layer, detail). Layers are checked in triage order."""
    if resp.behaviour != row["expected"]:
        if row["expected"] == "ANSWER" and resp.behaviour == "REFUSE":
            return False, "RETRIEVAL", "refused an in-scope question (nothing scored above threshold)"
        return False, "SCOPE", f"expected {row['expected']}, got {resp.behaviour}"
    if row["expected"] != "ANSWER":
        return True, "", ""
    stale_cites = [c for c in resp.citations if current[c.doc_id][0] != c.version]
    if stale_cites:
        c = stale_cites[0]
        return False, "CORPUS", f"cited superseded {c.doc_id} v{c.version} §{c.section}"
    if context_recall(row, resp) < 1:
        return False, "RETRIEVAL", "reference section not in top-k"
    cited = {(c.doc_id, c.section) for c in resp.citations}
    if not {(s["doc_id"], s["section"]) for s in row["sources"]} <= cited:
        return False, "RETRIEVAL", ("ranking: reference section was retrieved but ranked "
                                    "too low to be used in the answer")
    text = resp.text.lower()
    missing = [c["claim"] for c in row["reference_claims"]
               if not all(p.lower() in text for p in c["must_include"])]
    if missing:
        return False, "GENERATION", "missing claim: " + "; ".join(missing)
    return True, "", ""


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--golden", default=HERE / "golden_set.jsonl")
    ap.add_argument("--corpus", default=HERE / "corpus")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--refuse-below", type=float, default=1.0)
    ap.add_argument("--answer-from", type=int, default=2)
    ap.add_argument("--max-sentences", type=int, default=3)
    ap.add_argument("--include-superseded", action="store_true")
    args = ap.parse_args()

    config = {k: v for k, v in vars(args).items() if k not in ("golden", "corpus")}
    bot = Assistant(args.corpus, args.k, args.refuse_below, args.answer_from,
                    args.max_sentences, args.include_superseded)
    current = current_versions(args.corpus)
    rows = [json.loads(line) for line in Path(args.golden).read_text().splitlines() if line]

    results, stale = [], []
    for row in rows:
        why = stale_reason(row, current)
        if why:                         # the key is out of date: owner action, not a verdict
            stale.append({"id": row["id"], "owner": row["owner"], "reason": why})
            continue
        resp = bot.ask(row["question"])
        passed, layer, detail = score(row, resp, current)
        results.append({"id": row["id"], "risk": row["risk_tier"], "expected": row["expected"],
                        "got": resp.behaviour, "recall": context_recall(row, resp),
                        "passed": passed, "layer": layer, "detail": detail})

    print(f"\nGolden set {fingerprint(args.golden)} · {len(rows)} rows · config {config}\n")
    print(f"{'ROW':<10}{'RISK':<8}{'EXPECTED':<10}{'GOT':<10}{'RECALL':<8}RESULT")
    for r in results:
        recall = "-" if r["recall"] is None else f"{r['recall']:.2f}"
        verdict = "PASS" if r["passed"] else f"FAIL  [{r['layer']}] {r['detail']}"
        print(f"{r['id']:<10}{r['risk']:<8}{r['expected']:<10}{r['got']:<10}{recall:<8}{verdict}")

    for s in stale:
        print(f"{s['id']:<10}{'':<8}{'':<10}{'':<10}{'':<8}STALE GOLD  {s['reason']}")

    failed = [r for r in results if not r["passed"]]
    print(f"\nScored {len(results)} · passed {len(results) - len(failed)} · "
          f"failed {len(failed)} (high-risk: {sum(r['risk'] == 'high' for r in failed)}) · "
          f"stale {len(stale)}")
    for s in stale:
        print(f"  OWNER ACTION  {s['owner']}: re-review {s['id']} against the current version")

    report = HERE / "reports" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ.json")
    report.parent.mkdir(exist_ok=True)
    report.write_text(json.dumps({"golden_set_sha": fingerprint(args.golden), "config": config,
                                  "results": results, "stale": stale}, indent=2))
    print(f"  Evidence written to {report.relative_to(HERE)}")

    if failed:
        print("\nGATE: FAIL. Do not ship this configuration.")
        sys.exit(1)
    print("\nGATE: PASS.")


if __name__ == "__main__":
    main()
