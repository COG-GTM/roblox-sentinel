"""Reproduce and verify a slang-drift recall miss against the eval exemplar index.

Run with ``PYTHONPATH=src python eval/drift_check.py``.

Held-out messages below are never added to the index; they are the eval set.
Scam labels come from human moderators (see the triggering alert), not from this script.
"""

import sys
from pathlib import Path

from sentinel.sentinel_local_index import SentinelLocalIndex

EXEMPLAR_DIR = Path(__file__).parent / "exemplars"

# Moderator-labelled scam messages that Sentinel scored 0 (recall misses).
HELD_OUT_SCAM = [
    "im a dev drop ur login",
    "fr33 r0bux in bio",
    "beaming site in my status",
    "ill give u korblox for ur acc",
]

# Message quoted verbatim in the triggering alert (Sentry issue since deleted).
ALERT_MESSAGES = [
    "Player 1 you are a bad person",
]

# Held-out benign messages sharing vocabulary with the scam misses (precision check).
HELD_OUT_BENIGN = [
    "im a dev on this game lol",
    "drop ur user so i can friend u",
    "i put my fav game in my bio",
    "my status says im at school",
    "ill trade u my korblox leg for ur dominus",
    "fr this obby is so hard",
]


def load_lines(path):
    """Read non-empty lines from an exemplar file."""
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def build_index(scam, benign):
    """Build the eval index with the fixed model and seed."""
    return SentinelLocalIndex.from_texts(
        positive_texts=scam,
        negative_texts=benign,
        model_name="all-MiniLM-L6-v2",
        seed=0,
    )


def format_neighbors(explanation, n=3):
    """Render the top-n explain neighbours as a one-line string."""
    parts = []
    for nb in (explanation.get("neighbors") or [])[:n]:
        parts.append(f"[{nb['sign']}] {nb['neighbor']!r} ({nb['scaled_score']:.3f})")
    return "; ".join(parts)


def score_table(index, messages, label, **kwargs):
    """Score messages and collect per-message explain rows."""
    result = index.calculate_rare_class_affinity(messages, explain=True, **kwargs)
    rows = []
    for msg in messages:
        score = result.observation_scores[msg]
        expl = result.explanations[msg]
        rows.append(
            {
                "message": msg,
                "label": label,
                "score": score,
                "flagged": score > 0,
                "log_ratio": expl["contrastive"]["log_ratio_unclipped"],
                "neighbors": format_neighbors(expl),
            }
        )
    return rows


def print_rows(rows):
    """Print one block per scored message."""
    for r in rows:
        flag = (
            "FLAGGED"
            if r["flagged"]
            else "miss   " if r["label"] == "scam" else "ok     "
        )
        print(
            f"{r['message']!r:48} | {r['label']:6} | score={r['score']:.4f} | {flag} "
            f"| log_ratio={r['log_ratio']:+.4f}"
        )
        print(f"{'':48}   top-3: {r['neighbors']}")


def main():
    """Run the drift check and print the summary."""
    scam = load_lines(EXEMPLAR_DIR / "scam.txt")
    benign = load_lines(EXEMPLAR_DIR / "benign.txt")

    held_out = set(HELD_OUT_SCAM + HELD_OUT_BENIGN + ALERT_MESSAGES)
    leaked = held_out & set(scam + benign)
    if leaked:
        sys.exit(f"held-out messages must not be in the index: {sorted(leaked)}")

    print(f"index: {len(scam)} scam exemplars, {len(benign)} benign exemplars\n")
    index = build_index(scam, benign)

    scam_rows = score_table(index, HELD_OUT_SCAM, "scam")
    alert_rows = score_table(index, ALERT_MESSAGES, "alert")
    benign_rows = score_table(index, HELD_OUT_BENIGN, "benign")
    corpus_benign_rows = score_table(index, benign, "benign", prevent_exact_match=True)

    print("== held-out scam misses ==")
    print_rows(scam_rows)
    print("\n== alert message (label from alert; scored for reference) ==")
    print_rows(alert_rows)
    print("\n== held-out benign ==")
    print_rows(benign_rows)
    print("\n== indexed benign exemplars (prevent_exact_match=True), flagged only ==")
    flagged_corpus = [r for r in corpus_benign_rows if r["flagged"]]
    print_rows(flagged_corpus) if flagged_corpus else print("(none flagged)")

    caught = sum(r["flagged"] for r in scam_rows)
    fp_held_out = sum(r["flagged"] for r in benign_rows)
    fp_corpus = len(flagged_corpus)
    print("\n== summary ==")
    print(f"scam recall: {caught}/{len(scam_rows)}")
    print(
        f"alert message flagged: {sum(r['flagged'] for r in alert_rows)}/{len(alert_rows)}"
    )
    print(f"held-out benign false positives: {fp_held_out}/{len(benign_rows)}")
    print(f"indexed benign false positives: {fp_corpus}/{len(corpus_benign_rows)}")


if __name__ == "__main__":
    main()
