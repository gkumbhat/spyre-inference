"""Compare a parallel-encoder smoke run against an inline one.

Usage: python compare.py INLINE.json PARALLEL.json

text_only must match token for token (AC-5). Image prompts must be logprob-close
(AC-4), by the rule of vLLM's check_logprobs_close: at the first position where the
greedy tokens differ, each side's token must be in the other side's top-k, and the
comparison stops there because the two continuations no longer share a prefix.
"""

import json
import sys


def first_divergence(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main() -> int:
    inline = json.load(open(sys.argv[1]))["results"]
    parallel = json.load(open(sys.argv[2]))["results"]
    failed = False
    print(f"{'prompt':<14} {'result':<18} detail")
    for name in [k for k in inline if not k.startswith("_")]:
        a, b = inline[name], parallel[name]
        div = first_divergence(a["token_ids"], b["token_ids"])
        if div is None:
            print(f"{name:<14} {'exact':<18} {len(a['token_ids'])} tokens identical")
            continue
        if name == "text_only":
            failed = True
            print(f"{name:<14} {'FAIL (not exact)':<18} first difference at token {div}")
            continue
        if div >= min(len(a["logprobs"]), len(b["logprobs"])):
            print(f"{name:<14} {'close (length)':<18} identical up to token {div}")
            continue
        top_a, top_b = a["logprobs"][div], b["logprobs"][div]
        ta, tb = str(a["token_ids"][div]), str(b["token_ids"][div])
        close = ta in top_b and tb in top_a
        failed |= not close
        verdict = "close" if close else "FAIL (not close)"
        print(
            f"{name:<14} {verdict:<18} first difference at token {div}: "
            f"inline {ta} ({top_a[ta]:.3f}) vs parallel {tb} ({top_b[tb]:.3f})"
        )
    ba, bb = inline["_batched"]["token_ids"], parallel["_batched"]["token_ids"]
    same = sum(x == y for x, y in zip(ba, bb))
    print(f"{'_batched':<14} {'info':<18} {same}/{len(ba)} sequences identical")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
