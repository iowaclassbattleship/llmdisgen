import json
from pathlib import Path
from collections import defaultdict

OUT = Path("000_out")


def iter_run_jsons(base: Path):
    for f in base.rglob("*.json"):
        if f.stem == f.parent.parent.name or f.stem == f.parent.name:
            yield f


def empty():
    return {"cost": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "calls": 0}


def add(bucket, cost, prompt_tokens, completion_tokens):
    bucket["cost"] += cost
    bucket["prompt_tokens"] += prompt_tokens
    bucket["completion_tokens"] += completion_tokens
    bucket["calls"] += 1


by_model = defaultdict(empty)
by_level = defaultdict(empty)
by_condition = defaultdict(empty)  # (level, context_size, strategy)
totals = empty()

for path in iter_run_jsons(OUT):
    try:
        with open(path) as f:
            data = json.load(f)
        usage = data["response"]["usage"]
        model = data["model"]
        level = data.get("level", "?")
        strategy = data.get("prompt_strategy", "?")
        context_size = data.get("context_size", "?")
        cost = usage.get("cost", 0.0) or 0.0
        prompt_tokens = usage.get("prompt_tokens", 0) or 0
        completion_tokens = usage.get("completion_tokens", 0) or 0
    except Exception:
        continue

    add(by_model[model], cost, prompt_tokens, completion_tokens)
    add(by_level[level], cost, prompt_tokens, completion_tokens)
    add(by_condition[(level, context_size, strategy)], cost, prompt_tokens, completion_tokens)
    add(totals, cost, prompt_tokens, completion_tokens)


def print_table(rows, title):
    labels = [r[0] if isinstance(r[0], str) else "  ".join(r[0]) for r in rows]
    col = max(len(l) for l in labels)
    print(f"\n{title}")
    print(f"{'':>{col}} {'Calls':>6} {'Prompt tok':>12} {'Compl tok':>11} {'Cost (USD)':>12}")
    print("-" * (col + 45))
    for (_, s), label in zip(rows, labels):
        print(f"{label:>{col}} {s['calls']:>6} {s['prompt_tokens']:>12,} {s['completion_tokens']:>11,} {s['cost']:>12.4f}")


def main():
    # --- By model ---
    print_table(
        sorted(by_model.items(), key=lambda x: -x[1]["cost"]),
        "BY MODEL",
    )

    # --- By level ---
    print_table(
        sorted(by_level.items()),
        "BY LEVEL",
    )

    # --- By condition ---
    print_table(
        sorted(by_condition.items(), key=lambda x: x[0]),
        "BY LEVEL × CONTEXT × STRATEGY",
    )

    # --- Grand total ---
    print(f"\n{'TOTAL':>30} {totals['calls']:>6} {totals['prompt_tokens']:>12,} {totals['completion_tokens']:>11,} {totals['cost']:>12.4f}")


if __name__ == "__main__":
    main()