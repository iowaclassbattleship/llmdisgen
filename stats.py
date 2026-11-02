import sqlite3
import sys
import numpy as np
import pandas as pd
from scipy import stats

DB = "results.db"
df = None


def load(setname):
    con = sqlite3.connect(DB)
    query = "SELECT * FROM results WHERE component = 'f1'"
    params = []
    if setname:
        query += " AND setname = ?"
        params.append(setname)
    try:
        return pd.read_sql_query(query, con, params=params)
    finally:
        con.close()

METRICS = [
    "BERT_bert-base-uncased",
    "BERT_deberta-large-mnli",
    "blockmatch_bert-base-uncased",
    "blockmatch_deberta-large-mnli",
    "rouge_l",
]

METRIC_LABELS = {
    "BERT_bert-base-uncased":        "BERT bert-base-uncased       ",
    "BERT_deberta-large-mnli":       "BERT deberta-large-mnli      ",
    "blockmatch_bert-base-uncased":  "BlockMatch bert-base-uncased ",
    "blockmatch_deberta-large-mnli": "BlockMatch deberta-large-mnli",
    "rouge_l":                       "ROUGE-L                      ",
}


# ── Holm-Bonferroni correction ────────────────────────────────────────────────
def holm(ps):
    n = len(ps)
    order = np.argsort(ps)
    corrected = np.zeros(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = min(ps[idx] * (n - rank), 1.0)
        running_max = max(running_max, adj)
        corrected[idx] = running_max
    reject = corrected < 0.05
    return reject, corrected


def _sig(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


# ── Paper-level paired t-test helper ─────────────────────────────────────────
def paired_ttest_report(title, level, dim, val_a, val_b, label_a=None, label_b=None):
    label_a = label_a or val_a
    label_b = label_b or val_b
    W = 34
    print(f"\n{'─'*80}")
    print(f"{title}  [paper-level paired t-tests, Holm-Bonferroni]")
    print(f"{'─'*80}")
    h = f"  {'Metric':<{W}}  {label_a+' mean':>10}  {label_a+' SD':>8}  {label_b+' mean':>10}  {label_b+' SD':>8}  {'Δ':>7}  {'t':>7}  {'p_raw':>7}  {'p_Holm':>7}  sig"
    print(h)

    sub = df[df["level"] == level]
    raw_ps, rows = [], []
    for m in METRICS:
        paper_means = (
            sub[sub["metric_name"] == m]
            .groupby(["corpus_id", dim])["value"]
            .mean()
            .unstack(dim)
        )
        if val_a not in paper_means.columns or val_b not in paper_means.columns:
            continue
        pairs = paper_means[[val_a, val_b]].dropna()
        if len(pairs) < 2:
            continue
        a, b = pairs[val_a].values, pairs[val_b].values
        t, p = stats.ttest_rel(a, b)
        rows.append((m, a.mean(), a.std(ddof=1), b.mean(), b.std(ddof=1),
                     b.mean() - a.mean(), t, p, len(pairs)))
        raw_ps.append(p)

    _, corrected = holm(raw_ps)
    for i, (m, ma, sda, mb, sdb, diff, t, p, n) in enumerate(rows):
        print(f"  {METRIC_LABELS[m]:<{W}}  {ma:>10.4f}  {sda:>8.4f}  {mb:>10.4f}  {sdb:>8.4f}  {diff:>+7.4f}  {t:>7.3f}  {p:>7.4f}  {corrected[i]:>7.4f}  {_sig(corrected[i])}")


# ── Independent-samples t-test helper ────────────────────────────────────────
def indep_ttest_report(title, level_a, level_b, label_a=None, label_b=None):
    label_a = label_a or level_a
    label_b = label_b or level_b
    W = 34
    print(f"\n{'─'*80}")
    print(f"{title}  [independent-samples t-tests, Holm-Bonferroni]")
    print(f"{'─'*80}")
    print(f"  {'Metric':<{W}}  {label_a+' mean':>10}  {label_b+' mean':>10}  {'Δ':>7}  {'t':>7}  {'p_raw':>7}  {'p_Holm':>7}  sig")

    raw_ps, rows = [], []
    for m in METRICS:
        def paper_means(level, _m=m):
            return (
                df[(df["level"] == level) & (df["metric_name"] == _m)]
                .groupby("corpus_id")["value"]
                .mean()
                .values
            )
        a, b = paper_means(level_a), paper_means(level_b)
        if len(a) < 2 or len(b) < 2:
            continue
        t, p = stats.ttest_ind(a, b)
        rows.append((m, a.mean(), b.mean(), b.mean() - a.mean(), t, p))
        raw_ps.append(p)

    _, corrected = holm(raw_ps)
    for i, (m, ma, mb, diff, t, p) in enumerate(rows):
        print(f"  {METRIC_LABELS[m]:<{W}}  {ma:>10.4f}  {mb:>10.4f}  {diff:>+7.4f}  {t:>7.3f}  {p:>7.4f}  {corrected[i]:>7.4f}  {_sig(corrected[i])}")


def main(setname):
    global df
    df = load(setname)

    # ── Comparisons ───────────────────────────────────────────────────────────────
    paired_ttest_report(
        "Table 1 — SECTION: few-shot vs zero-shot",
        level="section", dim="prompt_strategy", val_a="zero", val_b="few",
        label_a="zero-shot", label_b="few-shot",
    )
    paired_ttest_report(
        "Table 2 — SECTION: medium vs small context",
        level="section", dim="context_size", val_a="small", val_b="medium",
        label_a="small", label_b="medium",
    )
    paired_ttest_report(
        "Table 3 — PARAGRAPH: few-shot vs zero-shot",
        level="paragraph", dim="prompt_strategy", val_a="zero", val_b="few",
        label_a="zero-shot", label_b="few-shot",
    )
    paired_ttest_report(
        "Table 4 — PARAGRAPH: medium vs small context",
        level="paragraph", dim="context_size", val_a="small", val_b="medium",
        label_a="small", label_b="medium",
    )
    paired_ttest_report(
        "Table 2b — SECTION: large vs medium context",
        level="section", dim="context_size", val_a="medium", val_b="large",
        label_a="medium", label_b="large",
    )
    paired_ttest_report(
        "Table 4b — PARAGRAPH: large vs medium context",
        level="paragraph", dim="context_size", val_a="medium", val_b="large",
        label_a="medium", label_b="large",
    )
    indep_ttest_report(
        "Table 5 — GRANULARITY: section vs paragraph",
        level_a="paragraph", level_b="section",
        label_a="paragraph", label_b="section",
    )


    # ── Coverage counts ───────────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("COVERAGE")
    print(f"{'─'*70}")
    for level in ["section", "paragraph"]:
        sub = df[df["level"] == level]
        if sub.empty:
            continue
        # Derive the expected count from the dimensions actually present, so the
        # denominator tracks the modelset and context sizes instead of a fixed design.
        n_models = sub["model_name"].nunique()
        n_papers = sub["corpus_id"].nunique()
        n_context = sub["context_size"].nunique()
        n_strategies = sub["prompt_strategy"].nunique()
        per_model = n_papers * n_context * n_strategies * len(METRICS)
        n_possible = n_models * per_model

        print(f"  {level}: {len(sub)}/{n_possible} F1 scores ({100*len(sub)/n_possible:.1f}%)")
        print(f"    ({n_models} models x {n_papers} papers x {n_context} context sizes "
              f"x {n_strategies} strategies x {len(METRICS)} metrics)")
        for model, n in sub.groupby("model_name").size().sort_values().items():
            print(f"    {model}: {n}/{per_model}")


    # ── Best / worst papers ───────────────────────────────────────────────────────
    per_paper = df.groupby("corpus_id")["value"].mean().rename("mean_f1").sort_values(ascending=False)
    best_id  = per_paper.idxmax()
    worst_id = per_paper.idxmin()


    def print_paper(corpus_id, label):
        print(f"\n{'='*60}")
        print(f"{label}: {corpus_id}  (mean F1 = {per_paper[corpus_id]:.4f})")
        print(f"{'='*60}")
        breakdown = (
            df[df["corpus_id"] == corpus_id]
            .groupby(["metric_name", "context_size", "prompt_strategy", "model_name"])["value"]
            .mean()
            .unstack("model_name")
            .round(4)
        )
        print(breakdown.to_string())


    print(f"\n{'─'*70}")
    print("BEST / WORST PAPERS (across all conditions)")
    print(f"{'─'*70}")
    print_paper(best_id, "BEST")
    print_paper(worst_id, "WORST")

    print(f"\n--- All papers ranked ---")
    print(per_paper.round(4).to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
