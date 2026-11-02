import argparse
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import product

DB = "results.db"

strategy_labels = {"few": "Few-Shot", "zero": "Zero-Shot"}
level_labels = {"section": "Section-Level", "paragraph": "Paragraph-Level"}
context_size_labels = {"small": "small", "medium": "medium", "large": "large"}

metric_labels = {
    "blockmatch_bert-base-uncased": "Blockmatch(bert-base-uncased)",
    "rouge_l": "rouge_l",
    "blockmatch_deberta-large-mnli": "Blockmatch(deberta-large-mnli)",
    "BERT_deberta-large-mnli": "BERT deberta-large-mnli",
    "BERT_bert-base-uncased": "BERT bert-base-uncased",
}

metric_colors = {
    "BERT_bert-base-uncased":        "#4CAF50",
    "blockmatch_bert-base-uncased":  "#1B5E20",
    "BERT_deberta-large-mnli":       "#FF9800",
    "blockmatch_deberta-large-mnli": "#E65100",
    "rouge_l":                       "#9E9E9E",
}


def run(setname=None, tables=False):
    FIGS = Path("report/img") / setname if setname else Path("report/img")
    FIGS.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(DB)
    query = "SELECT * FROM results WHERE component = 'f1'"
    params = []
    if setname:
        query += " AND setname = ?"
        params.append(setname)
    df = pd.read_sql_query(query, con, params=params)
    con.close()

    agg = (
        df.groupby(["level", "context_size", "prompt_strategy", "model_name", "metric_name"])["value"]
        .mean()
        .reset_index()
    )

    metrics = agg["metric_name"].unique()
    levels = agg["level"].unique()
    strategies = agg["prompt_strategy"].unique()
    sizes = agg["context_size"].unique()

    # ── Bar plots: model performance per level × context size ─────────────────
    for level, size in product(levels, sizes):
        n_models = agg["model_name"].nunique()
        n_metrics = agg["metric_name"].nunique()
        w = n_models * n_metrics * 0.25 + 2
        fig, axes = plt.subplots(1, 2, figsize=(w * 2, 5), sharey=True)
        fig.suptitle(
            f"Model performance comparisons for {level_labels[level]} and {context_size_labels[size]} context size"
        )

        for ax, strategy in zip(axes, ["few", "zero"]):
            subset = agg[
                (agg["level"] == level)
                & (agg["prompt_strategy"] == strategy)
                & (agg["context_size"] == size)
            ]
            if subset.empty:
                ax.set_visible(False)
                continue

            pivot = subset.pivot(index="model_name", columns="metric_name", values="value")
            ordered = [m for m in metric_colors if m in pivot.columns]
            pivot = pivot[ordered]
            colors = [metric_colors[m] for m in pivot.columns]
            pivot.plot(kind="bar", ax=ax, legend=False, color=colors, width=0.7)

            for container in ax.containers:
                ax.bar_label(container, fmt="%.2f", fontsize=5, rotation=90, padding=1)

            ax.set_title(strategy_labels[strategy])
            ax.set_ylabel("F1 (avg across papers)")
            ax.set_ylim(0, 1)
            ax.tick_params(axis="x", rotation=30)

        handles, labels = axes[0].get_legend_handles_labels()
        if not handles:
            handles, labels = axes[1].get_legend_handles_labels()
        if not handles:
            plt.close(fig)
            continue
        fig.legend(
            handles, labels, title="Metric",
            loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.12),
        )
        fig.tight_layout()

        fname = FIGS / f"{level}_{size}_few_vs_zero.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {fname}")

    # ── Tables: per-model scores by context size ──────────────────────────────
    if tables:
        for level, strategy, metric in product(levels, strategies, metrics):
            subset = agg[
                (agg["level"] == level)
                & (agg["prompt_strategy"] == strategy)
                & (agg["metric_name"] == metric)
            ]
            if subset.empty:
                continue

            table_data = subset.pivot(index="context_size", columns="model_name", values="value")
            table_data = table_data.map(lambda v: f"{v:.3f}")

            fig, ax = plt.subplots(
                figsize=(len(table_data.columns) * 1.8, len(table_data) * 0.6 + 1)
            )
            ax.axis("off")
            tbl = ax.table(
                cellText=table_data.values,
                rowLabels=table_data.index,
                colLabels=table_data.columns,
                cellLoc="center",
                loc="center",
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(10)
            tbl.auto_set_column_width(col=list(range(len(table_data.columns))))
            ax.set_title(
                f"{level_labels[level]} · {strategy_labels[strategy]} · {metric_labels[metric]}",
                pad=12,
            )
            fig.tight_layout()

            fname = FIGS / f"table_{level}_{strategy}_{metric}.png"
            fig.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"saved {fname}")

    # ── Bar plots: mean F1 by field of study ─────────────────────────────────
    if "fields_of_study" in df.columns and df["fields_of_study"].notna().any():
        fos_df = df.copy()
        fos_df["fields_of_study"] = fos_df["fields_of_study"].fillna("")
        fos_df["field"] = fos_df["fields_of_study"].apply(
            lambda x: list(set(f.strip() for f in x.split(",") if f.strip()))
        )
        fos_df = fos_df.explode("field").dropna(subset=["field"])
        fos_df = fos_df[fos_df["field"] != ""]

        per_paper = (
            fos_df.groupby(["corpus_id", "field", "level"])["value"]
            .mean()
            .reset_index()
        )

        for level in levels:
            sub = per_paper[per_paper["level"] == level]
            if sub.empty:
                continue

            field_means = (
                sub.groupby("field")["value"]
                .agg(mean="mean", count="count")
                .reset_index()
                .sort_values("mean", ascending=True)
            )

            fig, ax = plt.subplots(figsize=(8, max(4, len(field_means) * 0.45)))
            bars = ax.barh(field_means["field"], field_means["mean"],
                           color="#4CAF50", edgecolor="white")
            for bar, (_, row) in zip(bars, field_means.iterrows()):
                ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                        f"n={int(row['count'])}", va="center", ha="left", fontsize=8, color="#555")
            ax.set_xlabel("Mean F1 (avg across all conditions, metrics, models)")
            ax.set_title(f"Performance by field of study — {level_labels[level]}")
            ax.set_xlim(0, min(1.0, field_means["mean"].max() * 1.25))
            ax.axvline(sub["value"].mean(), color="#E65100", linestyle="--",
                       linewidth=1, label=f"Overall mean ({sub['value'].mean():.3f})")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fname = FIGS / f"fos_{level}.png"
            fig.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"saved {fname}")

            # Per-model FoS
            per_paper_model = (
                fos_df[fos_df["level"] == level]
                .groupby(["corpus_id", "field", "model_name"])["value"]
                .mean()
                .reset_index()
            )
            models_in_level = sorted(per_paper_model["model_name"].unique())
            model_colors = ["#4CAF50", "#FF9800", "#2196F3", "#E65100", "#9C27B0",
                            "#009688", "#F44336", "#795548"]
            color_map = {m: model_colors[i % len(model_colors)] for i, m in enumerate(models_in_level)}
            field_order = field_means["field"].tolist()
            n_fields = len(field_order)
            n_models = len(models_in_level)
            bar_height = 0.8 / n_models
            fig, ax = plt.subplots(figsize=(9, max(4, n_fields * n_models * 0.28 + 1)))
            for i, model in enumerate(models_in_level):
                model_sub = (
                    per_paper_model[per_paper_model["model_name"] == model]
                    .groupby("field")["value"].mean()
                )
                means = [model_sub.get(f, float("nan")) for f in field_order]
                y = [j + (i - n_models / 2 + 0.5) * bar_height for j in range(n_fields)]
                ax.barh(y, means, height=bar_height * 0.9, color=color_map[model], label=model)
            ax.set_yticks(range(n_fields))
            ax.set_yticklabels(field_order)
            ax.set_xlabel("Mean F1 (avg across all conditions and metrics)")
            ax.set_title(f"Performance by field of study per model — {level_labels[level]}")
            ax.set_xlim(0, min(1.0, field_means["mean"].max() * 1.3))
            ax.axvline(sub["value"].mean(), color="black", linestyle="--",
                       linewidth=1, alpha=0.4, label=f"Overall mean ({sub['value'].mean():.3f})")
            ax.legend(fontsize=7, loc="lower right")
            fig.tight_layout()
            fname = FIGS / f"fos_{level}_by_model.png"
            fig.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"saved {fname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", action="store_true", default=False)
    parser.add_argument("--setname", default=None)
    args = parser.parse_args()
    run(setname=args.setname, tables=args.tables)
