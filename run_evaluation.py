from tqdm import tqdm
from pathlib import Path
import sys
import sqlite3
from datasets import load_dataset as hf_load_dataset

base = Path("000_out")


def get_text(path: Path):
    with open(path, "r") as f:
        return f.read()


def get_human_discussion(corpus_id: str):
    return get_text(base / "discussions" / f"{corpus_id}.md")


def get_llm_discussions(corpus_id_dir: Path):
    results = []
    for level_dir in corpus_id_dir.iterdir():
        level = level_dir.name
        for strategy_dir in level_dir.iterdir():
            strategy = strategy_dir.name
            for size_dir in strategy_dir.iterdir():
                context_size = size_dir.name
                for model_dir in size_dir.iterdir():
                    model_name = model_dir.name

                    if level == "paragraph":
                        para_dirs = sorted(
                            [d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith("para_")],
                            key=lambda d: int(d.name.split("_")[1]),
                        )
                        if not para_dirs:
                            continue
                        d_ = "\n\n".join(
                            get_text(d / "response.md")
                            for d in para_dirs
                            if (d / "response.md").exists()
                        )
                    else:
                        response_path = model_dir / "response.md"
                        if not response_path.exists():
                            continue
                        d_ = get_text(response_path)

                    results.append({
                        "level": level,
                        "context_size": context_size,
                        "prompt_strategy": strategy,
                        "model_name": model_name,
                        "d_": d_,
                    })
    return results


def run(setname):
    fos_map = {}
    for ds_name in [
        "annamkiepura99/section-diss-gen-combined_v2",
        "annamkiepura99/paragraph-diss-gen-combined",
    ]:
        ds = hf_load_dataset(ds_name, split="train")
        for row in ds:
            fos_map[row["corpus_id"]] = ",".join(row.get("s2fieldsofstud") or [])

    con = sqlite3.connect("results.db")
    con.execute("""
        CREATE TABLE IF NOT EXISTS results (
            setname TEXT,
            corpus_id TEXT,
            level TEXT,
            context_size TEXT,
            prompt_strategy TEXT,
            model_name TEXT,
            metric_name TEXT,
            component TEXT,
            value REAL,
            fields_of_study TEXT,
            PRIMARY KEY (setname, corpus_id, level, context_size, prompt_strategy, model_name, metric_name, component)
        )
    """)

    already_done = set(con.execute(
        "SELECT corpus_id, level, context_size, prompt_strategy, model_name FROM results WHERE setname=?",
        (setname,)
    ).fetchall())

    pending = []
    for corpus_id_dir in (base / setname).iterdir():
        corpus_id = corpus_id_dir.stem
        for result in get_llm_discussions(corpus_id_dir):
            key = (corpus_id, result["level"], result["context_size"], result["prompt_strategy"], result["model_name"])
            if key not in already_done:
                pending.append((corpus_id, result))

    if not pending:
        print(f"  Evaluation: all results already in DB for setname={setname!r}, skipping.")
        con.close()
        return

    print(f"  Evaluation: {len(pending)} combinations to score, loading models...")
    from compare import BERTScore, block_match, rouge_l
    BERT_deberta = BERTScore.BERTScore("microsoft/deberta-large-mnli")
    BERT_bert = BERTScore.BERTScore("bert-base-uncased")
    comparator_functions = [
        ("BERT_deberta-large-mnli", lambda d, d_: BERT_deberta.metric(d_, d)),
        ("BERT_bert-base-uncased", lambda d, d_: BERT_bert.metric(d_, d)),
        ("blockmatch_deberta-large-mnli", lambda d, d_: block_match.metric(d, d_, BERT_deberta.metric)),
        ("blockmatch_bert-base-uncased", lambda d, d_: block_match.metric(d, d_, BERT_bert.metric)),
        ("rouge_l", lambda d, d_: rouge_l.metric(d_, d)),
    ]

    for corpus_id, result in tqdm(pending):
        fos = fos_map.get(corpus_id, "")
        d = get_human_discussion(corpus_id)
        level = result["level"]
        context_size = result["context_size"]
        prompt_strategy = result["prompt_strategy"]
        model_name = result["model_name"]
        d_ = result["d_"]
        metrics = {name: f(d, d_) for name, f in comparator_functions}

        rows = []
        for name, (p, r, f1) in metrics.items():
            # ROUGE-L exposes only an F-measure, so skip the components it does not
            # report rather than writing NULL rows.
            for component, value in (("precision", p), ("recall", r), ("f1", f1)):
                if value is None:
                    continue
                rows.append((setname, corpus_id, level, context_size, prompt_strategy,
                             model_name, name, component, value, fos))
        con.executemany("INSERT OR REPLACE INTO results VALUES (?,?,?,?,?,?,?,?,?,?)", rows)

    con.commit()
    con.close()


if __name__ == "__main__":
    run(sys.argv[1])
