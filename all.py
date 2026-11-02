import subprocess
import os
import disgen
import load_dataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--setname", required=True)
parser.add_argument("--dataset", choices=["annamkiepura99", "jessicalamjh"], required=True)
parser.add_argument("--level", choices=["section", "paragraph", "both"], default="both")
parser.add_argument("--n", type=int, default=20)
args = parser.parse_args()

SETNAME = args.setname
N = args.n

for level, path in [("section", "data/few_shot_examples.json"), ("paragraph", "data/few_shot_paragraph_examples.json")]:
    if not os.path.exists(path):
        subprocess.run(["python", "get_few_shot_examples.py", "--level", level], check=True)

if args.dataset == "annamkiepura99":
    levels = ["section", "paragraph"] if args.level == "both" else [args.level]
    for level in levels:
        print(f"\n=== Level: {level} ===")
        papers = load_dataset.get_papers_only(level=level)
        conditions = [(cs, strat) for cs in ["small", "medium", "large"] for strat in ["zero", "few"]]
        pending = [(cs, strat) for cs, strat in conditions
                   if disgen.has_pending_work(SETNAME, level, cs, strat, N, papers)]
        if not pending:
            print(f"  All outputs already exist for {level}. Skipping.")
            continue
        print(f"  Loading cited papers for {level} ({len(pending)}/{len(conditions)} conditions have pending work)...")
        cited = load_dataset.get_cited_papers(level=level)
        for context_size, strategy in pending:
            print(f"--- {level} / {context_size} / {strategy} ---")
            disgen.process_pre_2026(SETNAME, level, context_size, strategy, N,
                                    papers=papers, cited_papers=cited)

else:  # jessicalamjh — section + paragraph
    print("\n=== Loading post-2026 dataset (jessicalamjh) ===")
    papers = load_dataset.get_papers_post_2026()
    levels = ["section", "paragraph"] if args.level == "both" else [args.level]
    conditions = [(cs, strat) for cs in ["small", "medium", "large"] for strat in ["zero", "few"]]
    for level in levels:
        print(f"\n=== Level: {level} ===")
        pending = [(cs, strat) for cs, strat in conditions
                   if disgen.has_pending_work(SETNAME, level, cs, strat, N, papers, id_column="sample_id")]
        if not pending:
            print(f"  All outputs already exist for {level}. Skipping.")
            continue
        for context_size, strategy in pending:
            print(f"--- {level} / {context_size} / {strategy} ---")
            disgen.process_post_2026(SETNAME, level, context_size, strategy, N, papers=papers)

import run_evaluation
import visualization
run_evaluation.run(SETNAME)
visualization.run(SETNAME)
