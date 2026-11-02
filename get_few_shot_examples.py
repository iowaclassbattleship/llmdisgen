import json
from collections import defaultdict

import utils
import load_dataset

N_PER_FIELD = 3

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", choices=["section", "paragraph"], required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    defaults = {"section": "data/few_shot_examples.json", "paragraph": "data/few_shot_paragraph_examples.json"}
    out_path = args.out or defaults[args.level]

    papers = load_dataset.get_papers_only(level=args.level)

    examples = defaultdict(list)

    for i in range(len(papers)):
        fields = papers["s2fieldsofstud"][i]
        if not fields:
            continue

        if all(len(examples[f]) >= N_PER_FIELD for f in fields):
            continue

        try:
            sections = list(papers["sections"][i])
            _, discussion = utils.split_discussion(sections)
            if args.level == "paragraph":
                paras = utils.get_discussion_paragraphs(discussion)
                text = paras[0] if paras else None
            else:
                text = utils.build_discussion_body(discussion)
        except Exception:
            continue

        if not text.strip():
            continue

        for field in fields:
            if len(examples[field]) < N_PER_FIELD:
                examples[field].append(text)

    with open(out_path, "w") as f:
        json.dump(examples, f, indent=2)

    for field, exs in sorted(examples.items()):
        print(f"{field}: {len(exs)}")
