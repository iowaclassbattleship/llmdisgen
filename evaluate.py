from pathlib import Path
from collections import defaultdict
import compare
from llms import openai
import block_match
import json
import time
import sys
from tqdm import tqdm

experiment = sys.argv[1]

allowed_experiments = ["abstract_only", "full_paper"]
if experiment not in allowed_experiments:
    raise ValueError(f"{experiment} not in {allowed_experiments}")

papers_path = Path("papers")
ai_discussions_path = papers_path / experiment
data_out = Path("data")
data_out.mkdir(exist_ok=True)

evres = defaultdict(lambda: defaultdict(list))
discussions_path = papers_path / "discussion"
BERT = compare.BERTScore(model_type="bert-base-uncased")
discussion_txts = [f for f in discussions_path.iterdir()]
for discussion_txt in tqdm(discussion_txts):
    with open(discussion_txt, "r") as f:
        d = f.read()
    corpus_id = discussion_txt.name.split("_")[0]

    for model in openai.OpenAIWrapper.available_models:
        path = ai_discussions_path / f"{corpus_id}_discussion_{model}.txt"
        if path.exists():
            with open(path, "r") as f:
                dp = f.read()
            P, R, F1 = block_match.metric(dp, d, BERT.metric)
            evres[corpus_id][model].append({"P": P, "R": R, "F1": F1})
path = data_out / experiment
path.mkdir(exist_ok=True, parents=True)
with open(path / f"{int(time.time())}.json", "w") as f:
    json.dump(evres, f, indent=2)