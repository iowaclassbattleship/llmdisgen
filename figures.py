from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import utils
import json

runs = Path("runs")
out = Path("figs")
out.mkdir(exist_ok=True)

last_run = utils.get_runs()[-1]
last_run_file = f"metadata-{last_run}.json"

with open(runs / last_run_file, "r") as f:
    data = json.load(f)

b = { }

for d in data:
    for eval in d["evaluations"]:
        for score in eval["accuracy_scores"]:
            b.setdefault(eval["model"], []).append(score["result"])

models = list(b.keys())
binned_score = [np.mean(b[m]) for m in models]

plt.bar([k.split("/")[1] for k in models], binned_score)
plt.ylim(0, 1)
plt.title("mean BlockMatch BERTScore result for multiple LLM")
plt.savefig(out / f"{last_run}.png")