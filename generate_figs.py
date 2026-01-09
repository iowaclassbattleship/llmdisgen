import json
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm

experiment = sys.argv[1]

allowed_experiments = ["abstract_only", "full_paper"]
if experiment not in allowed_experiments:
    raise ValueError(f"{experiment} not in {allowed_experiments}")

figs = Path("figs") / experiment
figs.mkdir(exist_ok=True, parents=True)

input_path = Path("data") / experiment

datafiles = [f for f in input_path.iterdir()]
for datafile in tqdm(datafiles):
    ts = datafile.stem
    target = figs / f"{ts}.png"
    if target.exists():
        continue
    with open(datafile, "r") as f:
        data = json.load(f)
    d = defaultdict(list)

    for corpus_id, models in data.items():
        for model, entries in models.items():
            for entry in entries:
                p = entry.get("P")
                if p is not None:
                    d[model].append(p)

    models = np.array(list(d.keys()))
    means = np.array([np.mean(d[m]) if d[m] else 0 for m in models])

    order = np.argsort(means)

    plt.figure(figsize=(9, 4))
    plt.bar(models[order], means[order])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Mean Precision (P)")
    plt.title("Mean BERTScore Precision by Model")
    plt.tight_layout()
    plt.savefig(target, dpi=200)
    plt.close()
