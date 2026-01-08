import json
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from pathlib import Path

figs = Path("figs")
figs.mkdir(exist_ok=True)

with open("out.json", "r") as f:
    data = json.load(f)

# model -> list of P values
d = defaultdict(list)

for discussion_id, models in data.items():
    for model, entries in models.items():
        for entry in entries:  # entries is a list
            p = entry.get("P")
            if p is not None:
                d[model].append(p)


models = np.array(sorted(d.keys()))
means = np.array([np.mean(d[m]) for m in models])

order = np.argsort(means)

plt.figure(figsize=(9, 4))
plt.bar(models[order], means[order])
plt.xticks(rotation=45, ha="right")
plt.ylabel("Mean Precision (P)")
plt.title("Mean BERTScore Precision by Model")
plt.tight_layout()
plt.savefig(figs / "out.png", dpi=200)
plt.close()
