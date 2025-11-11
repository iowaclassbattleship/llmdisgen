import json
from pathlib import Path
import time

out = Path("runs")
out.mkdir(exist_ok=True)

def write_json(obj):
    filename = f"metadata-{int(time.time())}.json"
    with open(out / filename, "w") as f:
        json.dump(obj, f, indent=2)


def log_run(corpus_id, ollama_model, scorer_model):
    print("Running:")
    print(f"corpus_id: {corpus_id}")
    print(f"ollama model: {ollama_model}")
    print(f"comparison model: {scorer_model}")