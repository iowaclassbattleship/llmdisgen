import json
from pathlib import Path
import time

out = Path("runs")
out.mkdir(exist_ok=True)

filename = f"metadata-{int(time.time())}.json"

def write_json(obj):
    out_path = out / filename

    if out_path.exists():
        try:
            with open(out_path, "r") as f:
                data = json.load(f)

            if not isinstance(data, list):
                raise ValueError(f"Existing JSON in {out_path} is not a list")

        except json.JSONDecodeError:
            data = []
    else:
        data = []

    data.append(obj)

    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)


def log_run(corpus_id, ollama_model, scorer_model):
    print("Running:")
    print(f"corpus_id: {corpus_id}")
    print(f"ollama model: {ollama_model}")
    print(f"comparison model: {scorer_model}")