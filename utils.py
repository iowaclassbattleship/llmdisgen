import json
from pathlib import Path

def write_json(filename: Path, obj):
    with open(filename, "w") as f:
        json.dump(obj, f, indent=2)


def log_run(corpus_id, ollama_model, scorer_model):
    print("Running:")
    print(f"corpus_id: {corpus_id}")
    print(f"ollama model: {ollama_model}")
    print(f"comparison model: {scorer_model}")