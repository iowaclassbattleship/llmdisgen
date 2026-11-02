import requests
import utils
import json
from pathlib import Path

MODELFILE = Path("data/models.json")


def get_available_models():
    response = requests.get(
        url="https://openrouter.ai/api/v1/models", headers=utils.headers()
    )
    response.raise_for_status()

    result = response.json()["data"]

    return [
        {
            "id": x["id"],
            "name": x["name"],
            "context_length": x.get("context_length"),
            "pricing": x.get("pricing")
        }
        for x in result
    ]


def refresh():
    MODELFILE.parent.mkdir(exist_ok=True)
    with open(MODELFILE, "w") as f:
        json.dump(get_available_models(), f, indent=2)


def get_model_config(model_id):
    if not MODELFILE.exists():
        refresh()
    with open(MODELFILE) as f:
        models = json.load(f)
    for model in models:
        if model["id"] == model_id:
            return model
    raise ValueError(f"model_id {model_id} does not exist")


if __name__ == "__main__":
    refresh()
