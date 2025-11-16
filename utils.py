import json
from pathlib import Path
import time

out = Path("runs")
out.mkdir(exist_ok=True)

filename = f"metadata-{int(time.time())}.json"

# cited paper ids are noted as {{corpus_id}}
pattern = r"\{\{(.*?)\}\}"

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


def build_discussion_txt(section):
    txt = ""
    for subsection in section["subsections"]:
        paragraphs = subsection["paragraphs"]
        if len(paragraphs):
            txt += "".join(subsection["paragraphs"])
    return txt


def split_discussion(sections):
    for i, section in enumerate(sections):
        if section["header"].lower() == "discussion":
            discussion = sections.pop(i)
            return sections, discussion