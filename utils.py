import json
from pathlib import Path

def write_json(path: str, obj):
    if path.exists():
        try:
            with open(path, "r") as f:
                data = json.load(f)

            if not isinstance(data, list):
                raise ValueError(f"Existing JSON in {path} is not a list")

        except json.JSONDecodeError:
            data = []
    else:
        data = []

    data.append(obj)

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def build_discussion_txt(section, br="\n\n"):
    txt = ""
    for subsection in section["subsections"]:
        paragraphs = subsection["paragraphs"]
        if len(paragraphs):
            txt += br.join(paragraphs)
    return txt


def split_discussion(sections):
    for i, section in enumerate(sections):
        if section["header"].lower() == "discussion":
            discussion = sections.pop(i)
            return sections, discussion
        

def get_runs(path):
    return sorted([
        f.name.removesuffix(".json")
        for f in path.iterdir()
        if f.is_file() and f.name.endswith(".json")
    ])