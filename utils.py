import json
from pathlib import Path
import datasets
import compare
from collections import defaultdict
import block_match
from llms import openai
import time


def get_papers(level: str):
    if level not in ["paragraph", "section"]:
        raise ValueError(f"{level} not valid")
    papers = datasets.load_dataset(f"annamkiepura99/{level}-diss-gen-combined_v2")
    cited = datasets.load_dataset(f"annamkiepura99/{level}-cited-papers-combined_v2")

    return papers["train"], cited["train"]


def build_discussion_body(discussion):
    d = ""
    for subsection in discussion["subsections"]:
        if subsection["header"] != discussion["header"]:
            d += subsection["header"]
        for paragraph in subsection["paragraphs"]:
            d += paragraph + "\n\n"

    return d, discussion["papers_cited_discussion"]


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


def split_discussion(sections):
    for i, section in enumerate(sections):
        if section["header"].lower() == "discussion":
            discussion = sections.pop(i)
            return sections, discussion
    raise Exception("ALARM")


def get_runs(path):
    return sorted(
        [
            f.name.removesuffix(".json")
            for f in path.iterdir()
            if f.is_file() and f.name.endswith(".json")
        ]
    )
