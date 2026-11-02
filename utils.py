import json
import datasets
import os
from pathlib import Path
import tiktoken
import instruction


def headers():
    return {
        "Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json",
    }


def build_discussion_body(discussion, ret_papers_cited=False):
    d = ""
    for subsection in discussion["subsections"]:
        if subsection["header"] != discussion["header"]:
            d += subsection["header"]
        for paragraph in subsection["paragraphs"]:
            d += paragraph + "\n\n"

    if ret_papers_cited:
        return d, discussion["papers_cited_discussion"]
    return d


def write_json(path: str | Path, obj):
    if isinstance(path, str):
        path = Path(path)
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
    raise ValueError("no discussion section found")


def get_runs(path):
    return sorted(
        [
            f.name.removesuffix(".json")
            for f in path.iterdir()
            if f.is_file() and f.name.endswith(".json")
        ]
    )


def combine_sections_to_string(sections):
    s = ""
    for section in sections:
        if section["header"].lower() in ["figure", "header"]:
            continue
        if section["header"] != "X":
            s += "\n" + section["header"] + "\n\n"
        for subsection in section["subsections"]:
            if subsection["header"] != section["header"]:
                s += subsection["header"] + "\n\n"
            for paragraph in subsection["paragraphs"]:
                s += paragraph + "\n"
    return s


def get_cited_abstracts(matches):
    return [
        {"corpus_id": corpus_id, "title": title, "abstract": abstract}
        for corpus_id, title, abstract in zip(
            matches["corpus_id"], matches["title"], matches["abstract"]
        )
    ]


def concat_cited_papers(matches):
    a = []
    for match in matches:
        s = "\n"
        for section in match["sections"]:
            s += section["header"]
            s += "\n".join(section["paragraphs"])
        a.append(instruction.wrap(s, "cited_paper"))

    return "".join(a)


_enc = tiktoken.get_encoding("cl100k_base")


def get_token_count(s: str) -> int:
    return len(_enc.encode(s))


def get_discussion_paragraphs(discussion: dict) -> list[str]:
    """Non-empty paragraphs of a discussion section."""
    return [
        para
        for sub in discussion["subsections"]
        for para in sub["paragraphs"]
        if para.strip()
    ]


def get_paragraph_cited_ids(
    para_text: str,
    citations: list[dict],
    papers_cited_discussion: list[str] | None,
) -> set[str]:
    """Papers cited in one paragraph, found by matching citation markers in its text.

    Falls back to all papers cited in the discussion if no marker matches.
    """
    discussion_ids = set(papers_cited_discussion or [])
    matched = {
        c["matched_paper_id"]
        for c in citations
        if c["matched_paper_id"] in discussion_ids
        and c["citation_marker"].strip() in para_text
    }
    return matched if matched else discussion_ids
