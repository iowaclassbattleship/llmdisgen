import datasets

# sourced from https://huggingface.co/datasets/annamkiepura99
def get_papers_only(level: str):
    """Load only the small papers dataset (fast, ~seconds)."""
    if level not in ["paragraph", "section"]:
        raise ValueError(f"{level} not valid")
    suffix = "" if level == "paragraph" else "_v2"
    return datasets.load_dataset(f"annamkiepura99/{level}-diss-gen-combined{suffix}")["train"]


def get_cited_papers(level: str):
    """Load the large cited-papers dataset (slow, ~minutes)."""
    if level not in ["paragraph", "section"]:
        raise ValueError(f"{level} not valid")
    suffix = "" if level == "paragraph" else "_v2"
    cited_level = "paragraphs" if level == "paragraph" else level
    return datasets.load_dataset(f"annamkiepura99/{cited_level}-cited-papers-combined{suffix}")["train"]


def get_papers(level: str):
    return get_papers_only(level), get_cited_papers(level)


# sourced from https://huggingface.co/datasets/jessicalamjh
def get_papers_post_2026():
    papers = datasets.load_dataset(f"jessicalamjh/discussion-generation")
    return papers["train"]
