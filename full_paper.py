from llms import openai
from dotenv import load_dotenv
import utils
from tqdm import tqdm
from pathlib import Path
import time

experiment = "full_paper"

base = Path("papers")

experiment_out = base / experiment
experiment_out.mkdir(exist_ok=True, parents=True)

data_out = Path("out") / experiment
data_out.mkdir(exist_ok=True, parents=True)

N = 20

load_dotenv()

papers, cited_papers = utils.get_papers(level="section")


def build_paper_body_sans_discussion(sections):
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


def append_cited_papers_to_paper_body(s, matches, skip=["figure", "table"]):
    s += "\n\n" + "CITED_PAPERS:" + "\n"
    for match in matches:
        s += "\n\n" + match["corpus_id"] + ":" + match["title"] + "\n\n"
        for section in match["sections"]:
            if section["header"].lower() in skip:
                break
            if section["header"] != "X":
                s += "\n" + section["header"] + "\n\n"
            for paragraph in section["paragraphs"]:
                s += paragraph + "\n"

    return s


def run_prompting(corpus_id, s):
    for model in openai.OpenAIWrapper.available_models:
        out_path = experiment_out / f"{corpus_id}_discussion_{model}.txt"
        if out_path.exists():
            print(f"{out_path} exists, skipping...")
            continue

        llm = openai.OpenAIWrapper(model_name=model)
        print(f"prompting {model} for corpus_id:{corpus_id}")
        try:
            output = llm.prompt(
                "You are given a paper with the discussion part removed. Your job is to write a discussion based on the scientific text given to you."
                "The cited papers are found after the CITED_PAPERS magic string:"
                + "\n\n"
                + s
            )
        except Exception as e:
            print(e)
            output = ""
        with open(out_path, "w") as f:
            f.write(output)


for i in tqdm(range(min(N, len(papers["sections"])))):
    corpus_id = papers["corpus_id"][i]
    sections, discussion_section = utils.split_discussion(papers["sections"][i])

    # s_out_path = out / f"{corpus_id}.txt"
    d_out_path = base / "discussion" / f"{corpus_id}_discussion.txt"

    if not d_out_path.exists():
        d, papers_cited_discussion = utils.build_discussion_body(discussion_section)
        with open(d_out_path, "w") as f:
            f.write(d)

    s = build_paper_body_sans_discussion(sections)

    cited_ids = [citation["matched_paper_id"] for citation in papers["citations"][i]]

    matches = cited_papers.filter(lambda x: x["corpus_id"] in cited_ids)

    s = append_cited_papers_to_paper_body(s, matches)

    # with open(s_out_path, "w") as f:
    #     f.write(s)

    run_prompting(corpus_id, s)
