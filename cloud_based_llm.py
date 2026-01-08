from llms import openai
from dotenv import load_dotenv
import utils
from tqdm import tqdm
from pathlib import Path
import compare
import block_match
from collections import defaultdict
import json

out = Path("papers")
out.mkdir(exist_ok=True)

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


def build_discussion_body(discussion):
    d = ""
    for subsection in discussion["subsections"]:
        if subsection["header"] != discussion["header"]:
            d += subsection["header"]
        for paragraph in subsection["paragraphs"]:
            d += paragraph + "\n\n"

    return d, discussion["papers_cited_discussion"]


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
        out_path = out / "openai" / f"{corpus_id}_discussion_{model}.txt"
        if out_path.exists():
            print(f"{corpus_id}:{model} exists, skipping...")
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


def run_evaluations():
    evres = defaultdict(lambda: defaultdict(list))
    inp = out / "discussion"
    discussions = [f for f in inp.iterdir()]

    BERT = compare.BERTScore(model_type="bert-base-uncased")

    for discussion in tqdm(discussions):
        with open(discussion, "r") as f:
            d = f.read()
        corpus_id = discussion.name.split("_")[0]
        for model_family in ["openai"]:
            for model_generated_d in (out / model_family).iterdir():
                if model_generated_d.stat().st_size == 0:
                    continue
                if corpus_id in model_generated_d.name:
                    with open(model_generated_d, "r") as f:
                        dp = f.read()
                    P, R, F1 = block_match.metric(dp, d, BERT.metric)
                    evres[corpus_id][model_generated_d.stem.split("_")[-1]].append(
                        {"P": P, "R": R, "F1": F1}
                    )
    with open("out.json", "w") as f:
        json.dump(evres, f, indent=2)


for i in tqdm(range(min(N, len(papers["sections"])))):
    # sections sans discussion
    sections, discussion_section = utils.split_discussion(papers["sections"][i])

    s_out_path = out / f"{papers['corpus_id'][i]}.txt"
    d_out_path = out / "discussion" / f"{papers["corpus_id"][i]}_discussion.txt"

    if not d_out_path.exists():
        d, papers_cited_discussion = build_discussion_body(discussion_section)
        with open(d_out_path, "w") as f:
            f.write(d)

    s = build_paper_body_sans_discussion(sections)

    cited_ids = [citation["matched_paper_id"] for citation in papers["citations"][i]]

    matches = cited_papers.filter(lambda x: x["corpus_id"] in cited_ids)

    s = append_cited_papers_to_paper_body(s, matches)

    with open(s_out_path, "w") as f:
        f.write(s)

    run_prompting(papers["corpus_id"][i], s)
run_evaluations()
