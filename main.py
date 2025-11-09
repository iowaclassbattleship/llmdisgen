from datasets import load_dataset
from llama import OLlama
from compare import TextComparator
import utils
import re
import torch
from itertools import islice
import pandas as pd

model_types = [
    "bert-base-uncased",
    "microsoft/deberta-large-mnli"
]

ollama_models = [
    "unsloth/llama-3-8b-bnb-4bit",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
]

levels = ["sentence", "section"]
level = 1

pattern = r"\{\{(.*?)\}\}"

def build_discussion_txt(sections):
    txt = ""
    for section in sections:
        if section["header"].lower() == "discussion":
            for subsection in section["subsections"]:
                paragraphs = subsection["paragraphs"]
                if len(paragraphs):
                    txt += "".join(subsection["paragraphs"])
    return txt


ds = load_dataset(f"annamkiepura99/{levels[level]}-diss-gen-combined")
papers = ds["train"]
# cited_papers = load_dataset(f"annamkiepura99/{levels[level]}-cited-papers-combined")

metadata = []
for i in range(3):
    corpus_id = papers["corpus_id"][i]
    abstract = papers["abstract"][i]
    discussion_txt = build_discussion_txt(papers["sections"][i])

    evaluations = []
    for ollama_model in ollama_models:
        ollama_wrapper = OLlama(model_name=ollama_model)
        llm_dis = ollama_wrapper.prompt(abstract)

        accuracy_scores = []
        for model_type in model_types:
            c = TextComparator(model_type)
            utils.log_run(corpus_id, ollama_wrapper.model_name, c.model_name)
            P, R, F1 = c.score(discussion_txt, llm_dis)
            accuracy_scores.append({
                "model": c.model_name,
                "P": P.item(),
                "R": R.item(),
                "F1": F1.item()
            })

        evaluations.append({
            "model": ollama_wrapper.model_name,
            "llm_discussion": llm_dis,
            "accuracy_scores": accuracy_scores
        })

        del ollama_wrapper
        torch.cuda.empty_cache()

    metadata.append({
        "corpus_id": corpus_id,
        "level": levels[level],
        "method": "zero-shot",
        "actual_discussion": discussion_txt,
        "evaluations": evaluations,
        "cited_paper_ids": re.findall(pattern, discussion_txt)
    })

utils.write_json("metadata.json", metadata)