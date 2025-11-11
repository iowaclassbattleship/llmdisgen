from datasets import load_dataset
import llms.llama as llama
import compare
import utils
import re
import torch
from tqdm import tqdm

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
for i in tqdm(range(10), desc=f"Processing papers"):
    corpus_id = papers["corpus_id"][i]
    abstract = papers["abstract"][i]
    discussion_txt = build_discussion_txt(papers["sections"][i])

    evaluations = []
    for llama_model in llama.available_models:
        llama_wrapper = llama.Llama(model_name=llama_model)

        preprompt = (
            "You are given the abstract for a scientific research paper. Write ONLY a scientific discussion based on this abstract:"
        )

        llm_dis = llama_wrapper.prompt(preprompt + abstract)
        ollama_model_name = llama_wrapper.model_name

        del llama_wrapper
        torch.cuda.empty_cache()

        accuracy_scores = []
        for model_type in compare.available_models:
            c = compare.TextComparator(model_type)
            utils.log_run(corpus_id, ollama_model_name, c.model_name)
            P, R, F1 = c.score(discussion_txt, llm_dis)
            accuracy_scores.append({
                "model": c.model_name,
                "P": P.item(),
                "R": R.item(),
                "F1": F1.item()
            })

        evaluations.append({
            "model": ollama_model_name,
            "llm_discussion": llm_dis.split("\n"),
            "accuracy_scores": accuracy_scores
        })

    metadata.append({
        "corpus_id": corpus_id,
        "level": levels[level],
        "method": "zero-shot",
        "abstract": abstract,
        "discussion": discussion_txt,
        "evaluations": evaluations,
        "cited_paper_ids": re.findall(pattern, discussion_txt)
    })

utils.write_json(metadata)