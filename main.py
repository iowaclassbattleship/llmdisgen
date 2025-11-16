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


ds = load_dataset(f"annamkiepura99/{levels[level]}-diss-gen-combined")
papers = ds["train"]
# cited_papers = load_dataset(f"annamkiepura99/{levels[level]}-cited-papers-combined")

for i in tqdm(range(10), desc=f"Processing papers"):
    corpus_id = papers["corpus_id"][i]
    abstract = papers["abstract"][i]
    sections = papers["sections"][i]

    discussion_section = {}
    for i, section in enumerate(sections):
        if section["header"].lower() == "discussion":
            discussion_section = sections.pop(i)
    discussion_txt = utils.build_discussion_txt(discussion_section)

    evaluations = []
    for llama_model in llama.available_models:
        llama_wrapper = llama.Llama(model_name=llama_model)

        llm_dis = llama_wrapper.prompt(abstract)
        ollama_model_name = llama_wrapper.model_name

        del llama_wrapper
        torch.cuda.empty_cache()

        accuracy_scores = []
        for model_type in compare.available_models:
            c = compare.TextComparator(model_type)
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

    cited_paper_ids = re.findall(pattern, discussion_txt)
    citations = papers["citations"][i]

    utils.write_json({
        "corpus_id": corpus_id,
        "title": papers["title"][i],
        "externalids": papers["externalids"][i],
        "year": papers["year"][i],
        "level": levels[level],
        "sections": sections,
        "discussion": discussion_section,
        "discussion_txt": discussion_txt,
        "evaluations": evaluations,
    })