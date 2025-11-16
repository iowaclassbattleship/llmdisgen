from datasets import load_dataset
import llms.llama2 as llama2
import llms.llama3 as llama3
import llms.mistral as mistral
import compare
import utils
import re
import torch
from tqdm import tqdm
import metric

levels = ["sentence", "section"]
level = 1

ds = load_dataset(f"annamkiepura99/{levels[level]}-diss-gen-combined")
papers = ds["train"]
# cited_papers = load_dataset(f"annamkiepura99/{levels[level]}-cited-papers-combined")

modelconfigs = [
    (
        llama2.Llama2LLM,
        llama2.available_models
    ),
    (
        llama3.Llama3LLM,
        llama3.available_models
    ),
]

def get_accuracy_scores(d, dp):
    accuracy_scores = []
    for model_type in compare.available_models:
        c = compare.TextComparator(model_type)
        P, R, F1 = c.score(d, dp)
        accuracy_scores.append({
            "model": c.model_name,
            "P": P.item(),
            "R": R.item(),
            "F1": F1.item()
        })

    return accuracy_scores

for i in tqdm(range(5), desc=f"Processing papers"):
    sections, discussion = utils.split_discussion(papers["sections"][i])
    d = utils.build_discussion_txt(discussion)

    evaluations = []
    for llm_interface, models in modelconfigs:
        for model in models:
            w = llm_interface(model)

            prompt = f"Write the discussion based on the following abstract: {papers["abstract"][i]}"

            dp = w.prompt(prompt)
            ollama_model_name = w.model_name

            del w
            torch.cuda.empty_cache()

            accuracy_scores = get_accuracy_scores(d, dp)
            evaluations.append({
                "model": ollama_model_name,
                "discussion": {
                    "header": "Discussion",
                    "subsections": [
                        {
                            "header": "Discussion",
                            "paragraphs": dp.split("\n\n")
                        }
                    ]
                },
                "accuracy_scores": accuracy_scores,
                "P_mean": metric.bagged_score(accuracy_scores)
            })

    cited_paper_ids = re.findall(utils.pattern, d)
    citations = papers["citations"][i]

    utils.write_json({
        "corpus_id": papers["corpus_id"][i],
        "title": papers["title"][i],
        "externalids": papers["externalids"][i],
        "year": papers["year"][i],
        "level": levels[level],
        "sections": sections,
        "discussion": discussion,
        "discussion_txt": d,
        "evaluations": evaluations,
    })