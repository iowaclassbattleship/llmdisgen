from datasets import load_dataset
import llms.llama2 as llama2
import llms.llama3 as llama3
import llms.mistral as mistral
import compare.BERTScore as BERTScore
import compare.BLEU as BLEU
import compare.BlockMatch as BlockMatch
import utils
import re
import torch
from tqdm import tqdm
import metric

levels = ["sentence", "section"]
versions = ["", "_v2"]
level = 1
version = 1

ds = load_dataset(f"annamkiepura99/{levels[level]}-diss-gen-combined{versions[version]}")
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
    (
        mistral.MistralLLM,
        mistral.available_models
    )
]

def evaluate_section_level(d, llm, abstract: str, br="\n\n"):
    prompt = f"Write the discussion based on the following abstract: {abstract}"
    dp = llm.prompt(
        user_prompt=prompt,
        system_prompt=llm.system_prompt
    )
    ollama_model_name = llm.model_name

    del llm
    torch.cuda.empty_cache()

    accuracy_scores = []
    BERT = BERTScore.BERTScore(model_type=BERTScore.available_models[0])
    BM = BlockMatch.BlockMatch()
    P, R, F1 = BM.metric(dp.split("\n\n"), d.split(br), BERT.metric)
    accuracy_scores.append({
        "method": "BERTScore",
        "result": P
    })

    return {
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
    }

def evaluate_n_papers(n, level):
    for i in tqdm(range(n), desc=f"Processing papers"):
        sections, discussion = utils.split_discussion(papers["sections"][i])
        d = utils.build_discussion_txt(discussion)

        evaluations = []
        for llm_interface, models in modelconfigs:
            for model in models:
                llm = llm_interface(model)

                if level == "paragraph":
                    raise NotImplementedError
                elif level == "section":
                    evaluations.append(evaluate_section_level(d, llm, papers["abstract"][i]))

        cited_paper_ids = re.findall(utils.pattern, d)
        citations = papers["citations"][i]

        utils.write_json({
            "corpus_id": papers["corpus_id"][i],
            "title": papers["title"][i],
            "externalids": papers["externalids"][i],
            "year": papers["year"][i],
            "level": level,
            "sections": sections,
            "discussion": discussion,
            "discussion_txt": d,
            "evaluations": evaluations,
        })

if __name__ == "__main__":
    evaluate_n_papers(3, "section")