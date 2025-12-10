from datasets import load_dataset
import llms
import utils
import torch
from tqdm import tqdm
from dotenv import load_dotenv
from pathlib import Path
import time
from llms import llm_interfaces

load_dotenv()

allowed_levels = ["paragraph", "section"]
versions = ["", "_v2"]
level = 1
version = 1

def get_papers(level, version=0):
    if level not in allowed_levels:
        raise ValueError(f"{level} is not in {allowed_levels}")
    ds = load_dataset(f"annamkiepura99/{level}-diss-gen-combined{versions[version]}")
    papers = ds["train"]
    # cited_papers = load_dataset(f"annamkiepura99/{levels[level]}-cited-papers-combined")

    return papers

def generate_dp(llm, excerpt: str, br="\n\n"):
    prompt = f"Write the discussion based on the following scientific excerpt: {excerpt}"
    dp = llm.prompt(
        user_prompt=prompt,
        system_prompt=llm.system_prompt
    )

    return {
        "model": llm.model_name,
        "discussion": {
            "header": "Discussion",
            "subsections": [
                {
                    "header": "Discussion",
                    "paragraphs": dp.split(br)
                }
            ]
        },
    }

def generate_dp_for_n_papers(n, level):
    filename = f"{level}-{int(time.time())}.json"
    path = Path("runs") / level / "raw"
    path.mkdir(exist_ok=True, parents=True)

    papers = get_papers(level)
    for i in tqdm(range(n), desc=f"Processing papers"):
        sections, d = utils.split_discussion(papers["sections"][i])

        evaluations = []
        for llm_interface in llm_interfaces:
            for model in llm_interface.available_models:
                llm = llm_interface(model)
                evaluations.append(generate_dp(llm, papers["abstract"][i]))

        utils.write_json(path / filename, {
            "corpus_id": papers["corpus_id"][i],
            "title": papers["title"][i],
            "externalids": papers["externalids"][i],
            "year": papers["year"][i],
            "level": level,
            "sections": sections,
            "discussion": d,
            "evaluations": evaluations,
        })

if __name__ == "__main__":
    generate_dp_for_n_papers(2, level="section")