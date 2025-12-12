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


def evaluation_structure(model_name, paragraphs):
    return {
        "model": model_name,
        "discussion": {
            "header": "Discussion",
            "subsections": [
                {
                    "header": "Discussion",
                    "paragraphs": paragraphs
                }
            ]
        },
    }


def generate_db_paragraph_level(d, llm, excerpt: str, br="\n\n"):
    nparagraphs = len([p for p in [s for s in d["subsections"]]])

    prompt = (
        f"You are writing a paragraph-level scientific discussion based on the following manuscript: {excerpt}\n\n"
        "Here are the paragraphs you wrote so far:\n\n"
    )

    dp = []
    
    for _ in range(nparagraphs):
        r = llm.prompt(
            user_prompt=prompt,
            system_prompt=llm.system_prompt
        )

        dp.append(r)

        prompt += r + "\n\n"

    return "".join(dp)


def generate_dp_section_level(d, llm, excerpt: str, br="\n\n"):
    prompt = f"Write the section-level discussion based on the following scientific excerpt: {excerpt}"
    dp = llm.prompt(
        user_prompt=prompt,
        system_prompt=llm.system_prompt
    )

    return evaluation_structure(llm.model_name, dp.split(br))
    

def generate_dp_for_n_papers(n, level):
    filename = f"{int(time.time())}.json"
    path = Path("runs") / level / "raw"
    path.mkdir(exist_ok=True, parents=True)

    fs = {
        "paragraph": generate_db_paragraph_level,
        "section": generate_dp_section_level
    }

    papers = get_papers(level)

    for i in tqdm(range(n), desc=f"Processing papers"):
        excerpt = papers["abstract"][i]
        sections, d = utils.split_discussion(papers["sections"][i])

        evaluations = []
        for llm_interface in llm_interfaces:
            for model in llm_interface.available_models:
                llm = llm_interface(model)
                evaluations.append(
                    fs[level](d, llm, excerpt)
                )

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
    generate_dp_for_n_papers(10, level="paragraph")
