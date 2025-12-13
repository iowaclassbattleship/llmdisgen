from datasets import load_dataset
from llms import llama2, llama3, mistral
import utils
from tqdm import tqdm
from dotenv import load_dotenv
from pathlib import Path
import time

load_dotenv()

allowed_levels = ["paragraph", "section"]

def get_papers(level: str):
    if level not in allowed_levels:
        raise ValueError(f"{level} not in {allowed_levels}")
    papers = load_dataset(f"annamkiepura99/{level}-diss-gen-combined_v2")
    cited = load_dataset(f"annamkiepura99/{level}-cited-papers-combined_v2")

    return papers["train"], cited["train"]


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


def fn_by_level(level: str):
    if level == "paragraph": return generate_db_paragraph_level
    if level == "section": return generate_dp_section_level
    raise ValueError(f"Level {level} does not exist")
    

def generate_dp_for_n_papers(n, level, model_families):
    filename = f"{int(time.time())}.json"
    path = Path("runs") / level / "raw"
    path.mkdir(exist_ok=True, parents=True)

    papers = get_papers(level)

    for i in tqdm(range(n), desc=f"Processing papers"):
        excerpt = papers["abstract"][i]
        sections, d = utils.split_discussion(papers["sections"][i])

        evaluations = []
        for model_family in model_families:
            for model in model_family.available_models:
                llm = model_family(model)
                evaluations.append(
                    fn_by_level(level)(d, llm, excerpt)
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


def run_abstract_experiment(n, level: str, model_families):
    papers, cited_papers = get_papers(level)

    for i in tqdm(range(n), desc=f"Experiment 1"):
        _, d = utils.split_discussion(papers["sections"][i])
        abstract = papers["abstract"][i]

        cited_abstracts = []

        cited_corpus_ids = utils.get_cited_papers_from_text(abstract)

        for corpus_id in cited_corpus_ids:
            matches = cited_papers.filter(lambda x: x["corpus_id"] == corpus_id)
            if len(matches) > 0:
                cited_abstracts.append("Corpus ID: " + corpus_id + "\n" + matches[0]["abstract"])

        header = "Write a scientific discussion section for the following abstract. If the abstract cites papers these are also in the prompt"
        prompt = "\n\n".join([header] + [abstract] + cited_abstracts)

        llm_outputs = []
        for model_family in model_families:
            for model in model_family.available_models:
                llm = model_family(model)
                llm_outputs.append(
                    fn_by_level(level)(d, llm, prompt)
                )

        path = Path("runs") / "experiments" / "abstract" / level
        path.mkdir(exist_ok=True, parents=True)
        filename = f"{int(time.time())}.json"

        utils.write_json(path / filename, {
            "corpus_id": papers["corpus_id"][i],
            "title": papers["title"][i],
            "externalids": papers["externalids"][i],
            "year": papers["year"][i],
            "experiment": "abstract",
            "level": level,
            "discussion": d,
            "llm_outputs": llm_outputs,
        })
        

if __name__ == "__main__":
    n = 1000
    level = "section"
    model_families = [
        llama2.Llama2,
        llama3.Llama3,
        mistral.Mistral
    ]
    run_abstract_experiment(n, level, model_families)
