from pathlib import Path
from dotenv import load_dotenv
import utils
from tqdm import tqdm
from llms import openai

experiment = "abstract_only"

base = Path("papers")

out = base / experiment
out.mkdir(exist_ok=True, parents=True)

data_out = Path("out") / experiment
data_out.mkdir(exist_ok=True, parents=True)

N = 2

load_dotenv()

papers, cited_papers = utils.get_papers(level="section")


def run_prompting(corpus_id, s):
    for model in tqdm(openai.OpenAIWrapper.available_models):
        out.mkdir(exist_ok=True, parents=True)
        out_path = out / f"{corpus_id}_discussion_{model}.txt"
        if out_path.exists():
            continue

        llm = openai.OpenAIWrapper(model_name=model)
        try:
            output = llm.prompt(
                "You are given the abstract for a scientific paper. Write the discussion section for this paper based on the abstract and the abstracts of all the papers cited in this manuscript."
                "The cited paper abstracts are found after the CITED_PAPERS magic string:"
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
    abstract = papers["abstract"][i]
    _, discussion_section = utils.split_discussion(papers["sections"][i])

    cited_ids = [citation["matched_paper_id"] for citation in papers["citations"][i]]

    matches = cited_papers.filter(lambda x: x["corpus_id"] in cited_ids)

    cited_paper_abstracts = []
    prompt = abstract + "\n\n" + "CITED_PAPERS:"
    for match in matches:
        prompt += "\n\n" + match["title"] + "\n\n" + match["abstract"]

    run_prompting(corpus_id, prompt)
print("All done!")