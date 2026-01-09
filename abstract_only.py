from pathlib import Path
from dotenv import load_dotenv
import utils
from tqdm import tqdm
from llms import openai
from collections import defaultdict
import block_match
from compare import BERTScore
import json

experiment = "abstract_only"

base = Path("papers")

out = base / experiment
out.mkdir(exist_ok=True, parents=True)

data_out = Path("out") / experiment
data_out.mkdir(exist_ok=True, parents=True)

N = 1

load_dotenv()

papers, cited_papers = utils.get_papers(level="section")

def run_prompting(corpus_id, s):
    for model in openai.OpenAIWrapper.available_models:
        out_dir = out / "openai"
        out_dir.mkdir(exist_ok=True, parents=True)
        out_path = out_dir / f"{corpus_id}_discussion_{model}.txt"
        if out_path.exists():
            print(f"{corpus_id}:{model} exists, skipping...")
            continue

        llm = openai.OpenAIWrapper(model_name=model)
        print(f"prompting {model} for corpus_id:{corpus_id}")
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

evres = defaultdict(lambda: defaultdict(list))
discussions_path = base / "discussion"
ai_discussions_path = out / "openai"
BERT = BERTScore(model_type="bert-base-uncased")
for discussion_txt in discussions_path.iterdir():
    with open(discussion_txt, "r") as f:
        d = f.read()
    corpus_id = discussion_txt.stem.split("_")[0]

    for model in openai.OpenAIWrapper.available_models:
        path = ai_discussions_path / f"{corpus_id}_discussion_{model}.txt"
        if path.exists():
            with open(path, "r") as f:
                dp = f.read()
            P, R, F1 = block_match.metric(dp, d, BERT.metric)
            evres[corpus_id][model].append(
                {"P": P, "R": R, "F1": F1}
            )
with open(data_out / "out.json", "w") as f:
    json.dump(evres, f, indent=2)