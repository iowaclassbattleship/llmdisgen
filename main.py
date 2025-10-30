from datasets import load_dataset
from llama import Proompter
from compare import TextComparator

model_types = [
    "bert-base-uncased",
    "microsoft/deberta-xlarge-mnli",
    "microsoft/deberta-large-mnli"
]

ds = load_dataset("annamkiepura99/sentence-cited-papers-combined")

sample = ds["train"][0]

lines = []
discussion = ""
for section in sample["sections"]:
    if "discussion" not in section["header"].lower():
        lines.append(section["header"] + "\n")
        
        for paragraph in section["paragraphs"]:
            lines.append(paragraph + "\n\n")
    else:
        discussion = "Discussion\n" + "\n".join(section["paragraphs"])

p = Proompter()
prompt = "Reformulate this: " + discussion
llm_discussion = p.proompt(prompt)

with open("llm.txt", "w") as f:
    f.write(llm_discussion)

with open("prompt.txt", "w") as f:
    f.write(prompt)

# c = TextComparator(model_type=model_types[1])
# c.score(discussion, llm_discussion)