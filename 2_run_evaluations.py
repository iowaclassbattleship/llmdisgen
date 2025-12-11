from pathlib import Path
import json
from compare import BlockMatch, BERTScore
import utils

base = Path("runs")

if __name__ == "__main__":
    level = "section"
    input_path = base / level / "raw"
    if not input_path.exists():
        raise Exception(f"Path {input_path} does not exist")

    output_path = base / level / "scored"
    output_path.mkdir(exist_ok=True)
    
    for file in input_path.iterdir():
        output_file = output_path / file.name
        if output_file.exists():
            print(f"Scored file {output_file} already exists, skipping...")
            continue
    
        with open(file, "r") as f:
            data = json.load(f)
            
        for paper in data:
            d_paragraphs = []
            for subsection in paper["discussion"]["subsections"]:
                d_paragraphs.extend(subsection["paragraphs"])
            for eval in paper["evaluations"]:
                dp_paragraphs = []
                for subsection in eval["discussion"]["subsections"]:
                    dp_paragraphs.extend(subsection["paragraphs"])
                B = BlockMatch()
                BERT = BERTScore(BERTScore.available_models[0])
                P, R, F1 = B.metric(dp_paragraphs, d_paragraphs, BERT.metric)
                eval["accuracy_scores"] = [{
                    "metric": "BERTScore",
                    "score": {
                        "P": P,
                        "R": R,
                        "F1": F1
                    }
                }]

        with open(output_path / file.name, "w") as f:
            json.dump(data, f, indent=2)