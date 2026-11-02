import argparse
from pathlib import Path
import tiktoken

enc = tiktoken.get_encoding("o200k_base")


def tokens(text: str):
    return enc.encode(text)


def token_count(text: str):
    return len(tokens(text))


def main(input_dir: Path):
    for corpus_id_dir in input_dir.iterdir():
        for level_dir in corpus_id_dir.iterdir():
            for shot_dir in level_dir.iterdir():
                for size_dir in shot_dir.iterdir():
                    for model_dir in size_dir.iterdir():
                        path = model_dir / "prompt.md"
                        if path.exists():
                            with open(path, "r") as f:
                                prompt = f.read()
                                print(token_count(prompt))
                        else:
                            print(f"{path} does not exist")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("setname")
    args = parser.parse_args()
    main(Path("000_out") / args.setname)
