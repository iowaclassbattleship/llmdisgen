import json
import load_dataset

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", choices=["section", "paragraph"], required=True)
    args = parser.parse_args()

    papers, _ = load_dataset.get_papers(level=args.level)

    fields = set()
    for row in papers["s2fieldsofstud"]:
        for field in row:
            fields.add(field)

    fields = sorted(fields)
    print(json.dumps(fields, indent=2))
