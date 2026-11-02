import requests
from dotenv import load_dotenv
import utils
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import instruction
import get_models
import config
import load_dataset
import argparse

load_dotenv()

OUT_DIRECTORY = Path("000_out")
MAX_THREAD_WORKERS = 4


def prompt(prompt: str, model: str):
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=utils.headers(),
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        },
    )

    response.raise_for_status()
    data = response.json()

    return data


def write_human_discussion(corpus_id, d):
    out = OUT_DIRECTORY / "discussions"
    out.mkdir(parents=True, exist_ok=True)

    filepath = out / f"{corpus_id}.md"

    if not filepath.exists():
        with open(filepath, "w") as f:
            f.write(d)


_cited_index_cache = {}


def _get_cited_index(cited_papers):
    """Map corpus_id -> row position, built once per dataset (memoized).

    Reads only the corpus_id column (one fast pass) so per-paper lookups can use
    Dataset.select() (O(k)) instead of Dataset.filter() (full scan of ~669k rows).
    """
    key = getattr(cited_papers, "_fingerprint", id(cited_papers))
    index = _cited_index_cache.get(key)
    if index is None:
        index = {cid: i for i, cid in enumerate(cited_papers["corpus_id"])}
        _cited_index_cache[key] = index
    return index


def _select_cited(cited_papers, wanted_ids):
    """Return the subset of cited_papers whose corpus_id is in wanted_ids."""
    index = _get_cited_index(cited_papers)
    positions = [index[c] for c in wanted_ids if c in index]
    return cited_papers.select(positions)


def process_paper(papers, paper_idx, cited_papers, context_size):
    corpus_id = papers["corpus_id"][paper_idx]
    title = papers["title"][paper_idx]
    sections, d_ = utils.split_discussion(papers["sections"][paper_idx])
    write_human_discussion(corpus_id, utils.build_discussion_body(d_))

    cited_ids = set(
        citation["matched_paper_id"] for citation in papers["citations"][paper_idx]
    )
    matches = _select_cited(cited_papers, cited_ids)

    s = ""
    if context_size == "small":
        abstract = corpus_id + " - " + title + "\n"
        abstract += papers["abstract"][paper_idx] + "\n"
        s += instruction.wrap(abstract, "abstract")
        for a in utils.get_cited_abstracts(matches):
            cited_abstract = "\n" + a["corpus_id"] + " - " + a["title"] + "\n"
            cited_abstract += a["abstract"]
            s += "\n" + instruction.wrap(cited_abstract, "cited_abstract")
    elif context_size == "medium":
        paper_body = corpus_id + " - " + title + "\n"
        paper_body += utils.combine_sections_to_string(sections)
        s += instruction.wrap(paper_body, "paper")
        for a in utils.get_cited_abstracts(matches):
            cited_abstract = a["corpus_id"] + " - " + a["title"] + "\n"
            cited_abstract += a["abstract"]
            s += "\n" + instruction.wrap(cited_abstract, "cited_abstract")
    elif context_size == "large":
        paper_body = corpus_id + " - " + title + "\n"
        paper_body += utils.combine_sections_to_string(sections)
        s += instruction.wrap(paper_body, "paper")
        s += utils.concat_cited_papers(matches)
    else:
        raise ValueError(f"context_size {context_size} does not exist.")

    return corpus_id, title, s


def process_paper_for_paragraphs(papers, paper_idx, cited_papers, context_size):
    """Prepare paragraph-level generation inputs for one paper.

    Returns (corpus_id, paragraphs, para_cited_ids_list, cited_by_id, paper_part,
    fields_of_study): the ground-truth paragraphs, the cited IDs per paragraph,
    abstracts of all cited papers by ID, and the wrapped paper/abstract text.
    """
    corpus_id = papers["corpus_id"][paper_idx]
    title = papers["title"][paper_idx]
    sections = list(
        papers["sections"][paper_idx]
    )  # split_discussion mutates its argument
    sections, discussion = utils.split_discussion(sections)
    write_human_discussion(corpus_id, utils.build_discussion_body(discussion))

    paragraphs = utils.get_discussion_paragraphs(discussion)
    citations = papers["citations"][paper_idx]
    papers_cited_disc = discussion.get("papers_cited_discussion") or []

    para_cited_ids_list = [
        utils.get_paragraph_cited_ids(para, citations, papers_cited_disc)
        for para in paragraphs
    ]

    all_cited_ids = set(c["matched_paper_id"] for c in citations)
    matches = _select_cited(cited_papers, all_cited_ids)
    cited_by_id = {a["corpus_id"]: a for a in utils.get_cited_abstracts(matches)}

    if context_size == "small":
        abstract = corpus_id + " - " + title + "\n" + papers["abstract"][paper_idx]
        paper_part = instruction.wrap(abstract, "abstract")
    else:  # medium or large
        paper_body = (
            corpus_id
            + " - "
            + title
            + "\n"
            + utils.combine_sections_to_string(sections)
        )
        paper_part = instruction.wrap(paper_body, "paper")

    return (
        corpus_id,
        paragraphs,
        para_cited_ids_list,
        cited_by_id,
        paper_part,
        papers["s2fieldsofstud"][paper_idx],
    )


def process_paper_model(
    corpus_id,
    setname,
    level,
    prompt_strategy,
    context_size,
    prompt_body,
    model,
    fields_of_study,
):
    _, model_name = model["id"].split("/")
    out = (
        OUT_DIRECTORY
        / setname
        / corpus_id
        / level
        / prompt_strategy
        / context_size
        / model_name
    )
    out.mkdir(exist_ok=True, parents=True)
    filename = Path(f"{model_name}.json")

    if (out / filename).exists():
        print(f"paper {corpus_id} model {model['id']} skipped...")
        return

    p = instruction.build(
        prompt_body, level, context_size, prompt_strategy, fields_of_study
    )

    try:
        r = prompt(p, model=model["id"])
        if r is not None:
            try:
                with open(out / f"response.md", "w") as f:
                    f.write(r["choices"][0]["message"]["content"])
            except Exception as e:
                print(e)
            with open(out / f"prompt.md", "w") as f:
                f.write(p)
            with open(out / f"{filename}", "w") as f:
                json.dump(
                    {
                        "model": model["id"],
                        "level": level,
                        "prompt_strategy": prompt_strategy,
                        "context_size": context_size,
                        "response": r,
                    },
                    f,
                    indent=2,
                )
    except Exception as e:
        print(f"Failed paper {corpus_id} model {model['id']}: {e}")


def process_paper_model_paragraph(
    corpus_id,
    setname,
    prompt_strategy,
    context_size,
    paper_part,
    paragraphs,
    para_cited_ids_list,
    cited_by_id,
    model,
    fields_of_study,
):
    """Generate paragraphs sequentially, passing earlier generated ones as context.
    Paragraphs with a saved response are loaded instead of regenerated.
    """
    _, model_name = model["id"].split("/")
    out_base = (
        OUT_DIRECTORY
        / setname
        / corpus_id
        / "paragraph"
        / prompt_strategy
        / context_size
        / model_name
    )

    generated = []

    for para_idx, (_, cited_ids) in enumerate(zip(paragraphs, para_cited_ids_list)):
        para_out = out_base / f"para_{para_idx}"
        para_out.mkdir(exist_ok=True, parents=True)
        filename = f"{model_name}.json"

        if (para_out / filename).exists():
            print(f"paper {corpus_id} para {para_idx} model {model['id']} skipped...")
            try:
                with open(para_out / "response.md") as f:
                    generated.append(f.read())
            except Exception:
                generated.append("")
            continue

        cited_parts = [
            instruction.wrap(
                a["corpus_id"] + " - " + a["title"] + "\n" + a["abstract"],
                "cited_abstract",
            )
            for cid in cited_ids
            if (a := cited_by_id.get(cid)) is not None
        ]
        prompt_body = paper_part + (
            "\n" + "\n".join(cited_parts) if cited_parts else ""
        )
        preceding = "\n\n".join(generated) if generated else None
        p = instruction.build(
            prompt_body,
            "paragraph",
            context_size,
            prompt_strategy,
            fields_of_study,
            preceding,
        )

        try:
            r = prompt(p, model=model["id"])
            if r is not None:
                content = r["choices"][0]["message"]["content"]
                with open(para_out / "response.md", "w") as f:
                    f.write(content)
                with open(para_out / "prompt.md", "w") as f:
                    f.write(p)
                with open(para_out / filename, "w") as f:
                    json.dump(
                        {
                            "model": model["id"],
                            "level": "paragraph",
                            "para_idx": para_idx,
                            "prompt_strategy": prompt_strategy,
                            "context_size": context_size,
                            "response": r,
                        },
                        f,
                        indent=2,
                    )
                generated.append(content)
        except Exception as e:
            print(f"Failed para {para_idx} paper {corpus_id} model {model['id']}: {e}")
            generated.append("")


def has_pending_work(setname, level, context_size, prompt_strategy, n_papers, papers, id_column="corpus_id"):
    """Return True if at least one output file is missing for this condition."""
    n = min(n_papers, len(papers))
    for idx in range(n):
        corpus_id = papers[id_column][idx]
        for model_id in config.modelset:
            model_name = model_id.split("/")[1]
            out = (OUT_DIRECTORY / setname / corpus_id / level
                   / prompt_strategy / context_size / model_name / f"{model_name}.json")
            if not out.exists():
                return True
    return False


def process_pre_2026(setname, level, context_size, prompt_strategy, n_papers,
                     papers=None, cited_papers=None):
    if papers is None or cited_papers is None:
        papers, cited_papers = load_dataset.get_papers(level=level)

    tasks = []
    with ThreadPoolExecutor(max_workers=MAX_THREAD_WORKERS) as executor:
        for idx in range(min(n_papers, len(papers["sections"]))):
            if level == "section":
                corpus_id, _, prompt_body = process_paper(
                    papers, idx, cited_papers, context_size
                )
                for model_id in config.modelset:
                    model = get_models.get_model_config(model_id)
                    tasks.append(
                        executor.submit(
                            process_paper_model,
                            corpus_id,
                            setname,
                            level,
                            prompt_strategy,
                            context_size,
                            prompt_body,
                            model,
                            papers["s2fieldsofstud"][idx],
                        )
                    )
            else:  # paragraph
                (
                    corpus_id,
                    paragraphs,
                    para_cited_ids_list,
                    cited_by_id,
                    paper_part,
                    fos,
                ) = process_paper_for_paragraphs(
                    papers, idx, cited_papers, context_size
                )
                for model_id in config.modelset:
                    model = get_models.get_model_config(model_id)
                    tasks.append(
                        executor.submit(
                            process_paper_model_paragraph,
                            corpus_id,
                            setname,
                            prompt_strategy,
                            context_size,
                            paper_part,
                            paragraphs,
                            para_cited_ids_list,
                            cited_by_id,
                            model,
                            fos,
                        )
                    )

        for future in tqdm(as_completed(tasks), total=len(tasks)):
            try:
                future.result()
            except Exception as e:
                print(f"Task failed: {e}")


def _flatten_content_tree(items: list) -> str:
    """Recursively flatten content nodes (paragraphs or nested sections) to plain text."""
    parts = []
    for item in items:
        ct = item.get("content_type")
        if ct == "paragraph":
            parts.append(item["text"])
        elif ct == "section":
            if item.get("header"):
                parts.append(item["header"])
            parts.append(_flatten_content_tree(item.get("contents", [])))
    return "\n\n".join(p for p in parts if p)


def _sections_to_string(sections_json) -> str:
    if isinstance(sections_json, str):
        sections_json = json.loads(sections_json)
    return _flatten_content_tree(sections_json)


def _extract_discussion_text(contents: list) -> str:
    return _flatten_content_tree(contents)


def _extract_discussion_paragraphs(contents: list) -> list[str]:
    """Return a flat list of non-empty paragraph texts from a content tree."""
    paras = []
    for item in contents:
        ct = item.get("content_type")
        if ct == "paragraph":
            text = item.get("text", "").strip()
            if text:
                paras.append(text)
        elif ct == "section":
            paras.extend(_extract_discussion_paragraphs(item.get("contents", [])))
    return paras


def process_post_2026(setname, level, context_size, prompt_strategy, n_papers, papers=None):
    if papers is None:
        papers = load_dataset.get_papers_post_2026()

    tasks = []
    with ThreadPoolExecutor(max_workers=MAX_THREAD_WORKERS) as executor:
        for idx in range(min(n_papers, len(papers))):
            row = papers[idx]
            sample_id = row["sample_id"]
            ms = row["manuscript"]
            title = ms["title"]
            fos = ms["subjects"]

            all_done = all(
                (OUT_DIRECTORY / setname / sample_id / level / prompt_strategy / context_size
                 / get_models.get_model_config(m)["id"].split("/")[1]
                 / f"{get_models.get_model_config(m)['id'].split('/')[1]}.json").exists()
                for m in config.modelset
            )
            if all_done:
                print(f"paper {sample_id} skipped (all outputs exist)")
                continue

            gold_disc = json.loads(row["gold_discussion"])
            relevant_papers = json.loads(row["relevant_papers"])
            cited_ids = set(row["bibliography_entry_ids_referenced_in_gold_discussion"])

            discussion_text = _extract_discussion_text(gold_disc["contents"])
            write_human_discussion(sample_id, discussion_text)

            # all relevant papers, not only those cited in the discussion
            cited_refs = relevant_papers

            if context_size == "small":
                abstract_text = sample_id + " - " + title + "\n" + ms["abstract"]
                paper_part = instruction.wrap(abstract_text, "abstract")
            elif context_size in ("medium", "large"):
                paper_body = sample_id + " - " + title + "\n" + _sections_to_string(ms["contents"])
                paper_part = instruction.wrap(paper_body, "paper")
            else:
                raise ValueError(f"context_size '{context_size}' is invalid.")

            if level == "section":
                s = paper_part
                for ref_id, ref in cited_refs.items():
                    if context_size == "large" and ref.get("contents"):
                        cited_body = ref_id + " - " + ref.get("title", "") + "\n" + _sections_to_string(ref["contents"])
                        s += "\n" + instruction.wrap(cited_body, "cited_paper")
                    elif ref.get("abstract"):
                        cited_text = ref_id + " - " + ref.get("title", "") + "\n" + ref["abstract"]
                        s += "\n" + instruction.wrap(cited_text, "cited_abstract")

                for model_id in config.modelset:
                    model = get_models.get_model_config(model_id)
                    tasks.append(
                        executor.submit(
                            process_paper_model,
                            sample_id, setname, level, prompt_strategy, context_size, s, model, fos,
                        )
                    )

            else:  # paragraph
                paragraphs = _extract_discussion_paragraphs(gold_disc["contents"])
                if not paragraphs:
                    print(f"paper {sample_id} skipped (no paragraphs in discussion)")
                    continue

                # the dataset has no per-paragraph citation mapping
                para_cited_ids_list = [list(cited_refs.keys())] * len(paragraphs)

                cited_by_id = {
                    ref_id: {"corpus_id": ref_id, "title": ref.get("title", ""), "abstract": ref.get("abstract", "")}
                    for ref_id, ref in cited_refs.items()
                    if ref.get("abstract")
                }

                for model_id in config.modelset:
                    model = get_models.get_model_config(model_id)
                    tasks.append(
                        executor.submit(
                            process_paper_model_paragraph,
                            sample_id, setname, prompt_strategy, context_size,
                            paper_part, paragraphs, para_cited_ids_list, cited_by_id, model, fos,
                        )
                    )

        for future in tqdm(as_completed(tasks), total=len(tasks)):
            try:
                future.result()
            except Exception as e:
                print(f"Task failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate paper discussions with LLMs."
    )
    parser.add_argument("--setname", required=True)
    parser.add_argument("--level", choices=["section", "paragraph"], required=True)
    parser.add_argument(
        "--context_size", choices=["small", "medium", "large"], required=True
    )
    parser.add_argument("--strategy", choices=["zero", "few"], required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument(
        "--dataset", choices=["annamkiepura99", "jessicalamjh"], required=True
    )
    args = parser.parse_args()

    setname = args.setname
    level = args.level
    context_size = args.context_size
    prompt_strategy = args.strategy
    n_papers = args.n

    print(
        f"dataset={args.dataset}, level={level}, context_size={context_size}, "
        f"prompt_strategy={prompt_strategy}"
    )

    OUT_DIRECTORY.mkdir(exist_ok=True)

    if args.dataset == "annamkiepura99":
        process_pre_2026(setname, level, context_size, prompt_strategy, n_papers)
    else:
        process_post_2026(setname, level, context_size, prompt_strategy, n_papers)
