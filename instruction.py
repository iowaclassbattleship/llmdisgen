from pathlib import Path
import json

BASE = """You are a scientific writing assistant. You will be given a scientific paper and the abstracts (or full text) of papers it cites, each with a unique ID.

Your task is to write the discussion section for the paper.

Rules:
- Write in the style of a scientific discussion section: confident, precise, and analytical
- Integrate findings from the cited abstracts where genuinely relevant
- When a specific claim, contrast, or comparison draws on a cited paper, embed its ID inline using double curly braces, e.g. {{12345678}}
- Do NOT cite a paper unless the sentence meaningfully relies on it
- Do NOT cite every sentence — most claims from the focal paper itself need no citation
- Do NOT add references sections, headers, or any formatting — output only the discussion text
"""

CONTEXT_INSTRUCTIONS = {
    "small": (
        "You are given the abstract of a scientific paper followed by "
        "the abstracts of the papers cited in it.\n\n"
        "Write the scientific discussion for this paper.\n"
    ),
    "medium": (
        "You are given a scientific paper with the discussion section omitted, "
        "followed by the abstracts of the papers cited in it.\n\n"
        "Write the scientific discussion for this paper.\n"
    ),
    "large": (
        "You are given a scientific paper with the discussion section omitted, "
        "followed by the full text of the papers cited in it.\n\n"
        "Write the scientific discussion for this paper.\n"
    ),
}

PARAGRAPH_INSTRUCTION = (
    "You are given a scientific paper with the discussion section omitted, "
    "followed by the abstracts of the papers cited in this paragraph.\n\n"
    "Write exactly one paragraph of the discussion section. "
    "If preceding discussion paragraphs are provided, continue directly from them "
    "without repeating content.\n"
)

data_path = Path("data")


def wrap(s: str, key: str) -> str:
    return f"<{key}>\n{s}\n</{key}>"


def few_shot_examples(fos: list[str], N: int = 1, level: str = "section") -> str:
    fos_set = set(fos)
    b = "You are given a few examples of what is expected of you.\n\n"

    fname = "few_shot_paragraph_examples.json" if level == "paragraph" else "few_shot_examples.json"
    with open(data_path / fname) as f:
        examples_by_fos = json.load(f)

    for field_of_study in fos_set:
        examples = examples_by_fos.get(field_of_study)
        if examples is None:
            continue
        for example in examples[:N]:
            b += "\n" + wrap(example, "example")

    return b


def instruction_by_context_size(context_size: str) -> str:
    instruction = CONTEXT_INSTRUCTIONS.get(context_size)
    if instruction is None:
        raise ValueError(f"context_size '{context_size}' is invalid.")
    return instruction


def build(
    paper_body: str,
    level: str,
    context_size: str,
    prompt_strategy: str,
    fields_of_study: list[str],
    preceding_paragraphs: str | None = None,
) -> str:
    ctx_instruction = PARAGRAPH_INSTRUCTION if level == "paragraph" else instruction_by_context_size(context_size)
    parts = [BASE, ctx_instruction]

    if prompt_strategy == "few":
        parts.append(few_shot_examples(fields_of_study, level=level))

    parts.append(paper_body)

    if preceding_paragraphs:
        parts.append(wrap(preceding_paragraphs, "preceding_paragraphs"))

    return "\n".join(parts)
