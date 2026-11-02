# LLMDisGen

Benchmarking large language models on the **generation of scientific Discussion sections**.

Given a manuscript with its Discussion removed, plus the papers it cites, an LLM is asked to write the missing Discussion. The output is then scored against the human-written original using automated similarity metrics. The framework varies three dimensions (prompting strategy, input context size, and model choice) at two granularities: the whole section at once, or paragraph by paragraph.

This repository accompanies a semester thesis. The project description is in `Diss-Gen Student Project.pdf`.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

BERTScore runs `deberta-large-mnli` and `bert-base-uncased` locally via PyTorch, so the first evaluation downloads the model weights (a few GB in total). A GPU is strongly recommended.

All model calls are routed through [OpenRouter](https://openrouter.ai). Create a `.env` file in the repository root:

```
OPENROUTER_API_KEY=your_key_here
```

## Usage

Run a complete study end to end (generate → evaluate → visualize):

```bash
python all.py --setname main --dataset annamkiepura99
python all.py --setname 2026 --dataset jessicalamjh
```

Both commands skip any condition whose outputs already exist, so they are safe to re-run to
fill gaps. Restrict a run to one granularity with `--level section|paragraph`.

Individual stages:

```bash
# Generate one condition
python disgen.py --setname main --level section --context_size medium \
    --strategy zero --n 20 --dataset annamkiepura99

# Score outputs into results.db
python run_evaluation.py main

# Paired t-tests, coverage, best/worst papers
python stats.py main

# Regenerate figures into report/img/{setname}/
python visualization.py --setname main
```

## Experimental dimensions

| Dimension | Values |
|---|---|
| `level` | `section` · `paragraph` |
| `context_size` | `small` (abstract + cited abstracts) · `medium` (full paper, Discussion removed, + cited abstracts) · `large` (full paper, Discussion removed, + full cited texts) |
| `strategy` | `zero` (zero-shot) · `few` (few-shot, one example per field of study) |

Active models are listed in `config.py`.

## Metrics

- **BERTScore** — with `bert-base-uncased` and `microsoft/deberta-large-mnli` backbones
- **BlockMatch** — Hungarian-algorithm matching of 512-token blocks, scored by BERTScore
  ([arXiv:2405.01930](https://arxiv.org/abs/2405.01930)); removes BERTScore's 512-token input cap
- **ROUGE-L** — longest common subsequence, F-measure only

BERTScore and BlockMatch report precision, recall, and F1; ROUGE-L reports only an
F-measure. Analyses use the F1 component.

## Datasets

Both are loaded from HuggingFace at runtime and selected with `--dataset`:

- **`annamkiepura99`** — pre-2026 open-access papers, with cited papers resolved from a
  companion cited-papers corpus.
- **`jessicalamjh/discussion-generation`** — PubMed Central articles published in 2026,
  bundling each paper's cited works directly. Because they were published after the
  models' training data was collected, they allow checking for memorization of the source papers.

## Layout

```
all.py                 Full study: generate → evaluate → visualize
disgen.py              Generation orchestration
load_dataset.py        HuggingFace dataset loading
instruction.py         Prompt construction
run_evaluation.py      Scoring → results.db
stats.py               Significance tests and rankings
estimate_tokens.py     Prompt token counts
cost.py                API cost summary from saved responses
get_models.py          Refresh model catalog from OpenRouter
get_few_shot_examples.py  Build few-shot example pools
get_fields_of_study.py List fields of study in a dataset
visualization.py       Figures → report/img/{setname}/
config.py              Active model list
compare/               Metric implementations
data/                  Few-shot example pools (the model catalog `models.json` is fetched on first use, gitignored)
000_out/               Generated discussions and raw API responses (gitignored)
results.db             SQLite store of all metric scores
report/                Figures and tables (gitignored)
```

Generations are written under
`000_out/{setname}/{corpus_id}/{level}/{strategy}/{context_size}/{model_name}/`
as `prompt.md`, `response.md`, and a JSON file with the full API response. Paragraph-level
runs add a `para_{N}/` level, where each paragraph's prompt includes the previously
generated paragraphs.
