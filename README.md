# Does Every Instance Count Equally? Order Effects in LLM Multi-Instance Processing

<p align="center">
  <a href="https://arxiv.org/abs/2610.29057">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2610.29057-b31b1b.svg">
  </a>
  <a href="https://huggingface.co/datasets/jingxuanchen916/mip-order">
    <img alt="Hugging Face Dataset" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-yellow">
  </a>
</p>

Earlier work on instance count: [paper](https://aclanthology.org/2026.acl-long.1470/),
[mip-understand](https://huggingface.co/datasets/cjxhwyt/mip-understanding)

In multi-instance processing (MIP), a model reads many instances of a simple task in one prompt
and answers one question about all of them. Here the question is which label is the majority, and
the same instances come in different orders, so any change in the answer comes from the order alone.

## Task

Each prompt is a task description followed by the instances. A shortened example:

```text
Task: Determine the majority sentiment label among all provided instances.
...
Instance 1: a gory slash-fest

Instance 2: this is a heartfelt story ... it just is n't a very involving one .
...
Instance 2000: reactionary ideas about women
```

The expected answer is `{"answer": "positive"}`: 1,100 of the 2,000 instances are positive.

| Task | Each instance | Labels | Source |
|---|---|---|---|
| `math` | a grade-school math problem | the parity of its answer: `even_number` / `odd_number` | [GSM8K](https://huggingface.co/datasets/openai/gsm8k) |
| `ner` | a short text with one entity | `LOC` / `PER` | [WikiANN](https://huggingface.co/datasets/unimelb-nlp/wikiann) |
| `occurrence` | a tweet | the word "the" occurs `exactly_once` / `more_than_once` | [TweetEval](https://github.com/cardiffnlp/tweeteval) |
| `sentiment` | a movie-review phrase | `negative` / `positive` | [SST-2](https://huggingface.co/datasets/stanfordnlp/sst2) |
| `wsd` | a sentence containing "java" | `place` / `programming_language` | [CoarseWSD-20](https://github.com/danlou/bert-disambiguation) |

The orders, with M the majority and m the minority instances:

| `order_pattern` | Layout |
|---|---|
| `random` | shuffled, with `order_seed` 0, 1 or 2 |
| `majority_first`, `majority_last` | all M then all m; all m then all M |
| `majority_edges` | M at both ends, the m block in the middle |
| `majority_middle` | m at both ends, the M block in the middle |
| `soft_majority_edges` | like `majority_edges`, but the middle alternates M, m, M, m, … |
| `soft_majority_middle` | like `majority_middle`, but both ends alternate M, m, M, m, … |

## Data

```python
from datasets import load_dataset

core = load_dataset("cjxhwyt/mip-order", "core", split="test")
generalisability = load_dataset("cjxhwyt/mip-order", "generalisability", split="test")
```

- `core`: 270 prompts of 2,000 instances at 55:45, for every task, 2 majority labels, 9 orders and
  3 random selections of instances.
- `generalisability`: 1,800 prompts with 100 to 2,000 instances at 55:45 to 90:10, one selection.
  Its 55:45, 2,000-instance cell is the core condition at selection seed 1.

| Column | Content |
|---|---|
| `instances` | the instance texts, in prompt order |
| `instance_labels` | each instance's label |
| `majority_label` | the gold answer |
| `order_pattern`, `order_seed` | the order (see above) |
| `task`, `label_distribution`, `n_instances`, `selection_seed`, `setting` | the rest of the condition |
| `instance_answers` | `math`: each problem's answer; `occurrence`: how often "the" occurs; empty otherwise |
| `id` | `<task>/<distribution>_<majority label>_<order>_<n>_seed<selection seed>` |

Rows that share `task`, `selection_seed`, `label_distribution`, `n_instances` and `majority_label`
hold the same instances in the 9 orders.

## Evaluate a model

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY=...   # sent only to OpenRouter; another endpoint reads API_KEY
python run.py --model meta-llama/llama-4-scout --setting core      # or --base-url for e.g. vLLM
python score.py results/core/*/*.jsonl --by task order_pattern
```

`run.py` writes one line per prompt to `results/<setting>/<prompt>/<model>.jsonl` and skips the
rows already there when re-run. The defaults follow the paper: temperature 0, at most 20,000 output
tokens, and up to 3 attempts when no answer can be parsed. A response is correct when its
`"answer"` equals `majority_label`, ignoring case.

`--prompt` picks one of the paper's templates (`prompts.yaml`):

| `--prompt` | In the paper | Difference from `standard` |
|---|---|---|
| `standard` (default) | Standard | — |
| `hint` | Hint | adds that the majority cannot be read off a subset, so every instance must be processed |
| `suggested_strategy` | Suggested Strategy | adds a strategy: classify and count in batches, then sum the counts |
| `self_prompt` | Self-Prompt | two calls: the model first writes its own instruction without seeing the instances, then answers with it |

## Citation
```
@misc{chen2026doesinstancecountequally,
      title={Does Every Instance Count Equally? Order Effects in LLM Multi-Instance Processing}, 
      author={Jingxuan Chen and Mohammad Taher Pilehvar and Jose Camacho-Collados},
      year={2026},
      eprint={2610.29057},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2610.29057}, 
}
```
