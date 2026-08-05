#!/usr/bin/env python
"""Build prompt files for the fSRD depth-segmentation experiment (spec step 1 prep).

Implements the data-preparation half of INVESTIGATION.md section 5, step 1:

- ``prompts_wt2.txt``          : 128 WikiText-2 (raw, train) paragraphs, each of
                                 which tokenizes to >= 256 tokens under the
                                 Qwen3.5-0.8B tokenizer.
- ``prompts_wt2_holdout.txt``  : 20 further WikiText-2 paragraphs, disjoint from
                                 the 128 above, for the step-6 token-time
                                 forecast control (held out of every ridge fit).
- ``prompts_ts.txt``           : 64 TinyStories (validation) stories, each
                                 >= 256 tokens, as the domain contrast.

Sources (verified in INVESTIGATION.md):
  WikiText-2 raw train parquet from Salesforce/wikitext,
  TinyStories validation from roneneldan/TinyStories.
Both are fetched with the installed ``datasets`` library; nothing is written
outside this directory.
"""
from __future__ import annotations

import os

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(os.path.join(HERE, "models", "Qwen3.5-0.8B"))

    def long_enough(text: str) -> bool:
        return len(tok(text, truncation=False)["input_ids"]) >= 256

    # --- WikiText-2 raw, train split -------------------------------------
    wt2 = load_dataset(
        "parquet",
        data_files="https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/train-00000-of-00001.parquet",
        split="train",
    )
    picked: list[str] = []
    need = 128 + 20
    for row in wt2:
        text = row["text"].strip()
        # Paragraph-like: real prose lines, not section headers ("= ... =").
        if len(text) < 800 or text.startswith("="):
            continue
        text = " ".join(text.split())  # one prompt per line in the output file
        if long_enough(text):
            picked.append(text)
        if len(picked) >= need:
            break
    if len(picked) < need:
        raise SystemExit(f"only found {len(picked)} long WikiText-2 paragraphs, need {need}")

    with open(os.path.join(HERE, "prompts_wt2.txt"), "w") as fh:
        fh.write("\n".join(picked[:128]) + "\n")
    with open(os.path.join(HERE, "prompts_wt2_holdout.txt"), "w") as fh:
        fh.write("\n".join(picked[128:need]) + "\n")

    # --- TinyStories, validation split ------------------------------------
    ts = load_dataset("roneneldan/TinyStories", split="validation")
    stories: list[str] = []
    for row in ts:
        text = " ".join(row["text"].split())
        if len(text) < 1000:
            continue
        if long_enough(text):
            stories.append(text)
        if len(stories) >= 64:
            break
    if len(stories) < 64:
        raise SystemExit(f"only found {len(stories)} long TinyStories, need 64")

    with open(os.path.join(HERE, "prompts_ts.txt"), "w") as fh:
        fh.write("\n".join(stories) + "\n")

    print(f"wrote prompts_wt2.txt (128), prompts_wt2_holdout.txt (20), prompts_ts.txt (64)")


if __name__ == "__main__":
    main()
