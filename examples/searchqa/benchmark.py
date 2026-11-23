"""SearchQA input and scoring for the fixed EvoSkill benchmark."""

from __future__ import annotations

import hashlib
import json
import os
import re
import string
from collections import Counter
from pathlib import Path

# SHA-256 of the released IDs in split order, joined by LF without a trailing LF.
SPLITS = {
    "train": (400, "cded018312f19f27e1da3e019931fb4d18fbbdbc4fd6d1651b5a52f29dedd268"),
    "val": (200, "d2913aff1ba10bd9a577e79c24121108218b30136c0d5a7002eed36133babafe"),
    "test": (1400, "7fca98256e96c837eee2612da9b9d29fb9d4e5cb908c751ca9bfdfd2d10ab6fb"),
}


def load_splits(split_dir=None):
    default = Path(__file__).resolve().parents[3] / "SkillOpt" / "data" / "searchqa_split"
    data = Path(split_dir or os.environ.get("SEARCHQA_SPLIT_DIR") or default)
    splits = {}
    for name, (size, expected) in SPLITS.items():
        path = data / name / "items.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list) or len(rows) != size:
            raise ValueError(f"{path} must contain {size} SearchQA examples")
        ids = []
        for row in rows:
            if not isinstance(row, dict) or not all(
                isinstance(row.get(key), str) and row[key].strip() for key in ("id", "question", "context")
            ):
                raise ValueError(f"{path} contains a row without an ID, question or context")
            answers = row.get("answers")
            if not isinstance(answers, list) or not answers or not all(isinstance(g, str) for g in answers):
                raise ValueError(f"{path}: invalid reference answers for {row['id']}")
            ids.append(row["id"])
        digest = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
        if digest != expected:
            raise ValueError(f"{path}: IDs or order differ from the released {name} split")
        splits[name] = rows
    return splits


def format_question(row):
    context = row["context"]
    if len(context) > 6000:
        prefix = ""
        for doc in context.split("[DOC]"):
            candidate = prefix + "[DOC]" + doc if prefix else doc
            if len(candidate) > 6000:
                break
            prefix = candidate
        context = prefix or (context[:6000] + "\n...[truncated]")
    return f"## Context\n{context}\n\n## Question\n{row['question']}"


def _normalize(text):
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def _token_f1(pred_tokens, gold_tokens):
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    if not common:
        return 0.0
    precision, recall = common / len(pred_tokens), common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def score_response(response, answers):
    matches = re.findall(r"<answer>(.*?)</answer>", response, flags=re.IGNORECASE | re.DOTALL)
    if matches:
        answer = matches[-1].strip()
    else:
        answer = next((line.strip() for line in reversed(response.splitlines()) if line.strip()), "")
    prediction = _normalize(answer)
    golds = [_normalize(gold) for gold in answers]
    tokens = prediction.split()
    return {
        "predicted_answer": answer,
        "em": float(prediction in golds),
        "f1": max((_token_f1(tokens, gold.split()) for gold in golds), default=0.0),
    }
