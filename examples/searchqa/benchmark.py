"""Fixed benchmark manifests and open-book QA input/scoring for EvoSkill."""

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

FIXED_SPLITS = {
    "searchqa": SPLITS,
    "triviaqa": {
        "train": (1600, "f465cbaea5820ac4a01a9d0f5d6dc6b71668a78bf21cf783c687e3f181ac3445"),
        "val": (200, "c2b8957c0814fc54e381821f2efb6af200bae1e26605304ca3bac7c10747176c"),
        "test": (1000, "34c26854b2b9348c90fb740675bf45562937c1525f5a770824474e0b6df6397c"),
    },
    "hotpotqa": {
        "train": (1600, "925adec60c7d1488b925b5c5facaaeb9b0f2d60b155928db809fde778433a6d9"),
        "val": (200, "9022dabfa5ddd7a7df3a32eee1a8abbab8963ea807013f257dbefb1250d3d82b"),
        "test": (1000, "7b81664c3206ab4a9d4e3bd274a7bbe7b7d09976c74918970a02ecbb463b7439"),
    },
    "bird": {
        "train": (834, "bf49ab424f7627116e654bb8719fdee322031a35da199aa04b6c370ae0c281b3"),
        "val": (200, "2b3ab585858e168cb05ab1cf2ce8ceb4ecd24a4ae2d375d89ba9d94a362b19d1"),
        "test": (500, "706fcd80ba2577502ed013f49f22ceac5071af0dbfe58adb6fe22ee030b2487f"),
    },
}

def load_splits(split_dir=None, benchmark="searchqa"):
    default = Path(__file__).resolve().parents[3] / "SkillOpt" / "data" / f"{benchmark}_split"
    data = Path(split_dir or os.environ.get(f"{benchmark.upper()}_SPLIT_DIR") or default)
    splits = {}
    fields = ("id", "prompt", "db_id", "SQL") if benchmark == "bird" else ("id", "question", "context")
    for name, (size, expected) in FIXED_SPLITS[benchmark].items():
        path = data / name / "items.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list) or len(rows) != size:
            raise ValueError(f"{path} must contain {size} {benchmark} examples")
        ids = []
        for row in rows:
            if not isinstance(row, dict) or not all(
                isinstance(row.get(key), str) and row[key].strip() for key in fields
            ):
                raise ValueError(f"{path}: expected nonempty text for {fields}")
            if benchmark != "bird":
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
