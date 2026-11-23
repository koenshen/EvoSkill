"""Run native EvoSkill on the fixed SkillOpt SearchQA train/dev/test split."""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

EVOSKILL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EVOSKILL_ROOT))
from examples.searchqa.benchmark import format_question, load_splits, score_response


TASK = (
    "Answer the question using only its supplied context. Apply relevant project "
    "skills. Put the short answer in final_answer."
)
CONSTRAINTS = "Develop reusable text skills for answering from the supplied snippets."


def encode_items(rows):
    # The native loop expects a string gold (and lowercases it on training examples).
    # JSON preserves aliases, including embedded commas and Unicode letters.
    return [(format_question(row), json.dumps(row["answers"], ensure_ascii=False))
            for row in rows]


def initialize_workspace(path):
    path.mkdir(parents=True, exist_ok=False)
    (path / ".evoskill").mkdir()
    (path / ".gitignore").write_text(".evoskill/\n.cache/\n", encoding="utf-8")
    # The two generic seeds also ship in the native OfficeQA example.
    shutil.copytree(EVOSKILL_ROOT / ".claude" / "skills", path / ".claude" / "skills")
    for args in (["init", "-q", "-b", "main"], ["add", ".gitignore", ".claude/skills"],
                 ["commit", "-qm", "Initialize EvoSkill SearchQA run"]):
        subprocess.run(["git", *args], cwd=path, check=True)


def make_agents(workspace, harness):
    from src.agent_profiles.base_agent.base_agent import make_base_agent_options_from_task
    from src.agent_profiles.prompt_generator.prompt_generator import make_prompt_generator_options
    from src.agent_profiles.prompt_proposer.prompt_proposer import make_prompt_proposer_options
    from src.agent_profiles.skill_generator.skill_generator import make_skill_generator_options
    from src.agent_profiles.skill_proposer.skill_proposer import make_skill_proposer_options
    from src.harness import Agent
    from src.loop import LoopAgents
    from src.schemas import (AgentResponse, PromptGeneratorResponse, PromptProposerResponse,
                             SkillProposerResponse, ToolGeneratorResponse)

    kwargs = {"project_root": workspace, "model": harness.model}
    agent_kwargs = {"timeout_seconds": harness.timeout_seconds, "max_retries": harness.max_retries}
    base_factory = make_base_agent_options_from_task(TASK, **kwargs)

    def answer_options():
        options = base_factory()
        # allowed_tools controls approval. tools controls which tools exist in the session.
        options.tools = ["Skill", "StructuredOutput"]
        options.allowed_tools = list(options.tools)
        options.setting_sources = ["project"]
        options.extra_args["strict-mcp-config"] = None
        return options

    return LoopAgents(
        base=Agent(answer_options, AgentResponse, **agent_kwargs),
        skill_proposer=Agent(make_skill_proposer_options(**kwargs), SkillProposerResponse, **agent_kwargs),
        skill_generator=Agent(make_skill_generator_options(**kwargs), ToolGeneratorResponse, **agent_kwargs),
        prompt_proposer=Agent(make_prompt_proposer_options(**kwargs), PromptProposerResponse, **agent_kwargs),
        prompt_generator=Agent(make_prompt_generator_options(**kwargs), PromptGeneratorResponse, **agent_kwargs),
    )


async def evaluate_test(agent, rows, concurrency, out_dir):
    from src.evaluation import evaluate_agent_parallel

    pairs = encode_items(rows)
    results = await evaluate_agent_parallel(agent, pairs, max_concurrent=concurrency)
    records = []
    for row, (question, gold), result in zip(rows, pairs, results, strict=True):
        if (result.question, result.ground_truth) != (question, gold):
            raise ValueError(f"Test result order differs at {row['id']}")
        ok = result.trace is not None and result.trace.output is not None
        answer = result.trace.output.final_answer if ok else ""
        scores = score_response(answer or "", row["answers"])
        records.append({"id": row["id"], "response": answer, "agent_ok": ok,
                        "em": scores["em"] if ok else 0.0, "f1": scores["f1"] if ok else 0.0})
    with (out_dir / "test_results.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    if not any(record["agent_ok"] for record in records):
        raise RuntimeError("All test requests failed before producing an answer")
    return {"n": len(records), "em": sum(r["em"] for r in records) / len(records),
            "f1": sum(r["f1"] for r in records) / len(records)}


async def run(args, splits):
    from src.cli.config import EvolutionConfig, HarnessConfig
    from src.harness import set_sdk
    from src.loop import LoopConfig, SelfImprovingLoop
    from src.registry import ProgramManager

    harness = HarnessConfig()
    if args.model:
        harness.model = args.model
    set_sdk(harness.name)
    workspace = args.work_dir.expanduser().resolve()
    evolution = EvolutionConfig()
    config = LoopConfig(
        max_iterations=evolution.iterations, frontier_size=evolution.frontier_size,
        no_improvement_limit=evolution.no_improvement_limit, concurrency=evolution.concurrency,
        evolution_mode=evolution.mode, failure_sample_count=evolution.failure_samples,
        categories_per_batch=evolution.failure_samples, cache_dir=workspace / ".cache" / "runs",
    )
    initialize_workspace(workspace)
    manager = ProgramManager(cwd=workspace)
    agents = make_agents(workspace, harness)
    train = {"default": encode_items(splits["train"])}
    dev = [(question, answers, "default") for question, answers in encode_items(splits["val"])]

    def score_em(question, predicted, gold):
        return score_response(predicted or "", json.loads(gold))["em"]

    loop = SelfImprovingLoop(config, agents, manager, train, dev, scorer=score_em,
                             task_constraints=CONSTRAINTS)
    result = await loop.run()
    out_dir = workspace / ".evoskill"
    (out_dir / "train_result.json").write_text(json.dumps(asdict(result), indent=2) + "\n")
    manager.switch_to(result.best_program)
    test_metrics = await evaluate_test(agents.base, splits["test"],
                                       config.concurrency, out_dir)
    report = {"model": harness.model, "best_program": result.best_program,
              "dev_em": result.best_score, "test": test_metrics}
    (out_dir / "test_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, help="New directory for the isolated run")
    parser.add_argument("--split-dir", type=Path, help="Existing SkillOpt SearchQA payload")
    parser.add_argument("--model", help="Override the default Claude harness model")
    parser.add_argument("--check-data", action="store_true", help="Validate the split without running agents")
    args = parser.parse_args()
    if not args.check_data and args.work_dir is None:
        parser.error("--work-dir is required for a run")
    splits = load_splits(args.split_dir)
    if args.check_data:
        print(" ".join(f"{name}={len(rows)}" for name, rows in splits.items()))
        return
    asyncio.run(run(args, splits))


if __name__ == "__main__":
    main()
