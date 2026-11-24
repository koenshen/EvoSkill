"""Run native EvoSkill on a fixed QA or BIRD train/dev/test split."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

EVOSKILL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EVOSKILL_ROOT))
from examples.searchqa.benchmark import FIXED_SPLITS, format_question, load_splits, score_response
from examples.searchqa.bird import DEFAULT_DB_ROOT, database_path, evaluate_sql


TASK = (
    "Answer the question using only its supplied context. Apply relevant project "
    "skills. Put the short answer in final_answer."
)
CONSTRAINTS = "Develop reusable text skills for answering from the supplied snippets."
BIRD_TASK = (
    "Write the SQLite query requested in the supplied task. Apply relevant project "
    "skills. Return the SQL in final_answer, preserving literal values and case."
)
BIRD_CONSTRAINTS = "Develop reusable text skills for generating SQL from the supplied schema and evidence."


def encode_items(rows, benchmark="searchqa"):
    # The native agent only receives the first string of each pair. The second
    # holds the reference for the scorer and the optimizer's failure feedback.
    if benchmark == "bird":
        return [(row["prompt"], json.dumps({"db_id": row["db_id"], "SQL": row["SQL"]}, ensure_ascii=False))
                for row in rows]
    return [(format_question(row), json.dumps(row["answers"], ensure_ascii=False))
            for row in rows]


def initialize_workspace(path, benchmark="searchqa"):
    path.mkdir(parents=True, exist_ok=False)
    (path / ".evoskill").mkdir()
    (path / ".gitignore").write_text(".evoskill/\n.cache/\n", encoding="utf-8")
    # The two generic seeds also ship in the native OfficeQA example.
    shutil.copytree(EVOSKILL_ROOT / ".claude" / "skills", path / ".claude" / "skills")
    for args in (["init", "-q", "-b", "main"], ["add", ".gitignore", ".claude/skills"],
                 ["commit", "-qm", f"Initialize EvoSkill {benchmark} run"]):
        subprocess.run(["git", *args], cwd=path, check=True)


def gateway_environment(workspace, model):
    """Configure the existing OpenCode harness inside this run's workspace."""
    base_url = os.environ.get("OPENAI_COMPATIBLE_BASE_URL")
    if not model or not base_url or not os.environ.get("OPENAI_COMPATIBLE_API_KEY"):
        raise ValueError("Set OPENAI_COMPATIBLE_MODEL, OPENAI_COMPATIBLE_BASE_URL and OPENAI_COMPATIBLE_API_KEY")
    models = {}
    for role in ("target", "optimizer"):
        name = f"{role.upper()}_OPENAI_COMPATIBLE_MAX_TOKENS"
        budget = int(os.environ[name])
        if budget < 1:
            raise ValueError(f"{name} must be positive")
        # OpenCode uses context=0 for an unknown context window.
        models[role] = {
            "id": model, "tool_call": True, "limit": {"context": 0, "output": budget},
            "interleaved": {"field": "reasoning_content"},
        }
    provider = {
        "npm": "@ai-sdk/openai-compatible", "env": ["OPENAI_COMPATIBLE_API_KEY"],
        "options": {"baseURL": base_url.rstrip("/")},
        "models": models,
    }
    config = {
        "enabled_providers": ["evoskill"], "provider": {"evoskill": provider},
        "model": "evoskill/optimizer", "small_model": "evoskill/optimizer",
        "skills": {"paths": [str(workspace / ".claude" / "skills")]},
        "snapshot": False,  # The native loop already checkpoints candidates in Git.
        # Session title generation is UI bookkeeping and would bill extra requests.
        "agent": {"title": {"disable": True}},
    }
    state = workspace / ".evoskill" / "opencode"
    return {
        "OPENCODE_CONFIG_CONTENT": json.dumps(config),
        "OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX": str(max(m["limit"]["output"] for m in models.values())),
        "OPENCODE_DISABLE_EXTERNAL_SKILLS": "true",
        "OPENCODE_DISABLE_MODELS_FETCH": "true",
        "XDG_CONFIG_HOME": str(state / "config"), "XDG_DATA_HOME": str(state / "data"),
        "XDG_STATE_HOME": str(state / "state"), "XDG_CACHE_HOME": str(state / "cache"),
    }


def make_agents(workspace, harness, server_env=None, task=TASK):
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
    base_factory = make_base_agent_options_from_task(task, **kwargs)

    def answer_options():
        options = base_factory()
        if server_env is not None:
            return gateway_options(options, "target", {"skill": True})
        # allowed_tools controls approval. tools controls which tools exist in the session.
        options.tools = ["Skill", "StructuredOutput"]
        options.allowed_tools = list(options.tools)
        options.setting_sources = ["project"]
        options.extra_args["strict-mcp-config"] = None
        return options

    def gateway_options(options, role, tools):
        return {**options, "server_env": server_env, "provider_id": "evoskill",
                "model_id": role, "model": harness.model, "tools": tools}

    def optimizer(factory, schema):
        options = factory(**kwargs)
        if server_env is not None:
            options = gateway_options(options, "optimizer", options["tools"])
        return Agent(options, schema, **agent_kwargs)

    return LoopAgents(
        base=Agent(answer_options, AgentResponse, **agent_kwargs),
        skill_proposer=optimizer(make_skill_proposer_options, SkillProposerResponse),
        skill_generator=optimizer(make_skill_generator_options, ToolGeneratorResponse),
        prompt_proposer=optimizer(make_prompt_proposer_options, PromptProposerResponse),
        prompt_generator=optimizer(make_prompt_generator_options, PromptGeneratorResponse),
    )


def score_prediction(response, gold, benchmark, db_root):
    reference = json.loads(gold)
    if benchmark == "bird":
        return evaluate_sql(response, reference["SQL"], database_path(db_root, reference["db_id"]))
    return score_response(response, reference)


async def evaluate_test(agent, rows, concurrency, out_dir, benchmark="searchqa", db_root=None):
    from src.evaluation import evaluate_agent_parallel

    pairs = encode_items(rows, benchmark)
    results = await evaluate_agent_parallel(agent, pairs, max_concurrent=concurrency)
    records = []
    for row, (question, gold), result in zip(rows, pairs, results, strict=True):
        if (result.question, result.ground_truth) != (question, gold):
            raise ValueError(f"Test result order differs at {row['id']}")
        ok = result.trace is not None and result.trace.output is not None
        answer = result.trace.output.final_answer if ok else ""
        scores = score_prediction(answer or "", gold, benchmark, db_root)
        record = {"id": row["id"], "response": answer, "agent_ok": ok}
        if benchmark == "bird":
            record.update(db_id=row["db_id"], gold_sql=row["SQL"], **scores)
            if not ok:
                record.update(exec_status="agent_error", error="No structured agent answer")
            record.update(hard=record["ex"], soft=float(record["ex"]))
        else:
            record.update(em=scores["em"] if ok else 0.0, f1=scores["f1"] if ok else 0.0)
        records.append(record)
    with (out_dir / "test_results.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    if not any(record["agent_ok"] for record in records):
        raise RuntimeError("All test requests failed before producing an answer")
    metrics = ("ex",) if benchmark == "bird" else ("em", "f1")
    return {"n": len(records), **{k: sum(r[k] for r in records) / len(records) for k in metrics}}


async def run(args, splits):
    from src.cli.config import EvolutionConfig, HarnessConfig
    from src.harness import set_sdk
    harness = HarnessConfig()
    if args.model:
        harness.model = args.model
    workspace = args.work_dir.expanduser().resolve()
    server_env = None
    if args.harness == "openai-compatible":
        harness.name = "opencode"
        harness.model = args.model or os.environ.get("OPENAI_COMPATIBLE_MODEL")
        server_env = gateway_environment(workspace, harness.model)
    set_sdk(harness.name)
    from src.loop import LoopConfig, SelfImprovingLoop
    from src.loop.dev_gate import DevGate
    from src.registry import ProgramManager

    evolution = EvolutionConfig()
    config = LoopConfig(
        max_iterations=evolution.iterations, frontier_size=evolution.frontier_size,
        no_improvement_limit=evolution.no_improvement_limit, concurrency=args.workers or evolution.concurrency,
        evolution_mode=evolution.mode, failure_sample_count=evolution.failure_samples,
        categories_per_batch=evolution.failure_samples, cache_dir=workspace / ".cache" / "runs",
    )
    benchmark = args.benchmark
    db_root = args.db_root
    is_bird = benchmark == "bird"
    initialize_workspace(workspace, benchmark)
    manager = ProgramManager(cwd=workspace)
    agents = make_agents(workspace, harness, server_env, BIRD_TASK if is_bird else TASK)
    train = {"default": encode_items(splits["train"], benchmark)}
    dev = [(question, gold, "default") for question, gold in encode_items(splits["val"], benchmark)]

    metric = "ex" if is_bird else "em"

    def score(question, predicted, gold):
        return score_prediction(predicted or "", gold, benchmark, db_root)[metric]

    dev_gate = None
    if args.dev_early_stop:
        dev_gate = DevGate([row["id"] for row in splits["val"]], args.dev_order, args.dev_batch_size,
                           workspace / ".evoskill" / "dev_evaluations.json")
    loop = SelfImprovingLoop(config, agents, manager, train, dev, scorer=score,
                             task_constraints=BIRD_CONSTRAINTS if is_bird else CONSTRAINTS,
                             normalize_training_answers=not is_bird, dev_gate=dev_gate)
    result = await loop.run()
    out_dir = workspace / ".evoskill"
    (out_dir / "train_result.json").write_text(json.dumps(asdict(result), indent=2) + "\n")
    manager.switch_to(result.best_program)
    test_metrics = await evaluate_test(agents.base, splits["test"],
                                       config.concurrency, out_dir, benchmark, db_root)
    report = {"model": harness.model, "best_program": result.best_program,
              f"dev_{metric}": result.best_score, "test": test_metrics}
    if benchmark != "searchqa":
        report["benchmark"] = benchmark
    (out_dir / "test_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=FIXED_SPLITS, default="searchqa")
    parser.add_argument("--work-dir", type=Path, help="New directory for the isolated run")
    parser.add_argument("--split-dir", type=Path, help="Directory of existing fixed splits for the selected benchmark")
    parser.add_argument("--db-root", type=Path, default=Path(os.getenv("BIRD_DB_ROOT", DEFAULT_DB_ROOT)),
                        help="BIRD database root containing <db_id>/<db_id>.sqlite")
    parser.add_argument("--harness", choices=["claude", "openai-compatible"], default="claude")
    parser.add_argument("--model", help="Set the deployment for every model role")
    parser.add_argument("--workers", type=int, default=os.environ.get("SKILLOPT_WORKERS"),
                        help="Concurrent validation/test answers (default: SKILLOPT_WORKERS or native 4)")
    parser.add_argument("--check-data", action="store_true", help="Validate the split without running agents")
    parser.add_argument("--dev-early-stop", action="store_true",
                        help="Certify dev rejection when a candidate cannot enter the full frontier")
    parser.add_argument("--dev-order", choices=("hard-first", "easy-first"), default="hard-first",
                        help="Order candidates' dev items by historical mean EM")
    parser.add_argument("--dev-batch-size", type=int, default=4,
                        help="New dev items submitted before checking the rejection bound")
    args = parser.parse_args()
    if args.workers is not None and args.workers < 1:
        parser.error("--workers must be positive")
    if args.dev_batch_size < 1:
        parser.error("--dev-batch-size must be positive")
    if args.dev_early_stop and args.benchmark != "searchqa":
        parser.error("--dev-early-stop is supported for SearchQA only")
    if not args.check_data and args.work_dir is None:
        parser.error("--work-dir is required for a run")
    splits = load_splits(args.split_dir, args.benchmark)
    if args.benchmark == "bird":
        db_ids = {row["db_id"] for split in splits.values() for row in split}
        for db_id in sorted(db_ids):
            path = database_path(args.db_root, db_id)
            if not path.is_file():
                parser.error(f"BIRD database not found: {path}")
    if args.check_data:
        print(" ".join(f"{name}={len(rows)}" for name, rows in splits.items()))
        return
    asyncio.run(run(args, splits))


if __name__ == "__main__":
    main()
