"""Offline tests for the native fixed-benchmark entry point."""

import asyncio
import hashlib
import json
import sqlite3
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from examples.searchqa import benchmark, run
from examples.searchqa.bird import database_path, evaluate_sql, extract_sql
from src.evaluation.evaluate import EvalResult
from src.loop.config import LoopConfig
from src.loop.runner import SelfImprovingLoop


@pytest.fixture
def database(tmp_path):
    path = database_path(tmp_path, "cities")
    path.parent.mkdir()
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE cities (city TEXT)")
        db.executemany("INSERT INTO cities VALUES (?)", [("New York",), ("Lima",), ("Lima",)])
    return path


def bird_row(i):
    return {"id": str(i), "db_id": "cities", "prompt": "Schema and task " + str(i) + "x" * 7000,
            "SQL": "SELECT city FROM cities WHERE city = 'New York'"}


@pytest.mark.parametrize("task", ["triviaqa", "hotpotqa", "bird"])
def test_check_data_uses_matching_manifest_and_resources(tmp_path, monkeypatch, task):
    for name in ("train", "val", "test"):
        d = tmp_path / name
        d.mkdir()
        row = bird_row(name) if task == "bird" else {
            "id": name, "context": "context", "question": "question", "answers": ["one", "alias"],
        }
        (d / "items.json").write_text(json.dumps([row]))
    monkeypatch.setitem(benchmark.FIXED_SPLITS, task, {
        n: (1, hashlib.sha256(n.encode()).hexdigest()) for n in ("train", "val", "test")
    })
    monkeypatch.setenv(f"{task.upper()}_SPLIT_DIR", str(tmp_path))
    monkeypatch.setenv("SEARCHQA_SPLIT_DIR", str(tmp_path / "wrong"))
    monkeypatch.setattr(sys, "argv", ["run.py", "--check-data", "--benchmark", task, "--db-root", str(tmp_path)])
    if task == "bird":
        with pytest.raises(SystemExit):
            run.main()
        db = database_path(tmp_path, "cities")
        db.parent.mkdir()
        with sqlite3.connect(db) as conn:
            conn.execute("CREATE TABLE cities(city TEXT)")
    run.main()
    bad = json.loads((tmp_path / "val/items.json").read_text())
    bad[0]["id"] = "replacement"
    (tmp_path / "val/items.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="IDs or order"):
        run.main()


def test_sql_extraction_and_execution(database, tmp_path):
    sql = "SELECT 'it''s;a';"
    assert extract_sql("notes response ```sql\n" + sql + " SELECT 2;\n```") == sql
    gold = "SELECT city FROM cities ORDER BY city DESC"
    assert evaluate_sql("SELECT DISTINCT city FROM cities ORDER BY city", gold, database)["ex"] == 1
    assert evaluate_sql("SELECT 'new york'", "SELECT 'New York'", database)["ex"] == 0
    assert evaluate_sql("SELECT invalid", gold, database)["exec_status"] == "error"
    assert evaluate_sql("DELETE FROM cities", gold, database)["exec_status"] == "error"
    assert evaluate_sql(f"ATTACH DATABASE '{tmp_path}/outside.db' AS extra", gold, database)["exec_status"] == "error"
    assert not (tmp_path / "outside.db").exists()
    heavy = "WITH RECURSIVE cnt(x) AS (VALUES(1) UNION ALL SELECT x+1 FROM cnt) SELECT sum(x) FROM cnt"
    assert evaluate_sql("SELECT 1", heavy, database, timeout=0.001)["exec_status"] == "timeout"


def test_bird_test_denominator_includes_wrong_sql_and_agent_errors(database, tmp_path, monkeypatch):
    import src.evaluation

    answers = ["SELECT 'New York'", "SELECT 'no match'", "SELECT unknown", None]
    items = [bird_row(i) for i in range(len(answers))]

    async def replies(agent, pairs, max_concurrent):
        return [EvalResult(question=q, ground_truth=g, trace=None if answer is None else
                           SimpleNamespace(output=SimpleNamespace(final_answer=answer)))
                for (q, g), answer in zip(pairs, answers, strict=True)]

    monkeypatch.setattr(src.evaluation, "evaluate_agent_parallel", replies)
    summary = asyncio.run(run.evaluate_test(object(), items, 24, tmp_path, "bird", tmp_path))
    assert summary == {"n": 4, "ex": 0.25}
    results = [json.loads(line) for line in (tmp_path / "test_results.jsonl").read_text().splitlines()]
    assert [r["exec_status"] for r in results] == ["ok", "ok", "error", "agent_error"]
    assert [r["agent_ok"] for r in results] == [True, True, True, False]


def test_native_loop_preserves_case_at_train_and_dev(database, tmp_path):
    from src.schemas import AgentResponse

    current = Mock()
    current.cwd = tmp_path
    current.list_programs.return_value = ["base"]
    current.get_frontier.return_value = ["base"]
    current.get_frontier_with_scores.return_value = [("base", 1)]
    current.get_best_from_frontier.return_value = "base"
    current.select_from_frontier.return_value = "base"
    train = bird_row("train")
    dev = bird_row("dev")
    prediction = "SELECT city FROM cities WHERE city = 'new york'"
    seen = []

    async def answer(question):
        seen.append(question)
        return SimpleNamespace(output=SimpleNamespace(final_answer=prediction), total_cost_usd=0)

    def score(question, pred, gold):
        assert pred == prediction
        assert json.loads(gold)["SQL"] == train["SQL"]
        return run.score_prediction(pred, gold, "bird", tmp_path)["ex"]

    agent = SimpleNamespace(run=answer, response_model=AgentResponse, _get_options=lambda: {"model": "offline"})
    loop = SelfImprovingLoop(LoopConfig(max_iterations=1, cache_enabled=False), SimpleNamespace(base=agent),
                             current, {"default": run.encode_items([train], "bird")},
                             [(q, g, "default") for q, g in run.encode_items([dev], "bird")],
                             scorer=score, normalize_training_answers=False)
    loop._mutate_with_fallback = AsyncMock(return_value=None)
    asyncio.run(loop.run())
    current.update_frontier.assert_called_once_with("base", 0.0, max_size=3)
    loop._mutate_with_fallback.assert_awaited_once()
    assert seen == [dev["prompt"], train["prompt"]]


@pytest.mark.parametrize("task", ["triviaqa", "hotpotqa", "bird"])
def test_full_run_selects_native_scorer_and_writes_test(task, tmp_path, database, monkeypatch):
    from src import harness, loop, registry
    from src.loop.runner import LoopResult
    import src.evaluation

    work = tmp_path / "work"
    splits = {k: [bird_row(k)] if task == "bird" else [
        {"id": k, "question": k + "?", "context": "alias", "answers": ["main", "alias"]}
    ] for k in ("train", "val", "test")}
    monkeypatch.setattr(sys, "argv", ["run.py", "--benchmark", task, "--db-root", str(tmp_path),
                                      "--work-dir", str(work), "--model", "offline"])
    monkeypatch.setattr(run, "load_splits", lambda _, selected: splits if selected == task else {})
    monkeypatch.setattr(run, "initialize_workspace", lambda path, _: (path / ".evoskill").mkdir(parents=True))
    monkeypatch.setattr(registry, "ProgramManager", Mock())
    monkeypatch.setattr(harness, "set_sdk", Mock())
    agents = SimpleNamespace(base=object())
    monkeypatch.setattr(run, "make_agents", lambda *args: agents)
    engine = Mock()
    engine.return_value.run = AsyncMock(return_value=LoopResult([("best", 1)], "best", 1, 7))
    monkeypatch.setattr(loop, "SelfImprovingLoop", engine)
    answer = "SELECT 'New York'" if task == "bird" else "<answer>alias</answer>"

    async def replies(agent, pairs, max_concurrent):
        assert agent is agents.base
        question, gold = pairs[0]
        if task == "bird":
            assert question == splits["test"][0]["prompt"] and 'SELECT' not in question
        trace = SimpleNamespace(output=SimpleNamespace(final_answer=answer))
        return [EvalResult(question=question, ground_truth=gold, trace=trace)]

    monkeypatch.setattr(src.evaluation, "evaluate_agent_parallel", replies)
    run.main()
    q, g = engine.call_args.args[3]["default"][0]
    assert engine.call_args.kwargs["scorer"](q, answer, g) == 1
    assert engine.call_args.kwargs["normalize_training_answers"] is (task != "bird")
    report = json.loads((work / ".evoskill/test_summary.json").read_text())
    row = json.loads((work / ".evoskill/test_results.jsonl").read_text())
    assert row["id"] == "test" and row["agent_ok"]
    if task == "bird":
        assert report["test"] == {"n": 1, "ex": 1} and row["ex"] == row["hard"] == row["soft"] == 1
    else:
        assert report["test"] == {"n": 1, "em": 1, "f1": 1}
