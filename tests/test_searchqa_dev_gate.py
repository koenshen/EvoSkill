"""SearchQA dev admission and its interaction with native EvoSkill refinement."""

import asyncio
import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from examples.searchqa import run as searchqa
from src.loop.config import LoopConfig
from src.loop.dev_gate import DevGate
from src.loop.runner import SelfImprovingLoop


@pytest.mark.parametrize("order,first", [("hard-first", [1, 3]), ("easy-first", [0, 2])])
def test_bounds_order_and_history(tmp_path, order, first):
    ids = ["a", "b", "c", "d"]
    items = list(range(4))
    journal = tmp_path / "dev.json"
    gate = DevGate(ids, order, 2, journal)
    calls = []

    async def assess(label, scores, cutoff=None):
        async def score_batch(batch):
            calls.append(batch)
            return [scores[i] for i in batch]
        return await gate.evaluate(label, items, score_batch, cutoff=cutoff)

    ref = asyncio.run(assess("base", [1, 0, 1, 0]))
    assert calls == [items] and not ref.stopped and ref.lower == ref.upper == 0.5
    calls.clear()
    tie = asyncio.run(assess("loser", [0] * 4, 0.5))
    assert calls == [first] and tie.stopped and tie.upper == 0.5 and tie.lower == 0
    assert list(tie.observations) == [ids[i] for i in first]
    gate = DevGate(ids, order, 2, journal)
    gate.start(resume=True)
    calls.clear()
    asyncio.run(assess("loser", [0] * 4, 0.5))
    assert calls == [first] and set(gate.evaluations) == {"base", "loser"}
    assert json.loads(journal.read_text())["evaluations"]["loser"]["n_evaluated"] == 2
    calls.clear()
    winner = asyncio.run(assess("winner", [1] * 4, 0.5))
    assert len(calls) == 2 and not winner.stopped and winner.lower == winner.upper == 1
    with pytest.raises(ValueError, match="does not match"):
        DevGate(ids[::-1], order, 2, journal).start(resume=True)


def test_full_frontier_rejection_uses_bound_only(tmp_path):
    ids = [f"id-{i}" for i in range(8)]
    dev = [(str(i), "yes", "default") for i in range(8)]
    # With a full frontier, .625 beats the .50 floor even though its parent scored .75.
    correct = {
        "base": {0, 1, 2, 3}, "child-1": {0, 1, 2, 3, 4, 5},
        "child-2": set(), "child-3": {0, 1, 2, 3, 4},
    }
    current = "base"
    retained = {}
    evaluations = []

    def switch(name):
        nonlocal current
        current = name

    def update(name, score, max_size):
        if len(retained) >= max_size:
            worst = min(retained, key=retained.get)
            assert score > retained[worst]
            del retained[worst]
        retained[name] = score
        return True

    manager = Mock(cwd=tmp_path)
    manager.list_programs.return_value = ["base"]
    manager.switch_to.side_effect = switch
    manager.update_frontier.side_effect = update
    manager.get_frontier_with_scores.side_effect = lambda: sorted(retained.items(), key=lambda p: -p[1])
    manager.get_frontier.side_effect = lambda: list(retained)
    manager.get_best_from_frontier.return_value = "child-1"
    manager.select_from_frontier.side_effect = lambda *_: max(retained, key=retained.get)

    async def answer(question):
        if question == "training":
            answer = "no"
        else:
            evaluations.append((current, question))
            answer = "yes" if int(question) in correct[current] else "no"
        return SimpleNamespace(output=SimpleNamespace(final_answer=answer), total_cost_usd=0)

    async def mutate(parent, failures, iteration):
        switch(f"child-{iteration}")
        return current, "proposal", "reason"

    events = []
    gate = DevGate(ids, "hard-first", 4, tmp_path / ".evoskill/dev_evaluations.json")
    loop = SelfImprovingLoop(
        LoopConfig(max_iterations=3, frontier_size=2, cache_enabled=False),
        SimpleNamespace(base=SimpleNamespace(run=answer, _get_options=lambda: {"model": "offline"})), manager,
        {"default": [("training", "yes")]}, dev, scorer=lambda q,a,g: float(a == g),
        on_event=lambda kind,data: events.append((kind,data)), dev_gate=gate,
    )
    loop._mutate_with_fallback = mutate
    result = asyncio.run(loop.run())
    assert result.iterations_completed == 3 and result.best_score == 0.75
    assert manager.update_frontier.call_count == 3 and retained == {"child-1": 0.75, "child-3": 0.625}
    manager.discard.assert_called_once_with("child-2")
    assert len(evaluations) == 8 + 8 + 4 + 8
    partial, = [data for kind,data in events if kind == "dev_early_reject"]
    assert partial["score"] is None and partial["score_bounds"] == (0, 0.5)
    feedback = (tmp_path / ".evoskill/feedback_history.md").read_text().split("## child-2")[1]
    assert "DISCARDED" in feedback and "exact score unknown" in feedback and "+0.0000" not in feedback


@pytest.mark.parametrize("flags,expected", [
    ([], (False, "hard-first", 4)),
    (["--dev-early-stop"], (True, "hard-first", 4)),
    (["--dev-early-stop", "--dev-order", "easy-first", "--dev-batch-size", "2"], (True, "easy-first", 2)),
])
def test_searchqa_cli_defaults(monkeypatch, tmp_path, flags, expected):
    monkeypatch.setattr(sys, "argv", ["run.py", "--work-dir", str(tmp_path), *flags])
    monkeypatch.setattr(searchqa, "load_splits", lambda *args: {})
    job = AsyncMock()
    monkeypatch.setattr(searchqa, "run", job)
    searchqa.main()
    args = job.await_args.args[0]
    assert (args.dev_early_stop, args.dev_order, args.dev_batch_size) == expected


@pytest.mark.parametrize("enabled", [False, True])
def test_searchqa_run_gates_only_dev(monkeypatch, tmp_path, enabled):
    from src import harness, loop, registry
    from src.loop.runner import LoopResult

    args = SimpleNamespace(
        harness="claude", model="offline", workers=24, work_dir=tmp_path,
        dev_early_stop=enabled, dev_order="easy-first", dev_batch_size=4,
        benchmark="searchqa", db_root=None,
    )
    splits = {
        split: [
            {"id": f"{split}-{i}", "question": str(i), "context": "context", "answers": ["one"]}
            for i in range(n)
        ]
        for split, n in [("train", 400), ("val", 200), ("test", 1400)]
    }
    monkeypatch.setattr(harness, "set_sdk", Mock())
    monkeypatch.setattr(searchqa, "initialize_workspace", lambda path, _: (path / ".evoskill").mkdir())
    monkeypatch.setattr(registry, "ProgramManager", Mock())
    agents = SimpleNamespace(base=object())
    monkeypatch.setattr(searchqa, "make_agents", lambda *args: agents)
    engine = Mock()
    engine.return_value.run = AsyncMock(return_value=LoopResult([("best", 1)], "best", 1, 7))
    monkeypatch.setattr(loop, "SelfImprovingLoop", engine)
    test = AsyncMock(return_value={"n": 1400})
    monkeypatch.setattr(searchqa, "evaluate_test", test)
    asyncio.run(searchqa.run(args, splits))
    options = engine.call_args.kwargs["dev_gate"]
    if enabled:
        assert options.ids == [r["id"] for r in splits["val"]]
        assert options.order == "easy-first" and options.batch_size == 4
    else:
        assert options is None
    test.assert_awaited_once_with(agents.base, splits["test"], 24, tmp_path / ".evoskill", "searchqa", None)
