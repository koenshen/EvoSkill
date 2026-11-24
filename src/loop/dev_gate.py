"""Bounded, item-level candidate evaluation for finite binary dev scores."""

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar


Item = TypeVar("Item")


@dataclass
class DevAssessment:
    observations: dict[str, int]
    n_total: int

    @property
    def stopped(self) -> bool:
        return len(self.observations) < self.n_total

    @property
    def lower(self) -> float:
        return sum(self.observations.values()) / self.n_total

    @property
    def upper(self) -> float:
        remaining = self.n_total - len(self.observations)
        return (sum(self.observations.values()) + remaining) / self.n_total


class DevGate:
    def __init__(self, ids: list[str], order: str, batch_size: int, journal: Path):
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("The dev gate requires a nonempty list of unique dev IDs")
        if order not in ("hard-first", "easy-first") or batch_size < 1:
            raise ValueError("Invalid dev ordering or batch size")
        self.ids = list(ids)
        self.order = order
        self.batch_size = batch_size
        self.journal = journal
        self.evaluations: dict[str, dict] = {}

    def start(self, resume: bool) -> None:
        self.evaluations = {}
        if resume and self.journal.is_file():
            state = json.loads(self.journal.read_text(encoding="utf-8"))
            if (state["ids"], state["order"], state["batch_size"]) != (self.ids, self.order, self.batch_size):
                raise ValueError("Stored dev history does not match these IDs, ordering and batch size")
            self.evaluations = state["evaluations"]

    def _ordered_indices(self, program: str) -> list[int]:
        counts = dict.fromkeys(self.ids, 0)
        totals = counts.copy()
        for name, record in self.evaluations.items():
            # Retrying a named evaluation cannot contribute its own observations.
            if name != program:
                for item_id, score in record["observations"].items():
                    totals[item_id] += score
                    counts[item_id] += 1
        means = {i: totals[i] / counts[i] if counts[i] else 0.0 for i in self.ids}
        return sorted(range(len(self.ids)), key=lambda j: means[self.ids[j]], reverse=self.order == "easy-first")

    async def evaluate(
        self,
        program: str,
        items: list[Item],
        score_batch: Callable[[list[Item]], Awaitable[list[float]]],
        *,
        cutoff: float | None,
    ) -> DevAssessment:
        n = len(self.ids)
        if len(items) != n:
            raise ValueError("Dev item count differs from the fixed ID manifest")
        # The base and any candidate for an unfilled frontier need all scores.
        # Keep native evaluation order and concurrency when rejection is impossible.
        if program == "base" and cutoff is not None:
            raise ValueError("The initial dev assessment must be complete")
        indices = list(range(n)) if cutoff is None else self._ordered_indices(program)
        size = n if cutoff is None else self.batch_size
        result = DevAssessment({}, n)
        for offset in range(0, n, size):
            batch = indices[offset:offset + size]
            scores = await score_batch([items[i] for i in batch])
            for i, score in zip(batch, scores, strict=True):
                if score not in (0, 1):
                    raise ValueError("The dev gate requires binary per-item scores")
                result.observations[self.ids[i]] = int(score)
            # Equality is a rejection: the native frontier requires strictly better.
            if cutoff is not None and result.stopped and result.upper <= cutoff:
                break

        record = {
            "n_evaluated": len(result.observations), "n_total": n,
            "stopped": result.stopped, "cutoff": cutoff,
            "lower_bound": result.lower, "upper_bound": result.upper,
            "reason": "cannot_beat_frontier_floor" if result.stopped else "full_evaluation",
            "observations": result.observations,
        }
        evaluations = {**self.evaluations, program: record}
        state = {"ids": self.ids, "order": self.order, "batch_size": self.batch_size, "evaluations": evaluations}
        self.journal.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.journal.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        tmp.replace(self.journal)
        self.evaluations = evaluations
        return result
