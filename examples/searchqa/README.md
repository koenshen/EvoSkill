# EvoSkill on the fixed SearchQA, TriviaQA, HotpotQA and BIRD splits

Run from the EvoSkill checkout with Python 3.12 and the project's dependencies.
This adapter uses only EvoSkill code and the materialized SearchQA JSON files.
The default data directory is `../SkillOpt/data/searchqa_split` relative to the
EvoSkill checkout. Set `SEARCHQA_SPLIT_DIR` or pass `--split-dir` for another
location. All 400/200/1,400 IDs and their order are validated before any model calls.

`--benchmark` selects the dataset and defaults to `searchqa`. Each dataset reads
its own `../SkillOpt/data/<benchmark>_split/{train,val,test}/items.json` and
checks the SHA-256 digest of the ordered IDs for every split. Use `--split-dir`
or the corresponding `TRIVIAQA_SPLIT_DIR`, `HOTPOTQA_SPLIT_DIR`, or
`BIRD_SPLIT_DIR` to choose a different copy of that benchmark.

| Benchmark | Train / dev / test | Scoring |
|---|---|---|
| TriviaQA | 1,600 / 200 / 1,000 | Multi-reference EM and token F1, including all supplied aliases |
| HotpotQA | 1,600 / 200 / 1,000 | Answer EM and token F1 from the supplied passages |
| BIRD | 834 / 200 / 500 | SQLite execution accuracy (EX) |

The native agent, skill tools, training sample selection, proposer, generator
and frontier remain the same. The QA benchmarks send the existing context and
question through EvoSkill's QA formatter. BIRD sends the complete stored
`prompt` to the agent, which puts the SQL in its structured `final_answer`.
The reference SQL and database path are withheld from the answering agent.
Prediction and gold share a
30-second deadline in the same read-only database; EX compares sets of rows.
The native optimizer receives the scored training failures as usual.

To preflight on the CPU server without creating a workspace or making model calls:

```sh
for benchmark in triviaqa hotpotqa bird; do
  python examples/searchqa/run.py --benchmark "$benchmark" --check-data
done
```

For a full run, use the original gateway environment described below, set
`SKILLOPT_WORKERS=24`, and use a fresh work directory per benchmark:

```sh
python examples/searchqa/run.py --benchmark triviaqa --harness openai-compatible --work-dir /path/to/new-triviaqa-run
python examples/searchqa/run.py --benchmark hotpotqa --harness openai-compatible --work-dir /path/to/new-hotpotqa-run
python examples/searchqa/run.py --benchmark bird --harness openai-compatible --work-dir /path/to/new-bird-run
```

BIRD defaults to the sibling `bird-text-to-sql-260820/data_dev/dev_databases`;
`--db-root` or `BIRD_DB_ROOT` overrides the directory. The preflight checks that
each referenced `<db_id>/<db_id>.sqlite` exists. BIRD's `test_results.jsonl`
includes the extracted SQL, execution status and EX; `test_summary.json`
contains `dev_ex` for the best retained program and `test.ex`. All test items
remain in the denominator. The optional mild dev gate below is specific to SearchQA.

```sh
python examples/searchqa/run.py --check-data
```

The experiment command below uses the native Claude harness and its configured credentials.
Choose a new work directory outside the source checkout. Git needs a configured author
identity there for EvoSkill's candidate commits.

```sh
python examples/searchqa/run.py --work-dir /path/to/new-searchqa-run
```

The run seeds the isolated project with the shipped generic skills, cycles through a single
400-question training pool, scores candidates by EM on all 200 dev questions, checks out
the best frontier member, and evaluates it with the same answering agent on all 1,400 test
questions. The CLI defaults for the model, seven iterations, frontier size three, sampling,
and native stopping criteria apply. The answering agent can invoke project skills and return
structured answers. The proposer and generator use EvoSkill's native profiles.
`--model` overrides the model for all roles. Results are under the run's `.evoskill/` in
`train_result.json`, `test_results.jsonl`, and `test_summary.json`.

For an OpenAI-compatible Chat Completions gateway, use the existing OpenCode
harness and install the `opencode` CLI on PATH (tested with 1.18.x).
Set `OPENAI_COMPATIBLE_BASE_URL`, `OPENAI_COMPATIBLE_API_KEY`,
`OPENAI_COMPATIBLE_MODEL`, `TARGET_OPENAI_COMPATIBLE_MAX_TOKENS`,
`OPTIMIZER_OPENAI_COMPATIBLE_MAX_TOKENS` and `SKILLOPT_WORKERS` before launch.

```sh
python examples/searchqa/run.py --harness openai-compatible --work-dir /path/to/new-searchqa-run
```

The two role budgets limit generated tokens. Both aliases resolve to the exact
deployment name, including embedded slashes. `--workers` overrides the dev/test
concurrency. The base agent can load project skills and submit structured
answers. Optimizers use the native profiles, tools and structured schemas.

### Candidate dev early stopping

Add `--dev-early-stop` to the original SearchQA command to opt in. The defaults are
`--dev-order hard-first --dev-batch-size 4`; `--dev-order easy-first` reverses the
historical mean EM ordering. Ties keep the fixed manifest order. The initial base is
scored on the full dev split. The base and candidates for spare slots retain normal
concurrency. When rejection is possible, only the next batch is submitted. Only
observed item scores (including zero for a failed answer) enter its history.

An unfilled frontier accepts every candidate and needs its full score. Once full,
after `k` of `N` dev answers with `c` correct, the candidate is discarded when
`(c + N - k) / N <= lowest_frontier_score`. Equality fails EvoSkill's strict
admission test. Kept candidates receive exact scores. A partial rejection records
the measured interval in proposer feedback, with no exact score or parent delta.
This preserves the decision on fixed answers; later proposals can respond differently
to the bounded feedback. The seven-iteration loop, its native stopping criterion
and the full 1,400-item final test use their normal logic.

The run workspace's `.evoskill/dev_evaluations.json` records manifest IDs, ordering,
batch size, observed IDs/EM, total/evaluated counts, bounds, cutoff and reason for
each base or candidate assessment. The fixed SearchQA CLI requires a new workspace;
the native loop's continue mode can reload this journal for the same dev configuration.
In the archived original run the base and two candidates scored 107, 157 and 158 out
of 200, exactly filling the three slots. This rule would skip **0/600** dev answers
on that trajectory, for either order.
