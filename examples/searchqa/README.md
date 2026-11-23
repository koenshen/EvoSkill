# EvoSkill on the fixed SearchQA split

Run from the EvoSkill checkout with Python 3.12 and the project's dependencies.
This adapter uses only EvoSkill code and the materialized SearchQA JSON files.
The default data directory is `../SkillOpt/data/searchqa_split` relative to the
EvoSkill checkout. Set `SEARCHQA_SPLIT_DIR` or pass `--split-dir` for another
location. All 400/200/1,400 IDs and their order are validated before any model calls.

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
