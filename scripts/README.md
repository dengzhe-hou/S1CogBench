# scripts

Run every script from the repository root. Most of them belong to the paper runs and are kept at their original paths, because the accepted result records in `results/` cite them. To evaluate a new model, use `python -m s1cogbench` instead ([quickstart](../docs/QUICKSTART.md)).

| Files | Purpose |
| --- | --- |
| `fetch_public_sources.py` | Fetch the task inputs at their pinned revisions ([data guide](../docs/DATA.md)) |
| `build_latest_results.py`, `build_table2.py`, `add_table2_api_rows.py` | Rebuild the result tables from the accepted aggregate files, with no model calls |
| `build_homepage.py` | Prepare the project page in `homepage/` |
| `run_*.py`, `benchmark_paper_tasks.py` | The original task runners used for the paper's model results |
| `validate_cpu_reuse_24.py`, `audit_*.py`, `check_*.py` | Reference validation and protocol checks |
| `*.jl` | Mouselab helpers for the pinned Julia environment in `envs/` |
