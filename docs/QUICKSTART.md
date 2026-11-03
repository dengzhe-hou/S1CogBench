# Quickstart

Use Python 3.12 and run every command from the repository root.

## 1. Install the CPU task environment

```bash
git clone https://github.com/dengzhe-hou/S1CogBench.git
cd S1CogBench
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

These are the pins of the 24-task CPU validation (CPU PyTorch, NumPy 1.26, the original CogBench and bandit dependencies); the install took under three minutes on the lab cluster. A neural adapter adds its own model stack on top. Python 3.11 reproduces the 2026-10-26 validation bit for bit. Under Python 3.12 a few aggregates differ in their last digit or two, because Python 3.12 made the built-in float `sum()` compensated (see [REPRODUCTION.md](REPRODUCTION.md)).

## 2. Get the data

```bash
python scripts/fetch_public_sources.py all
python -m s1cogbench doctor
```

`fetch_public_sources.py all` downloads every public source at its pinned revision into `external/`, `data/` and `literature/`. Psych-101-test is gated: accept its conditions on Hugging Face, then follow the steps that `fetch_public_sources.py psych101` prints. Mouselab autonomous also needs Julia 1.4.1 (`fetch_public_sources.py julia`). `doctor` lists each of the 27 modes as runnable or names the missing input. On the lab cluster `all` took under seven minutes, most of it the 1.07 GB Mouselab archive and Julia's package registry; a fresh fetch gave files byte-identical to the inputs of the 2026-10-26 validation. See [DATA.md](DATA.md).

The sources can live outside the repository: pass `--data-root DIR` to `doctor` and `eval`, where `DIR` has the same `data/`, `external/`, `literature/` and `.tools/` layout. Missing entries are linked into the repository's ignored directories; an existing different entry stops the run.

## 3. Check the installation

```bash
python -m s1cogbench eval --smoke --output runs/smoke
```

Every runnable mode runs its original runner and scorer with `examples/uniform_reference_agent.py` for five adapter calls (Temporal discounting runs its 13-question questionnaire in full), and the calls and scores are compared with `s1cogbench/smoke_expected.json`. The last line says `SMOKE PASSED` or lists the differences in `runs/smoke/smoke_check.json`. It takes about two minutes on one CPU.

## 4. Evaluate a model

Copy a template from `examples/templates/` (see [MODELS.md](MODELS.md)), then:

```bash
S1_HF_MODEL=meta-llama/Llama-3.1-8B-Instruct \
python -m s1cogbench eval --adapter my_hf_agent.py --device cuda:0 \
    --model-label my-llama --tasks all --output runs/my-llama
```

`--tasks` takes `all`, task IDs (all modes of the task) or `task:mode` keys from `python -m s1cogbench list`. Each task runs in its own process through the runner and scorer that produced the accepted results. Outputs:

- `runs/my-llama/results.csv` and `results.json`: one row per mode for its Table 3 primary target, in the format of `results/aggregate/result_schema.json` and with the same comparison context as the accepted rows.
- `runs/my-llama/summary.json`: status, counts, primary value and seconds per mode.
- `runs/my-llama/<task>/<mode>/`: the original runner's raw outputs and summary. These contain source participant data for the gated tasks; do not redistribute them.

The full suite is about 910,000 adapter calls: 312,000 in the stationary bandits, 163,135 in Hybrid sampling, about 147,000 in Mouselab autonomous and 108,248 in Self-assessment (Mouselab autonomous and BART depend on the policy). The CPU reference adapters finish it in about 25 minutes on one CPU. A neural model's time is its own inference: the accepted 4B to 8B runs needed roughly 16 to 40 GPU-hours each at batch size one (shared GPUs, so only an order of magnitude).

## 5. Share a comparison

Keep the raw outputs and the result rows. Record the checkpoint revision, adapter, prompt, readout, action rule, precision, seed and device. Runtime records retain hardware and sharing context, and latency is not a human reaction-time measure. See [SUBMIT_RESULTS.md](SUBMIT_RESULTS.md). A local run is a user-supplied result until the maintainers validate it.

## The CPU Bayes example

The original single-task example still works and needs only `fetch_public_sources.py cogbench`:

```bash
python -m s1cogbench eval --adapter examples/bayes_event_agent.py --tasks bayesian_urn_inference \
    --output runs/bayes-posterior
```

Exact Bayes has full posterior MAE zero on all 400 queries. The original strict trimmed score is undefined when its q95 rule retains no errors; the scorer keeps it undefined. `scripts/benchmark_paper_tasks.py` keeps its export, predict and score entries for the probabilistic, Horizon and Mouselab tasks.
