<h1 align="center">S1CogBench</h1>

<h2 align="center">
  What System One Decision Models Can and Cannot Do on Cognitive Experiments
</h2>

<p align="center">
  <a href="https://arxiv.org/abs/2611.00911"><img src="https://img.shields.io/badge/arXiv-2611.00911-B31B1B" alt="arXiv paper 2611.00911"></a>
  <a href="https://huggingface.co/datasets/dengzhe-hou/S1CogBench"><img src="https://img.shields.io/badge/Hugging%20Face-Results-FFB000" alt="Aggregate results on Hugging Face"></a>
  <a href="docs/QUICKSTART.md"><img src="https://img.shields.io/badge/Docs-Quickstart-2878D0" alt="Quickstart documentation"></a>
  <a href="results/latest/table2.csv"><img src="https://img.shields.io/badge/Results-Table%203-2A8C82" alt="Table 3 as a CSV file"></a>
  <a href="task_catalog.json"><img src="https://img.shields.io/badge/Tasks-Catalogue-2A8C82" alt="Task catalogue"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-2878D0" alt="MIT license for the original project software"></a>
</p>

<p align="center">
  <a href="docs/assets/fig1.png">
    <img src="docs/assets/fig1.png" width="100%" alt="Six representative task interfaces above a diagram of model evaluation: task-specific adapters feed the model panel, which is scored separately on predicting human choices, executing its own actions, and reporting explicit probabilities.">
  </a>
</p>

<p align="center">
  <em>S1CogBench connects cognitive tasks to model evaluation.</em>
</p>

Decision-making in humans and AI involves multiple cognitive abilities. System One decision models produce fast, structured choices. Their cognitive strengths and limitations, however, remain poorly understood. We introduce S1CogBench, a benchmark of 24 established cognitive tasks and 27 evaluation modes, evaluating seven open models and the API model Jev. It combines human-response prediction, autonomous execution, and controlled task manipulations to reveal strengths and failures that aggregate scores miss. We find distinct behavioral profiles across the evaluated models. In these models, strong performance on some learning tasks coexists with failures to gather information or adapt to feedback, while high accuracy does not guarantee human-like responses to changing task demands. Shared readout controls show that model rankings depend partly on how choices are extracted. Separate latency comparisons with generative language models show that long responses are much slower than direct decisions, whereas matched one-token generation adds little overhead. To our knowledge, S1CogBench is the first benchmark of System One decision models on established cognitive experiments. It provides a common empirical basis for identifying their capabilities and limitations before integrating them into automated decision systems.

Read the paper on [arXiv](https://arxiv.org/abs/2611.00911). The original arXiv aggregate is also on [Hugging Face](https://huggingface.co/datasets/dengzhe-hou/S1CogBench).

## Add a model or inspect a protocol

Start with [the new-model guide](docs/ADD_MODEL.md). The plan command lists the original inputs, denominators and adapter requirements without importing a model or reading datasets.

```bash
python -m s1cogbench plan --tasks all --json
python -m s1cogbench plan --adapter my_agent.py --tasks all --json
```

Use [the comparison guide](docs/COMPARISON_GUIDE.md) to distinguish action rules and score targets. [Release scope](docs/RELEASE_SCOPE.md) separates the released aggregate from later manuscript evidence and tooling changes. Existing numeric results are preserved.

## Quick Start

**Run every mode through its original runner and scorer in about two minutes.** The smoke run makes a few calls per mode with a uniform reference adapter, on one CPU, with no model weights.

> [!CAUTION]
> Reproducing the 2026-10-26 CPU validation bit for bit needs Python 3.11. Under Python 3.12, 6 of the 27 modes differ in the last digits of floats, because Python 3.12 made the built-in float `sum()` compensated.

```bash
git clone https://github.com/dengzhe-hou/S1CogBench.git
cd S1CogBench
python3.12 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
```

> [!CAUTION]
> Psych-101-test is gated. Eight modes stop at `doctor` until you accept its conditions on Hugging Face and run `fetch_public_sources.py psych101`. Do not redistribute the file or the per-item outputs of those modes, which contain source responses.

```bash
python scripts/fetch_public_sources.py all     # public sources at pinned revisions
python -m s1cogbench doctor                    # which of the 27 modes can run here, and why not
```

```bash
python -m s1cogbench eval --smoke --output runs/smoke
```

The last line says `SMOKE PASSED`, or lists the differences in `runs/smoke/smoke_check.json`.

**Then evaluate your own model.** An adapter is a Python file with `create_agent(device)`, and a template is the shortest way to write one.

> [!CAUTION]
> `--output` must be a new directory inside the repository, for example `runs/my-model`. The original runners refuse to write anywhere else.

```bash
python -m s1cogbench eval --adapter my_agent.py --device cuda:0 --model-label my-model \
    --tasks all --output runs/my-model
```

[Read the full quickstart](docs/QUICKSTART.md) · [Write an adapter](docs/MODELS.md)

## Submit your model's scores

Evaluate your model, compare its `results.json` locally on the [project page](https://dengzhe-hou.github.io/S1CogBench/#explorer), and package a completed run for the [community leaderboard](https://dengzhe-hou.github.io/S1CogBench/#community).

```bash
# Fill examples/submission_metadata.json and save it as runs/my-model/metadata.json.
python -m s1cogbench package --run runs/my-model \
    --metadata runs/my-model/metadata.json --id my-model-20261111 \
    --output submissions/my-model-20261111.json
python -m s1cogbench check-submissions
```

Open a GitHub pull request with the submission JSON. After maintainer review and merge, the leaderboard updates automatically. Each task and action rule is compared separately using the original metric. Maintainer result snapshots are versioned, and no overall cognitive score is added. Read [the submission guide](docs/SUBMIT_RESULTS.md) for evidence and data-sharing requirements.

## Contribute a task

The 24 paper tasks stay fixed, and new paradigms join a separate extension track. An extension task is one file, `extensions/<task_id>/task.py`, with the task's source, terms, comparison context and trials. The command line finds it automatically, and its leaderboard appears in its own group. `--tasks all` still runs only the 27 paper modes.

```bash
cp -r extensions/_template extensions/my_task      # then fill in TASK and load_trials()
python -m s1cogbench eval --smoke --tasks my_task --output runs/my-task-smoke
```

Read [the task guide](docs/ADD_TASK.md) for the review criteria, and [CONTRIBUTING.md](CONTRIBUTING.md) for all ways to contribute.

## From a task source to a result row

[![Workflow from fetching the sources to result rows](docs/assets/workflow.svg)](docs/assets/workflow.svg)

`eval` runs each task in its own process, through the original runner and scorer that produced the accepted results, and writes `results.csv` and `results.json` with one row per mode for its Table 3 primary target. Each row has the [result schema](results/aggregate/result_schema.json) and the comparison context of the accepted rows, and raw per-item outputs stay under `runs/my-model/<task>/<mode>/`. There is no pooled score. Infinite NLL, the undefined all-invalid MAE and negative values remain in the result files.

[Read the protocols](docs/PROTOCOLS.md) · [Group results by context](results/aggregate/grouping_guide.md)

## Data & Models

| Resource | Where to start |
| --- | --- |
| 24 evaluated tasks in 27 modes, and 7 deferred paradigms | [Open the task catalogue](task_catalog.json) |
| 18 modes on public sources at pinned revisions | [Read the data guide](docs/DATA.md#fetch) |
| Mouselab autonomous, which also needs Julia 1.4.1 | [Set up Julia](docs/DATA.md#julia-for-mouselab-autonomous) |
| 8 modes on the gated Psych-101-test | [Request access](docs/DATA.md#gated-psych-101-test) |
| Accepted results and Table 3 | [All rows](results/latest/results.csv) · [Table 3](results/latest/table2.csv) |
| Weights of the evaluated models, not bundled | [Find the adapters](docs/MODELS.md#adapters-of-the-accepted-rows) |

The current release evaluates seven open models and Jev on 24 tasks (27 modes), with Claude on selected modes. See [the release guide](docs/LATEST_RESULTS.md) for the current scores and the retained five-model arXiv snapshot.

| Model | Access | Modes | Adapter or template |
| --- | --- | --- | --- |
| Laya English | Open weights | All 27 | [`laya_choice_agent.py`](examples/laya_choice_agent.py) |
| Intern-Decision-4B | Open weights | All 27 | [`intern_decision_choice_agent.py`](examples/intern_decision_choice_agent.py) |
| Llama-3.1-8B-Instruct | Open weights | All 27 | [By mode](docs/MODELS.md#adapters-of-the-accepted-rows) |
| Minitaur8B | Open weights | All 27 | [`minitaur8b_choice_agent.py`](examples/minitaur8b_choice_agent.py) |
| Qwen3.5-4B | Open weights | All 27 | [`qwen35_4b_choice_agent.py`](examples/qwen35_4b_choice_agent.py) |
| Decision-2.0-Nox-4B | Open weights | All 27 | [Native readout and checkpoint](docs/LATEST_RESULTS.md) |
| CLM-v0.1-8B | Open weights | All 27 | [Pooling readout and checkpoint](docs/LATEST_RESULTS.md) |
| Jev 1.14 | API | All 27 | [`decision_api_choice.py`](examples/templates/decision_api_choice.py) |
| Claude Opus 5.5 | API | 2 | [`generative_api_choice.py`](examples/templates/generative_api_choice.py) |
| Claude Sonnet 5.5 | API | 3 | [`generative_api_choice.py`](examples/templates/generative_api_choice.py) |
| Claude Haiku 5.5 | API | 15 | [`generative_api_choice.py`](examples/templates/generative_api_choice.py) |
| Your Hugging Face model | Your weights | Any | [`hf_causal_lm_choice.py`](examples/templates/hf_causal_lm_choice.py) |

> [!NOTE]
> All 27 modes run through the CLI with the CPU reference adapters. Among neural models, Llama-3.1-8B on a whole A100 is supported. In a pass of 20 calls per mode, all 23 comparable modes match the accepted runs exactly. On a MIG slice of an A100, prompts and identities match but probabilities differ by BF16 rounding steps, so the support does not extend to slices, and expect Llama numbers from a slice to differ from the accepted rows. The accepted results came from the original runners with model-specific adapters or published prompts, and the current seven-model score release does not extend this generic-CLI reproduction certification to every model. See [REPRODUCTION.md](docs/REPRODUCTION.md).

## Baselines

Each reference keeps its own target, fit and provenance. A human-action NLL is not an autonomous-return comparator, and there is no global reference score.

| Reference | Task modes | Ref. type | Provenance | Ref |
| --- | --- | --- | --- | --- |
| Uniform | All 27 | Control | Analytic expectation | [Definitions](docs/BASELINES.md#uniform-column) |
| Exact Bayes | Bayesian urn | Optimal | Analytic solution | [Guide](docs/BASELINES.md#bayesian-urn) |
| Multiple-cue formula reimplementation | Multiple-cue judgment | Human fit | Disclosed reimplementation | [Guide](docs/BASELINES.md) |
| WADD | Multi-attribute choice | Human fit | Disclosed reimplementation | [Guide](docs/BASELINES.md#recovered-references-for-the-existing-table-3-targets) |
| Median report of training participants | Probability coherence | Human | Independent fit | [Guide](docs/BASELINES.md#human-response-references) |
| GCM | SHJ, Medin, Weather | Human fit | Disclosed reimplementation | [Guide](docs/BASELINES.md#recovered-references-for-the-existing-table-3-targets) |
| RLWM archived fitted cognitive models | RLWM | Human fit | Archived replay | [Guide](docs/BASELINES.md#rlwm-and-horizon) |
| Rational conditional-response model | Recent probes | Human fit | Disclosed reimplementation | [Guide](docs/BASELINES.md#recovered-references-for-the-existing-table-3-targets) |
| SL (local refit) | WCST, both modes | Human fit | Official code | [Guide](docs/BASELINES.md) |
| Two-parameter Q-learning | Restless bandit | Algorithm | Disclosed reimplementation | [Guide](docs/BASELINES.md#restless-q-learning) |
| Rescorla-Wagner with separate learning rates | Instrumental learning | Algorithm | Disclosed reimplementation | [Guide](docs/BASELINES.md#autonomous-references) |
| RW3 (local formula fit) | Horizon, both modes | Human fit | Disclosed reimplementation | [Guide](docs/BASELINES.md#rlwm-and-horizon) |
| Thompson family | Stationary bandits | Algorithm | Official code and disclosed reimplementation | [Guide](docs/BASELINES.md#recovered-references-for-the-existing-table-3-targets) |
| Independent end-to-end ANN | Hybrid sampling | Human fit | Disclosed reimplementation | [Guide](docs/BASELINES.md) |
| MetaGreedy (official implementation) | Mouselab, prediction | Human fit | Official code | [Guide](docs/BASELINES.md) |
| MetaGreedy, archived fitted parameters | Mouselab, autonomous | Human fit | Disclosed reimplementation | [Guide](docs/BASELINES.md#mouselab-metagreedy) |
| Fixed pump threshold per balloon letter | BART | Algorithm | Independent fit | [Guide](docs/BASELINES.md#autonomous-references) |
| Later-reward rate of Psych-101 participants | Temporal discounting | Human | Human data summary | [Guide](docs/BASELINES.md#human-response-references) |
| Hybrid model fits to CogBench participants | Two-step | Human fit | Disclosed reimplementation | [Guide](docs/BASELINES.md#autonomous-references) |
| ToMi answer key | False-belief QA | Optimal | Answer key | [Guide](docs/BASELINES.md#false-belief-qa) |
| Other participants, leave one out | Politeness, Pragmatic interpretation | Human | Human data summary | [Guide](docs/BASELINES.md#human-response-references) |
| Human split-half | Nonliteral number | Human | Human data summary | [Guide](docs/BASELINES.md#human-response-references) |
| Lookup table | Self-assessment | Human fit | Disclosed reimplementation | [Guide](docs/BASELINES.md#human-response-references) |

[Reference values](results/aggregate/reference_results.json) · [Fit and exposure provenance](results/aggregate/baseline_provenance.json)

## Documentation

| I want to… | Guide |
| --- | --- |
| Run my first evaluation | [Quickstart](docs/QUICKSTART.md) |
| Write an adapter for my model | [Models and adapters](docs/MODELS.md) |
| Get the data and check its terms | [Data](docs/DATA.md) · [License status](docs/LICENSE_STATUS.md) |
| Read the protocols and scoring targets | [Protocols](docs/PROTOCOLS.md) |
| Look up a number in Table 3 | [Results guide](docs/RESULTS.md) · [The row behind each cell](results/latest/table2.json) |
| Compare results without mixing contexts | [Grouping guide](results/aggregate/grouping_guide.md) · [Metric applicability](results/aggregate/metric_status.json) |
| Understand a reference or the Uniform column | [Baselines](docs/BASELINES.md) |
| See what has been reproduced and what has not | [Reproduction](docs/REPRODUCTION.md) · [CPU validation](docs/CPU_REUSE_VALIDATION_24.md) |
| Read the Jev and Claude settings, time and cost | [API extension](docs/API_EXTENSION.md) · [Time and cost](docs/API_TIME_COST.md) |
| Read the latency, generation and readout controls | [Supplementary controls](docs/SUPPLEMENTARY_CONTROLS.md) · [Latency](docs/LATENCY.md) |
| Share my results | [Submit results](docs/SUBMIT_RESULTS.md) |
| Add a new cognitive task | [Task guide](docs/ADD_TASK.md) · [Extension tasks](extensions/README.md) |

<details>
<summary><strong>Repository structure</strong></summary>

```text
s1cogbench/   The command line: list, plan, doctor, eval, smoke
scripts/      Source fetching, the original task runners, Table 3 and homepage builders
src/          Task loaders, environments and scorers
examples/     Model adapters and templates
extensions/   Contributed tasks outside the 24-task paper suite, and a template
results/      Accepted aggregate results, references and Table 3
docs/         Guides and retained records
experiments/  Portable code of the supplementary controls
derived/      Inputs derived from published code
envs/         The pinned Julia environment for Mouselab
homepage/     The project page source
tests/        Offline tests of adapter templates and evaluation entry behavior
```

</details>

## Citation

If you use S1CogBench in your research, please cite.

```bibtex
@misc{hou2026s1cogbench,
  title         = {{S1CogBench}: What System One Decision Models Can and Cannot Do on Cognitive Experiments},
  author        = {Hou, Dengzhe and Yamada, Kazunori D},
  year          = {2026},
  eprint        = {2611.00911},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2611.00911}
}
```

## License

Original S1CogBench project software is licensed under [MIT](LICENSE). Third-party code, models and data keep their original terms, and gated participant data and model weights are not bundled. See [license status](docs/LICENSE_STATUS.md).
