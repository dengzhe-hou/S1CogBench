# Seven-model results

The active website release is `seven-model-20261110`. It contains Laya, Intern-Decision-4B, Llama-3.1-8B-Instruct, Minitaur8B, Qwen3.5-4B, Decision-2.0-Nox-4B and CLM-v0.1-8B. All seven cover 24 tasks and 27 primary evaluation modes. These are retained, accepted results, not new inference for this release.

The [main score table](../results/latest/table2.csv) contains all 189 open-model primary scores and keeps the existing API and reference columns. Every open-model cell was checked at the manuscript's displayed precision against the current accepted table. The original five-model arXiv aggregate and table remain unchanged under [results/aggregate](../results/aggregate/). The current website uses [results/latest](../results/latest/). The arXiv paper and separately maintained dataset mirrors have their own release history.

## Rebuild the release

```bash
python scripts/build_latest_results.py
python scripts/build_homepage.py
```

The first command reads accepted aggregate inputs and writes current JSON and CSV files. It checks all 189 displayed primary scores against their retained table definitions. It also exports 70 existing Horizon and WCST readout metrics. Neither command loads a model, obtains participant records or calls a provider. The homepage builder additionally requires the `markdown` package.

Each current row retains its original target, unit and aggregation. Accuracy is stored as a fraction even when a table prints a percentage. Positive infinite NLL remains `Infinity`. Temporal discounting remains a preference with no better direction. Bold and underlining in the main table mark the best and second-best open-model scores before rounding, including ties. They do not imply statistical significance.

## Readout comparisons

Horizon and WCST include legal argmax and categorical sampling for all seven open models. Horizon retains 31 timelines and 34,720 free choices per model and readout. WCST retains 75 participants and 12,754 choices. WCST accuracy, perseveration error and set-loss error use participant means. Horizon reward and optimal-choice rate retain the original free-choice means. The original five-model controls and the accepted Nox/CLM additions together supply the 70 metric rows in [readout_sources.json](../results/latest/readout_sources.json).

Sampling uses each model's own executed actions and feedback in its subsequent history. Source seeds, scoring and legal probability extraction remain unchanged. Nox and CLM's legal-argmax columns reuse the archived native choices whose equality was checked in the retained records. Nox's main native/argmax run used an A100 MIG allocation and its added sampling run used a whole A100. These results compare task performance and are not matched-hardware latency measurements.

Nox uses Decision-2.0-Nox-4B revision `ae269ecf34a0b3a89026bc1afd9c995279212fad`, with its native legal probabilities and no probability floor. CLM uses CLM-v0.1-8B revision `e939398d4556fcd9400c76fa8c5a513202f42b0a`, the official pooling path and first 2,048 tokens of each text. Nox retains the full state up to its 16,384-token context. Multiple-cue judgment retains the actual full-prefix or compact history used by each model. Model-specific readouts and protocol differences remain attached to the comparisons.

The source inputs and participant-level outputs retain their original access terms and are not included here. The community leaderboard labels these as maintainer results. External contributions continue through [the submission process](SUBMIT_RESULTS.md), with review before merge. The suite protocol identifier remains `arxiv-2611.00911`, while result releases are dated separately.
