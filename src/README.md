# src

Task loaders, environments and scorers used by the paper's evaluation runs. The command line in `s1cogbench/` and the runners in `scripts/` import these modules. They are kept at their original paths, because the accepted result records in `results/` cite them.

| Files | What they hold |
| --- | --- |
| `paper_*.py`, `horizon.py`, `wcst_task.py`, `rlwm_typed.py`, `subway_original.py`, `cogbench_expansion.py` | Original-protocol tasks and environments: Horizon, Hybrid sampling, Mouselab, probabilistic reasoning, Restless bandit, two-step, WCST, RLWM, Subway and the CogBench expansion |
| `psych_*.py`, `four_task_interfaces.py` | Psych-101 and four-task response prediction, with full and compact human histories |
| `*_models.py`, `batched_choices.py`, `choice_letter_tokens.py`, `integer_label_choice.py` | Shared model interfaces that turn a task state into legal-choice probabilities or events |
| `gcm_baseline.py`, `multi_attribute_wadd.py`, `pragmatic_cognitive_baselines.py` | Cognitive-model references |

To evaluate a model, start from [the quickstart](../docs/QUICKSTART.md) instead of calling these modules directly.
