# Contribute a task to the extension track

S1CogBench keeps its 24 paper tasks fixed. New paradigms join a separate extension track. An extension task is one Python file, `extensions/<task_id>/task.py`, plus any inputs its license lets you share. The command line finds it automatically, and models are evaluated, packaged and submitted the same way as on the paper tasks.

Extension tasks never change the paper suite. `--tasks all` still selects only the 27 paper modes, and the paper tables and maintainer results do not include extension tasks. Their leaderboards appear in a separate group on the project page.

If you would rather not write the task file, describe the paradigm in a [paradigm proposal](https://github.com/dengzhe-hou/S1CogBench/issues/new?template=propose_paradigm.yml). Anyone can later turn a proposal into a task under the criteria below.

## What a task needs before review

Maintainers review each task against these points. Address each one in the pull request.

1. **A cognitive or behavioral paradigm.** The paradigm comes from research on human cognition or behavior, and the source study ran it with human participants. A knowledge or skill test with an answer key but no participant study behind it does not qualify.
2. **A published source.** The paradigm and its human data, answer key or environment come from a peer-reviewed study or an established benchmark, cited in `TASK['source']`.
3. **Terms that allow the use.** The data license permits evaluation. Commit inputs only if the license permits redistribution. Otherwise `TASK['obtain']` explains how to retrieve them at a pinned revision. Gated or restricted participant records never enter a pull request.
4. **A new paradigm or a new study.** A new seed, prompt or condition of an existing paper or extension task is not a new task, and neither is the same data encoded another way. Data of an existing paradigm from a different study, with its own participants, is welcome as a separate task under its own ID, for example the paradigm name followed by the first author and year. Its leaderboard stays apart from that of the existing task.
5. **The original protocol.** Keep official splits, participant IDs and complete histories. State every change from the original in `TASK['adaptations']`, such as text display, supplied histories or a constrained response set.
6. **Full counts.** `n_inputs` and `n_scored` are the full source counts. Full runs stop when the counts differ.
7. **Visible information only.** `state`, `question` and `output_keys` hold only what a participant could see at that point. Targets, later feedback and hidden environment values stay with the scorer.
8. **A uniform control and a reference.** The pull request reports the score of `examples/uniform_reference_agent.py` on the full task, or the analytic uniform expectation for accuracy and reward targets. `TASK['reference']` gives at least one reference score with its provenance, such as a model fitted to training participants, an analytic solution or an answer key. Before merging, a maintainer reruns the full uniform control and compares it with the reported value. The reference is checked against its cited provenance and is not refitted.
9. **One task per pull request,** with the checks below passing.

## Write the task file

Copy `extensions/_template/` to `extensions/<task_id>/`. The directory name is the task ID. It uses lowercase letters, digits and underscores, and must differ from every paper task. Keep bundled inputs in a subdirectory such as `inputs/`, because the repository ignores directories named `data/`.

`TASK` describes the task, its source and its comparison context. The fields `protocol`, `mode`, `metric`, `metric_variant`, `scope`, `unit`, `aggregation` and `target` define which results can be compared, exactly as on the paper tasks ([PROTOCOLS.md](PROTOCOLS.md)). Change `protocol` whenever the protocol changes, so that results under the old and new versions are never ranked together.

Most tasks need only `load_trials(root)`, which returns the trials in source order and lets S1CogBench score them with a built-in metric.

| `metric` | Adapter method | Trial target | Score |
| --- | --- | --- | --- |
| `human_nll` | `choice_probabilities` | `observed`, the participant's key | Mean negative log probability of the observed key. A zero probability gives positive infinite NLL, with no epsilon. |
| `accuracy` | `choice_probabilities` | `correct`, the correct key | Share of trials where the model's choice is correct. The choice is the adapter's `last_native_choice` when it sets one, and otherwise the argmax with ties to the first key. |
| `probability_mae` | `posterior_probability` | `target`, a probability | Mean absolute error of the reported probability. An adapter may return `None` for an invalid report, which is kept in the records and excluded from the mean. |

A task whose source scorer is different, or that runs an environment on the model's own actions, defines `run(agent, out, limit, device, action_rule)` instead. It calls the adapter, writes its records under `out`, honours `limit` in smoke runs and `action_rule` in autonomous tasks, and returns the inputs, scored count, primary value and its denominator. It also sets `TASK['interface']`, `TASK['direction']` and, for environments, `TASK['autonomous'] = True`. The template shows the exact form.

## Check the task

```bash
python -m s1cogbench list                                   # the task appears under Extension tasks
python -m s1cogbench plan --tasks my_task --json            # comparison context and adapter methods
python -m s1cogbench doctor --tasks my_task                 # inputs present
python -m s1cogbench eval --smoke --tasks my_task --output runs/my-task-smoke
python -m s1cogbench eval --adapter examples/uniform_reference_agent.py \
    --tasks my_task --output runs/my-task-uniform           # full run, the uniform control
python -m unittest discover -s tests -p 'test_*.py'
```

The smoke run has no stored values for extension tasks, so it checks that the task runs without comparing numbers. The full uniform run checks the declared counts. Put its `results.json` value in the pull request.

## Open the pull request

Fork the repository, add `extensions/<task_id>/`, and open a pull request with the [extension-task template](../.github/PULL_REQUEST_TEMPLATE/extension_task.md). Before merging, maintainers check the source, terms, protocol, counts and reference, and rerun the full uniform control. A merged task appears in `list` and can receive model results through [SUBMIT_RESULTS.md](SUBMIT_RESULTS.md), and its leaderboard joins the Extension tasks group on the project page.

Model results for a new task go in a separate pull request, after the task is merged, so the task can be reviewed apart from any score.
