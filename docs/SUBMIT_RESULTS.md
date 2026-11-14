# Submit results to the community leaderboard

You can evaluate your own model and submit scores through a GitHub pull request. Maintainers review the run settings and supporting evidence, then merge the submission. GitHub Pages rebuilds the [community leaderboard](https://dengzhe-hou.github.io/S1CogBench/#community) automatically. You run the model on your own compute or provider account.

Maintainer result snapshots are versioned, and the original arXiv table stays available. Community entries are labeled separately and retain their checkpoint, prompt and readout. Each task, target and action rule has its own comparison. There is no cross-task total score. Ranks reflect measured values before display rounding, keep ties, and do not imply statistically significant differences. Preference and undefined results have no rank. Positive infinite NLL remains positive infinite.

## Run the original protocol

Follow [ADD_MODEL.md](ADD_MODEL.md), then run one or more complete task modes. Use `--tasks all` for the whole suite or select a task or mode with `--tasks`. A submission can cover a single complete mode. Smoke tests and incomplete mode runs cannot enter the leaderboard.

```bash
python -m s1cogbench plan --adapter my_agent.py --tasks all --json
python -m s1cogbench doctor --adapter my_agent.py
python -m s1cogbench eval --adapter my_agent.py --device cuda:0 \
    --model-label my-model --tasks all --output runs/my-model
```

The output directory must be new and inside the repository. A completed run writes `results.json`, `summary.json`, `run_config.json` and per-mode records. Retain all of them. The run may make paid API calls if your adapter uses a provider. Packaging and checking below make no model or API calls.

Extension tasks are submitted the same way. Select them with `--tasks extensions` or their IDs. Their rows keep the task's own protocol and appear in the Extension tasks group of the leaderboard, never among the paper tasks. See [ADD_TASK.md](ADD_TASK.md) to contribute a task.

## Package the saved scores

Copy [the metadata template](../examples/submission_metadata.json) to `runs/my-model/metadata.json`. Replace every field with the actual checkpoint revision, code revision, adapter, prompt, readout, precision, seeds, hardware and command. Set `model_name` to the exact label passed to `eval`. The `evidence_url` should link to reproducible instructions and evidence that you are permitted to share. Keep the adapter and prompt available at a fixed code revision. Describe how original sources were obtained.

```bash
python -m s1cogbench package --run runs/my-model \
    --metadata runs/my-model/metadata.json --id my-model-20261111 \
    --output submissions/my-model-20261111.json
python -m s1cogbench check-submissions
```

The package command reads the completed run, checks the aggregate against the saved mode result, and copies the aggregates and coverage counts. It preserves the original scoring denominator, which may count participants rather than actions. It does not copy raw trials, load an adapter, run inference, upload files or create a pull request.

## Open a pull request

Fork [the repository](https://github.com/dengzhe-hou/S1CogBench), add the generated JSON under `submissions/`, and open a pull request against `main`. Use the community-result pull request template and include your run evidence link. A maintainer checks the protocol, full counts, readout and evidence before merging. Material deviations should be discussed in the pull request instead of relabeling them as an existing protocol.

The automated check validates structure, counts and comparison contexts. It cannot establish that reported model outputs are genuine. Merging means the report was reviewed, not that the maintainers independently reran the model. Later corrections use another pull request, keeping Git history available. An error in someone else's entry can be reported with the [error report form](https://github.com/dengzhe-hou/S1CogBench/issues/new?template=correction.yml). Pending pull requests are not public leaderboard entries.

Only add the aggregate submission JSON. Restricted participant inputs and per-item participant outputs must not be placed in a public pull request or public artifact link. For gated sources, provide instructions for an authorized reader to obtain the original data and reproduce the score. Record access restrictions in the evidence documentation. Keep weights, credentials and provider keys out of the submission.

## Compare locally before submitting

Use the result explorer's “Add your results” button to open `results.json` or the packaged submission JSON. It stays in browser memory and is marked user-supplied. Select a matching task, mode, metric and comparison context. The local import does not upload or submit anything.

See [COMPARISON_GUIDE.md](COMPARISON_GUIDE.md) for the existing targets and action rules, [the result schema](../results/aggregate/result_schema.json) for aggregate fields, and [RELEASE_SCOPE.md](RELEASE_SCOPE.md) for the result release boundary.
