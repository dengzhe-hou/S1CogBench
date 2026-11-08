# Prepare a comparison file

Follow [ADD_MODEL.md](ADD_MODEL.md) to create a run and [COMPARISON_GUIDE.md](COMPARISON_GUIDE.md) to match the evaluated protocol. Native and sampled actions must remain distinct when viewing or submitting results.

The homepage can display a local comparison next to a matching accepted protocol. It does not upload files, run inference, publish results or add an official ranking.

1. Select the task, evaluation mode, metric and exact comparison context in the result explorer.
2. Download a matching result template. Keep its task, protocol, mode, metric, metric variant, scope, target, unit and aggregation only if your run actually matches them.
3. Fill the measured value, original value text, value status and full sample count. Preserve infinite and undefined values. Identify the model and disclose the actual checkpoint, prompt, readout, inference mode, precision, seed and runtime context in the accompanying notes and run record.
4. Use “Add your results” to load the JSON. It stays in browser memory and is marked user-supplied. “Download view” exports the selected rows without changing the accepted snapshot.

For a maintainable contribution, retain raw predictions, command, source revision, original input IDs, scorer output and per-condition sample counts. Explain material protocol deviations. Human-response NLL, autonomous reward, event-probability reports and controlled latency have different targets. Shared-device timing does not become an isolated speed comparison.

`python -m s1cogbench eval` writes `results.csv` and `results.json` in this format: one row per mode for its Table 3 primary target, with the comparison context of the accepted rows, the original value text, value status and sample count, and `status` `user_supplied_full_protocol` (or `user_supplied_incomplete` when a mode stopped). Add the run record (`run_config.json`) and the raw outputs under `<task>/<mode>/`.

Use `results/aggregate/result_schema.json` for the aggregate format. The homepage's import checks structural fields; it cannot verify that a submitted result was measured honestly, that all original inputs were used, or that checkpoint settings matched. Full acceptance uses the raw task-specific outputs and original scorer.

No public submission service or automatic acceptance endpoint is active.
