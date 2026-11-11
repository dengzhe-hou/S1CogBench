# Current seven-model release

`primary_sources.json` retains the accepted unrounded points and their original score denominators for 189 primary cells. `expected_printed` comes from the current accepted manuscript table. `readout_sources.json` retains the 70 complete Horizon and WCST legal-readout metric rows.

Run `python scripts/build_latest_results.py` from the repository root to regenerate this directory's `results.json`, `results.csv`, `table2.json` and `table2.csv`. This is aggregate-only table construction, with no model or API calls.

The generated result collection retains historical additional rows and adds the current primary and readout records. `release_id` identifies updated records. Original five-model files remain under [results/aggregate](../aggregate/). See [the release guide](../../docs/LATEST_RESULTS.md) for versions, readouts and interpretation.
