# Release scope

The active website release is `seven-model-20261110`. It publishes 189 accepted primary scores from seven open models across the original 27 modes, together with 70 existing Horizon and WCST legal-readout metrics. See [LATEST_RESULTS.md](LATEST_RESULTS.md) for the score sources and reproduction commands.

Current files live in `results/latest/`. The original five-model arXiv aggregate and table remain unchanged in `results/aggregate/`. Older result files are historical snapshots rather than the current website leaderboard. The suite protocol identifier stays `arxiv-2611.00911`, while result snapshots have separate release dates.

This release uses retained, accepted results. It makes no new model or paid API calls, changes no task sources or scoring definitions, and includes no restricted participant transcripts or model weights. The source tooling version remains `0.2.0rc1`. No separate package release or tag is created.

Follow [ADD_MODEL.md](ADD_MODEL.md) to integrate a model and [COMPARISON_GUIDE.md](COMPARISON_GUIDE.md) to match protocols. Existing measured neural reproduction and its limits remain in [REPRODUCTION.md](REPRODUCTION.md). Publishing accepted scores is separate from certifying all model adapters through the generic CLI.

Community contributions live separately under `submissions/`. After maintainer review and merge, GitHub Pages regenerates the per-target community leaderboard alongside the current maintainer results. The `package` and `check-submissions` commands read saved outputs only. Format checks and maintainer review are distinct from independent model reproduction.