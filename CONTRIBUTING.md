# Contributing to S1CogBench

There are three ways to contribute, and each one goes through a GitHub pull request reviewed by the maintainers.

| I want to… | Guide | What the pull request adds |
| --- | --- | --- |
| Evaluate my model | [docs/ADD_MODEL.md](docs/ADD_MODEL.md) | Nothing yet. Running a model needs only an adapter file on your side. |
| Put my model's scores on the leaderboard | [docs/SUBMIT_RESULTS.md](docs/SUBMIT_RESULTS.md) | One packaged JSON under `submissions/` |
| Add a new cognitive task | [docs/ADD_TASK.md](docs/ADD_TASK.md) | One directory under `extensions/` |

The 24 paper tasks and their protocols stay fixed, so results remain comparable with the paper. New tasks join the extension track, which has its own leaderboards.

Bug reports and corrections to existing documentation are welcome as issues or pull requests. Please never include restricted participant data, model weights, credentials or provider keys.
