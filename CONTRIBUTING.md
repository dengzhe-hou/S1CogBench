# Contributing to S1CogBench

Results and tasks reach the repository through GitHub pull requests reviewed by the maintainers. Error reports and paradigm suggestions use issue forms and need no code.

| I want to… | Guide | What I send |
| --- | --- | --- |
| Evaluate my model | [docs/ADD_MODEL.md](docs/ADD_MODEL.md) | Nothing yet. Running a model needs only an adapter file on your side. |
| Put my model's scores on the leaderboard | [docs/SUBMIT_RESULTS.md](docs/SUBMIT_RESULTS.md) | A pull request with one packaged JSON under `submissions/` |
| Add a new cognitive task | [docs/ADD_TASK.md](docs/ADD_TASK.md) | A pull request with one directory under `extensions/` |
| Suggest a paradigm without writing code | [Paradigm proposal](https://github.com/dengzhe-hou/S1CogBench/issues/new?template=propose_paradigm.yml) | An issue with the source, participants, data and license |
| Report an error in the documentation, code, data, results or leaderboard | [Error report](https://github.com/dengzhe-hou/S1CogBench/issues/new?template=correction.yml) | An issue with the location, the correct value and its evidence |

The 24 paper tasks and their protocols stay fixed, so results remain comparable with the paper. New tasks join the extension track, which has its own leaderboards.

A fix can also go straight to a pull request. Please never include restricted participant data, model weights, credentials or provider keys in an issue or pull request.
