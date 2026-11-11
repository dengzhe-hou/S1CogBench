# Community results

Add one JSON file per completed run through a pull request. Follow [the submission guide](../docs/SUBMIT_RESULTS.md) to package the saved CLI outputs. The filename must match `submission_id`.

Each submission can cover one or more complete task modes. Partial runs and smoke runs are not leaderboard entries. Undefined scores stay visible without a rank, and positive infinite NLL stays positive infinite. Task protocols, action rules and targets remain separate. There is no overall score.

The automated check validates the file structure, original counts and comparison contexts. Maintainers review the settings and supporting evidence before merging. A merged entry is a reviewed community report, not an independent rerun of the model. GitHub Pages then updates the community leaderboard automatically.

Only publish aggregates and shareable evidence. Keep restricted participant inputs and per-item outputs under the original provider's access terms. Do not place them in a public pull request or public artifact link. For restricted tasks, describe how an authorized maintainer can reproduce the score from the original source without redistributing that source.
