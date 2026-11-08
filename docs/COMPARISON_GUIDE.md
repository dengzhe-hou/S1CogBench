# Compare results under the same protocol

Use `python -m s1cogbench plan --tasks all --json` to inspect the evaluation targets. Human-response prediction, autonomous execution and event reports are separate comparisons. Do not pool them into a cognitive score.

Match the task, protocol, mode, metric, metric variant, scope, target, unit and aggregation. Also inspect checkpoint, prompt, probability extraction, context retention, action rule and full sample counts. Matching column names alone does not establish comparable inputs or readouts. Latency additionally requires matched hardware and timing boundaries.

`--action-rule` affects autonomous action execution. It does not resample a person's recorded response in prediction modes. WCST categorical sampling now executes sampled legal actions and uses their feedback in subsequent prompts. It retains the original participant IDs and NumPy sampling convention with seed `10000 + subject_id`. Native probabilities and executed actions are stored separately. No probability floor is added.

Categorical sampling requires actual option probabilities. An adapter declaring `PROBABILITY_READOUT=False` cannot use a text-parser placeholder distribution for sampling. The evaluator rejects that combination before creating an agent or evaluation output.

Restless bandit retains its source-native runner. Requesting sampling for it stops before adapter import or inference. Other supported autonomous rules are listed by `plan`.

New sampled results receive a distinct protocol key when the original fixed key did not encode the rule. Previously written files and published aggregate values are unchanged. An old `sample` label alone cannot establish which actions were executed. Inspect the raw decisions before using such a record.

The public aggregate is the released arXiv snapshot. This tooling update does not add later manuscript tables or certify new neural inference. Follow [REPRODUCTION.md](REPRODUCTION.md) for existing measured reproduction limits and [RELEASE_SCOPE.md](RELEASE_SCOPE.md) for version boundaries.
