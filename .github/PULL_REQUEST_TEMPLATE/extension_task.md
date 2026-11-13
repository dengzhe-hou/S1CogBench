# Extension task

Add one paradigm under `extensions/<task_id>/`. Follow [the task guide](https://github.com/dengzhe-hou/S1CogBench/blob/main/docs/ADD_TASK.md). Model results for the task go in a later pull request.

- [ ] The source study or benchmark is cited in `TASK['source']`.
- [ ] The data license permits this use, and only redistributable inputs are committed.
- [ ] The task is a new paradigm, not a new seed, prompt, condition or transcription of an existing task.
- [ ] Official splits, participant IDs and histories are kept, and every change is listed in `TASK['adaptations']`.
- [ ] `n_inputs` and `n_scored` are the full source counts, and a full uniform run completes.
- [ ] Trials show the model only what a participant could see at that point.
- [ ] `TASK['reference']` gives at least one reference score with its provenance.
- [ ] `python -m unittest discover -s tests -p 'test_*.py'` passes.

Uniform control on the full task (value and n from `results.json`):

Reference score and how it was obtained:

Differences from the original protocol:
