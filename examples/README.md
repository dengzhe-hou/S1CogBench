# examples

Adapters connect a model to the benchmark. To add your own model, copy a file from `templates/` and follow [the model guide](../docs/ADD_MODEL.md).

| Files | What they are |
| --- | --- |
| `templates/` | Starting points for a Hugging Face causal LM, a decision API and a generative API |
| `uniform_reference_agent.py`, `uniform_choice_agent.py`, `restless_random_agent.py` | Uniform and random controls |
| `<model>_<interface>_agent.py` | The adapters of the evaluated models, such as `laya_choice_agent.py` or `llama8b_event_agent.py`. The interface part names the task family the adapter serves. Some names carry the date of the run that used them. |
| `bayes_event_agent.py`, `cpu_protocol_validation_agent.py` | Reference and validation agents |
| `submission_metadata.json` | The metadata template for [submitting results](../docs/SUBMIT_RESULTS.md) |

The paper adapters stay at their original paths, because the accepted result records in `results/` cite them.
