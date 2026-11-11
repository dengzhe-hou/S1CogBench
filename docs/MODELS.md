# Models and adapters

The current seven-model maintainer results include Nox and CLM. Their checkpoint and native readout details are in [LATEST_RESULTS.md](LATEST_RESULTS.md). Earlier adapter-validation records below retain their original scope.

See [ADD_MODEL.md](ADD_MODEL.md) for the single integration workflow and [COMPARISON_GUIDE.md](COMPARISON_GUIDE.md) for action-rule scope. The model list below describes the released aggregate snapshot. Current seven-model paper evidence is indexed separately in [RELEASE_SCOPE.md](RELEASE_SCOPE.md).

The released arXiv snapshot evaluated five frozen open models. The list comprises Laya English, Intern-Decision-4B, Llama-3.1-8B-Instruct, Minitaur8B and Qwen3.5-4B. Jev 1.14 (`typesafe/jev-1.14`, snapshot `typesafe/jev-1.14-20261014`) was evaluated through its decision API on all 27 modes, with the readout of `examples/templates/decision_api_choice.py`. Claude 5.5 (Opus, Sonnet and Haiku) was evaluated through its API on selected modes, read from a strict final `FINAL:` line of its text; `examples/templates/generative_api_choice.py` generalizes that parser to any legal keys. Claude returns no option probabilities, so its human-NLL cells are n.a. The exception is Politeness, Pragmatic interpretation, Multi-attribute choice and Horizon prediction, where Claude Haiku states each option's probability on a final JSON line (two-decimal values divided by their sum, as for Jev, no floor). Those cells are stated probabilities, a different readout from the open models' internal probabilities. Settings and records are in [API_EXTENSION.md](API_EXTENSION.md), and time and cost per target in [API_TIME_COST.md](API_TIME_COST.md).

Weights are not bundled. Obtain them, and any inference code, from the original providers under their terms.

## Roles in the paper

| Model | Role | Recorded interface |
| --- | --- | --- |
| Laya English | Target structured decision model | Local checkpoint and SDK choice or numerical head |
| Intern-Decision-4B | Target structured decision model | Official checkpoint inference implementation and legal-symbol probabilities |
| Llama-3.1-8B-Instruct | General-purpose language-model control | Task-specific original or constrained-response readout, identified per result |
| Minitaur-8B | Human-behavior-trained control | Checkpoint-specific response formatting and recorded readout |
| Qwen3.5-4B | Backbone-family comparison for Intern-Decision | Released checkpoint with explicit choice formatting |

Same-family models are not assumed to have matched training data or training compute.

## Evaluate your own model

An adapter is a Python file that defines `create_agent(device)`. The returned object implements the methods the modes call:

| Method | Called by | Returns |
| --- | --- | --- |
| `choice_probabilities(state, question, output_keys)` | 25 modes | one probability per legal key, in the given order, finite and summing to 1 |
| `last_native_choice` (attribute set by the call above) | optional; required by both WCST modes and by the stationary bandits under the native rule | one of `output_keys` |
| `posterior_probability(state, question)` | Bayesian urn, nonliteral number | a probability in [0, 1]; `None` marks an invalid nonliteral report |
| `confidence_probability(state, chosen, original_specifications)` | Restless bandit | a probability in [0, 1] |

`output_keys` can hold a single key. In Mouselab prediction, 63 of the 20,363 requests offer only STOP. The only valid answer there is `[1.0]`, so an adapter can return it without calling a model that rejects one-option questions. `python -m s1cogbench list` prints the methods each mode calls. A module-level `PROBABILITY_READOUT = False` declares that the adapter has no action probabilities (for example a model read from generated text); the CLI then leaves human-NLL targets undefined instead of scoring a placeholder.

Only the visible state, the typed question and the ordered legal keys reach the adapter. Targets, human responses, hidden environment values and future feedback stay with the scorer. Autonomous modes act on `last_native_choice` when the adapter sets it and otherwise on the argmax of the probabilities, with ties taking the first key; `--action-rule sample` samples from the probabilities in supported autonomous modes, including WCST. Restless bandit keeps its source-native rule and rejects a sampling request before inference. See `python -m s1cogbench plan --tasks all --json`. Native choice and sampling are different execution conditions and are recorded in each result row.

```bash
python -m s1cogbench eval --adapter my_agent.py --device cuda:0 --model-label my-model \
    --tasks all --output runs/my-model
```

## Templates

Copy one of these and adjust it:

| Template | Readout |
| --- | --- |
| [`examples/templates/hf_causal_lm_choice.py`](../examples/templates/hf_causal_lm_choice.py) | Hugging Face causal LM. Temperature-one softmax over the legal replies: next-token logits when every label is one token, otherwise the summed log-probability of each complete reply with its end-of-turn token. Native choice is the argmax. Set `S1_HF_MODEL`. |
| [`examples/templates/decision_api_choice.py`](../examples/templates/decision_api_choice.py) | Decision API in the Jev style. The native choice is the action; the returned two-decimal probabilities are divided by their returned sum, with no floor. Key from `S1_DECISION_API_KEY`. |
| [`examples/templates/generative_api_choice.py`](../examples/templates/generative_api_choice.py) | Text-generation API (Anthropic Messages) read through the retained strict `FINAL: X` line parser, generalized to the legal keys. No action probabilities; NLL targets are n.a. Key from `ANTHROPIC_API_KEY`. |

The API templates read their keys from the environment only and never write them to files or logs. Every request is billed by the provider, and the full suite is large (Self-assessment alone has 108,248 responses). A smoke run makes five calls per mode:

```bash
python -m s1cogbench eval --smoke --adapter my_decision_agent.py --output runs/api-smoke
```

Add `--smoke-calls N` for more calls per mode. The decision and generative templates were checked by replaying every saved request and response of the accepted Jev and Claude Haiku runs through them, without new API calls: identical request bodies, native choices, probabilities and parsed answers (see [REPRODUCTION.md](REPRODUCTION.md)). The Hugging Face template was run on CPU with a small local model; it has not been run against the accepted Llama, Minitaur or Qwen rows, which used the model-specific adapters below. [REPRODUCTION.md](REPRODUCTION.md) reports a short pass of the Llama adapters through the CLI.

## Adapters of the accepted rows

| Model | Local setup | Example adapter |
| --- | --- | --- |
| Laya English | `models/laya-english`; original Laya SDK at `external/laya` | `examples/laya_choice_agent.py` |
| Intern-Decision-4B | `models/intern-decision-4b`, including the checkpoint's official `inference.py` | `examples/intern_decision_choice_agent.py` |
| Llama-3.1-8B-Instruct | `models/llama-3.1-8b-instruct` | `examples/llama8b_choice_agent.py` |
| Minitaur8B | `models/minitaur8b` or `S1COGBENCH_MINITAUR_MODEL_DIR` with the evaluated provider revision | `examples/minitaur8b_choice_agent.py` |
| Qwen3.5-4B | `models/qwen3.5-4b` | `examples/qwen35_4b_choice_agent.py` |

The Intern and Qwen examples record their model IDs and evaluated revisions. Several tasks used task-specific variants of these adapters (event, integer-sequence and Restless files in `examples/`); the task runner recorded in each accepted row names the one used. Preserve each adapter's precision, temperature, prompt and decision rule when reproducing its results. `requirements.txt` installs the CPU task environment only; the neural adapters need their own PyTorch, Transformers, SDK and checkpoint dependencies. The CLI starts each task's worker with its own `PYTHONPATH`, so code outside the active environment is not visible to an adapter. List its directories in `S1COGBENCH_EXTRA_PATH`, separated as in `PYTHONPATH`; the CLI's own adapter check and every worker then import from them.

Through the CLI, Llama-3.1-8B on a whole A100 is the one neural setup supported so far. The other adapters in this table have not run through it, and a MIG slice gives different Llama numbers. See [REPRODUCTION.md](REPRODUCTION.md).

Llama used these adapters by mode:

| Modes | Adapter |
| --- | --- |
| Bayesian urn | `examples/llama8b_event_agent.py` |
| Nonliteral number | `examples/llama8b_social_numeric_agent.py` |
| Restless bandit | `examples/llama8b_restless_agent.py` |
| Probability coherence, Self-assessment | `examples/llama_integer_choice_agent.py` |
| Horizon and WCST, both modes each | No adapter: the published Llama prompts and readouts. `llama8b_choice_agent.py` through the CLI is a different condition |
| All other modes | `examples/llama8b_choice_agent.py` |

## Reference adapters

| File | Use |
| --- | --- |
| `examples/uniform_reference_agent.py` | Uniform legal probabilities, first legal key as native choice, 0.5 reports. Used by `--smoke`. |
| `examples/cpu_protocol_validation_agent.py` | The deterministic plumbing policy of the 2026-10-26 CPU validation. |
| `examples/bayes_event_agent.py` | Exact visible-information Bayes for the Bayesian urn (posterior MAE 0). |

None of these is a research baseline.
