# Run the public 400-query CPU example

These commands are for a Linux server, from the release repository root, with Python 3.11 and Git installed. They run the analytical Bayes reference, so no neural weights, GPU or paid API is required. Original project software is MIT licensed. Original-provider data terms are preserved.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu.txt
python scripts/fetch_public_sources.py cogbench
python scripts/benchmark_paper_tasks.py list
python scripts/benchmark_paper_tasks.py export \
  --task probabilistic-posterior \
  --output runs/example_posterior_export
python scripts/benchmark_paper_tasks.py run \
  --task probabilistic-posterior \
  --adapter-file examples/bayes_event_agent.py \
  --device cpu \
  --output runs/example_bayes_posterior
python scripts/benchmark_paper_tasks.py score \
  --task probabilistic-posterior \
  --predictions runs/example_bayes_posterior/predictions.jsonl \
  --output runs/example_bayes_posterior_score
```

Use fresh output directory names. The runner intentionally refuses to overwrite an existing run. The fetch command obtains the exact CogBench revision recorded in `public_sources.json` (8e70d3399193bfad9ef5e63dc9aa6feeccabdf72). It does not download all other task assets.

The generated protocol contains 200 runs and two conditions, giving 400 queries. These are new simulated urn observations, not redistributed human records. The declared first 100 runs are separately reported at the original paper scale. Requests expose visible priors, likelihoods and observations; scoring targets are separate. Bayes has zero posterior MAE. The original strict 95th-percentile error trimming can retain no observations for exact Bayes, so its original trimmed score is undefined rather than an invented perfect score. Read the result's named metric, scope and undefined fields.

# Add a model

The real Python adapter entry is a local file defining `create_agent(device)`. An event-probability adapter returns an object with this method.

```python
class MyEventAgent:
    def posterior_probability(self, state, question):
        # Compute P(the event named by question | visible state).
        # Return a finite scalar in [0, 1].
        ...

def create_agent(device):
    return MyEventAgent()
```

Save your implementation as `examples/my_event_agent.py`, then run the same `run` command with that adapter path and a new output directory. This is an interface skeleton, not a ready model implementation. `examples/bayes_event_agent.py` is the existing runnable reference. Neural `*_event_agent.py` adapters require their provider's weights and inference environment; `requirements-cpu.txt` installs only the analytical CPU example's dependencies.

For legal-action tasks the factory returns an object with a different method.

```python
class MyChoiceAgent:
    def choice_probabilities(self, state, question, output_keys):
        # Return one finite, non-negative probability per supplied key,
        # in exactly that order. Probabilities must sum to 1.
        ...

def create_agent(device):
    return MyChoiceAgent()
```

Predictive Horizon and Mouselab accept this interface through `predict`. Generative Horizon, Two-step and Mouselab accept it through `run`. `--action-rule native` and `--action-rule sample` are separate protocols. The native rule uses the adapter's non-null `last_native_choice` when supplied, otherwise probability argmax in key order; the executed source is recorded. Do not treat a categorical choice probability as the probability that an action succeeds.

# Which commands exist

`scripts/benchmark_paper_tasks.py` supplies `list`, `export`, `score`, `predict`, `run` and `report`.

| Task key | Export / score | Predict | Run |
| --- | --- | --- | --- |
| probabilistic-posterior | yes | no | yes |
| horizon-predictive | yes | yes | no |
| mouselab-predictive | yes | yes | no |
| horizon-generative | no | no | yes |
| twostep-generative | no | no | yes |
| mouselab-generative | no | no | yes |

All 27 modes of the 24 evaluated tasks also run through one entry point, `python -m s1cogbench eval --adapter <file> --tasks all`, which dispatches each mode to the task-specific runner recorded in `tasks.json` (see `docs/QUICKSTART.md`). The seven deferred catalogue tasks have no runner.

The separate `scripts/validate_cpu_reuse_24.py --output runs/cpu_reuse24_v1` driver executed 24 tasks and 27 named modes in a new Python environment using supplied original inputs. It uses a CPU reference agent, not the five evaluated neural checkpoints. The complete validation requires additional dependencies and original assets, including task-specific Julia dependencies; follow `docs/CPU_REUSE_VALIDATION_24.md`. This command is not a replacement for neural evaluation or a claim that gated data can be downloaded without provider approval.

# Data and comparison boundaries

Original data, official splits, complete visible histories, legal actions and task-specific scoring remain attached to each named protocol. Psych-101 test transcripts require provider access and are not bundled. Other original repositories, archives, model weights and source terms are separate. Original S1CogBench project software is MIT licensed; that license grants nothing for upstream code, models or data, which keep their own terms.

Filter comparisons by task, protocol, mode, metric variant, scope, target, unit and aggregation. Keep original numerical and binned NLL separate. Keep full and compact histories separate. Human-response agreement is distinct from objective accuracy and autonomous reward. Latency depends on the saved hardware and interface and is not human reaction time. The website provides no aggregate cognitive score.
