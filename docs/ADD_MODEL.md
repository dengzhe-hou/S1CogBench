# Evaluate a new model

The entry point is `python -m s1cogbench eval`. A model adapter defines `create_agent(device)` and returns an object with the methods listed in [MODELS.md](MODELS.md). Keep model loading inside the factory so importing the module does not start inference.

Inspect the full suite before installing a model or obtaining participant data.

```bash
python -m s1cogbench plan --tasks all --json
python -m s1cogbench plan --adapter my_agent.py --tasks all --json
```

The adapter check parses Python and checks the factory signature. It does not import the file, create an agent, verify its numerical outputs or certify a paper reproduction. Mode records list required methods, input paths, full denominators, action rules and comparison contexts. Counts denote their stated units, not a common inference-call budget.

Copy an [adapter template](../examples/templates/) and follow [QUICKSTART.md](QUICKSTART.md) for dependencies and data. `doctor` checks runtime availability. Adapter-dependent `doctor` imports the module, so its top-level code must not load a model.

```bash
python -m s1cogbench doctor --adapter my_agent.py
python -m s1cogbench eval --adapter my_agent.py --device cuda:0 \
  --model-label my-model --tasks all --output runs/my-model
```

`eval` performs real inference and may incur provider charges. Use the original full protocol for reported comparisons. A diagnostic smoke run checks integration and is not a complete benchmark result. Retain checkpoint, readout, action rule, histories and source scoring.

After a complete run, compare `results.json` locally on the project page. Use `python -m s1cogbench package` to prepare the aggregate submission and open a GitHub pull request following [SUBMIT_RESULTS.md](SUBMIT_RESULTS.md). Raw participant records stay local under their original access terms.
