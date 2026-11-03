# Decision-time and readout controls

This directory contains the execution and merge code for the supplementary P2 and P3 studies described in [the supplementary controls](../../docs/SUPPLEMENTARY_CONTROLS.md). These scripts do not add results to the accepted aggregate snapshot.

## Required sources and runtime

Run from the S1CogBench repository root. Use the original source materials associated with [Namazova et al.](https://arxiv.org/abs/2508.07887) and the source links in [the catalogue](../../task_catalog.json). Obtain and retain their redistribution and access terms. Do not substitute a browser demonstration, a generated timeline, or a different participant split.

The task helpers expect the following original layout:

```text
external/paper_matched_centaur_evaluation/
  horizon/horizon_centaur_one.py
  horizon/horizon_llama_one.py
  horizon/data/in/timeline_structure.csv
  wcst/Data/json_wcst_dataset.npy
  wcst/generative_predictive_llms_eval/generative_centaur_70B.py
  wcst/generative_predictive_llms_eval/generative_llama_70B.py
```

The source and participant files are not redistributed in this directory. The existing `python scripts/fetch_public_sources.py cogbench` command fetches only its declared CogBench revision and does not acquire the materials above. See [data access](../../docs/DATA.md).

Set up the frozen provider checkpoints and SDKs described in [models and adapters](../../docs/MODELS.md). The main model runtime and the Intern/Qwen runtime have different matching dependency versions; they are not installed by `requirements.txt`. The current main runtime uses Python 3.11, PyTorch 2.10.0+cu128 and Transformers 5.3.0. The Intern/Qwen runtime uses Python 3.12, PyTorch 2.9.1+cu128 and Transformers 5.14.1, together with the matching fused linear-attention implementation. The code records the actual configuration. These version notes do not constitute a clean-machine installation recipe.

Use an allocated CUDA device and explicit CPU and memory limits appropriate to your scheduler. The examples below use `cuda:0`; a visible device setting selects a device and does not impose a resource limit. Weights, private deployment scripts, credentials, participant transcripts, and raw outputs are not bundled.

## P2, generated reasoning

Choose a fresh output directory and a verified main or Qwen runtime. `$ROOT` below is the absolute path of this repository.

```bash
ROOT="$PWD"
PY_MAIN=/path/to/verified-main-environment/bin/python

"$PY_MAIN" -B experiments/decision_controls/p2/run_p2.py \
  --model llama --task horizon --phase pilot \
  --root "$ROOT" --output runs/p2-llama-horizon/pilot --device cuda:0

"$PY_MAIN" -B experiments/decision_controls/p2/run_p2.py \
  --model llama --task horizon --phase full \
  --root "$ROOT" --reuse-pilot runs/p2-llama-horizon/pilot \
  --output runs/p2-llama-horizon/full --device cuda:0
```

Use `--task wcst` for the other task and `--model qwen` with its verified runtime for Qwen. The Qwen runtime must expose the same fused kernel implementation as the retained configuration. The second command requires a successful 200-decision measurement prefix and checks its exact original state, trial identity, and legal options before reusing responses.

Generation retains the original per-model task prompt family and changes the response instruction to reasoning followed by exactly one legal `FINAL: X` line. Llama retains the original tokenizer handling, including its special-token behavior. Qwen uses its native thinking-enabled template. Both use greedy BF16 generation with a 1,024-new-token maximum and no answer retry. Complete state, generated text, token IDs, counts, elapsed time, truncation, and invalid-response flags are saved to the chosen output directory. No probabilities or human-response NLL are invented from the text.

Invalid responses remain recorded. Original Horizon has no reward transition for an illegal action. The source Centaur WCST formatter likewise cannot advance an invalid key, while the original Llama WCST feedback path explicitly supports an invalid answer. The runner preserves those original differences and marks unfinishable trajectories incomplete rather than reporting a score from only the successful subset.

Independent original participants can be partitioned without reducing the full sample:

```bash
"$PY_MAIN" -B experiments/decision_controls/p2/prepare_shards.py \
  --root "$ROOT" --model llama --task horizon --parts 4 \
  --pilot-dir "$ROOT/runs/p2-llama-horizon/pilot" \
  --results-root "$ROOT/runs/p2-llama-horizon/shards" \
  --output runs/p2-llama-horizon/partition.json

"$PY_MAIN" -B experiments/decision_controls/p2/run_p2_shard.py \
  --manifest runs/p2-llama-horizon/partition.json --shard-id s000 --device cuda:0

# Run each remaining manifest shard once, on separately admitted resources.
"$PY_MAIN" -B experiments/decision_controls/p2/merge_shards.py \
  --manifest runs/p2-llama-horizon/partition.json \
  --root "$ROOT" --output runs/p2-llama-horizon/merged
```

The merge verifies the complete original participant union, trial identities, action-to-reward or feedback mapping, and exactly-once reuse of the measurement prefix before producing full-task metrics. A partial shard is not a full-task result.

## P3, matched legal readout

```bash
"$PY_MAIN" -B experiments/decision_controls/p3/run_matched.py \
  --root "$ROOT" --model llama8b \
  --output runs/p3-llama8b --tasks horizon wcst \
  --readouts argmax sample --device cuda:0
```

The other model IDs are `laya`, `intern_decision_4b`, `minitaur8b`, and `qwen35_4b`. Use each model's verified runtime and SDK. `--conditions horizon:argmax wcst:sample` can select explicitly named conditions when exact compatible results for the complementary conditions have already been retained. It is not authorization to treat a partial set as the full comparison.

Both legal readouts use the same frozen model, complete source history, original participant seed, and task scoring. Argmax ties use legal-option order. Sampling uses the original per-participant NumPy seed. Llama and Minitaur legal conditioning is disclosed separately from old full-vocabulary decoding. Original per-model prompt families remain, so this is a common-readout comparison rather than a claim that all prompts are identical.

## Validation and licensing

The published files are the same runner implementations used by the current study. Direct preparation checks cover syntax, command arguments, participant partition, response parsing, and prefix reuse. Complete neural execution and result acceptance remain in progress. No clean-machine end-to-end neural reproduction is claimed here.

Original S1CogBench software is covered by the repository's [MIT license](../../LICENSE). Upstream task code, data, SDKs, and checkpoints retain their own terms.
