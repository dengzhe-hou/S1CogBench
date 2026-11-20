# Tooling changelog

## 2026-11-20

`--adapter hf:<model>[@<revision>]` runs a Hugging Face causal language model through `examples/templates/hf_causal_lm_choice.py` without an adapter file, in `plan`, `doctor` and `eval`. It sets `S1_HF_MODEL` and `S1_HF_REVISION` for the workers, records the spec as `adapter_spec` in `run_config.json`, and labels the results with the model name unless `--model-label` is given. Adapter files work as before and keep the default label `user_model`. `examples/colab_lite.ipynb` uses it to run the lite targets in Colab.

`doctor` and `eval` load the config of an `hf:` model once before any worker starts. A wrong id, a gated repository without access, an offline machine or an architecture that the installed Transformers lacks now stops the run with one message, instead of failing in every worker.

The Hugging Face template's base-model path now ends the prompt with `Answer:` and puts one space before each label. Before, the trailing space of `Answer: ` merged with the label's token in byte-level BPE tokenizers such as SmolLM2's, and choice modes stopped with "A legal label changes the tokenization of the prompt". Chat models take the chat-template path, which is unchanged, and no accepted result used the base-model path. A Restless bandit confidence reply that is not a number from 0 to 1 now raises an error that shows the reply, instead of a TypeError inside the runner, which has no rule for invalid reports.

## Candidate tooling changes for 2026-11-08

This source update uses package version `0.2.0rc1`. It does not create a separate release tag.

The new `plan` command lists the original 27 modes, required adapter methods, full scored units, input paths and comparison contexts. Optional adapter inspection checks syntax and the factory signature without importing the file or loading a model. It does not validate inference outputs.

WCST autonomous sampling now executes the submitted categorical probabilities and supplies that action to the following feedback history. Previously this CLI path always executed the native choice even when sampling was requested. The existing native and human-history paths retain their choices. Sampling keeps the original participant ID and documented NumPy seed convention, with no probability floor. This fix does not rewrite archived results or repeat model inference.

Text-only adapters declaring no probability readout cannot sample their parser placeholders. This is rejected before agent creation.

Restless bandit sampling is explicitly unsupported by its retained runner and now fails before adapter import. New sampled rows use separate protocol keys where a fixed key previously did not record the rule. Plans show effective action rules, and result metadata retain both the requested and effective rule.

The documentation gives one model-integration workflow and separates the historical public aggregate from later manuscript evidence. Existing published aggregate files remain unchanged.
