# Data and original resources

Every task keeps its source participant IDs, official splits, complete histories and task-specific evaluation rules. Nothing is resplit, and official test participants are not replaced by training participants. [task_catalog.json](../task_catalog.json) records each task's papers, resource links, split, metrics and adaptation notes, along with its source collections, legal responses, feedback and scorer entry points. It holds the 24 evaluated tasks ([selected_tasks_24.json](../selected_tasks_24.json)) and seven retained but deferred paradigms. Seeds, conditions, and repeated trials do not create additional tasks, and grammar and logical-reasoning item domains form one self-assessment paradigm.

## Fetch

```bash
python scripts/fetch_public_sources.py list    # sources, access, license, modes
python scripts/fetch_public_sources.py all     # every source; gated ones print their steps
python -m s1cogbench doctor                    # which modes can now run
```

[public_sources.json](../public_sources.json) pins each source and maps its files to the layout the task loaders read (`data/`, `external/`, `literature/`, `.tools/`, all git-ignored). GitHub sources come as archives of the pinned commits, so git is not needed. Files already present are kept, so the command can be rerun. `--data-root DIR` writes the same layout elsewhere; pass the same `--data-root` to `doctor` and `eval`.

| Source | Upstream and pin | Access | Modes | Terms stated upstream |
| --- | --- | --- | --- | --- |
| `cogbench` | github.com/juliancodaforno/CogBench @ 8e70d33 | public | Bayesian urn, Restless, Instrumental, Horizon (both), BART, Temporal discounting, Two-step | none stated |
| `evolve` | github.com/allenanie/EVOLvE @ d9143f7 | public | Stationary bandits | MIT |
| `centaur_evaluation` | github.com/snamazova/centaur_evaluation @ d788dac | public | Horizon (both), WCST (both) | none stated |
| `coggym` | github.com/lance-ying/coggym @ a1cd9df: evaluation code and 26 EML files | public subset | False-belief QA, Politeness, Pragmatic interpretation, Nonliteral number | code MIT; data CC BY 4.0 compilation, original-study terms apply |
| `tomi` | github.com/facebookresearch/ToMi @ dea2bca, `tomi_balanced_story_types.zip` | public | False-belief QA | CC BY-NC 4.0 |
| `rlwm` | OSF wgzb6, file a58vd (Collins and Frank 2012) | public | RLWM | none set |
| `hybrid` | github.com/simonedambrogio/HybridModellingProject @ 3d343be (v1.0.0), 40 `X.csv`/`y.csv` files | public | Hybrid sampling | MIT |
| `four_task` | OSF er9ms (two Qualtrics files); marcelbinz/Llama-3.1-Centaur-70B @ f8038fb (three files) | public | Self-assessment | OSF none set; Centaur repository Apache-2.0 |
| `callaway` | OSF 6venh, `model-code.zip` (1.07 GB, downloaded once; 39 members kept) | public | Mouselab (both) | none set |
| `julia` | Julia 1.4.1 official build and the Manifest in `envs/mouselab-julia141` | public | Mouselab, autonomous | MIT |
| `psych101` | Hugging Face `marcelbinz/Psych-101-test` @ 88acfc1, `prompts_testing_t1.jsonl` | **gated** | Multiple-cue, Multi-attribute, Probability coherence, SHJ, Medin, Weather, Recent probes, Self-assessment (grammar half) | CC BY-ND 4.0; not to be redistributed |

"None stated" means the upstream has no license statement, not that reuse is permitted. Each source keeps its own terms; this repository bundles none of these files.

Two inputs have no published file and are derived from the published code: the Hybrid held-out folds (`data/hybrid_paper_matched/fold_assignments.csv`, built by the fetch script from the subjects' row counts with the 8-fold contiguous split of the authors' code) and the Mouselab participant folds (`derived/mouselab_callaway2022_official/folds.json`, written by `scripts/recover_mouselab_official_folds.jl` with Julia 1.4.1 from the original `MersenneTwister(123)` split).

## Gated: Psych-101-test

1. Sign in at huggingface.co and accept the conditions of [marcelbinz/Psych-101-test](https://huggingface.co/datasets/marcelbinz/Psych-101-test). Approval is automatic.
2. Create a read token and run `hf auth login`, or export `HF_TOKEN` in the current shell only.
3. Run `python scripts/fetch_public_sources.py psych101`. It downloads revision 88acfc17 (91,981,882 bytes) to `data/psych101/prompts_testing_t1.jsonl`.

The provider asks that the file not be redistributed. Do not commit it, and do not share the per-item outputs of these eight modes, which contain source responses. The regular test file is not a substitute for the logical-reasoning half of Self-assessment; that half comes from the `four_task` source.

## Julia for Mouselab autonomous

`fetch_public_sources.py julia` downloads the official Julia 1.4.1 build into `.tools/julia-1.4.1`, copies `envs/mouselab-julia141/{Project,Manifest}.toml` and runs `Pkg.instantiate()` into `.tools/julia-depot-1.4.1`. This clones the General registry and needs network access once. The first Mouselab run then precompiles the packages (about 100 seconds on the lab cluster; later starts take about 20). The other 26 modes do not need Julia.

## What does not run

- The seven deferred catalogue tasks are not part of the 27 modes. Two of them lack source inputs: Subway has no complete original reward records, and population-comparison confidence has no mapping from each participant's items to the country alternatives.
- Without Psych-101-test access, the eight modes listed above stop at `doctor`.
- Without Julia 1.4.1, Mouselab autonomous stops at `doctor`; Mouselab prediction does not need it.

Derived compact transcripts preserve original events, order, feedback and samples but change the textual presentation; they are separate protocols (multiple-cue judgment uses the compact history in the CLI, as in the accepted Intern, Laya and Qwen rows).
