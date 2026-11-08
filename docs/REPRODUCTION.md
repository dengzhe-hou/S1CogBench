# Reproduction of accepted reference values

This record describes the released arXiv snapshot and its measured validation. Later manuscript evidence is versioned separately as described in [RELEASE_SCOPE.md](RELEASE_SCOPE.md). New offline tooling tests do not extend the neural-reproduction claims recorded below.


`python -m s1cogbench eval` dispatches every mode to the original runner and scorer. To check that it changes no protocol, prompt, history, sample, seed or scorer, reference adapters were run through it on the inputs of the 2026-10-26 CPU validation (rebuilt on the lab cluster from the project snapshot), on CPU through Slurm on c01, with the pins of `requirements.txt`.

## CPU reference: 27/27 modes identical

`examples/cpu_protocol_validation_agent.py` is the deterministic policy of the 2026-10-26 validation (`examples/bayes_event_agent.py` for the Bayesian urn, as there). Each mode must have the same full input and original scored counts, the same Table 3 primary value (exact floating-point equality), the same original summary, and the same per-item output files row by row (responses, predictions, decisions, trajectories, trial and score tables). Excluded are timing fields, absolute paths, the `--model-label` text that the CogBench scorers store as `engine`, and two labels the validation driver wrote itself: the interface name of the nonliteral sliders (whose fixed 50 % report the driver supplied inline; the CLI asks the adapter) and the WCST full-vocabulary status sentence.

The validation ran Python 3.11.5. Under Python 3.11 (3.11.15) the CLI reproduces it in 27 of 27 modes. Under Python 3.12, 21 modes are identical and 6 differ only in float values that come from the built-in `sum()` (per-row expected reports, per-episode reward sums and NLL aggregates), by at most 3e-14 relative. Python 3.12 made float `sum()` compensated; the inputs, choices, trajectories, counts and every other field are identical.

BART: the accepted five-model rows use 20 runs (900 balloons); the validation ran the 10-run code default. The CLI runs the accepted 20 runs, and its first 10 are compared with the validation's 10.

## Uniform: 18/18 reproducible values match Table 3

Table 3's Uniform column is the expected score of uniform choice among the legal options (control jobs of 2026-10-29). `examples/uniform_reference_agent.py` (uniform probabilities, first legal key as native choice, 0.5 reports) was run through the CLI on all 27 modes. For human-NLL targets the CLI's own score is the Uniform value ("scorer"). For accuracy and numeric-report targets the expectation is recomputed from the CLI's scored items ("items"): the probability mass on the correct key, or the mean absolute error of a uniform report against each item's posterior or human value. 15 of these 18 agree to 1e-12. The others match to the printed precision: Horizon and WCST differ by the scorers' epsilons (-log(p + 1e-8), -log(p + 1e-12)), the rest by accumulated float rounding over up to 108,248 rows. The other 9 Uniform values are analytic expectations over environments and action sequences; a run of one fixed policy is one sample of them, so no CLI run reproduces them.

| Mode | CPU reference: validation | CLI | Python 3.11 | Python 3.12 | Uniform: Table 3 | reproduced |
| --- | --- | --- | --- | --- | --- | --- |
| Bayesian urn | 0 | 0 | identical | identical | 0.349467975288 | 0.349467975288 (items; exact) |
| Multiple-cue judgment | 3.11768307131 | 3.11768307131 | identical | identical | 2.19722457734 | 2.19722457734 (scorer; exact) |
| Multi-attribute choice | 0.848432094793 | 0.848432094793 | identical | identical | 0.69314718056 | 0.69314718056 (scorer; exact) |
| Probability coherence | 31.9943089431 | 31.9943089431 | identical | float sums only, rel. 4e-16 | 32.1338525316 | 32.1338525316 (items; exact) |
| SHJ categories | 0.5 | 0.5 | identical | identical | 0.5 | 0.5 (items; exact) |
| Medin categories | 0.49746031746 | 0.49746031746 | identical | identical | 0.5 | 0.5 (items; exact) |
| Weather prediction | 0.4975 | 0.4975 | identical | identical | 0.5 | 0.5 (items; exact) |
| RLWM | 0.337831452817 | 0.337831452817 | identical | identical | 0.333333333333 | 0.333333333333 (items; exact) |
| Recent probes | 0.500529941706 | 0.500529941706 | identical | identical | 0.5 | 0.5 (items; exact) |
| WCST, prediction | 1.93737332575 | 1.93737332575 | identical | identical | 1.38629436112 | 1.38629436112 (scorer; diff 4e-12) |
| WCST, autonomous | 0.242106605256 | 0.242106605256 | identical | identical | 0.25 | analytic, not a run output |
| Restless bandit | 0.52 | 0.52 | identical | identical | 0.5 | analytic, not a run output |
| Instrumental learning | 0.754166666667 | 0.754166666667 | identical | identical | 0.75 | analytic, not a run output |
| Horizon, prediction | 0.844671936594 | 0.844671936594 | identical | identical | 0.69314718056 | 0.69314716056 (scorer; diff 2e-08) |
| Horizon, autonomous | 0.53202764977 | 0.53202764977 | identical | identical | 0.5 | analytic, not a run output |
| Stationary bandits | 0.114804487179 | 0.114804487179 | identical | float sums only, rel. 3e-14 | 0.125 | analytic, not a run output |
| Hybrid sampling | 1.87695419954 | 1.87695419954 | identical | float sums only, rel. 2e-16 | 1.38629436112 | 1.38629436112 (scorer; exact) |
| Mouselab, prediction | 2.21698328053 | 2.21698328053 | identical | identical | 1.63363484604 | 1.63363484604 (scorer; exact) |
| Mouselab, autonomous | 9.28526315789 | 9.28526315789 | identical | identical | 7.00775792911 | analytic, not a run output |
| BART | 0.0822222222222 (10 runs) | 0.0822222222222 (first 10 of 20) | identical | identical | 0.837727864668 | analytic, not a run output |
| Temporal discounting | 0.7 | 0.7 | identical | identical | 0.5 | analytic, not a run output |
| Two-step | 0.5 | 0.5 | identical | identical | 0.496289328684 | analytic, not a run output |
| False-belief QA | 0.499499499499 | 0.499499499499 | identical | identical | 0.5 | 0.5 (items; exact) |
| Politeness | 1.36396314629 | 1.36396314629 | identical | float sums only, rel. 3e-15 | 1.01391209922 | 1.01391209922 (scorer; exact) |
| Pragmatic interpretation | 1.7236120365 | 1.7236120365 | identical | float sums only, rel. 4e-15 | 1.28226831751 | 1.28226831751 (scorer; exact) |
| Nonliteral number | 27.1976666667 | 27.1976666667 | identical | identical | 34.7690393182 | 34.7690393182 (items; exact) |
| Self-assessment | 2.81619849315 | 2.81619849315 | identical | float sums only, rel. 2e-16 | 2.03927233032 | 2.03927233032 (scorer; diff 2e-12) |

## Adapter templates

No API call was made. Every saved exchange of the accepted 2026-10-30 runs was replayed through the templates with a local transport:

- `decision_api_choice.py`: 47,505 Jev choice responses in 10 tasks, all with identical request bodies and identical native choice, raw and normalized probabilities; 330 Jev nonliteral event answers, identical bodies and reports.
- `generative_api_choice.py`: 20,132 Claude Haiku responses in 5 tasks (letter and digit keys), all with identical message bodies and identical parsed answer and invalid flag (12 invalid).
- `hf_causal_lm_choice.py`: a five-call smoke run on CPU with Qwen3-0.6B (float32) through 15 modes covering the chat template, multi-token integer replies, numeric reports and Restless confidence. It has not been compared with the accepted open-model rows, which used model-specific adapters.

`python tests/test_adapter_templates.py` runs the parser and readout cases offline.

## Table 3

`scripts/add_table2_api_rows.py` added 110 API rows and nine references to the aggregate files from the closed acceptance records. The Uniform column (27 rows), the eight older references, the analytic Bayesian reference, the five open-model temporal cells, the RLWM argmax accuracy and the Horizon greedy replay followed, each read from its record. Each row validates against the result schema, and all earlier rows keep their JSON text and CSV bytes. Each n equals the accepted n of its comparison context. Only the two clipped Jev rows have no open-model row in the same context ([API_EXTENSION.md](API_EXTENSION.md)).

`python scripts/build_table2.py` rebuilds the whole table from `results.json` and `reference_results.json`. The paper prints the Opus and Sonnet columns in a sentence of its appendix. All 297 cells, with targets, directions, marks, reference types and domains, match the table in the paper's latest PDF, and the bold and gray marks match its source. The 171 Uniform, open-model and reference values recorded when the table was generated are equal exactly, and the eight older references equal their higher-precision numbers in [BASELINES.md](BASELINES.md).

## First real-model pass

Llama-3.1-8B-Instruct ran through `python -m s1cogbench eval --smoke --smoke-calls 20` in every mode, with the software versions of the accepted runs (torch 2.10.0+cu128, transformers 5.3.0, accelerate 1.13.0) on one `3g.40gb` MIG slice of an A100 80GB. The job took 6 minutes 34 seconds, about 0.11 GPU-hours, and made no paid call.

The real adapter passes through the original runner of every mode. The one failure is an adapter mismatch: `examples/llama8b_event_agent.py` reads an urn question and stops on nonliteral number. `examples/llama8b_social_numeric_agent.py`, the numeric report of the accepted run, runs that mode.

The first 20 decisions of 23 modes were compared with the saved accepted outputs. A mode was to match if legal keys, native choice, identity and every probability were within 1e-3. Prompts, legal keys, input tokens and source IDs are identical everywhere they are recorded. Only nonliteral number matches, exactly. The other 22 modes miss the tolerance: probabilities differ by up to 0.10, native choices change in 11 of the compared decisions, and two of 20 Bayesian reports differ (0.60 against 0.53). Identical counts decisions with no difference at all. In Restless bandit the 10 confidence reports are identical and the 10 choice probabilities are not.

| Mode | Decisions | Identical | Largest probability or report difference | Native choice changes | Identity checked by |
| --- | ---: | ---: | ---: | ---: | --- |
| Multi-attribute choice | 20 | 8 | 0.046 | 0 | source ID, input tokens |
| SHJ categories | 20 | 2 | 0.062 | 0 | source ID, input tokens |
| Medin categories | 20 | 11 | 0.052 | 0 | source ID, input tokens |
| Weather prediction | 20 | 5 | 0.013 | 0 | source ID, input tokens |
| Recent probes | 20 | 5 | 0.043 | 1 | source ID, input tokens |
| False-belief QA | 20 | 10 | 0.062 | 1 | source ID, input tokens |
| Politeness | 20 | 7 | 0.060 | 0 | source ID, input tokens |
| Pragmatic interpretation | 20 | 1 | 0.078 | 0 | source ID, input tokens |
| Multiple-cue judgment, compact history | 20 | 0 | 0.049 | 1 | source ID, input tokens |
| RLWM | 20 | 1 | 0.093 | 2 | source ID, input tokens |
| Mouselab, prediction | 20 | 0 | 0.076 | 1 | prompt, input tokens |
| Hybrid sampling | 20 | 0 | 0.101 | 1 | input tokens |
| Two-step | 20 | 6 | 0.030 | 0 | prompt, input tokens |
| BART | 20 | 8 | 0.060 | 0 | prompt, input tokens |
| Temporal discounting | 9 | 5 | 0.007 | 0 | prompt, input tokens |
| Instrumental learning | 20 | 7 | 0.031 | 0 | prompt, input tokens |
| Stationary bandits | 20 | 4 | 0.037 | 0 | input tokens |
| Mouselab, autonomous | 20 | 0 | 0.026 | 0 | prompt, input tokens |
| Bayesian urn | 20 | 18 | 0.070 | n.a. | prompt, source ID, input tokens |
| Restless bandit | 20 | 10 | 0.052 | 0 | prompt, input tokens |
| Probability coherence | 20 | 0 | 0.091 | 2 | source ID, input tokens |
| Self-assessment | 20 | 0 | 0.059 | 2 | source ID, input tokens |
| Nonliteral number | 20 | 20 | 0.000 | n.a. | source ID, input tokens |

Every difference between log probabilities lies on the 1/32 grid of BF16 logits, so these are rounding steps of the logits, mostly 1/16 or 1/8, and not a change of prompt or scorer. The saved accepted run on a whole A100 80GB and the latency study on a whole A100-SXM4-40GB are identical on all 2,576 decisions they share, so this slice is the outlier. The whole-card rerun below shows that the CLI is not the cause. Expect Llama numbers from a MIG slice to differ from the accepted rows.

Four modes cannot be compared. The accepted Llama rows of Horizon and WCST used the published Llama prompts and readouts, not an adapter, so `llama8b_choice_agent.py` through the CLI is a different condition. All four ran through their runners. Twenty decisions also say nothing about a whole-task score.

## Whole-card rerun

The same pass ran again on a whole A100-SXM4-40GB with MIG disabled and driver 615.71.09, through the same CLI, adapters and prompts. The modes took about 5 minutes.

All 23 compared modes, 449 decisions, are identical to the saved accepted outputs, with a largest difference of exactly 0 in every probability and report. This includes the 22 modes that missed the tolerance on the MIG slice and nonliteral number, which already matched. Native choices, legal keys, input tokens and the recorded prompts and source IDs match as well.

Autonomous Mouselab ran in a second start of the same VM. In the first start its Julia server stopped, because the copy of the sources prepared for that run left out the fetched file `mdps/base/4O8nSiCFkbB`. That was a mistake in preparing the run, not a failure of the CLI. The other results are as in the first pass: the event adapter stops on nonliteral number, and the four Horizon and WCST modes ran but cannot be compared.

The Python environment was the one of the latency study (torch 2.10.0+cu128, transformers 5.3.0, tokenizers 0.22.2, NumPy 1.26.4, safetensors 0.7.0), with the CLI's other dependencies added at the versions of the first pass.

This rules out the CLI, the adapters and the prompts as the source of the differences on the MIG slice. It changes the slice and the driver together (595.58.03 on the slice, 615.71.09 here), so it does not say which of the two is responsible.

Llama-3.1-8B on a whole A100 is therefore supported through the CLI. This means that its example adapters run in all 27 modes, and that the 23 comparable modes reproduce the accepted outputs exactly for the decisions compared. In Horizon and WCST it means only that the modes run, because their accepted rows used published prompts. Nonliteral number needs `examples/llama8b_social_numeric_agent.py`. Support does not extend to MIG slices or to the other neural models, and this short pass does not repeat any whole task.

## Not covered

Apart from the short passes above, these checks run reference policies through the original scorers. They do not repeat the five-model neural inference, do not acquire the gated inputs independently (the validation inputs were reused), and do not compare the Hugging Face template with accepted rows.
