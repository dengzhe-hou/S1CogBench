# CPU execution and scoring validation

A separate source copy and independent package environments completed all 24 selected tasks in 27 named modes using the supplied original inputs and a legal-output reference policy. Every mode passed complete-input and original-scored-count checks.

The environment used Python 3.11.5, CPU PyTorch 2.10.0, and a fresh Julia 1.4.1 depot. These checks establish task execution and scoring with those inputs. They do not repeat inference for five neural checkpoints, acquire gated records independently, or grant redistribution rights.

All 27 modes now run through `python -m s1cogbench eval`, and running this validation's reference policy through it reproduces the table below; see [REPRODUCTION.md](REPRODUCTION.md). The earlier `scripts/benchmark_paper_tasks.py` keeps Horizon prediction and execution, Mouselab prediction and execution, Two-step execution, and probabilistic-posterior reporting. Source data and runtimes follow [DATA.md](DATA.md). Runtime records retain hardware and sharing context, and latency is not a human reaction-time measure.

| Task | Mode | Full inputs | Original scored count | Result |
| --- | --- | ---: | ---: | --- |
| Bayesian urn inference | primary_protocol | 400 | 400 | complete |
| Multiple cue judgment | primary_protocol | 6136 | 6136 | complete |
| Multi attribute choice | primary_protocol | 576 | 576 | complete |
| Probability coherence judgment | primary_protocol | 2460 | 2460 | complete |
| SHJ category learning | primary_protocol | 3664 | 3664 | complete |
| Medin category learning | primary_protocol | 6972 | 6972 | complete |
| Weather prediction | primary_protocol | 400 | 400 | complete |
| Reinforcement learning and working memory | primary_protocol | 52911 | 52911 | complete |
| Recent probes | primary_protocol | 3774 | 3774 | complete |
| Wisconsin Card Sorting Test | human_predictive | 12754 | 12754 | complete |
| Wisconsin Card Sorting Test | autonomous_generative | 12754 | 12754 | complete |
| Nonstationary bandit | primary_protocol | 1600 | 1600 | complete |
| Instrumental learning | primary_protocol | 960 | 960 | complete |
| Horizon | human_predictive | 6720 | 6720 | complete |
| Horizon | autonomous_generative | 34720 | 34720 | complete |
| Stationary multiarmed bandit | primary_protocol | 16 | 312000 | complete |
| Hybrid information sampling | primary_protocol | 163135 | 163135 | complete |
| Mouselab tree | human_predictive | 20363 | 20363 | complete |
| Mouselab tree | autonomous_generative | 23750 | 23750 | complete |
| Balloon Analogue Risk Task | primary_protocol | 450 | 450 | complete |
| Temporal discounting | primary_protocol | 1 | 1 | complete |
| Two-step decision task | primary_protocol | 4000 | 4000 | complete |
| False belief story qa | primary_protocol | 5994 | 5994 | complete |
| Politeness pragmatics | primary_protocol | 128 | 128 | complete |
| Pragmatic language interpretation | primary_protocol | 170 | 170 | complete |
| Nonliteral number interpretation | primary_protocol | 330 | 330 | complete |
| Performance self assessment | primary_protocol | 108248 | 108248 | complete |

BART ran the 10-run code default here (450 balloons). The accepted five-model rows, and `python -m s1cogbench eval`, use the 20 runs (900 balloons) of the accepted rows.
