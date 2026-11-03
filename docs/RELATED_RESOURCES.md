# Related evaluation resources

This is a factual protocol comparison from the cited papers and their official resources. No performance ranking is implied.

| Resource | Evaluation source and split | Mode and visible history | Scoring targets and references | Reusable material |
| --- | --- | --- | --- | --- |
| CogBench (Coda-Forno et al., 2024) | Seven experiments; 40 language models; generated tasks and a fixed questionnaire. | Experiment prompts; feedback-dependent choices in sequential tasks. | Task performance and ten behavioral indices; human and random references. | Experiment runners, task scorers, and model adapters. |
| CogGym, Sept. 2026 v1 (Ying et al., 2026) | 258 experiments; 50 models; matched source trials and human responses. | Single-turn, non-interactive trials; JSON answers or verbalized response distributions. | Mean-response and distribution alignment; human split-half and random references. | EML specifications, model administration, and human-replication interfaces. |
| Psych-101 / Centaur (Binz et al., 2025) | 160 experiments; 90%/10% participant split; additional held-out tasks. | Human-history prediction; model-generated trajectories in Horizon and Two-step. | Response NLL and simulation statistics; domain-specific cognitive models. | Transcripts, model adapters, and analysis code; gated test-set access. |
| BanditBench / EVOLvE (Nie et al., 2025) | Context-free and contextual bandits; generated interaction trajectories. | Interactive choices; raw or summary histories; algorithm-guided conditions. | Reward and regret; UCB, Thompson sampling, and LinUCB references. | Bandit environments, agent interfaces, and evaluation runners. |
| System1-Arena (Chen et al., 2026) | Public-source and generated decision tasks and closed-loop environments. | Typed probability requests; decision models and general language models. | Probability quality, task outcomes, latency, cost, and reasoning budgets. | Shared decision interfaces and evaluation resources. |
| TypedBench (Sharma et al., 2026) | Generated evaluation suites with controlled wording, options, and length. | Static decision requests under matched presentation changes. | Calibration, framing sensitivity, asymmetric costs, and latency. | Evaluation suites. |
| DecisionPivot (Zhuang et al., 2026) | Generated task families with controlled diagnostic conditions. | Decision models and general language-model comparisons. | Probability ranking, calibration, and selective coverage. | Diagnostic task families and evaluation resources. |
| CogArena (Hou et al., 2026) | 13 procedurally generated paradigms. | Language and vision-language models. | Measurement consistency and predictive validity. | | |
| CogArena (Fernandez et al., 2026) | Ten browser-based behavioral experiments. | Multimodal agents interacting through an execution scaffold. | Task completion, performance, and behavioral signatures. | Interactive experiment implementations and agent evaluation code. |
| S1CogBench | 24 selected tasks; five open models and Jev; source participant and fold rules where applicable; fitted replay labeled separately. | Human prediction, autonomous execution, and event reports distinguished; original and compact histories retained. | Target-specific NLL, return, event Brier, and source condition effects; reference provenance disclosed. | Source mappings, adapters, scorers; CPU reference pipelines verified for 24 tasks in 27 modes. |

Reported experiments and the selected task inventory are different counting units. Numerical comparisons require matching the source split, information history, execution mode, response decoding, scoring target, and reference fit. The S1CogBench CPU checks use separately provided original inputs. They do not establish fresh neural-model deployment, independent public retrieval of restricted inputs, or redistribution permission. Model, metric, and cognitive-baseline completion remain separately recorded. The S1CogBench row describes the five-model suite; the [API extension](API_EXTENSION.md) is separate.

## References

- Marcel Binz, Elif Akata, Matthias Bethge, and others. 2025. A foundation model to predict and capture human cognition. Nature, 644(8078):1002-1009.
- Tianyu Chen, Xinyang Liu, Ruyi An, Zhendong Wang, and Mingyuan Zhou. 2026. System1-Arena: Which Decisions Can Be Left to System 1? arXiv:2610.21244.
- Julian Coda-Forno, Marcel Binz, Jane X Wang, and Eric Schulz. 2024. CogBench: a large language model walks into a psychology lab. In Proceedings of the 41st International Conference on Machine Learning, PMLR 235:9076-9108.
- Kianté Fernandez, Caitlin Chen, Jialu Zhou, Bartek Sadowski, Anthony C. Miceli, and Ian Krajbich. 2026. CogArena: Benchmarking Multimodal Agents on Interactive Behavioral Experiments. In Advances in Neural Information Processing Systems, Evaluations and Datasets Track.
- Allen Nie, Yi Su, Bo Chang, Jonathan Lee, Ed H. Chi, Quoc V Le, and Minmin Chen. 2025. EVOLvE: Evaluating and optimizing LLMs for in-context exploration. In Proceedings of the 42nd International Conference on Machine Learning, PMLR 267:46346-46376.
- Rahul Sharma, Andrew B. Duncan, Gaétan Marceau Caron, and Sebastian J. Vollmer. 2026. TypedBench: A Benchmark for Calibration, Framing Sensitivity, and Cost in System One Decision Models. arXiv:2610.11392.
- Lance Ying, Jinzhou Wu, Yingshan Susan Wang, and others. 2026. CogGym: Towards Large-Scale Comparative Evaluation of Human and Machine Cognition. arXiv:2609.21259.
- Xialie Zhuang, Mingcong Lei, Zunhai Su, and others. 2026. DecisionPivot: A Diagnostic Benchmark for Decision Models in LLM Systems. arXiv:2610.28522.
