# Paper details

The paper's appendix summarizes the points below and refers here for the full text. Tables are named rather than numbered, because numbers change between paper versions.

## Reference construction

The paper marks each reference in the main results table as an existing fit or policy (†), an adapted policy (\*), or a value computed from source data (¶). The appendix table of reference constructions lists what each one is fitted on and how it is built.

**Adapted policies (\*).** Mouselab MetaGreedy reuses the 95 participant parameter sets archived for the policy family of Callaway et al. (2022), with no new fitting, and plays the same 23,750 synthetic trees as the open models. Restless runs a published two-parameter Q-learning model in the CogBench environment (Coda-Forno et al., 2024). Training reward on 100 independent environments selects α = 0.44 and β = 100, the largest grid value, both fixed before test. Its source fits human responses instead, so this is not a human fit.

**Computed from source data (¶).** These references are computed on CPU from the saved source data and original environments, with no model or API call, using the same targets, aggregation, and scorer as the open models.

**Existing fits and policies (†).** These are existing fits or policies scored on the same items and targets. They are WADD for multi-attribute choice, GCM for SHJ, Medin, and weather prediction, the rational conditional-response model for recent probes, and the Thompson family for stationary bandits. WADD, GCM, and the rational model follow the published families (Binz et al., 2025). The fits use source training participants and are reimplementations rather than the authors' own programs.

Recent probes shows the current letter set in the prompt, so its fitted model answers every item correctly, which does not show a working-memory mechanism. The stationary bandit conditions are the public FullBench conditions used for the open models, not the publication's horizon and seed protocol.

## Probability extraction

The human-response NLL entries score probabilities over the task's legal response support. For a causal decoder with answer-token logits z, the legal distribution is q(a | x, A) = exp z_k(a) / Σ_{b ∈ A} exp z_k(b).

The token index k(a) is checked after the complete actual prompt, not from an isolated label. Single-letter task keys are retained. Otherwise, the letter adapter maps the ordered legal options to consecutive letters. The numeric-label adapter retains the original integer keys and verifies that each complete candidate is one distinct next token. No multi-token answer is silently scored as its first token. Model-specific event and numerical reports are separate targets and do not become answer distributions by this construction.

Laya uses the SDK's returned probabilities, rounded to four decimals, divided by their sum over the legal keys. Intern and Qwen use their legal-choice adapter distributions. Nox and Jev retain the native choice separately and normalize the returned legal probabilities by their sum, without adding a floor. Jev's two-decimal rounding can therefore still produce exact zeros. CLM supplies a softmax over its scaled cosine scores. Haiku's NLL uses the probability distribution stated in JSON, rather than one obtained from token likelihoods. These extraction rules are part of each evaluated combination of model and interface.

In WCST, Llama and Minitaur also record the full-vocabulary probability and its original diagnostic NLL. Their full-vocabulary next-token argmax determines the recorded prediction, but the main results table uses the separately computed legal-ABCD NLL. The full-vocabulary diagnostic is not substituted for that table entry. The same distinction applies to saved original decoder diagnostics in the other tasks. A native action, a sampled action, and the probability assigned to a human response need not agree.

## Invalid outputs and zeros

The existing scorers and their denominators are unchanged. An invalid executed action remains in its task's error count. A malformed response in a generative closed-loop extension ends that trajectory rather than supplying an invented action. Missing or incomplete probability targets stay unreported. No new smoothing is applied to make an infinite primary score finite. In particular, the epsilons of the source WCST and Horizon metrics apply only to those two targets, not to the other unsmoothed targets. Native choices and raw rounded probabilities are retained alongside any normalized legal distribution.

## Timing boundary and generation prefixes

The controlled single-decision timer brackets tokenization, adapter checks, model forward, and readout with CUDA synchronization. It excludes model loading and the 20 warm-up calls per mode. Forward-only CUDA-event timing is recorded separately. These are batch-one local measurements, not API round trips. The original scalar causal-decoder decision forward disables the KV cache. The historical free-text generation condition enables its within-request KV cache. No cross-request prefix cache is introduced into those two local decoder adapters. CLM's separately disclosed serving path does reuse prefixes. Warm-up does not guarantee that Laya's TileLang path has compiled every later input shape.

Accepted 4,096-token-cap generation prefixes:

| Horizon prefix | Attempts | Valid | Input mean | Output mean | P50 (s) | P95 (s) |
|---|---:|---:|---:|---:|---:|---:|
| Llama | 200 | 197 | 317.30 | 197.97 | 2.958 | 4.650 |
| Qwen | 200 | 200 | 341.05 | 475.25 | 16.330 | 50.001 |

Token counts are actual means, not the cap. All attempts, including Llama's three capped invalid responses and first-call overhead, remain in the timing records. Greedy BF16 generation is CUDA-synchronized and includes tokenization and parsing. Model loading is outside the timer. Qwen's native thinking switch is disabled in this condition, while the prompt permits free-text reasoning. These prefixes are not full autonomous evaluations.

The 80 and 316 ratios in the latency section divide these generation medians by the corresponding Horizon single-decision medians. They compare recorded inference protocols. Their prompts, actual input lengths, output lengths, action histories, warm-up treatment, and cache use are not all identical, so these ratios are not paired estimates of an isolated interface effect or a decomposition of all serving costs. No time to first token or provider-only latency is inferred from API responses.

## Same-input one-token timing control

We use the first 200 saved P1 requests from each of its six input modes for each checkpoint. This gives 2,400 paired inputs and 4,800 timed calls, with 20 warm-up pairs per model and mode (480 additional calls). Each pair contains a direct legal-choice forward pass and `generate(max_new_tokens=1)` with greedy decoding and no legal-token mask. Both paths use the same complete prompt and actual token IDs, batch size one, BF16 weights, and `use_cache=False`. There is no cross-request prefix cache. Pair order alternates with the original input index. Timing includes formatting, tokenization, the common legal-token boundary checks, synchronized model execution, and readout or decoding. Loading and warm-up are excluded. Llama uses its original PyTorch 2.10.0/CUDA 12.8 and Transformers 5.3.0 environment. Qwen uses PyTorch 2.9.1/CUDA 12.8, Transformers 5.14.1, and the original FLA chunk kernel. Both run on the same full A100-SXM4-40GB. CPU source replay verifies each actual token prefix, output token and legal mapping, as well as the saved timing arithmetic, without new model calls.

The matched one-token path has little additional overhead in these measurements. It does not reproduce the original multi-token free-text protocol, nor establish that a single unconstrained token supplies a valid decision. In total, 524/1,200 Llama and 936/1,200 Qwen timed outputs are outside the legal token set. We do not discard these requests, retry them, substitute legal actions, or change the task scorers. Thus the original long-generation ratios remain cross-protocol comparisons, not estimates of an 80- or 316-fold cost of the generation function itself. Every attempt remains in the latency summaries. The control's code and per-request timing records are not included in this repository. The summary is in [the one-token timing table](manuscript_20261109/appendix_details/one_token_timing.csv).

## CLM window control

CLM's official vLLM pooling server keeps the first 2,048 tokens of each text. The control keeps the last 2,048 tokens instead. It tests window placement, not an untruncated CLM configuration. The scores do change, although the sign of every listed comparison with Uniform stays the same. These are paired protocol scores from saved runs, not a new fit or an interval estimate.

| Target | Uniform | First window | Last window |
|---|---:|---:|---:|
| Weather prediction (accuracy) | 0.5000 | 0.5250 | 0.5250 |
| Multi-attribute choice (NLL) | 0.6931 | 1.0391 | 1.1978 |
| Restless bandit (accuracy) | 0.5000 | 0.5262 | 0.5212 |
| Probability coherence (MAE) | 32.1339 | 42.1268 | 37.7622 |
| Recent probes (accuracy) | 0.5000 | 0.5591 | 0.5604 |
| SHJ categories (accuracy) | 0.5000 | 0.5041 | 0.5060 |
| Multiple-cue judgment (NLL) | 2.1972 | 2.6753 | 2.5248 |
| Medin categories (accuracy) | 0.5000 | 0.5043 | 0.5027 |
| WCST prediction (NLL) | 1.3863 | 2.3678 | 2.6163 |
| WCST autonomous (accuracy) | 0.2500 | 0.2571 | 0.2525 |
| Self-assessment (NLL) | 2.0393 | 3.0939 | 2.7424 |

Accuracy is a fraction, NLL is in nats, and probability-coherence MAE uses the source percentage-point scale. Higher accuracy and lower NLL/MAE are better. All 11 comparisons with Uniform retain their direction. This does not establish invariance of model rankings or cognitive effects.
