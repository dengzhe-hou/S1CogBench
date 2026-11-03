# API decision time and cost

These tables cover the targets of Table 3 that each API model completed on its full original sample. Each decision is one saved generation request, and calibration requests that count toward the full run are included.

Latency is the client round trip of that request, so it includes network and provider service time. Jev was called through OpenRouter and Claude through the Anthropic API, so these times do not compare model speed. Claude times exclude the separate token count request. Quantiles use NumPy linear interpolation.

Costs are usage prices, not invoices. Jev costs are the provider reported charges. Claude costs use 0.10 and 0.50 USD per million input and output tokens for Haiku, 2 and 10 for Sonnet, and 4 and 20 for Opus. Output tokens are the mean per decision for Claude only and include thinking.

The same rows at full precision, with timers and source records, are in [api_time_cost_per_target.csv](../results/aggregate/api_time_cost_per_target.csv).

## By model

| Model | Targets | Decisions | Median (s) | P95 (s) | Lowest target median (s) | Highest target median (s) | USD per 1,000 | Lowest target (USD per 1,000) | Highest target (USD per 1,000) | Output tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Jev 1.14 | 27 | 864,813 | 0.35 | 0.41 | 0.33 | 0.38 | 0.0448 | 0.0172 | 0.251 | 94 |
| Claude Opus 5.5 | 2 | 730 | 3.59 | 5.34 | 3.50 | 3.73 | 8.96 | 7.71 | 10.5 | 289 |
| Claude Sonnet 5.5 | 3 | 2,330 | 3.08 | 4.99 | 2.10 | 3.41 | 5.39 | 2.63 | 6.24 | 253 |
| Claude Haiku 5.5 | 15 | 92,803 | 2.60 | 7.65 | 1.92 | 6.58 | 0.336 | 0.174 | 1.10 | 442 |

Jev's output tokens are the provider's `usage.output_tokens`, which count the returned structured response. Generated reasoning is not in them. Claude's output tokens include thinking.

Pooled values use all decisions of a model, except one Jev WCST request with no saved latency. A target median or target cost is the value of one target. Total usage-priced cost: Jev 1.14 38.77 USD, Claude Opus 5.5 6.54 USD, Claude Sonnet 5.5 12.56 USD, Claude Haiku 5.5 31.15 USD.

## Per target

### Jev 1.14

| Target | Decisions | Median (s) | P95 (s) | Mean (s) | Total (USD) | USD per 1,000 | Output tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Bayesian urn | 400 | 0.343 | 0.394 | 0.357 | 0.0078 | 0.01940 | 20 |
| Multiple-cue judgment | 6,136 | 0.379 | 0.445 | 0.385 | 1.5372 | 0.2505 | 93 |
| Multi-attribute choice | 576 | 0.348 | 0.394 | 0.353 | 0.0480 | 0.08333 | 31 |
| Probability coherence | 2,460 | 0.359 | 0.414 | 0.370 | 0.3656 | 0.1486 | 817 |
| SHJ categories | 3,664 | 0.359 | 0.401 | 0.363 | 0.6058 | 0.1653 | 31 |
| Medin categories | 6,300 | 0.339 | 0.380 | 0.346 | 0.5222 | 0.08289 | 31 |
| Weather prediction | 400 | 0.349 | 0.391 | 0.366 | 0.0561 | 0.1403 | 31 |
| RLWM | 52,911 | 0.342 | 0.399 | 0.352 | 1.8143 | 0.03429 | 38 |
| Recent probes | 3,774 | 0.335 | 0.370 | 0.342 | 0.2742 | 0.07265 | 31 |
| WCST, prediction | 12,754 | 0.380 | 0.748 | 0.414 | 1.7977 | 0.1409 | 45 |
| WCST, autonomous | 12,754 | 0.378 | 0.506 | 0.403 | 1.7750 | 0.1392 | 45 |
| Restless bandit | 1,600 | 0.354 | 0.410 | 0.368 | 0.1037 | 0.06483 | 26 |
| Instrumental learning | 960 | 0.337 | 0.381 | 0.344 | 0.0447 | 0.04652 | 31 |
| Horizon, prediction | 6,720 | 0.356 | 0.430 | 0.372 | 0.1559 | 0.02320 | 31 |
| Horizon, autonomous | 34,720 | 0.341 | 0.404 | 0.358 | 0.8233 | 0.02371 | 31 |
| Stationary bandits | 312,000 | 0.347 | 0.414 | 0.357 | 12.1607 | 0.03898 | 143 |
| Hybrid sampling | 163,135 | 0.342 | 0.403 | 0.351 | 3.9879 | 0.02445 | 50 |
| Mouselab, prediction | 20,363 | 0.330 | 0.370 | 0.341 | 0.5388 | 0.02646 | 60 |
| Mouselab, autonomous | 101,739 | 0.343 | 0.406 | 0.353 | 2.5676 | 0.02524 | 60 |
| BART | 2,566 | 0.340 | 0.385 | 0.346 | 0.1218 | 0.04745 | 31 |
| Temporal discounting | 11 | 0.341 | 0.362 | 0.340 | 0.0002 | 0.01715 | 31 |
| Two-step | 4,000 | 0.330 | 0.366 | 0.338 | 0.1124 | 0.02809 | 31 |
| False-belief QA | 5,994 | 0.329 | 0.367 | 0.340 | 0.1040 | 0.01735 | 31 |
| Politeness | 128 | 0.338 | 0.393 | 0.343 | 0.0029 | 0.02296 | 36 |
| Pragmatic interpretation | 170 | 0.334 | 0.379 | 0.339 | 0.0041 | 0.02423 | 44 |
| Nonliteral number | 330 | 0.341 | 0.383 | 0.345 | 0.0176 | 0.05328 | 20 |
| Self-assessment | 108,248 | 0.355 | 0.424 | 0.364 | 9.2164 | 0.08514 | 122 |

### Claude Opus 5.5

| Target | Decisions | Median (s) | P95 (s) | Mean (s) | Total (USD) | USD per 1,000 | Output tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Bayesian urn | 400 | 3.501 | 4.446 | 3.553 | 3.0832 | 7.708 | 312 |
| Nonliteral number | 330 | 3.735 | 5.887 | 3.922 | 3.4542 | 10.47 | 261 |

### Claude Sonnet 5.5

| Target | Decisions | Median (s) | P95 (s) | Mean (s) | Total (USD) | USD per 1,000 | Output tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Bayesian urn | 400 | 2.104 | 2.943 | 2.160 | 1.0524 | 2.631 | 190 |
| Restless bandit | 1,600 | 3.411 | 5.307 | 3.565 | 9.9886 | 6.243 | 281 |
| Nonliteral number | 330 | 2.788 | 3.866 | 2.912 | 1.5156 | 4.593 | 197 |

### Claude Haiku 5.5

| Target | Decisions | Median (s) | P95 (s) | Mean (s) | Total (USD) | USD per 1,000 | Output tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Bayesian urn | 400 | 2.215 | 4.380 | 2.523 | 0.0861 | 0.2152 | 357 |
| Multi-attribute choice | 576 | 4.245 | 6.985 | 4.511 | 0.3527 | 0.6123 | 736 |
| SHJ categories | 3,664 | 6.471 | 13.948 | 7.041 | 4.0344 | 1.101 | 1,257 |
| Medin categories | 6,300 | 6.577 | 10.260 | 6.718 | 5.0806 | 0.8065 | 1,212 |
| Weather prediction | 400 | 5.252 | 9.295 | 5.637 | 0.3667 | 0.9168 | 997 |
| RLWM | 52,911 | 1.987 | 4.157 | 2.327 | 11.5051 | 0.2174 | 260 |
| Recent probes | 3,774 | 1.922 | 3.689 | 2.169 | 1.0879 | 0.2883 | 158 |
| Horizon, prediction | 6,720 | 3.048 | 5.258 | 3.316 | 1.7440 | 0.2595 | 421 |
| BART | 7,424 | 5.205 | 9.348 | 5.504 | 4.3640 | 0.5878 | 885 |
| Temporal discounting | 12 | 3.983 | 4.815 | 3.961 | 0.0037 | 0.3093 | 571 |
| Two-step | 4,000 | 3.776 | 6.834 | 4.060 | 1.3358 | 0.3340 | 535 |
| False-belief QA | 5,994 | 2.545 | 5.080 | 2.735 | 1.0445 | 0.1743 | 290 |
| Politeness | 128 | 3.069 | 4.954 | 3.117 | 0.0299 | 0.2336 | 363 |
| Pragmatic interpretation | 170 | 2.274 | 4.165 | 2.497 | 0.0333 | 0.1960 | 285 |
| Nonliteral number | 330 | 3.019 | 5.586 | 3.202 | 0.0861 | 0.2608 | 259 |

## Notes per target

- Jev 1.14, Bayesian urn: latency is the source-loop timer around the agent call, so it also covers the local writes of the request, response and usage files.
- Jev 1.14, WCST, prediction: call 8728 has no saved latency (the original run stopped at its parse guard before writing the record); latency over 12,753 decisions, cost over all 12,754.
- Jev 1.14, Restless bandit: two requests per trial (choice and confidence); latency is the source-loop timer around the agent call, so it also covers the local writes of the request, response and usage files.
- Jev 1.14, Horizon, autonomous: 6,179 decisions reused from an earlier continuation run with their original measured latency.
- Jev 1.14, Stationary bandits: all 480 runs complete; N and cost checked against the usage ledger and the acceptance record.
- Claude Sonnet 5.5, Restless bandit: two requests per trial (choice and confidence).
- Claude Haiku 5.5, Multi-attribute choice: stated-probability protocol.
- Claude Haiku 5.5, SHJ categories: 12 responses truncated at max_tokens (invalid, scored wrong) are included.
- Claude Haiku 5.5, RLWM: 531 calibration responses count toward the full run. The acceptance record confirms N and this cost, and also lists 0.0043 USD of held reservations outside this total.
- Claude Haiku 5.5, Horizon, prediction: stated-probability protocol; 69 calibration responses plus 6,651 production responses.
- Claude Haiku 5.5, Politeness: stated-probability protocol; full sample answered with 1 invalid response.
- Claude Haiku 5.5, Pragmatic interpretation: stated-probability protocol.

## Timers

The timer differs by run. All wrap the generation request; none includes the token count request.

| Timer | Targets |
| --- | --- |
| source loop time.monotonic around agent call: wraps the decisions POST plus local writes of the request/response/usage files (agent-only timer not saved for this mode) | Jev 1.14: Bayesian urn · Restless bandit |
| agent perf_counter around the decisions POST only; reused rows carry the original measurement | Jev 1.14: Horizon, prediction · WCST, prediction · WCST, autonomous · Horizon, autonomous |
| perf_counter around the decisions POST only | Jev 1.14: Probability coherence · Nonliteral number · Stationary bandits · Mouselab, autonomous · RLWM · Hybrid sampling · Self-assessment |
| perf_counter around the generation POST only (count_tokens precedes the timer) | Jev 1.14: Multiple-cue judgment · Multi-attribute choice · SHJ categories · Medin categories · Weather prediction · Recent probes · Mouselab, prediction · Politeness · Pragmatic interpretation · False-belief QA · Instrumental learning · Two-step · BART · Temporal discounting; Claude Haiku 5.5: SHJ categories · Medin categories · Weather prediction · Recent probes · False-belief QA · Two-step · BART · Temporal discounting |
| perf_counter around the Messages POST only (count_tokens precedes the timer) | Claude Haiku 5.5: Bayesian urn · Nonliteral number; Claude Sonnet 5.5: Bayesian urn · Nonliteral number · Restless bandit; Claude Opus 5.5: Bayesian urn · Nonliteral number |
| perf_counter around the Messages POST only (count_tokens timed separately) | Claude Haiku 5.5: Politeness · Pragmatic interpretation · Multi-attribute choice · Horizon, prediction |
| perf_counter around the Messages POST only; count_tokens_seconds is recorded separately and excluded | Claude Haiku 5.5: RLWM |

## Not in these tables

Not in the main table: not completed on the full original sample. Timing and cost not computed for these.

| Model | Target | Saved decisions | Full sample | Why |
| --- | --- | ---: | --- | --- |
| Claude Opus 5.5 | Probability coherence | 0 | 2,460 | canceled before generation |
| Claude Opus 5.5 | WCST, autonomous | 200 | 12,754 | 200-call pilot prefix only (n.f. in Table 3) |
| Claude Opus 5.5 | Restless bandit | 436 | 1,600 | partial prefix: stopped by the authors on 2026-10-30 |
| Claude Opus 5.5 | Horizon, autonomous | 200 | 34,720 | 200-call pilot prefix only (n.f. in Table 3) |
| Claude Sonnet 5.5 | Probability coherence | 0 | 2,460 | canceled before generation |
| Claude Sonnet 5.5 | WCST, autonomous | 200 | 12,754 | 200-call pilot prefix only (n.f. in Table 3) |
| Claude Sonnet 5.5 | Horizon, autonomous | 200 | 34,720 | 200-call pilot prefix only (n.f. in Table 3) |
| Claude Haiku 5.5 | Multiple-cue judgment | 63 | 6,136 | calibration-only rows of the four-row verbalized-probability quote; no full run |
| Claude Haiku 5.5 | Probability coherence | 181 | 2,460 | partial prefix: stopped by the authors on 2026-10-30 |
| Claude Haiku 5.5 | WCST, prediction | 129 | 12,754 | calibration-only rows of the four-row verbalized-probability quote; no full run |
| Claude Haiku 5.5 | WCST, autonomous | 200 | 12,754 | 200-call pilot prefix only (n.f. in Table 3) |
| Claude Haiku 5.5 | Restless bandit | 1,508 | 1,600 | partial: 9 of 10 runs complete; run 0 ended at an invalid response after 68 responses; no full-sample score |
| Claude Haiku 5.5 | Instrumental learning | 843 | 10 runs (960 decisions for Jev) | partial: 8 of 10 runs complete (full_protocol_reached false); no Table 3 score |
| Claude Haiku 5.5 | Horizon, autonomous | 200 | 34,720 | 200-call pilot prefix only (n.f. in Table 3) |
| Claude Haiku 5.5 | Stationary bandits | 0 | 312,000 | dropped by the authors before any request |
| Claude Haiku 5.5 | Mouselab, prediction | 205 | 20,363 | calibration-only rows of the four-row verbalized-probability quote; no full run |
| Claude Haiku 5.5 | Mouselab, autonomous | 1,566 | 23,750 episodes | calibration only (runs 0-237, 1,566 saved responses); the production run was not started |
