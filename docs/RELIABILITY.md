# Reliability against uniform choice

The reliability count is the number of the 26 directional Table 3 targets on which a model beats uniform choice with a 95% percentile bootstrap interval. The interval is of the model-minus-Uniform difference over each target's sampling units (participants, runs, episodes, items or story blocks), with 10,000 resamples, as in the paper's interval table. It is not a cognitive score. Targets differ in units, sample sizes and difficulty, and the intervals are not adjusted for multiple comparisons. Compare values within a row of Table 3 for anything beyond this count.

| Model | Access | Reliably better (of 26) | Contains zero | Reliably worse | No interval | Lite (of 8) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Laya | open | 7 | 8 | 9 | 2 | 1 |
| Intern-Decision-4B | open | 21 | 3 | 1 | 1 | 7 |
| Llama3.1-8B | open | 12 | 3 | 11 | 0 | 3 |
| Minitaur8B | open | 17 | 3 | 4 | 2 | 4 |
| Qwen3.5-4B | open | 11 | 8 | 5 | 2 | 1 |
| Decision-2.0-Nox-4B | open | 23 | 0 | 3 | 0 | 8 |
| CLM-v0.1-8B | open | 11 | 4 | 9 | 2 | 2 |
| Jev 1.14 | API | 20 | 0 | 1 | 5 | 6 |

"No interval" means the score is +∞ in every resample, or every unit gives the same value, so only the sign of the point difference exists. Nox ran through the CLI on an A100 MIG slice (3g.40gb), and CLM on a whole A100 with its official server, which keeps the first 2,048 tokens of each text. Their rows are not yet in the accepted aggregate files. [reliability.csv](../results/aggregate/reliability.csv) and [reliability.json](../results/aggregate/reliability.json) give the difference and interval for every model and target.

## Lite targets

Eight targets are cheap to run: politeness, pragmatic interpretation, multi-attribute choice, instrumental learning, false-belief QA, two-step, the Restless bandit and Horizon prediction. Nox runs them in about 50 minutes on an A100 MIG slice, whereas the full suite took about 52 hours on the same slice. On the eight models above, the lite count ranks the models as the full count does (Spearman 0.98, Kendall 0.96). The targets were chosen with these same models, so this agreement is in-sample, and a new model can rank differently.

```bash
python -m s1cogbench eval --tasks lite --adapter path/to/adapter.py --device cuda:0 --output runs/my_model
python -m s1cogbench score runs/my_model
```

`score` recomputes each lite target from the saved records, and the result must equal the value the run wrote in `mode_result.json`. It then writes `reliability.json` into the run directory, with the interval and class of each target. The seed of each target is fixed, `[20261103, Table 3 row index]`. On the Nox and CLM runs, `score` gives the same class as the published interval table for all 16 lite cells.
