# S1CogBench homepage

This page uses a Leaderboard-First Nerfies design: a centered paper header, the overview, Table 3 as a full-width leaderboard, grey part bands, boxed figures, a BibTeX block and a plain footer.

`python scripts/build_homepage.py` copies the current `results/latest` scores and shared archive resources to `homepage/data` and writes `homepage/docs` from `docs/`. It needs the `markdown` package. Both folders are build outputs and are not tracked.

The leaderboard renders `data/table2.json`, which `scripts/build_latest_results.py` builds from accepted aggregate inputs. Bold and underlining mark the best and second-best scores among seven open models, and gray marks scores not better than Uniform. There is no pooled cognitive score, no overall ranking and no rank column. Browser-local imports stay in the browser and are never uploaded.

The paper links go to arXiv. `.github/workflows/pages.yml` builds and publishes this folder to GitHub Pages.

The separate community section reads `data/community.json`, built from reviewed JSON files under `submissions/` and matching current maintainer rows. It ranks each original target separately, retains ties and infinities, and leaves preference or undefined values unranked. Merging a submission triggers the Pages workflow. No public upload server is required. See [the submission guide](../docs/SUBMIT_RESULTS.md).
