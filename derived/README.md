# derived

Inputs that have no published file and are derived from published code. `mouselab_callaway2022_official/folds.json` holds the Mouselab participant folds, written by `scripts/recover_mouselab_official_folds.jl` with Julia 1.4.1 from the original split. `scripts/fetch_public_sources.py` copies it into `data/`, where the Mouselab modes read it. See [the data guide](../docs/DATA.md).
