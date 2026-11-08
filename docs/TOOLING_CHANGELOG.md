# Candidate tooling changes for 2026-11-08

This source update uses package version `0.2.0rc1`. It does not create a separate release tag.

The new `plan` command lists the original 27 modes, required adapter methods, full scored units, input paths and comparison contexts. Optional adapter inspection checks syntax and the factory signature without importing the file or loading a model. It does not validate inference outputs.

WCST autonomous sampling now executes the submitted categorical probabilities and supplies that action to the following feedback history. Previously this CLI path always executed the native choice even when sampling was requested. The existing native and human-history paths retain their choices. Sampling keeps the original participant ID and documented NumPy seed convention, with no probability floor. This fix does not rewrite archived results or repeat model inference.

Text-only adapters declaring no probability readout cannot sample their parser placeholders. This is rejected before agent creation.

Restless bandit sampling is explicitly unsupported by its retained runner and now fails before adapter import. New sampled rows use separate protocol keys where a fixed key previously did not record the rule. Plans show effective action rules, and result metadata retain both the requested and effective rule.

The documentation gives one model-integration workflow and separates the historical public aggregate from later manuscript evidence. Existing published aggregate files remain unchanged.
