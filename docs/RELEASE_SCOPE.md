# Release scope

This tooling update improves model integration and protocol reporting. It does not change the benchmark task selection, source data, metrics or existing result values. Its package version is `0.2.0rc1`.

The GitHub repository and arXiv paper are public. The aggregate files retain the released arXiv snapshot, with five open models and the recorded API coverage. Later manuscript evidence is versioned separately and is not automatically included in a tooling update. Do not treat the historical aggregate as every later paper revision.

Use [ADD_MODEL.md](ADD_MODEL.md) to integrate a model and [COMPARISON_GUIDE.md](COMPARISON_GUIDE.md) to identify matching protocols. New neural runs remain user-supplied until their full records and settings have been checked.

The new `plan` command performs no model or API call. Its adapter inspection checks Python syntax and the factory signature without importing the file. It is not a neural-reproduction test. Existing measured validation and its limitations remain in [REPRODUCTION.md](REPRODUCTION.md).

This source update includes no new inference, restricted participant transcripts, model weights or new manuscript result tables. No separate GitHub release or tag is created by this update.
