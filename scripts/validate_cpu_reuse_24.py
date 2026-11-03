#!/usr/bin/env python3
"""Server-only full-source CPU reference validation, not a neural-model benchmark.
Each named mode reports installation, full input, original scoring and E2E separately.
Do not redistribute private artifacts containing gated human source information.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REFERENCE = ROOT / "examples/cpu_protocol_validation_agent.py"
LABEL = "cpu_protocol_validation_reference_not_research_baseline"
STAGES = ("environment_install", "full_input_parse",
          "legal_output_and_original_scoring", "full_end_to_end")
EXPECTED_TYPED = {
    "multiple_cue_judgment": 6136, "multi_attribute_choice": 576,
    "shj_category_learning": 3664, "medin_category_learning": 6972,
    "weather_prediction": 400, "rlwm": 52911, "recent_probes": 3774,
    "politeness_pragmatics": 128, "pragmatic_language_interpretation": 170,
    "nonliteral_number_interpretation": 330, "false_belief_story_qa": 5994,
}
MODES = {t: ("human_predictive", "autonomous_generative")
         for t in ("horizon", "mouselab_tree", "wcst")}
COGBENCH = {"instrumental_learning": "instrumental", "bart": "bart",
            "temporal_discounting": "temporal"}


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                     allow_nan=False) + "\n")


def module(name):
    value = importlib.import_module(name)
    if not Path(value.__file__).resolve().is_relative_to(ROOT):
        raise RuntimeError(f"Production code imported instead of new copy: {value.__file__}")
    return value


def agent():
    return module("examples.cpu_protocol_validation_agent").create_agent("cpu")


class Receipt:
    def __init__(self, task, mode, out, environment):
        self.path = out / "validation_receipt.json"
        self.start = time.monotonic()
        self.data = {"task": task, "mode": mode, "status": "running",
                     "started_utc": now(), "policy": LABEL,
                     "stages": {s: {"status": "not_run"} for s in STAGES},
                     "environment": environment}
        self.mark("environment_install", "passed_python_core",
                  note="Native/task dependencies must also succeed during this mode.")

    def mark(self, stage, status, **fields):
        self.data["stages"][stage] = {"status": status, "utc": now(), **fields}
        save(self.path, self.data)

    def inputs(self, count, expected, **fields):
        if count != expected:
            raise ValueError(f"Full source count {count} != expected {expected}")
        self.mark("full_input_parse", "passed", count=count, expected=expected, **fields)

    def scored(self, count, expected, **fields):
        if count != expected:
            raise ValueError(f"Full scoring count {count} != expected {expected}")
        self.mark("legal_output_and_original_scoring", "passed",
                  count=count, expected=expected, **fields)

    def finish(self):
        for s in ("full_input_parse", "legal_output_and_original_scoring"):
            assert self.data["stages"][s]["status"] == "passed", s
        self.mark("environment_install", "passed",
                  note="Fresh Python core and complete named-mode execution succeeded.")
        self.mark("full_end_to_end", "passed",
                  scope="full CPU-reference pipeline for this mode, not five neural models")
        self.data.update(status="complete", finished_utc=now(),
                         seconds=time.monotonic()-self.start,
                         max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        save(self.path, self.data)

    def fail(self, exc):
        if isinstance(exc, ModuleNotFoundError):
            failed_stage = "environment_install"
        elif self.data["stages"]["full_input_parse"]["status"] != "passed":
            failed_stage = "full_input_parse"
        elif self.data["stages"]["legal_output_and_original_scoring"]["status"] != "passed":
            failed_stage = "legal_output_and_original_scoring"
        else:
            failed_stage = "full_end_to_end"
        self.mark(failed_stage, "failed", error=repr(exc))
        self.data.update(status="failed", error=repr(exc), traceback=traceback.format_exc(),
                         finished_utc=now(), seconds=time.monotonic()-self.start,
                         max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        save(self.path, self.data)


def unique(ids, expected):
    assert len(ids) == len(set(ids)) == expected, "Full unique source IDs required"


def typed(task, mode, out, r):
    if task == "shj_category_learning":
        m = module("src.psych_typed")
        trials = m.load_shj_trials()
        inventory, score, summarize = m.source_inventory(trials), m.score_prediction, m.summarize
    elif task in {"medin_category_learning", "weather_prediction"}:
        m = module("src.psych_category_tasks")
        trials = m.load_category_trials("medin" if task.startswith("medin") else "weather")
        inventory, score, summarize = m.inventory(trials), m.score, m.summarize_category
    elif task == "rlwm":
        m = module("src.rlwm_typed")
        original = m.read_source()
        trials = m.build_trials(original)
        inventory, score, summarize = m.inventory(original, trials), m.score, m.summarize
    else:
        name = ("src.psych_compact" if task == "multiple_cue_judgment" else
                "src.psych_extended" if task in {"multi_attribute_choice", "recent_probes"}
                else "src.psych_social_tasks")
        m = module(name)
        trials = m.load_trials(task)
        inventory, score, summarize = m.inventory(trials), m.score, m.summarize
    expected = EXPECTED_TYPED[task]
    if task == "medin_category_learning":
        assert inventory["response_kinds"] == {
            "learning/category_choice": 6300,
            "test/category_choice": 336,
            "test/typicality_rating": 336,
        }, inventory["response_kinds"]
    if task == "rlwm":
        assert inventory["source_rows"] == 53573, inventory["source_rows"]
    unique([t.id for t in trials], expected)
    save(out / "source_inventory.json", inventory)
    r.inputs(len(trials), expected, source_inventory="source_inventory.json",
             builder=str(Path(m.__file__).relative_to(ROOT)))
    reference = agent()
    predict = module("scripts.run_psych_typed").predict_original_scalar
    rows = []
    with (out / "private_reference_responses.jsonl").open("w") as stream:
        for i, t in enumerate(trials):
            if getattr(t, "kind", None) == "single-slider":
                start = time.monotonic()
                p = module("src.social_numeric_models").parse_percentage("50")
                prediction = {"reported_probability": p,
                              "raw_answer": {"cpu_reference_percentage": "50"},
                              "batch_elapsed_seconds": time.monotonic()-start,
                              "inference_interface": "cpu_reference_explicit_percentage"}
            else:
                prediction = predict(reference, t, "cpu")
            row = score(t, prediction)
            row["runner_batch_id"] = i
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            rows.append(row)
    result = summarize(rows, expected)
    assert result["status"] == "complete"
    save(out / "original_summary.json", result)
    r.scored(len(rows), expected, scorer=f"{score.__module__}.{score.__name__}",
             aggregator=f"{summarize.__module__}.{summarize.__name__}",
             original_summary="original_summary.json",
             note="Original NA/undefined metrics are preserved.")


def four(task, mode, out, r):
    m = module("scripts.run_four_task_responses_20261018")
    argument = ("probability_coherence_judgment" if task == "probability_coherence_judgment"
                else "answer_self_assessment")
    expected = 2460 if argument.startswith("probability") else 108248
    ids, kinds = [], Counter()
    for trials, metadata in m.sessions(argument):
        for t in trials:
            ids.append(t.id)
            kinds[f"{t.experiment}:{t.response_kind}"] += 1
    unique(ids, expected)
    r.inputs(len(ids), expected, builder="original sessions iterator", kinds=dict(kinds))
    dest = out / "original_runner"
    m.run(SimpleNamespace(task=argument, output=dest, adapter_file=REFERENCE,
                          model_label=LABEL, device="cpu"))
    s = json.loads((dest / "summary.json").read_text())
    assert s["status"] == "complete" and s["full_scope_complete"]
    r.scored(s["responses"], expected, scorer="src.four_task_interfaces.score_response",
             original_summary="original_runner/summary.json",
             note="All original integers retained; mixed-unit MAE is not a total score.")


def posterior(task, mode, out, r):
    m = module("src.paper_probabilistic")
    records = m.load_records()
    unique([x["request"]["id"] for x in records], 400)
    r.inputs(len(records), 400, builder="src.paper_probabilistic.load_records", runs=200)
    a = module("examples.bayes_event_agent").create_agent("cpu")
    predictions = []
    with (out / "reference_predictions.jsonl").open("w") as stream:
        for row in records:
            q = row["request"]
            prediction = {"id": q["id"], "reported_event_probability":
                          float(a.posterior_probability(q["state"], q["question"]))}
            stream.write(json.dumps(prediction, allow_nan=False)+"\n")
            predictions.append(prediction)
    scored = m.score_rows(records, predictions)
    save(out / "original_summary.json", m.score_records(records, predictions))
    r.scored(len(scored), 400, scorer="original score_rows / score_records",
             policy="existing visible-information analytic Bayes reference",
             note="Exact Bayes strict-trimmed score may be undefined; no repair.")


def prediction(task, mode, out, r):
    m = module("scripts.benchmark_paper_tasks")
    arg = "horizon-predictive" if task == "horizon" else "mouselab-predictive"
    expected = 6720 if task == "horizon" else 20363
    records = m.load_records(arg)
    unique([x["request"]["id"] for x in records], expected)
    r.inputs(len(records), expected, builder=f"original load_records({arg})",
             participants=m.TASKS[arg]["participants"],
             scope="Original human prediction; not autonomous return.")
    del records
    dest = out / "original_runner"
    m.predict_task(SimpleNamespace(task=arg, output=dest, adapter_file=REFERENCE, device="cpu"))
    s = json.loads((dest / "summary.json").read_text())
    r.scored(s["decisions"], expected, scorer="original independent score_task",
             original_summary="original_runner/summary.json")


def wcst(task, mode, out, r):
    m = module("src.wcst_task")
    scorer = module("scripts.run_wcst_models")
    data = m.test_data()
    r.inputs(sum(map(len, data.values())), 12754, participants=len(data),
             builder="original test_data", formatter="Centaur task formatter",
             note="Neural tokenizer/model-specific formatters are not tested.")
    a, parts = agent(), []
    original_mode = "predictive" if mode == "human_predictive" else "generative"
    with (out / "reference_decisions.jsonl").open("w") as stream:
        for sid, source in data.items():
            dialogue, rows = m.Dialogue("centaur", original_mode), []
            for i, row in enumerate(source):
                state = dialogue.prompt(row)
                p = a.choice_probabilities(state, m.QUESTION, list(m.LETTERS))
                key = m.LETTERS.index(a.last_native_choice)+1
                assert len(p) == 4 and abs(sum(p)-1) < 1e-6
                record = {"subject_id": sid, "trial": i, "key": key,
                          "human_key": int(row["user_key"]), "ground_key": int(row["ground_key"]),
                          "legal_probabilities": p, "native_choice": a.last_native_choice}
                stream.write(json.dumps(record, allow_nan=False)+"\n")
                rows.append(record)
                actual = int(row["user_key"]) if original_mode == "predictive" else key
                dialogue.update(row, actual,
                                a.last_native_choice if original_mode == "generative" else None)
            part = {"subject_id": sid, "mode": original_mode, "trials": len(rows),
                    **m.task_metrics(source, [x["key"] for x in rows])}
            if original_mode == "predictive":
                part.update(scorer.predictive_metrics(rows, False))
            parts.append(part)
    count = sum(x["trials"] for x in parts)
    save(out / "original_summary.json", {
        "status": "complete", "mode": original_mode, "responses": count,
        "participants": len(parts), "per_participant": parts,
        "original_full_vocabulary_nll": None,
        "original_full_vocabulary_nll_status": "Not produced by the CPU typed reference.",
    })
    r.scored(count, 12754, scorer="original task_metrics / predictive_metrics",
             original_summary="original_summary.json")


def hybrid(task, mode, out, r):
    m = module("scripts.run_hybrid_open_prediction")
    export = out / "full_export"
    m.export(export)
    info = json.loads((export / "summary.json").read_text())
    r.inputs(info["requests"], 163135, folds=160,
             builder="original raw-X/y export and original fold assignments")
    dest = out / "original_runner"
    m.run(dest, export, REFERENCE, "cpu", False, 1)
    s = json.loads((dest / "summary.json").read_text())
    assert s["status"] == "complete"
    r.scored(s["heldout_actions"], 163135,
             scorer="original held-out NLL, subject/fold aggregation and full alignment checks",
             original_summary="original_runner/summary.json")


def generated(task, mode, out, r):
    dest = out / "original_runner"
    dest.mkdir()
    if task == "nonstationary_bandit":
        m = module("src.paper_restless")
        s = m.run_restless(dest, REFERENCE, "cpu")
        r.inputs(s["queries"], 1600, runs=s["runs"], trials=s["trials"],
                 builder="original query and Gaussian environment",
                 scope="Complete own-history action/confidence trajectory, not human data.")
        r.scored(s["queries"], 1600, scorer="original official_score",
                 original_summary="original_runner/summary.json")
    elif task == "two_step":
        m = module("src.paper_twostep")
        m.run_twostep(dest, REFERENCE, "cpu", "native")
        s = json.loads((dest / "summary.json").read_text())
        r.inputs(s["n_decisions"], 4000, runs=s["n_runs"], trials=s["n_trials"],
                 builder="original version1 environment,100runs x20trials")
        r.scored(s["n_decisions"], 4000, scorer="original CogBench score/OLS",
                 original_summary="original_runner/summary.json")
    elif task == "horizon":
        m = module("src.paper_horizon")
        m.run_generative(dest, REFERENCE, "cpu", "native")
        s = json.loads((dest / "summary.json").read_text())
        assert s["n_rows"] == 74400 and s["n_participants"] == 31
        r.inputs(s["n_free_choices"], 34720, timeline_rows=s["n_rows"],
                 builder="original31participant autonomous timeline")
        r.scored(s["n_free_choices"], 34720,
                 scorer="original notebook optimal-choice definition",
                 original_summary="original_runner/summary.json")
    else:
        m = module("src.paper_mouselab_generative")
        people = json.loads(
            (ROOT / "data/mouselab_callaway2022_official/trials.json").read_text()).keys()
        slots = list(m.episode_schedule(people))
        r.inputs(len(slots), 23750, builder="original95x25x10 episode schedule",
                 scope="Simulation slots, not human trajectories.")
        required = [
            ROOT / ".tools/julia-1.4.1/bin/julia",
            ROOT / ".tools/mouselab-julia141-env/Project.toml",
            ROOT / ".tools/mouselab-julia141-env/Manifest.toml",
            m.MODEL_COPY, m.SERVER,
        ]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            r.mark("environment_install", "blocked_native_runtime", missing=missing)
            raise FileNotFoundError("Fresh-copy Julia/model/server assets missing: "+repr(missing))
        s = m.run_mouselab_generative(dest, REFERENCE, "cpu", "native")
        r.scored(s["simulations"], 23750,
                 scorer="original Julia rollout/STOP score and original check_terminal",
                 original_summary="original_runner/summary.json",
                 native_runtime="new validation project/depot; do not reuse production dependency directories")


def bandit(task, mode, out, r):
    m = module("scripts.run_bandit_open_models")
    bench = m.FullBench(num_threads=1)
    specs = m.specifications(bench)
    expected = sum(x["horizon"] for x in specs)*m.N_TRAJECTORIES
    r.inputs(len(specs), 16, input_unit="complete original condition specifications",
             scheduled_episodes=480, expected_scored_steps=expected,
             builder="original FullBench specifications/seeds",
             history="official SH full prior-feedback statistics",
             scope="Full schedule parsed;312000 generated steps are checked only after rollout.")
    a, episodes = agent(), []
    with (out / "reference_episodes.jsonl").open("w") as stream:
        for spec, env in zip(specs, bench.verbal_envs):
            for repeat in range(m.N_TRAJECTORIES):
                row = m.run_episode(env, spec, repeat, a, "summary", "native", out)
                episodes.append(row)
                stream.write(json.dumps(row, allow_nan=False)+"\n")
                stream.flush()
    assert len(episodes) == 480
    count = sum(x["steps"] for x in episodes)
    save(out / "original_summary.json", {
        "status": "complete", "conditions": 16, "episodes": len(episodes),
        "steps": count, "history": "summary", "action_rule": "native",
        "per_episode_original_reward_and_pseudo_regret": episodes,
        "source_protocol": m.SOURCE_LABEL,
        "note": "Original helper executed; different reward scales are not pooled.",
    })
    r.scored(count, 312000, scorer="original run_episode reward/pseudo-regret",
             original_summary="original_summary.json")


def expansion(task, mode, out, r):
    import pandas as pd
    m = module("src.cogbench_expansion")
    arg = COGBENCH[task]
    runs = m.DEFAULT_RUNS[arg]
    expected = runs*{"temporal": 1, "bart": 45, "instrumental": 96}[arg]
    frames = []
    with (out / "reference_requests.jsonl").open("w") as stream:
        bridge = m.ChoiceLLM(agent(), arg, stream, "native", "cpu")
        for i in range(runs):
            frame = m.run_one(arg, bridge, i, 10000+i, "1", 500)
            frame.to_csv(out / f"run_{i:04d}.csv", index=False)
            frames.append(frame)
        frame = pd.concat(frames, ignore_index=True)
        frame.to_csv(out / "reference_trials.csv", index=False)
        r.inputs(len(frame), expected, original_runs=runs, generated_calls=len(bridge.records),
                 builder=f"original run_one({arg})",
                 scope="all original code-default runs, not held-out human observations")
        r.mark("legal_output_and_original_scoring", "running",
               note="Full original scorer; unchanged Instrumental DE settings may take substantial CPU time.")
        scores = m.score_runs(arg, frame, LABEL, 10000)
        scores.to_csv(out / "original_scores.csv", index=False)
        summary = m.summary_metrics(arg, frame, bridge.records)
    save(out / "original_summary.json", {
        "status": "complete", "task": arg, "original_scoring": "complete",
        "original_scores_file": "original_scores.csv", **summary,
    })
    r.scored(len(frame), expected, scorer="original get_scores and original fit settings",
             original_summary="original_summary.json",
             undefined_score_note="Undefined original score CSV values are retained.")


def environment_receipt(out):
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("Use the newly created validation venv.")
    cfg = (Path(sys.prefix) / "pyvenv.cfg").read_text().lower()
    if "include-system-site-packages = false" not in cfg:
        raise RuntimeError("The venv must not inherit system site packages.")
    versions = {d.metadata["Name"].lower(): d.version
                for d in importlib.metadata.distributions()}
    missing = [name for name in ("numpy", "pandas", "scipy", "statsmodels", "torch")
               if name not in versions]
    if missing:
        raise RuntimeError("Required CPU dependencies not installed: "+repr(missing))
    check = subprocess.run([sys.executable, "-m", "pip", "check"], text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out / "pip_check.log").write_text(check.stdout)
    if check.returncode:
        raise RuntimeError("Fresh environment dependency conflict; see pip_check.log")
    code = [
        "external/CogBench/Experiments",
        "external/paper_matched_centaur_evaluation/horizon",
        "external/paper_matched_centaur_evaluation/wcst",
        "external/coggym_social_evaluation", "external/EVOLvE",
        "external/callaway2022_archive/planning/model",
    ]
    for name in code:
        path = ROOT / name
        if path.exists() and not path.resolve().is_relative_to(ROOT):
            raise RuntimeError("Copy source code; do not borrow production code: "+name)
    data = [
        "data/psych101/prompts_testing_t1.jsonl",
        "data/rlwm/CollinsFrank_2012_EJoN.csv",
        "data/mouselab_callaway2022_official",
        "data/hybrid_paper_matched/fold_assignments.csv",
        "data/tomi_original_2019/tomi_balanced_story_types",
        "literature/four_task_resolution_20261018",
        "literature/task_catalog_20261017/raw",
        "external/HybridModellingProject/Data/fit/data",
    ]
    value = {
        "verified_utc": now(), "status": "clean_venv_core_dependency_check_passed",
        "python": sys.version, "executable": sys.executable,
        "prefix": sys.prefix, "base_prefix": sys.base_prefix,
        "include_system_site_packages": False, "dependency_versions": versions,
        "source_copy": str(ROOT), "pip_check_returncode": 0,
        "code_locations": [{"path": x, "resolved": str((ROOT/x).resolve()),
                            "exists": (ROOT/x).exists()} for x in code],
        "raw_input_locations": [{"path": x, "resolved": str((ROOT/x).resolve()),
                                 "exists": (ROOT/x).exists()} for x in data],
        "data_acquisition": "Existing authorized inputs, not an independent provider-download test.",
        "restricted_data": "Do not redistribute Psych-101 or private validation response artifacts.",
        "julia": "Separate native environment; Python installation alone does not certify it.",
        "memory_note": "worker maxRSS excludes separate Julia child; actual tree limits are external.",
    }
    save(out / "environment.json", value)
    return value


def worker(task, out, environment):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    torch.set_num_threads(1)
    failures = []
    for mode in MODES.get(task, ("primary_protocol",)):
        directory = out / mode
        directory.mkdir(parents=True, exist_ok=False)
        r = Receipt(task, mode, directory, environment)
        try:
            if task in EXPECTED_TYPED:
                typed(task, mode, directory, r)
            elif task in {"probability_coherence_judgment", "performance_self_assessment"}:
                four(task, mode, directory, r)
            elif task == "bayesian_urn_inference":
                posterior(task, mode, directory, r)
            elif task in {"horizon", "mouselab_tree"} and mode == "human_predictive":
                prediction(task, mode, directory, r)
            elif task == "wcst":
                wcst(task, mode, directory, r)
            elif task == "hybrid_information_sampling":
                hybrid(task, mode, directory, r)
            elif task == "stationary_multiarmed_bandit":
                bandit(task, mode, directory, r)
            elif task in COGBENCH:
                expansion(task, mode, directory, r)
            else:
                generated(task, mode, directory, r)
            r.finish()
        except Exception as exc:
            r.fail(exc)
            failures.append({"task": task, "mode": mode, "error": repr(exc)})
            traceback.print_exc()
    save(out / "task_result.json", {
        "task": task, "status": "failed" if failures else "complete",
        "declared_modes": list(MODES.get(task, ("primary_protocol",))),
        "failures": failures,
    })
    return 1 if failures else 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--task", default="all")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--environment", type=Path, help=argparse.SUPPRESS)
    args = p.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to(ROOT):
        raise ValueError("Write validation artifacts inside the new source copy.")
    if args.worker:
        env = json.loads(args.environment.read_text())
        raise SystemExit(worker(args.task, out, env))
    tasks = json.loads((ROOT / "selected_tasks_24.json").read_text())
    # 2026-10-26: restored the verified original list-of-dictionaries schema.
    # A printed extracted-ID list had been mistaken for the source format.
    task_ids = [x["task_id"] for x in tasks]
    assert len(task_ids) == len(set(task_ids)) == 24
    chosen = task_ids if args.task == "all" else [args.task]
    if not set(chosen).issubset(task_ids):
        raise ValueError("Task outside approved24.")
    out.mkdir(parents=True, exist_ok=False)
    runtime = out / ".runtime"
    for name in ("home", "tmp", "cache", "matplotlib"):
        (runtime/name).mkdir(parents=True)
    os.environ.update({
        "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1", "JULIA_NUM_THREADS": "1",
        "TF_NUM_INTRAOP_THREADS": "1", "TF_NUM_INTEROP_THREADS": "1",
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        "HOME": str(runtime/"home"), "TMPDIR": str(runtime/"tmp"),
        "XDG_CACHE_HOME": str(runtime/"cache"), "MPLCONFIGDIR": str(runtime/"matplotlib"),
        "PYTHONPATH": str(ROOT),
    })
    environment_receipt(out)
    ordered = [t for t in chosen if t != "instrumental_learning"]
    if "instrumental_learning" in chosen:
        ordered.append("instrumental_learning")
    results = []
    for task in ordered:
        directory = out / task
        directory.mkdir()
        command = [sys.executable, "-B", str(Path(__file__).resolve()),
                   "--worker", "--task", task, "--output", str(directory),
                   "--environment", str(out/"environment.json")]
        with (directory/"worker.log").open("w") as log:
            result = subprocess.run(command, cwd=ROOT, env=os.environ.copy(),
                                    stdout=log, stderr=subprocess.STDOUT)
        path = directory / "task_result.json"
        state = json.loads(path.read_text()) if path.exists() else {
            "task": task, "status": "worker_failed_before_receipt"}
        results.append({**state, "returncode": result.returncode})
        save(out/"progress.json", {"status": "running", "updated_utc": now(),
                                  "planned_tasks": len(chosen), "finished_tasks": len(results),
                                  "results": results})
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)
    completed = sum(x["status"] == "complete" and x["returncode"] == 0 for x in results)
    summary = {
        "status": "complete" if completed == len(chosen) else "incomplete",
        "completed_tasks": completed, "planned_tasks": len(chosen),
        "approved_task_count": 24, "finished_utc": now(), "results": results,
        "full24_cpu_reference_validation_complete": len(chosen) == 24 and completed == 24,
        "scope": "Full named CPU-reference input/output/original-score pipelines in fresh Python venv.",
        "not_certified": ["Five neural models/tokenizers/readouts",
                          "Independent gated input acquisition or redistribution",
                          "New strong-baseline performance or cognitive superiority"],
    }
    save(out/"summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    raise SystemExit(0 if summary["status"] == "complete" else 1)


if __name__ == "__main__":
    main()
