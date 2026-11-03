#!/usr/bin/env python3
"""Run the original WCST sequential-learning baseline, including its full fit.

The model and training objective are executed directly from the official
notebook. Only grid execution is parallelized; its order and tie rule stay fixed.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from datetime import datetime, timezone
import itertools
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
import traceback
import urllib.request

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "external/paper_matched_centaur_evaluation/wcst"
PUBLISHED = (0.967, 0.656, 0.41, 0.05)
ARCHIVE_BASE = "https://raw.githubusercontent.com/snamazova/centaur_evaluation/main/wcst/Data/simulated_data/"
TRAIN = None
CORE = None
NLL = None


def now():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def notebook_functions():
    notebook = json.loads((SOURCE / "SL_model_fitting_and_eval.ipynb").read_text())
    namespace = {"np": np}
    for cell, names in ((6, {"match", "rule_type", "wcst_sl_choice_prob"}),
                        (9, {"neg_log_likelihood_sl"})):
        tree = ast.parse("".join(notebook["cells"][cell]["source"]))
        selected = [node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name in names]
        assert {node.name for node in selected} == names
        exec(compile(ast.Module(body=selected, type_ignores=[]),
                     f"official_notebook_cell_{cell}", "exec"), namespace)
    return namespace


def load_split():
    # Trusted original public object-array data, not user-supplied pickle input.
    data = np.load(SOURCE / "Data/json_wcst_dataset.npy", allow_pickle=True)
    ids = np.unique([int(row["subject_id"]) for row in data])
    test_ids = ids[:int(len(ids) * 0.2)]
    train_ids = ids[int(len(ids) * 0.2):]
    test_set, train_set = set(test_ids), set(train_ids)
    train = np.array([row for row in data if row["subject_id"] in train_set])
    test = np.array([row for row in data if row["subject_id"] in test_set])
    assert (len(ids), len(train_ids), len(test_ids), len(train), len(test)) == (375, 300, 75, 50467, 12754)
    assert train_set.isdisjoint(test_set)
    # The original objective resets on subject changes, so noncontiguous rows
    # would alter the intended fit; verify the actual official file grouping.
    for rows in (train, test):
        groups = [sid for sid, _ in itertools.groupby(int(x["subject_id"]) for x in rows)]
        assert len(groups) == len(set(groups))
    return train, test, train_ids, test_ids


def subject_average(values):
    return float(np.mean([np.mean(v) for v in values.values()]))


def evaluate(rows, params, mode, outdir, namespace):
    outdir.mkdir(parents=True, exist_ok=True)
    metrics = {name: defaultdict(list) for name in
               ("correct", "aligned", "persev", "setloss", "loglikelihood",
                "human_correct", "human_persev", "human_setloss")}
    state, last_sid = {}, None
    prev_m_rule = prev_h_rule = prev_m_ok = prev_h_ok = None
    generative = mode == "generative"
    with (outdir / "trials.jsonl").open("w") as handle:
        for row in rows:
            sid = int(row["subject_id"])
            if sid != last_sid:
                np.random.seed(10000 + sid)
                state = {}
                prev_m_rule = prev_h_rule = prev_m_ok = prev_h_ok = None
                last_sid = sid
            key, probs, _ = namespace["wcst_sl_choice_prob"](
                row, state, r=params[0], p=params[1], d=params[2], f=params[3],
                sample_choice=generative, update_from="model" if generative else "human")
            assert np.isfinite(probs).all() and np.all(probs >= 0) and np.isclose(probs.sum(), 1)
            key, human_key, ground_key = int(key), int(row["user_key"]), int(row["ground_key"])
            m_ok, h_ok = int(key == ground_key), int(human_key == ground_key)
            m_rule = namespace["rule_type"](row["key_cards"][key], row["stimulus"])
            h_rule = namespace["rule_type"](row["key_cards"][human_key], row["stimulus"])
            record = {
                "subject_id": sid, "trial_index": len(metrics["correct"][sid]),
                "source_step": int(row["step"]), "choice": key, "human_choice": human_key,
                "ground_key": ground_key, "probabilities": probs.tolist(),
                "correct": m_ok, "aligned": int(key == human_key),
                "persev": int(prev_m_ok == 0 and m_ok == 0 and m_rule == prev_m_rule),
                "setloss": int(prev_m_ok == 1 and m_ok == 0 and m_rule != prev_m_rule),
                "loglikelihood": float(np.log(probs[human_key - 1] + 1e-12)),
                "human_correct": h_ok,
                "human_persev": int(prev_h_ok == 0 and h_ok == 0 and h_rule == prev_h_rule),
                "human_setloss": int(prev_h_ok == 1 and h_ok == 0 and h_rule != prev_h_rule),
            }
            for metric in metrics:
                metrics[metric][sid].append(record[metric])
            handle.write(json.dumps(record) + "\n")
            prev_m_rule, prev_h_rule, prev_m_ok, prev_h_ok = m_rule, h_rule, m_ok, h_ok
    for name, values in metrics.items():
        np.save(outdir / f"{name}.npy", dict(values))
    summary = {
        "status": "complete", "mode": mode, "parameters": list(map(float, params)),
        "subjects": len(metrics["correct"]), "trials": len(rows),
        "mean_subject_accuracy": subject_average(metrics["correct"]),
        "mean_subject_alignment": subject_average(metrics["aligned"]),
        "mean_subject_perseveration": subject_average(metrics["persev"]),
        "mean_subject_setloss": subject_average(metrics["setloss"]),
        "human_mean_subject_accuracy": subject_average(metrics["human_correct"]),
        "human_mean_subject_perseveration": subject_average(metrics["human_persev"]),
        "human_mean_subject_setloss": subject_average(metrics["human_setloss"]),
        "seed": "10000 + subject_id; original NumPy RandomState sampling",
        "environment": "original human stimulus/ground-key timeline; model-dependent feedback in generative mode",
    }
    if not generative:
        total_nll = -float(sum(sum(v) for v in metrics["loglikelihood"].values()))
        summary.update(mean_subject_nll=-subject_average(metrics["loglikelihood"]),
                       total_nll=total_nll, trial_weighted_nll=total_nll / len(rows))
    else:
        summary["loglikelihood_note"] = "Saved human-action likelihood conditioned on model history is diagnostic only, not predictive human-trajectory NLL."
    save_json(outdir / "summary.json", summary)
    print(json.dumps({"evaluation": str(outdir), **summary}), flush=True)
    return summary, metrics


def compare_archives(outdir, local):
    """Missing downloads are reported; a failed archive match never changes data."""
    archive_dir = outdir / "author_archives"
    archive_dir.mkdir(exist_ok=True)
    mapping = {
        "SL_model_correct_generative.npy": ("generative", "correct"),
        "SL_model_persev_generative.npy": ("generative", "persev"),
        "SL_model_setloss_generative.npy": ("generative", "setloss"),
        "SL_model_loglikelihood_predictive.npy": ("predictive", "loglikelihood"),
    }
    results = {}
    for filename, (mode, metric) in mapping.items():
        path = archive_dir / filename
        url = ARCHIVE_BASE + filename
        try:
            if not path.exists():
                with urllib.request.urlopen(url, timeout=45) as response:
                    path.write_bytes(response.read())
        except (OSError, urllib.error.URLError) as exc:
            results[filename] = {"source": url, "status": "download_failed", "error": str(exc)}
            continue
        archived = np.load(path, allow_pickle=True).item()
        computed = local[mode][metric]
        ids_match = set(archived) == set(computed)
        shared = sorted(set(archived) & set(computed))
        sizes_match = all(len(archived[s]) == len(computed[s]) for s in shared)
        exact, maximum_error = False, None
        if ids_match and sizes_match:
            differences = np.concatenate([np.asarray(archived[s]) - np.asarray(computed[s]) for s in shared])
            maximum_error = float(np.max(np.abs(differences)))
            exact = bool(np.all(differences == 0))
        results[filename] = {
            "source": url, "status": "compared", "archive_subjects": len(archived),
            "subject_ids_equal": ids_match, "subject_trial_counts_equal": sizes_match,
            "all_values_exactly_equal": exact, "max_abs_error": maximum_error,
            "author_mean_of_subject_means": subject_average(archived),
            "local_mean_of_subject_means": subject_average(computed),
        }
    save_json(outdir / "archive_comparison.json", results)
    return results


def grid_job(item):
    index, params = item
    return index, params, float(NLL(params, TRAIN))


def run(args):
    global TRAIN, CORE, NLL
    outdir = Path(args.output).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    if (outdir / "config.json").exists():
        raise FileExistsError("Use a new output directory; completed/failed experiment records are not overwritten.")
    started = time.time()
    namespace = notebook_functions()
    CORE, NLL = namespace["wcst_sl_choice_prob"], namespace["neg_log_likelihood_sl"]
    TRAIN, test, train_ids, test_ids = load_split()
    config = {
        "started_utc": now(), "pid": os.getpid(), "workers": args.workers,
        "source_notebook": str(SOURCE / "SL_model_fitting_and_eval.ipynb"),
        "source_data": str(SOURCE / "Data/json_wcst_dataset.npy"),
        "source_url": "https://github.com/snamazova/centaur_evaluation/tree/main/wcst",
        "train_subject_ids": train_ids.tolist(), "test_subject_ids": test_ids.tolist(),
        "train_trials": len(TRAIN), "test_trials": len(test),
        "published_parameters": list(PUBLISHED), "grid_candidates": 2835,
        "local_optimization": {"method": "L-BFGS-B", "maxiter": 400,
                               "bounds": [[.001, .999], [.001, .999], [.05, 5.0], [.05, 5.0]]},
        "implementation_change": "6-or-fewer CPU workers evaluate the unchanged original grid; product ordering and first-minimum tie rule retained.",
        "data_rule": "Keep every official row. Correctness follows user_key == ground_key as in original code, not the separate is_correct field.",
        "training_anomalies": {"ground_key_zero": sum(int(x["ground_key"]) == 0 for x in TRAIN),
                               "is_correct_disagreements": sum(bool(x["is_correct"]) != (x["user_key"] == x["ground_key"]) for x in TRAIN)},
    }
    save_json(outdir / "config.json", config)
    progress = {"status": "running", "stage": "published_evaluation", "pid": os.getpid(), "updated_utc": now()}
    save_json(outdir / "progress.json", progress)
    published_summaries, published_metrics = {}, {}
    for mode in ("predictive", "generative"):
        published_summaries[mode], published_metrics[mode] = evaluate(
            test, PUBLISHED, mode, outdir / "published_parameters" / mode, namespace)
    archived = compare_archives(outdir, published_metrics)
    save_json(outdir / "published_parameters" / "summary.json", published_summaries)

    candidates = list(itertools.product(np.linspace(.05, .95, 9), np.linspace(.05, .95, 9),
                                       np.geomspace(.5, 5.0, 7), np.geomspace(.5, 5.0, 5)))
    assert len(candidates) == 2835
    best_nll, best_params = float("inf"), None
    grid_started = time.time()
    progress.update(stage="full_grid_fit", grid_completed=0, grid_total=len(candidates), updated_utc=now())
    save_json(outdir / "progress.json", progress)
    with (outdir / "grid.jsonl").open("w") as handle, mp.get_context("fork").Pool(args.workers) as pool:
        for index, params, nll in pool.imap(grid_job, enumerate(candidates), chunksize=1):
            if nll < best_nll:
                best_nll, best_params = nll, params
            handle.write(json.dumps({"index": index, "parameters": list(params), "train_nll": nll}) + "\n")
            handle.flush()
            if (index + 1) % 10 == 0 or index == len(candidates) - 1:
                elapsed = time.time() - grid_started
                progress.update(grid_completed=index + 1, best_train_nll=best_nll,
                                best_parameters=list(best_params), grid_elapsed_seconds=elapsed,
                                grid_eta_seconds=elapsed / (index + 1) * (len(candidates) - index - 1), updated_utc=now())
                save_json(outdir / "progress.json", progress)
                print(json.dumps(progress), flush=True)
    save_json(outdir / "grid_best.json", {"parameters": list(best_params), "train_nll": best_nll,
                                         "candidates": len(candidates), "seconds": time.time() - grid_started})
    progress.update(stage="lbfgsb_refinement", updated_utc=now())
    save_json(outdir / "progress.json", progress)
    result = minimize(NLL, np.array(best_params), args=(TRAIN,),
                      bounds=[(.001, .999), (.001, .999), (.05, 5.0), (.05, 5.0)],
                      method="L-BFGS-B", options={"maxiter": 400})
    fit = {"success": bool(result.success), "message": str(result.message), "parameters": result.x.tolist(),
           "train_nll": float(result.fun), "nfev": int(result.nfev), "iterations": int(result.nit),
           "train_nll_per_trial": float(result.fun / len(TRAIN)),
           "test_nll": float(NLL(result.x, test)), "optimizer": "L-BFGS-B", "maxiter": 400}
    fit["test_nll_per_trial"] = fit["test_nll"] / len(test)
    save_json(outdir / "fit.json", fit)
    progress.update(stage="refitted_evaluation", updated_utc=now())
    save_json(outdir / "progress.json", progress)
    refit_summaries = {}
    for mode in ("predictive", "generative"):
        refit_summaries[mode], _ = evaluate(test, result.x, mode, outdir / "refitted_parameters" / mode, namespace)
    save_json(outdir / "summary.json", {
        "status": "complete" if result.success else "completed_with_nonconverged_fit",
        "published_parameters": published_summaries, "refitted_parameters": refit_summaries,
        "fit": fit, "archive_comparison": archived, "wall_seconds": time.time() - started,
        "completed_utc": now(), "scope": "Full original 300-subject fit and 75-subject predictive/generative evaluation; no reduced configuration.",
    })
    progress.update(status="complete" if result.success else "completed_with_nonconverged_fit",
                    stage="finished", wall_seconds=time.time() - started, updated_utc=now())
    save_json(outdir / "progress.json", progress)
    print(json.dumps(progress), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=6, choices=range(1, 7))
    arguments = parser.parse_args()
    try:
        run(arguments)
    except Exception:
        output = Path(arguments.output).resolve()
        output.mkdir(parents=True, exist_ok=True)
        (output / "failure.txt").write_text(traceback.format_exc())
        raise
