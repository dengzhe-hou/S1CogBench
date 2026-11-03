"""P2 original full Horizon/WCST, generation control, pilot-to-full reuse.

The pilot records 200 calls/task. Full execution replays the exact saved prefix
without neural calls, then continues at original full sample. A prefix mismatch
fails explicitly. No scientific setting or old result is changed.
"""
from pathlib import Path
import argparse
import datetime
import json
import random
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "p3"))
from generation_agent import GenerationAgent
from task_runtime import configure_root, horizon_question, load_horizon, write_json


class PilotFinished(Exception):
    pass


class InvalidAction(Exception):
    pass


class CallRecorder:
    def __init__(self, agent, task, output, phase, reuse=None, participant_ids=None, expected_decisions=None):
        self.agent, self.task, self.output, self.phase = agent, task, output, phase
        self.rows, self.reused = [], 0
        self.expected_decisions = expected_decisions or (34720 if task == "horizon" else 12754)
        self.cache = []
        if reuse is not None:
            path = Path(reuse) / task / "calls.jsonl"
            if not path.exists():
                raise FileNotFoundError("Pilot cache missing: " + str(path))
            self.cache = [json.loads(line) for line in path.open()]
            if participant_ids is not None:
                field = "participant_id" if task == "horizon" else "subject_id"
                self.cache = [x for x in self.cache if int(x["source_identity"][field]) in set(participant_ids)]
        self.stream = (output / "calls.jsonl").open("w")
        self.neural = 0
        self.invalid_count = self.truncated_count = self.total_input_tokens = self.total_output_tokens = 0
        self.total_new_inference_seconds = 0.0

    def call(self, prompt, question, keys, identity):
        index = len(self.rows)
        if self.phase == "pilot" and index >= 200:
            raise PilotFinished()
        if index < len(self.cache):
            record = dict(self.cache[index])
            if record["state"] != prompt or record["source_identity"] != identity or record["output_keys"] != keys:
                raise ValueError("Saved pilot prefix differs from full original task state/identity/legal keys.")
            record["pilot_query_index"] = record["query_index"]
            record["reused_pilot_response"] = True
            self.reused += 1
        else:
            if self.agent.model_name == "llama":
                record = self.agent.decide_raw_prompt(prompt, keys)
            else:
                record = self.agent.decide(prompt, question, keys)
            record.update(source_identity=identity, reused_pilot_response=False, question=question)
            self.neural += 1
        record.update(query_index=index, task=self.task)
        self.rows.append(record)
        self.invalid_count += int(record["invalid"])
        self.truncated_count += int(record["truncated"])
        self.total_input_tokens += int(record["input_tokens"])
        self.total_output_tokens += int(record["output_tokens"])
        if not record["reused_pilot_response"]:
            self.total_new_inference_seconds += float(record["inference_seconds"])
        if len(self.rows) % 10 == 0:
            progress = {
                "status": "running", "event": "P2_PROGRESS", "phase": self.phase,
                "task": self.task, "model": self.agent.model_name,
                "attempted_decisions": len(self.rows),
                "expected_decisions": 200 if self.phase == "pilot" else self.expected_decisions,
                "new_neural_calls": self.neural, "reused_pilot_calls": self.reused,
                "invalid_outputs": self.invalid_count, "truncated_outputs": self.truncated_count,
                "input_tokens_total": self.total_input_tokens, "output_tokens_total": self.total_output_tokens,
                "new_neural_inference_seconds": self.total_new_inference_seconds,
                "last_input_tokens": record["input_tokens"], "last_output_tokens": record["output_tokens"],
                "last_inference_seconds": record["inference_seconds"], "source_identity": identity,
            }
            write_json(self.output / "progress_call.json", progress)
            print(json.dumps(progress), flush=True)
        self.stream.write(json.dumps(record, allow_nan=False) + "\n")
        self.stream.flush()
        if len(self.rows) == 200 and self.phase == "pilot":
            self.pilot_receipt()
        return record

    def pilot_receipt(self):
        import numpy as np
        rows = self.rows[:200]
        write_json(self.output / "PILOT_200_MEASURED.json", {
            "status": "measured_generation_prefix_not_full_task_score",
            "task": self.task, "calls": len(rows),
            "model": self.agent.model_name, "generation_batch_size": 1,
            "output_tokens_mean": float(np.mean([x["output_tokens"] for x in rows])),
            "input_tokens_mean": float(np.mean([x["input_tokens"] for x in rows])),
            "latency_p50_seconds": float(np.quantile([x["inference_seconds"] for x in rows], .5)),
            "latency_p95_seconds": float(np.quantile([x["inference_seconds"] for x in rows], .95)),
            "invalid": sum(x["invalid"] for x in rows),
            "truncated": sum(x["truncated"] for x in rows),
            "reuse_full_prefix": "Full phase verifies exact original state/identity and reuses these responses without neural calls.",
        })
        print(json.dumps({"event": "PILOT_200_MEASURED", "model": self.agent.model_name, "task": self.task}), flush=True)

    def close(self):
        self.stream.close()


def call_statistics(rows):
    import numpy as np
    if not rows:
        return {"latency_p50_seconds": None, "latency_p95_seconds": None}
    return {
        "latency_p50_seconds": float(np.quantile([x["inference_seconds"] for x in rows], .5)),
        "latency_p95_seconds": float(np.quantile([x["inference_seconds"] for x in rows], .95)),
        "input_tokens_mean": float(np.mean([x["input_tokens"] for x in rows])),
        "output_tokens_mean": float(np.mean([x["output_tokens"] for x in rows])),
        "new_neural_inference_seconds": float(sum(x["inference_seconds"] for x in rows if not x["reused_pilot_response"])),
        "generation_batch_size": 1,
        "timing_note": "All per-call records retained; reused pilot response timings are original measurements, not new full-phase GPU time.",
    }


def select_participants(all_ids, requested):
    if requested is None:
        return list(all_ids)
    requested = [int(x) for x in requested]
    if not requested or len(set(requested)) != len(requested) or not set(requested).issubset(set(all_ids)):
        raise ValueError("Participant shard must contain distinct original IDs.")
    return [sid for sid in all_ids if sid in set(requested)]


def run_horizon(agent, root, output, phase, reuse, participants=None):
    import numpy as np
    import pandas as pd
    family = "llama" if agent.model_name == "llama" else "centaur"
    official = load_horizon(root, family)
    timeline = pd.read_csv(root / "external/paper_matched_centaur_evaluation/horizon/data/in/timeline_structure.csv")
    if (timeline.participant_id.nunique(), len(timeline), int(timeline.type.eq("free").sum())) != (31, 74400, 34720):
        raise ValueError("Original full Horizon count mismatch.")
    all_ids = [int(x) for x in timeline.participant_id.unique()]
    selected = select_participants(all_ids, participants)
    selected_data = timeline[timeline.participant_id.isin(selected)]
    expected_decisions = int(selected_data.type.eq("free").sum())
    expected_rows = len(selected_data)
    is_full_task = selected == all_ids
    seeds = official.generate_seeds(num_seeds=32)
    question = horizon_question(root)
    recorder = CallRecorder(agent, "horizon", output, phase, reuse, selected, expected_decisions)
    frames, failures = [], []
    write_json(output / "metadata.json", {
        "task": "horizon", "phase": phase, "prompt_family": family,
        "expected_participants": len(selected), "selected_participant_ids": selected,
        "expected_source_rows": expected_rows, "expected_model_decisions": expected_decisions,
        "original_full_participants": 31, "original_full_source_rows": 74400,
        "original_full_decisions": 34720, "is_full_task": is_full_task,
        "participant_seeds": {str(int(p)): seeds[int(p)-1] for p in timeline.participant_id.unique()},
        "agent": agent.metadata,
        "invalid_transition": "Original environment defines no reward for invalid action; retain call and mark trajectory incomplete. No substitute action or retry.",
    })
    try:
        for participant in selected:
            part = timeline[timeline.participant_id.eq(participant)]
            free_rows = part[part.type.eq("free")].sort_values(["block", "game", "trial_num_block"]).reset_index(drop=True)
            current = 0

            class Pipe:
                def __call__(self, prompt):
                    nonlocal current
                    row = free_rows.iloc[current]
                    identity = {k: int(row[k]) for k in ["participant_id", "block", "game", "trial_num_block"]}
                    record = recorder.call(prompt, question, ["H", "I"], identity)
                    current += 1
                    if record["invalid"]:
                        raise InvalidAction(record["invalid_reason"])
                    return [{"generated_text": prompt + record["choice"]}]

            try:
                result = official.simulate_participant_by_block(part, participant, seeds, None, None, Pipe())
                expected = part.rename(columns={"trial_num_block": "trial_num"}).set_index(
                    ["participant_id", "block", "game", "trial_num"]).sort_index()
                actual = result.set_index(["participant_id", "block", "game", "trial_num"]).sort_index()
                pd.testing.assert_index_equal(actual.index, expected.index)
                free = actual.is_free.astype(bool)
                np.testing.assert_array_equal(free, expected.type.eq("free"))
                np.testing.assert_array_equal(actual.loc[~free, "choice"], expected.loc[~free, "choice"])
                np.testing.assert_array_equal(actual.loc[~free, "reward"], expected.loc[~free, "reward"])
                routed = np.where(actual.choice.eq("H"), expected.reward_H, expected.reward_I)
                np.testing.assert_array_equal(actual.loc[free, "reward"], routed[free])
                result.to_csv(output / f"participant_{int(participant)}.csv", index=False)
                frames.append(result)
            except InvalidAction as exc:
                failure = {"participant_id": int(participant), "completed_or_attempted_decisions": current, "error": str(exc)}
                failures.append(failure)
                write_json(output / f"participant_{int(participant)}_incomplete.json", failure)
            write_json(output / "progress.json", {
                "status": "running", "phase": phase, "participants_complete": len(frames),
                "participants_incomplete": failures, "attempted_decisions": len(recorder.rows),
                "new_neural_calls": recorder.neural, "reused_pilot_calls": recorder.reused,
                "expected_full_decisions": 34720, "expected_shard_decisions": expected_decisions,
            })
    except PilotFinished:
        pass
    finally:
        recorder.close()
    complete = phase == "full" and not failures and len(frames) == len(selected) and len(recorder.rows) == expected_decisions
    summary = {
        "status": "complete" if complete else ("pilot_measured" if phase == "pilot" and len(recorder.rows) == 200 else "incomplete"),
        "task": "horizon", "phase": phase, "attempted_decisions": len(recorder.rows),
        "expected_full_decisions": 34720, "expected_shard_decisions": expected_decisions,
        "selected_participant_ids": selected, "is_full_task": is_full_task,
        "participants_complete": len(frames), "failures": failures,
        "invalid_outputs": sum(x["invalid"] for x in recorder.rows), "truncated_outputs": sum(x["truncated"] for x in recorder.rows),
        "new_neural_calls": recorder.neural, "reused_pilot_calls": recorder.reused,
        "full_task_metrics_defined": complete and is_full_task, "selected_participant_metrics_defined":complete,
    }
    if complete:
        full = pd.concat(frames, ignore_index=True)
        free = full[full.is_free.astype(bool)].copy()
        full.to_csv(output / "trials.csv", index=False)
        free["optimal"] = ((free.choice.eq("H") & free.reward_mean_H.gt(free.reward_mean_I)) |
                           (free.choice.eq("I") & free.reward_mean_I.gt(free.reward_mean_H)))
        summary.update(mean_reward_free_additional_metric=float(free.reward.mean()),
                       mean_optimal_choice_rate=float(free.optimal.mean()))
        by_trial = free.groupby(["participant_id", "horizon", "trial_num"]).optimal.mean()
        by_trial.groupby(["horizon", "trial_num"]).agg(["mean", "sem"]).to_csv(output / "optimal_choice_by_trial.csv")
    summary.update(call_statistics(recorder.rows))
    write_json(output / "summary.json", summary)
    return summary


def run_wcst(agent, root, output, phase, reuse, participants=None):
    import numpy as np
    import torch
    from src.wcst_task import Dialogue, LETTERS, QUESTION, task_metrics, test_data
    data = test_data()
    all_ids = list(data)
    selected = select_participants(all_ids, participants)
    data = {sid:data[sid] for sid in selected}
    expected_decisions = sum(map(len, data.values()))
    is_full_task = selected == all_ids
    family = "llama" if agent.model_name == "llama" else "centaur"
    recorder = CallRecorder(agent, "wcst", output, phase, reuse, selected, expected_decisions)
    summaries, failures = [], []
    write_json(output / "metadata.json", {
        "task": "wcst", "phase": phase, "prompt_family": family,
        "participants": len(selected), "selected_participant_ids": selected,
        "expected_model_decisions": expected_decisions, "original_full_participants":75,
        "original_full_decisions":12754, "is_full_task":is_full_task,
        "seed": "10000 + original subject_id", "agent": agent.metadata,
        "split": "Original ordered first20% participants, exactly75; source stimulus/rule timeline.",
        "invalid_transition": "Llama original source renders invalid answer and SWITCH feedback; Centaur formatter cannot advance invalid key, so mark that trajectory incomplete.",
    })
    try:
        for sid, rows in data.items():
            random.seed(10000 + sid); np.random.seed(10000 + sid); torch.manual_seed(10000 + sid); torch.cuda.manual_seed_all(10000 + sid)
            dialogue, records = Dialogue(family, "generative"), []
            with (output / f"{sid}_trials.jsonl").open("w") as stream:
                for i, row in enumerate(rows):
                    prompt = dialogue.prompt(row, agent.tokenizer)
                    identity = {"subject_id": int(sid), "trial": i, "source_step": int(row["step"])}
                    response = recorder.call(prompt, QUESTION, list(LETTERS), identity)
                    letter = response["choice"]
                    key = LETTERS.index(letter) + 1 if letter is not None else None
                    record = {"subject_id": int(sid), "trial": i, "source_step": int(row["step"]),
                              "human_key": int(row["user_key"]), "ground_key": int(row["ground_key"]),
                              "ground_rule": int(row["ground_rule"]), "stimulus": list(row["stimulus"]),
                              "executed_key": key, "letter": letter,
                              "feedback": "REPEAT" if key == row["ground_key"] else "SWITCH",
                              "invalid": response["invalid"], "truncated": response["truncated"],
                              "call_index": response["query_index"]}
                    stream.write(json.dumps(record, allow_nan=False) + "\n"); stream.flush(); records.append(record)
                    if key is None and family != "llama":
                        failures.append({"subject_id": int(sid), "attempted_decisions": len(records), "error": response["invalid_reason"]})
                        break
                    dialogue.update(row, key, letter or "?")
            if len(records) == len(rows):
                part = {"subject_id": int(sid), "trials": len(records),
                        **task_metrics(rows, [x["executed_key"] for x in records])}
                summaries.append(part)
                write_json(output / f"{sid}_summary.json", part)
            write_json(output / "progress.json", {
                "status": "running", "phase": phase, "participants_complete": len(summaries),
                "failures": failures, "attempted_decisions": len(recorder.rows),
                "new_neural_calls": recorder.neural, "reused_pilot_calls": recorder.reused,
                "expected_full_decisions": 12754, "expected_shard_decisions": expected_decisions,
            })
    except PilotFinished:
        pass
    finally:
        recorder.close()
    complete = phase == "full" and not failures and len(summaries) == len(selected) and len(recorder.rows) == expected_decisions
    summary = {
        "status": "complete" if complete else ("pilot_measured" if phase == "pilot" and len(recorder.rows) == 200 else "incomplete"),
        "task": "wcst", "phase": phase, "attempted_decisions": len(recorder.rows), "expected_full_decisions": 12754,
        "expected_shard_decisions":expected_decisions, "selected_participant_ids":selected, "is_full_task":is_full_task,
        "participants_complete": len(summaries), "failures": failures,
        "invalid_outputs": sum(x["invalid"] for x in recorder.rows), "truncated_outputs": sum(x["truncated"] for x in recorder.rows),
        "new_neural_calls": recorder.neural, "reused_pilot_calls": recorder.reused,
        "full_task_metrics_defined": complete and is_full_task, "selected_participant_metrics_defined":complete,
    }
    if complete:
        summary["per_participant"] = summaries
        summary["metrics"] = {
            field: {"participant_mean": float(np.mean([x[field] for x in summaries])),
                    "trial_weighted_mean": float(np.average([x[field] for x in summaries], weights=[x["trials"] for x in summaries]))}
            for field in ["accuracy", "perseveration_error", "set_loss_error"]
        }
    summary.update(call_statistics(recorder.rows))
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["llama", "qwen"], required=True)
    parser.add_argument("--phase", choices=["pilot", "full"], required=True)
    parser.add_argument("--root", type=Path, default=HERE.parent / "root")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-pilot", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--task", choices=["both", "horizon", "wcst"], default="both")
    args = parser.parse_args()
    if args.phase == "pilot" and args.reuse_pilot is not None:
        parser.error("Pilot cannot reuse an existing pilot.")
    if args.phase == "full" and args.reuse_pilot is None:
        parser.error("Full phase must reuse its measured pilot prefix; no duplicate neural calls.")
    root = configure_root(args.root)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    agent = GenerationAgent(args.model, args.device, root)
    results = {}
    for task, fn in [("horizon", run_horizon), ("wcst", run_wcst)]:
        if args.task not in ["both", task]:
            continue
        output = args.output / task
        output.mkdir()
        results[task] = fn(agent, root, output, args.phase, args.reuse_pilot)
        print(json.dumps(results[task]), flush=True)
    required = "complete" if args.phase == "full" else "pilot_measured"
    status = required if all(x["status"] == required for x in results.values()) else "incomplete"
    write_json(args.output / "RUN_STATUS.json", {
        "status": status, "phase": args.phase, "model": args.model,
        "tasks": results, "total_seconds": time.perf_counter() - started,
        "recorded_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scope": "Fixed full original tasks; pilot alone is not full-task completion.",
    })
    if status == "incomplete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
