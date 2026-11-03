"""Merge accepted disjoint P2 full shards against the original full task.

Run on an admitted CPU job. Raw generation outputs remain in their shard files;
this writes full scientific metrics and an explicit raw-call file index.
"""
from pathlib import Path
import argparse
import datetime
import json
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(HERE.parent/"p3"))
from prepare_shards import original_units,validate_partition
from task_runtime import configure_root,write_json


def identity(record,task):
    row=record["source_identity"]
    keys=["participant_id","block","game","trial_num_block"] if task=="horizon" else ["subject_id","trial","source_step"]
    return tuple(int(row[k]) for k in keys)


def pilot_records(plan,pilot_dir):
    pilot=Path(pilot_dir or plan["pilot_dir"])
    receipt=json.loads((pilot/plan["task"]/"PILOT_200_MEASURED.json").read_text())
    run=json.loads((pilot/"RUN_STATUS.json").read_text())
    if receipt["calls"]!=200 or receipt["model"]!=plan["model"] or receipt["task"]!=plan["task"]:
        raise ValueError("One complete matching200decision pilot is required.")
    if run["phase"]!="pilot" or run["status"]!="pilot_measured":
        raise ValueError("Pilot status is not successful measured phase.")
    rows=[json.loads(x) for x in (pilot/plan["task"]/"calls.jsonl").open()]
    if len(rows)!=200:raise ValueError("Pilot call count mismatch.")
    result={identity(row,plan["task"]):row for row in rows}
    if len(result)!=200:raise ValueError("Pilot duplicated original decision identities.")
    return result,str(pilot)


def scan_calls(plan,locations,pilot):
    import math
    choices={}; times=[]; inputs=[]; outputs=[]
    invalid=truncated=reused=new=0
    new_seconds=0.
    sources=[]; reused_ids=set()
    for shard,folder in locations:
        path=folder/plan["task"]/"calls.jsonl"
        sources.append({"shard_id":shard["shard_id"],"path":str(path)})
        local=0
        with path.open() as f:
            for line in f:
                row=json.loads(line); key=identity(row,plan["task"])
                if key in choices:raise ValueError("Duplicated original decision across shards.")
                if row["task"]!=plan["task"]:raise ValueError("Call task mismatch.")
                field="participant_id" if plan["task"]=="horizon" else "subject_id"
                if int(row["source_identity"][field]) not in shard["participant_ids"]:
                    raise ValueError("Call belongs to another shard participant.")
                legal=["H","I"] if plan["task"]=="horizon" else list("ABCD")
                if row["output_keys"]!=legal:raise ValueError("Call legal action order differs.")
                if row["choice"] is not None and row["choice"] not in legal:raise ValueError("Illegal parsed action.")
                if bool(row["invalid"])!=(row["choice"] is None):raise ValueError("Invalid flag/action mismatch.")
                if row["truncated"] and not row["invalid"]:raise ValueError("Truncation was not treated as invalid.")
                if int(row["output_tokens"])>1024:raise ValueError("Generation allowance exceeded.")
                if not math.isfinite(row["inference_seconds"]) or row["inference_seconds"]<=0:raise ValueError("Missing finite call timing.")
                if int(row["generation_batch_size"])!=1:raise ValueError("P2 batch1 setting changed.")
                choices[key]=row["choice"]
                times.append(float(row["inference_seconds"]));inputs.append(int(row["input_tokens"]));outputs.append(int(row["output_tokens"]))
                invalid+=int(row["invalid"]);truncated+=int(row["truncated"])
                if row["reused_pilot_response"]:
                    if key not in pilot:raise ValueError("Unrecognized reused pilot decision.")
                    original=pilot[key]
                    for name in ["state","model_prompt","raw_text","generated_token_ids","choice","input_tokens","output_tokens","invalid","truncated"]:
                        if row[name]!=original[name]:raise ValueError("Reused pilot response changed: "+name)
                    reused_ids.add(key);reused+=1
                else:
                    new+=1;new_seconds+=float(row["inference_seconds"])
                local+=1
        if local!=shard["expected_decisions"]:raise ValueError("Shard call count mismatch.")
    if len(choices)!=plan["expected_full_decisions"]:raise ValueError("Full call union count mismatch.")
    if reused!=200 or reused_ids!=set(pilot):raise ValueError("The single pilot prefix was not reused exactly once across shards.")
    import numpy as np
    metrics={"latency_p50_seconds":float(np.quantile(times,.5)),"latency_p95_seconds":float(np.quantile(times,.95)),
             "input_tokens_mean":float(np.mean(inputs)),"output_tokens_mean":float(np.mean(outputs)),
             "invalid_outputs":invalid,"truncated_outputs":truncated,"new_neural_calls":new,"reused_pilot_calls":reused,
             "new_neural_inference_seconds":new_seconds,"generation_batch_size":1,
             "timing_context":"Distributed batch1; actual GPU UUID and kernel metadata retained per shard. P1 is the exclusive same-GPU performance comparison."}
    return choices,metrics,sources


def merge_horizon(root,plan,locations,choices,output):
    import numpy as np
    import pandas as pd
    source=pd.read_csv(root/"external/paper_matched_centaur_evaluation/horizon/data/in/timeline_structure.csv")
    frames=[]
    for shard,folder in locations:
        frame=pd.read_csv(folder/"horizon"/"trials.csv")
        if len(frame)!=shard["expected_source_rows"]:raise ValueError("Horizon shard source rows mismatch.")
        if set(map(int,frame.participant_id.unique()))!=set(shard["participant_ids"]):raise ValueError("Horizon shard participant union mismatch.")
        frames.append(frame)
    full=pd.concat(frames,ignore_index=True)
    keys=["participant_id","block","game","trial_num"]
    expected=source.rename(columns={"trial_num_block":"trial_num"}).set_index(keys).sort_index()
    actual=full.set_index(keys).sort_index()
    pd.testing.assert_index_equal(actual.index,expected.index)
    free=actual.is_free.astype(bool)
    np.testing.assert_array_equal(free,expected.type.eq("free"))
    np.testing.assert_array_equal(actual.loc[~free,"choice"],expected.loc[~free,"choice"])
    np.testing.assert_array_equal(actual.loc[~free,"reward"],expected.loc[~free,"reward"])
    if not actual.choice.isin(["H","I"]).all():raise ValueError("Horizon full trajectory contains illegal action.")
    routed=np.where(actual.choice.eq("H"),expected.reward_H,expected.reward_I)
    np.testing.assert_array_equal(actual.loc[free,"reward"],routed[free])
    decision_keys=set()
    for index,row in actual.loc[free].iterrows():
        key=tuple(map(int,index))
        if choices.get(key)!=row.choice:raise ValueError("Horizon executed choice differs from saved generated answer.")
        decision_keys.add(key)
    if decision_keys!=set(choices):raise ValueError("Horizon scored/call identities do not match.")
    order={int(sid):i for i,sid in enumerate(plan["original_participant_ids"])}
    full["_source_participant_order"]=full.participant_id.map(order)
    full=full.sort_values(["_source_participant_order","block","game","trial_num"]).drop(columns="_source_participant_order")
    full.to_csv(output/"trials.csv",index=False)
    free_rows=full[full.is_free.astype(bool)].copy()
    free_rows["optimal"]=((free_rows.choice.eq("H") & free_rows.reward_mean_H.gt(free_rows.reward_mean_I)) |
                          (free_rows.choice.eq("I") & free_rows.reward_mean_I.gt(free_rows.reward_mean_H)))
    curves=free_rows.groupby(["participant_id","horizon","trial_num"]).optimal.mean()
    curves.groupby(["horizon","trial_num"]).agg(["mean","sem"]).to_csv(output/"optimal_choice_by_trial.csv")
    return {"participants":31,"rows":len(full),"decisions":len(free_rows),
            "mean_reward_free_additional_metric":float(free_rows.reward.mean()),
            "mean_optimal_choice_rate":float(free_rows.optimal.mean()),
            "checks":"Exact full original keys/forced actions/observed reward routing and generated-choice alignment."}


def merge_wcst(root,plan,locations,choices,output):
    import numpy as np
    from src.wcst_task import LETTERS,task_metrics,test_data
    data=test_data();parts=[];seen_ids=set();decision_keys=set()
    for shard,folder in locations:
        for sid in shard["participant_ids"]:
            if sid in seen_ids:raise ValueError("Duplicated WCST participant.")
            seen_ids.add(sid)
            rows=data[sid]
            path=folder/"wcst"/f"{sid}_trials.jsonl"
            records=[json.loads(line) for line in path.open()]
            if len(records)!=len(rows):raise ValueError("Missing WCST original trials.")
            keys=[]
            for i,(record,source) in enumerate(zip(records,rows)):
                if int(record["subject_id"])!=sid or record["trial"]!=i or record["source_step"]!=int(source["step"]):
                    raise ValueError("WCST source identity mismatch.")
                for field in ["human_key","ground_key","ground_rule"]:
                    src="user_key" if field=="human_key" else field
                    if record[field]!=int(source[src]):raise ValueError("WCST source target changed.")
                if record["stimulus"]!=list(source["stimulus"]):raise ValueError("WCST original stimulus changed.")
                key=record["executed_key"]
                if key is not None and key not in [1,2,3,4]:raise ValueError("WCST action invalid type/value.")
                letter=LETTERS[key-1] if key is not None else None
                call_key=(sid,i,int(source["step"]))
                if call_key not in choices or choices[call_key]!=letter:raise ValueError("WCST response/executed action mismatch.")
                if record["feedback"]!=("REPEAT" if key==source["ground_key"] else "SWITCH"):raise ValueError("WCST own-action feedback misrouting.")
                if bool(record["invalid"])!=(key is None):raise ValueError("WCST invalid flag changed.")
                keys.append(key);decision_keys.add(call_key)
            metrics=task_metrics(rows,keys)
            saved=json.loads((folder/"wcst"/f"{sid}_summary.json").read_text())
            for field,value in metrics.items():
                if abs(float(saved[field])-value)>1e-12:raise ValueError("WCST participant score differs from original full-trial scoring.")
            part={"subject_id":sid,"trials":len(rows),**metrics}
            parts.append(part)
            with (output/f"{sid}_trials.jsonl").open("w") as out:
                for record in records:out.write(json.dumps(record,allow_nan=False)+"\n")
    if seen_ids!=set(data) or decision_keys!=set(choices):raise ValueError("WCST full union incomplete.")
    parts.sort(key=lambda x:plan["original_participant_ids"].index(x["subject_id"]))
    metrics={field:{"participant_mean":float(np.mean([x[field] for x in parts])),
                    "trial_weighted_mean":float(np.average([x[field] for x in parts],weights=[x["trials"] for x in parts]))}
             for field in ["accuracy","perseveration_error","set_loss_error"]}
    return {"participants":75,"decisions":sum(x["trials"] for x in parts),"per_participant":parts,"metrics":metrics,
            "checks":"All original split participants/trial identities/targets/stimuli/actions/feedback; original task_metrics recomputed."}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--results-root",type=Path)
    parser.add_argument("--pilot-dir",type=Path)
    parser.add_argument("--root",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    plan=json.loads(args.manifest.read_text());validate_partition(plan)
    root=configure_root(args.root or plan["root"])
    units,_=original_units(root,plan["task"])
    if units!=plan["original_units"]:raise ValueError("Partition does not match current original source participants/counts/seeds.")
    result_root=args.results_root or Path(plan["results_root"])
    locations=[]
    for shard in plan["shards"]:
        folder=result_root/shard["output_relative"]
        status=json.loads((folder/"SHARD_STATUS.json").read_text())
        if status["status"]!="complete" or status["scope"]!="original_participant_shard":
            raise ValueError("Shard is not complete; preserve it without a full-task completion claim.")
        if status["model"]!=plan["model"] or status["task"]!=plan["task"] or status["shard_id"]!=shard["shard_id"]:
            raise ValueError("Shard model/task/identity mismatch.")
        if status["participant_ids"]!=shard["participant_ids"] or status["expected_decisions"]!=shard["expected_decisions"]:
            raise ValueError("Shard participant/count plan changed.")
        meta=json.loads((folder/plan["task"]/"metadata.json").read_text())
        agent=meta["agent"]
        if agent["max_new_tokens"]!=1024 or agent["do_sample"] is not False:
            raise ValueError("P2 generation settings changed.")
        locations.append((shard,folder))
    pilot,pilot_dir=pilot_records(plan,args.pilot_dir)
    choices,timing,sources=scan_calls(plan,locations,pilot)
    args.output.mkdir(parents=True,exist_ok=False)
    fn=merge_horizon if plan["task"]=="horizon" else merge_wcst
    summary=fn(root,plan,locations,choices,args.output)
    summary.update(status="complete",phase="full",scope="all_original_participants",
                   full_task_metrics_defined=True,model=plan["model"],task=plan["task"],**timing)
    write_json(args.output/"summary.json",summary)
    write_json(args.output/"CALL_FILES.json",{"raw_calls_preserved_in_shards":sources,"pilot_dir":pilot_dir})
    receipt={"status":"accepted","task":plan["task"],"model":plan["model"],
             "full_original_participants":plan["original_participant_ids"],
             "decisions":plan["expected_full_decisions"],"source_rows":plan["expected_full_source_rows"],
             "shards":len(locations),"pilot_calls_reused_exactly_once":200,
             "no_duplicate_or_omitted_original_decisions":True,
             "summary":str(args.output/"summary.json"),"raw_call_index":str(args.output/"CALL_FILES.json"),
             "recorded_utc":datetime.datetime.now(datetime.timezone.utc).isoformat()}
    write_json(args.output/"P2_FULL_SHARDS_MERGED_ACCEPTED_20261028.json",receipt)
    print(json.dumps(receipt))

if __name__=="__main__":main()
