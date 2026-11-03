"""Run one approved full P2 participant shard after its single shared pilot."""
from pathlib import Path
import argparse
import datetime
import json
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from prepare_shards import validate_partition
from run_p2 import run_horizon,run_wcst
from generation_agent import GenerationAgent
sys.path.insert(0,str(HERE.parent/"p3"))
from task_runtime import configure_root,write_json


def pilot_for_shard(plan,participant_ids,pilot_dir=None):
    pilot=Path(pilot_dir or plan["pilot_dir"])
    task=plan["task"]
    receipt=pilot/task/"PILOT_200_MEASURED.json"
    status=pilot/"RUN_STATUS.json"
    if not receipt.exists() or not status.exists():
        raise FileNotFoundError("Wait for the single model/task pilot to complete; do not create a new pilot.")
    info=json.loads(receipt.read_text()); run=json.loads(status.read_text())
    if info["calls"]!=200 or info["model"]!=plan["model"] or info["task"]!=task:
        raise ValueError("Pilot measurement does not match model/task200decision plan.")
    if run["phase"]!="pilot" or run["status"]!="pilot_measured":
        raise ValueError("Pilot did not finish successfully; preserve error and do not automatically bypass it.")
    calls=[json.loads(x) for x in (pilot/task/"calls.jsonl").open()]
    if len(calls)!=200:raise ValueError("Pilot call count is not200.")
    field="participant_id" if task=="horizon" else "subject_id"
    selected=[x for x in calls if int(x["source_identity"][field]) in set(participant_ids)]
    return (pilot if selected else None),{"global_pilot_dir":str(pilot),"global_pilot_calls":200,
        "pilot_responses_assigned_to_this_shard":len(selected),
        "pilot_query_indices":[int(x["query_index"]) for x in selected]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--shard-id",required=True)
    parser.add_argument("--root",type=Path)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--pilot-dir",type=Path)
    parser.add_argument("--device",default="cuda:0")
    args=parser.parse_args()
    plan=json.loads(args.manifest.read_text());validate_partition(plan)
    shard=next((x for x in plan["shards"] if x["shard_id"]==args.shard_id),None)
    if shard is None:raise ValueError("Unknown manifest shard.")
    reuse,pilot_info=pilot_for_shard(plan,shard["participant_ids"],args.pilot_dir)
    root=configure_root(args.root or plan["root"])
    output=args.output or Path(plan["results_root"])/shard["output_relative"]
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/"SHARD_PLAN.json",{"partition":plan,"shard":shard,"pilot_assignment":pilot_info,
        "root_used":str(root),"manifest_path":str(args.manifest),"is_full_task":False})
    started=time.perf_counter()
    agent=GenerationAgent(plan["model"],args.device,root)
    task_dir=output/plan["task"];task_dir.mkdir()
    fn=run_horizon if plan["task"]=="horizon" else run_wcst
    result=fn(agent,root,task_dir,"full",reuse,participants=shard["participant_ids"])
    complete=result["status"]=="complete" and result["attempted_decisions"]==shard["expected_decisions"]
    if result["reused_pilot_calls"]!=pilot_info["pilot_responses_assigned_to_this_shard"]:
        raise ValueError("Assigned pilot responses were not reused exactly once.")
    status={"status":"complete" if complete else "incomplete","scope":"original_participant_shard",
        "is_full_task":False,"model":plan["model"],"task":plan["task"],"shard_id":shard["shard_id"],
        "participant_ids":shard["participant_ids"],"expected_decisions":shard["expected_decisions"],
        "expected_source_rows":shard["expected_source_rows"],"result":result,"pilot_assignment":pilot_info,
        "wall_seconds":time.perf_counter()-started,
        "recorded_utc":datetime.datetime.now(datetime.timezone.utc).isoformat()}
    write_json(output/"SHARD_STATUS.json",status)
    print(json.dumps({k:v for k,v in status.items() if k not in ["result","pilot_assignment"]}),flush=True)
    if not complete:raise SystemExit(2)

if __name__=="__main__":main()
