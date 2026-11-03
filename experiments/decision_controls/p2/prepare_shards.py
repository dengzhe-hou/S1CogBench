"""Exact disjoint participant partition for the full P2 tasks."""
from pathlib import Path
import argparse
import csv
import datetime
import json
import random
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"p3"))
from task_runtime import configure_root, write_json


def original_units(root, task):
    root=Path(root)
    if task=="horizon":
        source=root/"external/paper_matched_centaur_evaluation/horizon/data/in/timeline_structure.csv"
        units={}
        with source.open() as f:
            for row in csv.DictReader(f):
                sid=int(row["participant_id"])
                unit=units.setdefault(sid,{"participant_id":sid,"source_rows":0,"decisions":0})
                unit["source_rows"]+=1
                unit["decisions"]+=int(row["type"]=="free")
        if (len(units),sum(x["source_rows"] for x in units.values()),sum(x["decisions"] for x in units.values()))!=(31,74400,34720):
            raise ValueError("Original Horizon full counts do not match.")
        rng=random.Random(42)
        seeds=[rng.randint(1,100000) for _ in range(32)]
        for sid,unit in units.items(): unit["seed"]=seeds[sid-1]
    elif task=="wcst":
        configure_root(root)
        from src.wcst_task import test_data
        data=test_data()
        units={int(sid):{"participant_id":int(sid),"source_rows":len(rows),"decisions":len(rows),"seed":10000+int(sid)} for sid,rows in data.items()}
        source=root/"external/paper_matched_centaur_evaluation/wcst/Data/json_wcst_dataset.npy"
        if len(units)!=75 or sum(x["decisions"] for x in units.values())!=12754:
            raise ValueError("Original WCST full counts do not match.")
    else: raise ValueError("Task must be horizon or wcst.")
    return list(units.values()),str(source)


def validate_partition(plan):
    units={int(x["participant_id"]):x for x in plan["original_units"]}
    seen=[]
    for shard in plan["shards"]:
        ids=[int(x) for x in shard["participant_ids"]]
        if not ids or len(ids)!=len(set(ids)): raise ValueError("Empty or duplicated shard participant.")
        if not set(ids).issubset(units): raise ValueError("Non-source participant in shard.")
        if shard["expected_decisions"]!=sum(units[sid]["decisions"] for sid in ids): raise ValueError("Shard decision count mismatch.")
        if shard["expected_source_rows"]!=sum(units[sid]["source_rows"] for sid in ids): raise ValueError("Shard source row count mismatch.")
        seen.extend(ids)
    if len(seen)!=len(units) or len(set(seen))!=len(seen) or set(seen)!=set(units):
        raise ValueError("Shards must partition all original participants exactly once.")
    if sum(x["expected_decisions"] for x in plan["shards"])!=plan["expected_full_decisions"]: raise ValueError("Partition total decision mismatch.")
    if sum(x["expected_source_rows"] for x in plan["shards"])!=plan["expected_full_source_rows"]: raise ValueError("Partition total row mismatch.")
    return True


def build_plan(root,task,model,parts,pilot_dir,results_root):
    units,source=original_units(root,task)
    if not 1<=parts<=len(units): raise ValueError("Invalid shard count.")
    buckets=[[] for _ in range(parts)]; loads=[0]*parts
    order={x["participant_id"]:i for i,x in enumerate(units)}
    for unit in sorted(units,key=lambda x:(-x["decisions"],order[x["participant_id"]])):
        i=min(range(parts),key=lambda i:(loads[i],i))
        buckets[i].append(unit)
        loads[i]+=unit["decisions"]
    shards=[]
    for i,bucket in enumerate(buckets):
        bucket.sort(key=lambda x:order[x["participant_id"]])
        shards.append({"shard_id":f"s{i:03d}","participant_ids":[x["participant_id"] for x in bucket],
                       "expected_decisions":sum(x["decisions"] for x in bucket),
                       "expected_source_rows":sum(x["source_rows"] for x in bucket),
                       "output_relative":f"s{i:03d}"})
    plan={"schema":"s1cogbench-p2-original-participant-partition-v1","model":model,"task":task,
          "recorded_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"source_file":source,
          "root":str(root),"pilot_dir":str(pilot_dir),"results_root":str(results_root),
          "original_participant_ids":[x["participant_id"] for x in units],"original_units":units,
          "expected_full_decisions":sum(x["decisions"] for x in units),
          "expected_full_source_rows":sum(x["source_rows"] for x in units),"shards":shards,
          "scientific_scope":"All original participants/trials/seeds retained; only independent participant work is distributed.",
          "generation_settings":{"max_new_tokens":1024,"do_sample":False,"batch_size":1},
          "pilot_reuse":"One shared200decision pilot per model/task. Each shard reuses only matching original participant prefix records; disjoint shards require no response prefix.",
          "status":"prepared_not_launched"}
    validate_partition(plan)
    return plan


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--task",choices=["horizon","wcst"],required=True)
    parser.add_argument("--model",choices=["llama","qwen"],required=True)
    parser.add_argument("--parts",type=int,required=True)
    parser.add_argument("--pilot-dir",type=Path,required=True)
    parser.add_argument("--results-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError("Preserve previous partition plan.")
    plan=build_plan(args.root,args.task,args.model,args.parts,args.pilot_dir,args.results_root)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_json(args.output,plan)
    print(json.dumps({"status":"prepared_not_launched","manifest":str(args.output),
                      "shards":[{"id":x["shard_id"],"participants":len(x["participant_ids"]),"decisions":x["expected_decisions"]} for x in plan["shards"]]}))

if __name__=="__main__":main()
