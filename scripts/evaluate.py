"""Evaluate saved predictions against an explicitly aligned GT NPZ."""
import argparse
from pathlib import Path
from uav_visual_tracking.common.provenance import write_json
from uav_visual_tracking.data.results import load_result
from uav_visual_tracking.evaluation.metrics import evaluate_gt_file

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--result",required=True,type=Path)
parser.add_argument("--gt",required=True,type=Path)
parser.add_argument("--output",required=True,type=Path)
args=parser.parse_args()
if args.output.exists():raise FileExistsError("Do not overwrite existing evaluations")
data,manifest=load_result(args.result)
metrics=evaluate_gt_file(data,manifest,args.gt)
args.output.parent.mkdir(parents=True,exist_ok=True)
write_json(args.output,metrics)
print(metrics)
