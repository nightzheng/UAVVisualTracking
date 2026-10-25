"""Evaluate aligned saved results; use --plan for strict complete-parent macro scores."""
import argparse
import json
from pathlib import Path
from uav_visual_tracking.common.provenance import write_json
from uav_visual_tracking.data.results import load_result
from uav_visual_tracking.evaluation.entry import load_gt,evaluate_sequence,evaluate_plan


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result',type=Path);p.add_argument('--gt',type=Path)
    p.add_argument('--model',choices=['B0','B1'])
    p.add_argument('--plan',type=Path,help='Complete explicit plan JSON; paths relative to plan file')
    p.add_argument('--epsilon',type=float,choices=[5.,10.],default=10.)
    p.add_argument('--query-mode',choices=['first','strided'],default='first')
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args(argv)
    if a.output.exists():p.error('Use a new output file')
    if a.plan:
        if a.result or a.gt or a.model:p.error('--plan cannot be combined with single-result arguments')
        report=evaluate_plan(json.loads(a.plan.read_text('utf8')),a.plan.resolve().parent,a.epsilon,a.query_mode)
    else:
        if not all([a.result,a.gt,a.model]):p.error('Provide --result, --gt and --model, or --plan')
        d,m=load_result(a.result)
        report=evaluate_sequence(d,m,load_gt(a.gt),a.model,a.epsilon,a.query_mode)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    # Exclusive create also protects against a concurrent writer.
    with a.output.open('x',encoding='utf8') as f:json.dump(report,f,indent=2,ensure_ascii=False,allow_nan=False)
    print(json.dumps(dict(status=report.get('status','sequence_complete'),output=a.output.name)))
    return 2 if report.get('status')=='refused_incomplete_plan' else 0


if __name__=='__main__':raise SystemExit(main())
