"""Check recorded Stage A hashes without rerunning the technical audit."""
import argparse
import json
from pathlib import Path
from uav_visual_tracking.common.provenance import sha256, write_json

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--legacy-root",required=True,type=Path)
parser.add_argument("--baseline",required=True,type=Path)
parser.add_argument("--output",required=True,type=Path)
args=parser.parse_args()
if args.output.exists():raise FileExistsError("Do not overwrite integrity evidence")
baseline=json.loads(args.baseline.read_text('utf-8'))['source_snapshot']
changed=[]
for relative,expected in baseline.items():
    path=args.legacy_root/relative
    if not path.is_file():changed.append(dict(file=relative,status='missing'))
    elif sha256(path)!=expected:changed.append(dict(file=relative,status='changed'))
new=[]
for path in args.legacy_root.rglob('*'):
    if path.is_file() and not any(part in {'.git','__pycache__'} for part in path.parts):
        relative=path.relative_to(args.legacy_root).as_posix()
        if relative not in baseline:new.append(relative)
report=dict(checked_existing_files=len(baseline),changed=changed,new_files=new,unchanged=not changed and not new,
            bytecode_and_git_excluded=True,comparison='Stage A full-file hash baseline; not a new technical audit')
write_json(args.output,report)
print(json.dumps(report,ensure_ascii=False,indent=2))
if not report['unchanged']:raise SystemExit(1)
