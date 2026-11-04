import hashlib
import json
from pathlib import Path

import os

ARCH = Path(os.environ.get("UAV_WORKBENCH_ARCHIVE", ".local/research_archive")).resolve()
STAGE = ARCH / 'stage_u3d_visual_inspection_01'
SETTINGS = Path(__file__).resolve().parent

def read(path):return json.loads(Path(path).read_text(encoding='utf8'))


def save(path,data):Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
