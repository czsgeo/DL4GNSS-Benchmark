#!/bin/bash
# Train every net in config.json's plans with N-way parallelism.
# Usage: bash run_training.sh [n_workers]
set -u
cd "$(dirname "$0")"
N=${1:-4}
mkdir -p nets logs
../.venv/bin/python - <<'PYEOF' > jobs.txt
import json
cfg = json.load(open('config.json'))
for name, pl in cfg['plans'].items():
    for s in pl['seeds']:
        print(pl['arch'], pl['task'], pl['family'], s)
PYEOF
echo "jobs: $(wc -l < jobs.txt)"
xargs -d '\n' -P "$N" -n 1 bash _one.sh < jobs.txt
echo "ALL DONE $(date)" >> logs/runner.log
echo "ALL DONE"
