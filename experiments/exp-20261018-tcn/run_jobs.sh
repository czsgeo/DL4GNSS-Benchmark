#!/bin/bash
# 4 concurrent workers; one log per job. Single-threaded BLAS to avoid
# oversubscribing the 8-core host while inference experiments also run.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
grep -v '^$' jobs.txt | xargs -P 4 -n 4 bash -c 't=$0; a=$1; f=$2; s=$3; ../.venv/bin/python train_tcn.py "$t" "$a" "$f" "$s" > "logs/${t}_${f}_s${s}.log" 2>&1; echo "exit $? ${t}_${f}_s${s}" >> logs/driver.status'
