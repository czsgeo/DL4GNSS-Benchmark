#!/bin/bash
# Round-1 controlled GRU driver.
#
# The original run_training.sh (xargs -P4 over all 66 jobs) was stopped because
# the non-cuDNN GRU path costs ~4 h/seed, so the full 16-seed GRU plan (~16 h)
# does not fit the budget. Decision (recorded in state/response-plan.md notes and
# in out/summary): scope the M6 alternative-architecture baseline to
#   denoise in 07..14  (already trained / already running -> 8 seeds)
#   impute  in 07..10  (trained here -> 4 seeds)
# and drop impute 11..14. This driver trains ONLY those 4 impute nets, in
# parallel with the still-running denoise jobs (8 cores, ~4 free).
#
# It touches logs/gru-done only after every gru job (denoise + impute) is gone,
# which is the gate the experiment driver waits on before running alt-arch.
set -u
cd "$(dirname "$0")"
LOG=logs/impute-driver.log
mkdir -p logs
echo "=== gru-impute driver start $(date) ===" >> "$LOG"

for s in 20261007 20261008 20261009 20261010; do
  echo "gru impute in $s"
done | xargs -d '\n' -P 4 -n 1 bash _one.sh >> "$LOG" 2>&1
echo "=== gru-impute training launched-jobs done $(date); waiting for denoise arms ===" >> "$LOG"

# Wait until no train_all.py process (denoise or impute) remains.
while pgrep -f "train_all.py .* gru " >/dev/null 2>&1; do sleep 60; done
echo "=== all gru jobs finished $(date) ===" >> "$LOG"
touch logs/gru-done
echo "=== gru-done sentinel written $(date) ===" >> "$LOG"
