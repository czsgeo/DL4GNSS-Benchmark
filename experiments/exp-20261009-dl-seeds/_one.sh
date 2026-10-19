#!/bin/bash
# Train one net. $1 = "arch task family seed"
read -r arch task family seed <<< "$1"
log=logs/runner.log
echo "[$(date +%H:%M:%S)] START $arch $task $family $seed" >> "$log"
../.venv/bin/python train_all.py "$task" "$arch" "$family" "$seed" >> logs/train.log 2>&1
echo "[$(date +%H:%M:%S)] END   $arch $task $family $seed rc=$?" >> "$log"
