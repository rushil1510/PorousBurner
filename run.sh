#!/usr/bin/env bash
# Run GA optimizer and append best parameters and fitness score to log.txt
output=$(python3 ga_optimizer.py)
best=$(echo "$output" | grep 'Best parameters')
fitness=$(echo "$output" | grep 'Fitness score')
echo "$(date '+%Y-%m-%d %H:%M:%S') | $best | $fitness" >> log.txt
