#!/usr/bin/env bash
# Run GA optimizer and append runtime (in seconds) with best parameters and fitness score to log.txt

# record start time in nanoseconds
start_time=$(date +%s%N)

output=$(python3 ga_optimizer.py)

# record end time in nanoseconds
end_time=$(date +%s%N)

# compute elapsed time in seconds with millisecond precision
elapsed_ns=$((end_time - start_time))
elapsed=$(awk "BEGIN {printf \"%.3f\", ${elapsed_ns}/1000000000}")

best=$(echo "$output" | grep 'Best parameters')
fitness=$(echo "$output" | grep 'Fitness score')

echo "${elapsed}s | $best | $fitness" >> log.txt