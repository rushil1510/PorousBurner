#!/bin/bash

# This script runs the GA optimizer and logs the results.

# Run the parse_dataset_to_csv.py script to convert .dat files to .csv
python parse_dataset_to_csv.py

# Run the GA optimizer and append the results to log.txt
python ga_optimizer.py >> log.txt

# Print the last line of the log file to the console
tail -n 1 log.txt

echo "GA optimization completed. Results appended to log.txt"