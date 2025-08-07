#!/bin/bash

# Create a log file with timestamp (optional for uniqueness)
LOGFILE="dates_analysis_$(date +'%Y-%m-%d_%H-%M-%S').log"

# Run the Python script and redirect stdout and stderr to the log file
python dates_analysis.py \
  --input_folder ../sukkari-RGB/ \
  --output_csv results_sukkari.csv \
  --output_dir ./sukkari \
  > "$LOGFILE" 2>&1
