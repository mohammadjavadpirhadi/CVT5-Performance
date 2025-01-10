#!/bin/bash

# Define the number of iterations
iterations=10

# Compare these two
# ffmpeg -y -loglevel error -i "{input_video}" -s 224x224 -c:a aac -filter:v fps=2 -movflags faststart -f null -
# ffmpeg -y -loglevel error -i "{input_video}" -s 224x224 -c:v libx264 -x264-params bframes=0 -c:a aac -filter:v fps=2 -movflags faststart -f null -

# Prompt the user for the command
read -p "Enter the command to run: " command

# Initialize variables to store times
times=()

# Run the command multiple times
for ((i=1; i<=iterations; i++)); do
    echo "Running iteration $i..."

    # Run the command and capture the real time using 'time', handling spaces correctly
    result=$( (time -p eval "$command") 2>&1 | grep real | awk '{print $2}' )

    # Check if the command executed successfully
    if [ $? -ne 0 ]; then
        echo "Error running the command on iteration $i."
        continue
    fi

    # Store the time in the array
    times+=($result)
    echo "Run $i: $result seconds"
done

# Check if there are valid results to process
if [ ${#times[@]} -eq 0 ]; then
    echo "No valid execution times recorded. Please check the command and try again."
    exit 1
fi

# Calculate average and standard deviation using awk
awk -v times="${times[*]}" '
BEGIN {
    split(times, arr, " ")
    n = length(arr)
    sum = 0
    for (i = 1; i <= n; i++) {
        sum += arr[i]
    }
    avg = sum / n
    sumsq = 0
    for (i = 1; i <= n; i++) {
        sumsq += (arr[i] - avg) ^ 2
    }
    stddev = sqrt(sumsq / n)
    printf "Average: %.3f seconds\n", avg
    printf "Standard Deviation: %.3f seconds\n", stddev
}'
