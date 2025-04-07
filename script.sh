#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# # Install the required Python package
# echo "Installing 'uv' package..."
# python3 -m pip install uv

# Run the data preparation script
echo "Running data preparation..."
python3 src/data_preparation.py 

# Run the training script
echo "Running training..."
python3 src/train.py

# Run the evaluation script
echo "Running evaluation..."
python3 src/evaluate.py

echo "Pipeline execution completed successfully!"