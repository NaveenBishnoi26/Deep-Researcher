#!/bin/bash

# Activate virtual environment
source ../.venv/bin/activate

# Set PYTHONPATH to include the current directory
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run Streamlit
streamlit run frontend.py 