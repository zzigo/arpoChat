#!/bin/bash

# Set development environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)
export LOG_LEVEL="debug"

# Note: Set HF_TOKEN environment variable before running this script
# export HF_TOKEN="your-token-here"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Start the development server
uvicorn api.main:app --host 0.0.0.0 --port 8002 --reload --log-level debug