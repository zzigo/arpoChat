#!/bin/bash

# Check if HF_TOKEN is provided as an argument
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN environment variable is not set"
    echo "Please set your HF_TOKEN environment variable first:"
    echo "export HF_TOKEN='your_hugging_face_token'"
    exit 1
fi

# Set development environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)
export LOG_LEVEL="debug"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Kill any existing process on port 8004
lsof -ti:8004 | xargs kill -9 2>/dev/null || true

# Start the development server
echo "Starting server on http://localhost:8004"
uvicorn api.main:app --host 0.0.0.0 --port 8004 --reload --log-level debug 