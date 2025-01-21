#!/bin/sh

# Set PYTHONPATH to the project root
export PYTHONPATH=$(dirname "$0")

# Function to check if a process is running by PID
check_process() {
    pid=$1
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        return 0  # Process is running
    else
        return 1  # Process is not running
    fi
}

# Run API
API_PID_FILE=api_pid.txt
if test -f "$API_PID_FILE"; then
    pid=$(cat "$API_PID_FILE")
    if check_process "$pid"; then
        echo "API is already running with PID $pid"
    else
        echo "Stale PID file detected. Removing $API_PID_FILE..."
        rm "$API_PID_FILE"
    fi
fi

if ! test -f "$API_PID_FILE"; then
    echo "Running Inference API..."
    nohup poetry run python data_ml_assignment/api/main.py > api.log 2>&1 &
    echo $! > "$API_PID_FILE"
    echo "API started successfully."
fi

# Run Streamlit
STREAMLIT_PID_FILE=streamlit_pid.txt
if test -f "$STREAMLIT_PID_FILE"; then
    pid=$(cat "$STREAMLIT_PID_FILE")
    if check_process "$pid"; then
        echo "Streamlit dashboard is already running with PID $pid"
    else
        echo "Stale PID file detected. Removing $STREAMLIT_PID_FILE..."
        rm "$STREAMLIT_PID_FILE"
    fi
fi

if ! test -f "$STREAMLIT_PID_FILE"; then
    echo "Running Streamlit Dashboard..."
    nohup poetry run python -W ignore -m streamlit run --server.port 8001 dashboard.py > streamlit.log 2>&1 &
    echo $! > "$STREAMLIT_PID_FILE"
    echo "Streamlit dashboard started successfully."
fi