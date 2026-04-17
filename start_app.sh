#!/bin/bash

# 1. Ensure we are in the main project directory
cd "$(dirname "$0")"

echo "🎸 Initializing Deep Chord Project (Unified Stack)..."

# 2. LOAD AND ACTIVATE CONDA
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lab

if [ $? -ne 0 ]; then
    echo "❌ ERROR: Could not activate Conda environment 'lab'."
    read
    exit 1
fi

# 3. START UNIFIED SERVER (Backend + Frontend on port 8000)
echo "-> Starting Unified Server on http://localhost:8000..."
python src/web/main.py &
SERVER_PID=$!

# Give the PyTorch model 3 seconds to load into GPU
sleep 3

echo "---------------------------------------------------"
echo "✅ System ready and operational!"
echo "---------------------------------------------------"
echo "Press CTRL+C in this window to shut down everything."

# 4. MAGIC: Automatically open the browser on Linux!
xdg-open http://localhost:8000 2>/dev/null &

# 5. SHUTDOWN MANAGEMENT
trap "echo -e '\n🛑 Shutting down...'; kill $SERVER_PID 2>/dev/null; exit" INT
wait $SERVER_PID

echo "⚠️ WARNING: The server has closed unexpectedly."
read
