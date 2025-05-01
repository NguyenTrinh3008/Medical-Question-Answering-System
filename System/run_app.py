import os
import sys
import subprocess

# Disable Streamlit's file watcher
os.environ["STREAMLIT_SERVER_WATCH_DIRS"] = "false"

# Run the Streamlit app
subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
