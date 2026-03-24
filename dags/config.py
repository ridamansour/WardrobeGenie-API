"""
config.py
Global configuration variables for WardrobeGenie.
"""

import os
from pathlib import Path

# This automatically finds the root folder of your project dynamically.
# (Assuming config.py is inside wardrobegenie/dags/)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Define core directories relative to the root
DATA_DIR = PROJECT_ROOT / "data" / "fashionpedia_coco"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# We cast them to strings here so they format perfectly into your Bash commands later
PROJECT_ROOT_STR = str(PROJECT_ROOT)
DATA_DIR_STR = str(DATA_DIR)