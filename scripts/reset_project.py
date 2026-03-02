"""
ChurnGuard Project Reset Utility

Clears generated artifacts (models, results, reports) to enable clean re-runs.
Useful during development when testing pipeline changes.

Author: Aditya
Version: 1.0.0
"""

import shutil
import os


FOLDERS_TO_RESET = [
    "models",
    "results",
    "reports",
    "data/processed"
]


def reset_folders():
    """
    Remove and recreate project artifact directories.
    
    This function clears all generated files from previous runs including:
    - Trained models
    - Batch scoring results
    - Visualization reports
    - Processed feature data
    
    Raw data (data/raw/) is preserved to avoid accidental data loss.
    """
    for folder in FOLDERS_TO_RESET:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)
        print(f"Reset: {folder}")
    
    print("\nProject folders reset successfully.")
    print("Next steps:")
    print("  1. python -m scripts.generate_data")
    print("  2. python -m src.models")
    print("  3. python -m batch.score_all_customers")


if __name__ == "__main__":
    reset_folders()