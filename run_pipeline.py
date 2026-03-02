"""
ChurnGuard Pipeline Runner

Executes the complete ML pipeline from data generation to batch scoring.
Useful for testing full pipeline or resetting project state.

Author: Aditya
Version: 1.0.0
"""

from scripts.reset_project import reset_folders
from scripts.generate_data import generate_synthetic_data
from src.models import ChurnModelPipeline
from batch.score_all_customers import batch_score


def run():
    """Execute complete ML pipeline."""
    print("Resetting project...")
    reset_folders()

    print("Generating data...")
    generate_synthetic_data()

    print("Training model...")
    pipeline = ChurnModelPipeline()
    pipeline.train()

    print("Running batch scoring...")
    batch_score()

    print("Pipeline complete.")


if __name__ == "__main__":
    run()