"""Top-level entrypoint for model training and evaluation.

Runs the comprehensive ML pipeline in src/ml/train_and_evaluate.py.
"""

from src.ml.train_and_evaluate import run_pipeline

if __name__ == "__main__":
    run_pipeline("config/train_config.yaml")
