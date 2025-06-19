# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# © 2025 Mahesh Mutukula. All rights reserved.
# This file is part of the GenAI QA Eval Framework.


import yaml
from utils.report_utils import plot_confusion_matrix
from utils.ml_utils import invoke_sagemaker_endpoint
from config.credentials import get_credentials_manager

# Load configuration
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)
ML_CONFIG = config["ml"]

# Sample test data (replace with actual test data or fixtures)
y_true = [1, 0, 1, 1, 0, 0, 1, 0]  # Ground truth labels
y_pred = [1, 0, 1, 0, 0, 1, 1, 0]  # Predicted labels
labels = ["Not Eligible", "Eligible"]  # Class labels for rebate eligibility model

def generate_confusion_matrix():
    """Generate and save confusion matrix for ML model."""
    output_path = "reports/visualizations/confusion_matrix.png"
    plot_confusion_matrix(y_true, y_pred, labels, output_path)
    print(f"Confusion matrix saved to {output_path}")

if __name__ == "__main__":
    generate_confusion_matrix()