# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# © 2025 Mahesh Mutukula. All rights reserved.
# This file is part of the GenAI QA Eval Framework.

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import logging
from typing import Dict, List
from sklearn.metrics import confusion_matrix

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def plot_confusion_matrix(y_true: List[int], y_pred: List[int], labels: List[str], output_path: str):
    """Plot and save confusion matrix."""
    try:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Confusion matrix saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to plot confusion matrix: {str(e)}")
        raise

def plot_metric_trends(metrics: List[Dict[str, float]], output_path: str):
    """Plot metric trends over test runs."""
    try:
        df = pd.DataFrame(metrics)
        plt.figure(figsize=(10, 6))
        for column in df.columns:
            if column.endswith("_score") or column.endswith("_metric"):
                plt.plot(df.index, df[column], label=column)
        plt.title("Metric Trends")
        plt.xlabel("Test Run")
        plt.ylabel("Score")
        plt.legend()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Metric trends saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to plot metric trends: {str(e)}")
        raise

def save_json_report(results: Dict[str, Any], output_path: str):
    """Save test results as JSON."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"JSON report saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save JSON report: {str(e)}")
        raise
