import yaml
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Load configuration
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)
ML_CONFIG = config["ml"]

# Sample test data (replace with actual test data or fixtures)
y_true = [1, 0, 1, 1, 0, 0, 1, 0]  # Ground truth labels
y_scores = [0.9, 0.1, 0.8, 0.7, 0.2, 0.3, 0.95, 0.05]  # Predicted probabilities

def generate_roc_curve():
    """Generate and save ROC curve for ML model."""
    output_path = "reports/visualizations/roc_curve.png"
    
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    
    # Save plot
    plt.savefig(output_path)
    plt.close()
    print(f"ROC curve saved to {output_path}")

if __name__ == "__main__":
    generate_roc_curve()