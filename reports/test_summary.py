import json
import os
import yaml
from utils.report_utils import save_json_report

# Load configuration
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)
REPORT_CONFIG = config["reporting"]

# Sample test results (replace with actual results from test runs)
test_results = {
    "llm": {
        "relevancy_score": 0.85,
        "hallucination_score": 0.15,
        "tests_passed": 8,
        "tests_failed": 1
    },
    "nlp": {
        "entity_extraction": {
            "precision": 0.88,
            "recall": 0.82,
            "f1": 0.85
        },
        "intent_detection": {
            "accuracy": 0.92
        },
        "tests_passed": 6,
        "tests_failed": 0
    },
    "ml": {
        "adherence_model": {
            "precision": 0.87,
            "recall": 0.81,
            "f1": 0.84
        },
        "risk_score_model": {
            "mse": 0.08,
            "r2": 0.78
        },
        "tests_passed": 10,
        "tests_failed": 2
    },
    "end_to_end": {
        "tests_passed": 3,
        "tests_failed": 0
    }
}

def generate_test_summary():
    """Generate and save JSON test summary."""
    output_path = os.path.join(REPORT_CONFIG["output_dir"], "test_summary.json")
    save_json_report(test_results, output_path)
    print(f"Test summary saved to {output_path}")

if __name__ == "__main__":
    generate_test_summary()