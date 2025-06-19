# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# © 2025 Mahesh Mutukula. All rights reserved.
# This file is part of the GenAI QA Eval Framework.

import pytest
import json
import yaml
from utils.ml_utils import invoke_sagemaker_endpoint, evaluate_classification, evaluate_regression
from config.credentials import get_credentials_manager

# Load configuration
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)
ML_CONFIG = config["ml"]

# Load fixtures
with open("tests/fixtures/ml_fixtures.json", "r") as f:
    ML_FIXTURES = json.load(f)

@pytest.fixture(scope="module")
def credentials():
    """Provide credentials for tests."""
    return get_credentials_manager()

@pytest.mark.parametrize("test_case", [tc for tc in ML_FIXTURES if tc["model"] == "adherence"])
def test_adherence_model(test_case, credentials):
    """Test medication adherence classification model."""
    payload = test_case["input"]
    expected_label = test_case["expected_label"]
    
    # Invoke SageMaker endpoint
    response = invoke_sagemaker_endpoint(ML_CONFIG["sagemaker_endpoints"]["adherence"], payload)
    predicted_label = response["prediction"]
    
    # Validate prediction
    assert predicted_label == expected_label, (
        f"Expected label '{expected_label}', got '{predicted_label}'"
    )
    
    # Evaluate metrics (assuming multiple test cases for metrics)
    y_true = [test_case["expected_label"]]
    y_pred = [predicted_label]
    results = evaluate_classification(
        y_true, y_pred,
        min_precision=ML_CONFIG["evaluation"]["classification"]["min_precision"],
        min_recall=ML_CONFIG["evaluation"]["classification"]["min_recall"],
        min_f1=ML_CONFIG["evaluation"]["classification"]["min_f1"]
    )
    
    assert results["precision_pass"], f"Precision too low: {results['precision']}"
    assert results["recall_pass"], f"Recall too low: {results['recall']}"
    assert results["f1_pass"], f"F1-score too low: {results['f1']}"

@pytest.mark.parametrize("test_case", [tc for tc in ML_FIXTURES if tc["model"] == "risk_score"])
def test_risk_score_model(test_case, credentials):
    """Test user risk score regression model."""
    payload = test_case["input"]
    expected_score = test_case["expected_score"]
    
    # Invoke SageMaker endpoint
    response = invoke_sagemaker_endpoint(ML_CONFIG["sagemaker_endpoints"]["risk_score"], payload)
    predicted_score = response["prediction"]
    
    # Validate prediction (within tolerance)
    assert abs(predicted_score - expected_score) < 0.1, (
        f"Expected score {expected_score}, got {predicted_score}"
    )
    
    # Evaluate metrics
    y_true = [expected_score]
    y_pred = [predicted_score]
    results = evaluate_regression(
        y_true, y_pred,
        max_mse=ML_CONFIG["evaluation"]["regression"]["max_mse"],
        min_r2=ML_CONFIG["evaluation"]["regression"]["min_r2"]
    )
    
    assert results["mse_pass"], f"MSE too high: {results['mse']}"
    assert results["r2_pass"], f"R2 too low: {results['r2']}"