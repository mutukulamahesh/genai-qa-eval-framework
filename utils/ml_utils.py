import boto3
from typing import List, Dict, Any
from sklearn.metrics import precision_recall_f1_score, mean_squared_error, r2_score
import numpy as np
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def invoke_sagemaker_endpoint(endpoint_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Invoke SageMaker endpoint with input payload."""
    try:
        sagemaker = boto3.client("sagemaker-runtime")
        response = sagemaker.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Body=json.dumps(payload)
        )
        result = json.loads(response["Body"].read().decode("utf-8"))
        logger.info(f"SageMaker response: {result}")
        return result
    except Exception as e:
        logger.error(f"SageMaker invocation failed: {str(e)}")
        raise

def evaluate_classification(
    y_true: List[int],
    y_pred: List[int],
    min_precision: float = 0.85,
    min_recall: float = 0.80,
    min_f1: float = 0.82
) -> Dict[str, float]:
    """Evaluate classification model performance."""
    try:
        precision, recall, f1, _ = precision_recall_f1_score(y_true, y_pred, average="weighted")
        results = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "precision_pass": precision >= min_precision,
            "recall_pass": recall >= min_recall,
            "f1_pass": f1 >= min_f1
        }
        logger.info(f"Classification evaluation results: {results}")
        return results
    except Exception as e:
        logger.error(f"Classification evaluation failed: {str(e)}")
        raise

def evaluate_regression(
    y_true: List[float],
    y_pred: List[float],
    max_mse: float = 0.1,
    min_r2: float = 0.75
) -> Dict[str, float]:
    """Evaluate regression model performance."""
    try:
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        results = {
            "mse": mse,
            "r2": r2,
            "mse_pass": mse <= max_mse,
            "r2_pass": r2 >= min_r2
        }
        logger.info(f"Regression evaluation results: {results}")
        return results
    except Exception as e:
        logger.error(f"Regression evaluation failed: {str(e)}")
        raise