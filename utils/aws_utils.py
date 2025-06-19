import boto3
from botocore.exceptions import ClientError
import logging
from typing import Dict, Any
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_aws_client(service: str, region: str = "us-east-1") -> Any:
    """Initialize AWS client with secure credentials."""
    try:
        session = boto3.Session(
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=region
        )
        return session.client(service)
    except Exception as e:
        logger.error(f"Failed to initialize AWS client for {service}: {str(e)}")
        raise

def invoke_lambda(function_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Invoke AWS Lambda function."""
    try:
        lambda_client = get_aws_client("lambda")
        response = lambda_client.invoke(
            FunctionName=function_name,
            InvocationType="RequestResponse",
            Payload=json.dumps(payload)
        )
        return json.loads(response["Payload"].read().decode("utf-8"))
    except ClientError as e:
        logger.error(f"Lambda invocation failed: {str(e)}")
        raise

def invoke_api_gateway(api_url: str, payload: Dict[str, Any], headers: Dict[str, str] = None) -> Dict[str, Any]:
    """Invoke API Gateway endpoint (placeholder for custom implementation)."""
    try:
        import requests
        response = requests.post(api_url, json=payload, headers=headers or {})
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"API Gateway invocation failed: {str(e)}")
        raise
