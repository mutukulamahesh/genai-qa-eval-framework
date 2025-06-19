import pytest
import json
import yaml
from utils.llm_utils import query_chatbot
from utils.nlp_utils import extract_entities, detect_intent
from utils.ml_utils import invoke_sagemaker_endpoint
from utils.aws_utils import invoke_api_gateway
from config.credentials import get_credentials_manager

# Load configuration
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)
LLM_CONFIG = config["llm"]
NLP_CONFIG = config["nlp"]
ML_CONFIG = config["ml"]
AWS_CONFIG = config["aws"]

# Load fixtures
with open("tests/fixtures/llm_fixtures.json", "r") as f:
    LLM_FIXTURES = json.load(f)

@pytest.fixture(scope="module")
def credentials():
    """Provide credentials for tests."""
    return get_credentials_manager()

def test_end_to_end_flow(credentials):
    """Test end-to-end flow: query -> chatbot -> NLP -> ML -> response."""
    test_case = LLM_FIXTURES[0]  # Use first fixture for simplicity
    query = test_case["query"]
    expected_response = test_case["expected_response"]
    expected_intent = "check_eligibility"
    expected_entities = [{"text": "ibuprofen", "label": "DRUG"}]
    
    # Step 1: Query chatbot via API Gateway
    payload = {"query": query, "context": test_case.get("context", "")}
    response = invoke_api_gateway(AWS_CONFIG["api_gateway_url"], payload)
    chatbot_response = response["response"]
    
    assert expected_response.lower() in chatbot_response.lower(), (
        f"Expected '{expected_response}' in chatbot response, got '{chatbot_response}'"
    )
    
    # Step 2: Extract entities from chatbot response
    extracted_entities = extract_entities(chatbot_response, NLP_CONFIG["entity_extraction"]["model"])
    assert any(
        e["text"] == expected_entities[0]["text"] and e["label"] == expected_entities[0]["label"]
        for e in extracted_entities
    ), f"Expected entity {expected_entities}, got {extracted_entities}"
    
    # Step 3: Detect intent
    detected_intent = detect_intent(chatbot_response, NLP_CONFIG["intent_detection"]["model"])
    assert detected_intent == expected_intent, (
        f"Expected intent '{expected_intent}', got '{detected_intent}'"
    )
    
    # Step 4: Invoke ML model (e.g., eligibility)
    ml_payload = {"features": [1.0, 2.0, 3.0]}  # Example input
    ml_response = invoke_sagemaker_endpoint(ML_CONFIG["sagemaker_endpoints"]["eligibility"], ml_payload)
    assert ml_response["prediction"] in [0, 1], f"Invalid ML prediction: {ml_response['prediction']}"