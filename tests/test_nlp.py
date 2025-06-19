import pytest
import json
import yaml
from utils.nlp_utils import extract_entities, validate_entities, detect_intent

# Load configuration
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)
NLP_CONFIG = config["nlp"]

# Load fixtures
with open("tests/fixtures/nlp_fixtures.json", "r") as f:
    NLP_FIXTURES = json.load(f)

@pytest.mark.parametrize("test_case", NLP_FIXTURES)
def test_entity_extraction(test_case):
    """Test entity extraction from text."""
    text = test_case["text"]
    expected_entities = test_case["expected_entities"]
    
    # Extract entities
    extracted_entities = extract_entities(text, model_name=NLP_CONFIG["entity_extraction"]["model"])
    
    # Validate entities
    results = validate_entities(extracted_entities, expected_entities)
    
    assert results["precision"] >= NLP_CONFIG["entity_extraction"]["min_precision"], (
        f"Precision too low: {results['precision']}"
    )
    assert results["recall"] >= NLP_CONFIG["entity_extraction"]["min_recall"], (
        f"Recall too low: {results['recall']}"
    )
    assert results["f1"] >= NLP_CONFIG["entity_extraction"]["min_f1"], (
        f"F1-score too low: {results['f1']}"
    )

@pytest.mark.parametrize("test_case", NLP_FIXTURES)
def test_intent_detection(test_case):
    """Test intent detection from text."""
    text = test_case["text"]
    expected_intent = test_case["expected_intent"]
    
    # Detect intent
    detected_intent = detect_intent(text, model_name=NLP_CONFIG["intent_detection"]["model"])
    
    assert detected_intent == expected_intent, (
        f"Expected intent '{expected_intent}', got '{detected_intent}'"
    )