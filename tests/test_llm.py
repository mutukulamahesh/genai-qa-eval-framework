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
from utils.llm_utils import query_chatbot, evaluate_llm_response
from config.credentials import get_credentials_manager

# Load configuration
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)
LLM_CONFIG = config["llm"]

# Load fixtures
with open("tests/fixtures/llm_fixtures.json", "r") as f:
    LLM_FIXTURES = json.load(f)

@pytest.fixture(scope="module")
def credentials():
    """Provide credentials for tests."""
    return get_credentials_manager()

@pytest.mark.parametrize("test_case", LLM_FIXTURES)
def test_chatbot_response(test_case, credentials):
    """Test chatbot response correctness and DeepEval metrics."""
    query = test_case["query"]
    expected_response = test_case["expected_response"]
    context = test_case.get("context", None)
    
    # Query chatbot
    response = query_chatbot(
        query=query,
        context=context,
        lambda_function=LLM_CONFIG["lambda_function"],
        api_key=credentials.get_openai_api_key()
    )
    
    # Functional validation
    assert response is not None, "Chatbot returned no response"
    assert expected_response.lower() in response.lower(), (
        f"Expected '{expected_response}' in response, got '{response}'"
    )
    
    # DeepEval metrics
    results = evaluate_llm_response(
        query=query,
        response=response,
        context=context,
        min_relevancy=LLM_CONFIG["evaluation"]["relevancy_threshold"],
        max_hallucination=LLM_CONFIG["evaluation"]["hallucination_threshold"]
    )
    
    assert results["relevancy_pass"], f"Relevancy score too low: {results['relevancy_score']}"
    assert results["hallucination_pass"], f"Hallucination score too high: {results['hallucination_score']}"

def test_chatbot_edge_cases(credentials):
    """Test chatbot with malformed or out-of-scope inputs."""
    edge_cases = [
        {"query": "", "expected": "Please provide a valid query"},
        {"query": "What's the weather?", "expected": "I'm sorry, I can only assist with rebate and medication queries"},
        {"query": "12345!@#", "expected": "Invalid input"}
    ]
    
    for case in edge_cases:
        response = query_chatbot(
            query=case["query"],
            lambda_function=LLM_CONFIG["lambda_function"],
            api_key=credentials.get_openai_api_key()
        )
        assert response is not None, f"Chatbot failed to handle edge case: {case['query']}"
        assert case["expected"].lower() in response.lower(), (
            f"Expected '{case['expected']}' in response, got '{response}'"
        )