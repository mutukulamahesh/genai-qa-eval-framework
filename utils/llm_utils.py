# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# © 2025 Mahesh Mutukula. All rights reserved.
# This file is part of the GenAI QA Eval Framework.

import json
import logging
from typing import Optional, Dict, Any
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancy, Hallucination
import boto3
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_llm_chain(api_key: str, model_name: str = "gpt-3.5-turbo") -> LLMChain:
    """Initialize LangChain with OpenAI LLM."""
    try:
        llm = OpenAI(api_key=api_key, model_name=model_name)
        prompt = PromptTemplate(
            input_variables=["query", "context"],
            template="Given the context: {context}\nAnswer the query: {query}"
        )
        return LLMChain(llm=llm, prompt=prompt)
    except Exception as e:
        logger.error(f"Failed to initialize LLM chain: {str(e)}")
        raise

def query_chatbot(
    query: str,
    context: Optional[str] = None,
    lambda_function: Optional[str] = None,
    api_key: Optional[str] = None
) -> str:
    """Query the chatbot via Lambda or local LangChain."""
    try:
        if lambda_function:
            # Invoke AWS Lambda function
            lambda_client = boto3.client("lambda")
            payload = {"query": query, "context": context or ""}
            response = lambda_client.invoke(
                FunctionName=lambda_function,
                InvocationType="RequestResponse",
                Payload=json.dumps(payload)
            )
            response_payload = json.loads(response["Payload"].read().decode("utf-8"))
            return response_payload.get("response", "")
        else:
            # Local LangChain query
            chain = initialize_llm_chain(api_key)
            return chain.run(query=query, context=context or "")
    except ClientError as e:
        logger.error(f"Lambda invocation failed: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Chatbot query failed: {str(e)}")
        raise

def evaluate_llm_response(
    query: str,
    response: str,
    context: Optional[str] = None,
    min_relevancy: float = 0.8,
    max_hallucination: float = 0.2
) -> Dict[str, float]:
    """Evaluate LLM response using DeepEval metrics."""
    try:
        relevancy_metric = AnswerRelevancy()
        hallucination_metric = Hallucination()
        
        relevancy_score = evaluate(relevancy_metric, query=query, response=response, context=context)
        hallucination_score = evaluate(hallucination_metric, response=response, context=context)
        
        results = {
            "relevancy_score": relevancy_score,
            "hallucination_score": hallucination_score,
            "relevancy_pass": relevancy_score >= min_relevancy,
            "hallucination_pass": hallucination_score <= max_hallucination
        }
        logger.info(f"LLM evaluation results: {results}")
        return results
    except Exception as e:
        logger.error(f"LLM evaluation failed: {str(e)}")
        raise