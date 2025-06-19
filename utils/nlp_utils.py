import spacy
from typing import List, Dict, Tuple
from sklearn.metrics import precision_recall_f1_score
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_nlp_model(model_name: str = "en_core_web_sm") -> spacy.language.Language:
    """Load spaCy model for entity extraction."""
    try:
        return spacy.load(model_name)
    except Exception as e:
        logger.error(f"Failed to load spaCy model {model_name}: {str(e)}")
        raise

def extract_entities(text: str, model_name: str = "en_core_web_sm") -> List[Dict[str, str]]:
    """Extract medical entities from text using spaCy."""
    try:
        nlp = load_nlp_model(model_name)
        doc = nlp(text)
        entities = [
            {"text": ent.text, "label": ent.label_}
            for ent in doc.ents
            if ent.label_ in ["DRUG", "SYMPTOM", "DIAGNOSIS"]
        ]
        logger.info(f"Extracted entities: {entities}")
        return entities
    except Exception as e:
        logger.error(f"Entity extraction failed: {str(e)}")
        raise

def validate_entities(
    extracted: List[Dict[str, str]],
    expected: List[Dict[str, str]]
) -> Dict[str, float]:
    """Validate extracted entities against ground truth."""
    try:
        y_true = [ent["label"] for ent in expected]
        y_pred = [ent["label"] for ent in extracted]
        
        # Ensure same length for metrics
        if len(y_true) != len(y_pred):
            logger.warning("Mismatch in entity counts; padding with None")
            max_len = max(len(y_true), len(y_pred))
            y_true.extend(["None"] * (max_len - len(y_true)))
            y_pred.extend(["None"] * (max_len - len(y_pred)))
        
        precision, recall, f1, _ = precision_recall_f1_score(y_true, y_pred, average="weighted")
        results = {"precision": precision, "recall": recall, "f1": f1}
        logger.info(f"Entity validation results: {results}")
        return results
    except Exception as e:
        logger.error(f"Entity validation failed: {str(e)}")
        raise

def detect_intent(text: str, model_name: str = "simple_classifier") -> str:
    """Simple intent detection (placeholder for Hugging Face or custom model)."""
    try:
        # Placeholder: Replace with Hugging Face or custom classifier
        intents = ["check_eligibility", "resolve_dispute", "medication_query"]
        for intent in intents:
            if intent.replace("_", " ") in text.lower():
                return intent
        return "unknown"
    except Exception as e:
        logger.error(f"Intent detection failed: {str(e)}")
        raise