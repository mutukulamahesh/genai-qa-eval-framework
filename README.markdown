#  AI/ML Test Framework

A Python-based test framework for validating and evaluating the AI/ML models and LLM chatbot in the PWP Rebate platform, a cloud-native AWS-hosted system for personalized patient engagement, pharma tenant onboarding, and rebate tracking.

## Features
- **LLM Testing**: Validates the LangChain + OpenAI chatbot for response correctness, relevancy, and hallucination using DeepEval.
- **NLP Testing**: Tests entity extraction (e.g., drugs, symptoms) and intent detection using spaCy.
- **ML Testing**: Evaluates SageMaker-hosted models (medication adherence, rebate eligibility, user risk scores) for accuracy, precision, recall, and regression metrics.
- **End-to-End Testing**: Validates the full pipeline: user query → chatbot → NLP → ML → response.
- **Reporting**: Generates HTML reports (`pytest-html`), Allure dashboards, and custom visualizations (confusion matrices, ROC curves).

## Directory Structure
```
pwp_rebate_test_framework/
├── config/                    # Configuration files
│   ├── config.yaml           # Test parameters (endpoints, thresholds)
│   └── credentials.py        # Secure credential management
├── tests/                     # Test scripts and fixtures
│   ├── test_llm.py           # LLM chatbot tests
│   ├── test_nlp.py           # NLP pipeline tests
│   ├── test_ml_models.py     # ML model tests
│   ├── test_end_to_end.py    # End-to-end integration tests
│   └── fixtures/             # Test data
├── utils/                    # Utility functions
│   ├── llm_utils.py          # LLM interaction helpers
│   ├── nlp_utils.py          # NLP validation helpers
│   ├── ml_utils.py           # ML model testing helpers
│   ├── aws_utils.py          # AWS service interactions
│   └── report_utils.py       # Reporting and visualization helpers
├── reports/                  # Generated reports and visualizations
├── requirements.txt           # Dependencies
├── setup.py                  # Package setup
└── README.md                 # This file
```

## Prerequisites
- Python 3.9+
- AWS account with access to Lambda, SageMaker, and API Gateway
- OpenAI API key
- Environment variables: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `OPENAI_API_KEY`

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd pwp_rebate_test_framework
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```
3. Install the package:
   ```bash
   pip install .
   ```
4. Set environment variables:
   ```bash
   export AWS_ACCESS_KEY_ID="your_aws_access_key"
   export AWS_SECRET_ACCESS_KEY="your_aws_secret_key"
   export OPENAI_API_KEY="your_openai_api_key"
   ```
   Or use a `.env` file:
   ```
   AWS_ACCESS_KEY_ID=your_aws_access_key
   AWS_SECRET_ACCESS_KEY=your_aws_secret_key
   OPENAI_API_KEY=your_openai_api_key
   ```

## Usage
1. Configure `config/config.yaml` with your AWS endpoints and thresholds.
2. Run tests:
   ```bash
   pytest tests/ -v --html=reports/pytest_report.html --alluredir=reports/allure_results
   ```
3. Generate visualizations:
   ```bash
   python reports/visualizations/confusion_matrix.py
   python reports/visualizations/roc_curve.py
   ```
4. Generate JSON summary:
   ```bash
   python reports/test_summary.py
   ```
5. View Allure report:
   ```bash
   allure serve reports/allure_results
   ```

## Testing Scope
- **LLM Tests**: Validate chatbot responses for rebate eligibility, medication queries, and claim disputes.
- **NLP Tests**: Verify entity extraction (e.g., DRUG, SYMPTOM) and intent detection (e.g., check_eligibility).
- **ML Tests**: Evaluate SageMaker models for classification (adherence, eligibility) and regression (risk scores).
- **End-to-End Tests**: Test the full pipeline from query to response.
- **Metrics**: Precision, recall, F1-score, AUC-ROC, MSE, R², DeepEval metrics (relevancy, hallucination).

## HIPAA Compliance
- Use synthetic data in `tests/fixtures/` to avoid real patient data.
- Ensure logs and reports do not expose sensitive information.

## Contributing
- Add new test cases to `tests/fixtures/`.
- Enhance utilities in `utils/` for additional validations.
- Submit pull requests with clear descriptions.

## License
MIT License

## Contact
For issues or questions, contact Mahesh Mutukula at your.mmk.mutu@gmail.com.