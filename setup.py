from setuptools import setup, find_packages

setup(
    name="pwp_rebate_test_framework",
    version="0.1.0",
    description="Test framework for validating AI/ML models and LLM chatbot in PWP Rebate platform",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(exclude=["tests", "reports"]),
    install_requires=[
        "pytest==7.4.0",
        "pytest-html==4.1.1",
        "allure-pytest==2.13.2",
        "langchain==0.2.0",
        "openai==1.30.0",
        "deepeval==0.21.0",
        "boto3==1.34.0",
        "spacy==3.7.0",
        "scikit-learn==1.3.0",
        "pandas==2.0.0",
        "numpy==1.24.0",
        "matplotlib==3.7.0",
        "seaborn==0.13.0",
        "requests==2.31.0",
        "pyyaml==6.0.0",
        "python-dotenv==1.0.0",
    ],
    extras_require={
        "dev": [
            "black==23.3.0",
            "flake8==6.0.0",
            "pytest-cov==4.0.0"
        ]
    },
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)