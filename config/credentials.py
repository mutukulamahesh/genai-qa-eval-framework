import os
import logging
from typing import Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CredentialsManager:
    """Securely manage AWS and OpenAI credentials from environment variables."""
    
    def __init__(self):
        """Initialize with default environment variable names."""
        self._aws_access_key_env = "AWS_ACCESS_KEY_ID"
        self._aws_secret_key_env = "AWS_SECRET_ACCESS_KEY"
        self._openai_api_key_env = "OPENAI_API_KEY"
    
    def get_aws_credentials(self) -> Dict[str, str]:
        """Retrieve AWS credentials from environment variables."""
        try:
            access_key = os.getenv(self._aws_access_key_env)
            secret_key = os.getenv(self._aws_secret_key_env)
            
            if not access_key or not secret_key:
                raise ValueError("AWS credentials not found in environment variables")
            
            return {
                "access_key_id": access_key,
                "secret_access_key": secret_key
            }
        except Exception as e:
            logger.error(f"Failed to retrieve AWS credentials: {str(e)}")
            raise
    
    def get_openai_api_key(self) -> str:
        """Retrieve OpenAI API key from environment variable."""
        try:
            api_key = os.getenv(self._openai_api_key_env)
            
            if not api_key:
                raise ValueError("OpenAI API key not found in environment variable")
            
            return api_key
        except Exception as e:
            logger.error(f"Failed to retrieve OpenAI API key: {str(e)}")
            raise
    
    def update_env_vars(
        self,
        aws_access_key_env: Optional[str] = None,
        aws_secret_key_env: Optional[str] = None,
        openai_api_key_env: Optional[str] = None
    ):
        """Update environment variable names for credentials."""
        if aws_access_key_env:
            self._aws_access_key_env = aws_access_key_env
        if aws_secret_key_env:
            self._aws_secret_key_env = aws_secret_key_env
        if openai_api_key_env:
            self._openai_api_key_env = openai_api_key_env
        logger.info("Updated environment variable names for credentials")

def get_credentials_manager() -> CredentialsManager:
    """Factory function to get CredentialsManager instance."""
    return CredentialsManager()