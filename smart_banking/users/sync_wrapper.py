# utils/sync_wrapper.py
import asyncio
from asyncio.log import logger
import logging

from .intent_classifier import BankingAssistant, BankingResponse
from .model_manager import ModelManager
from .models import User
#from .query_processor import QueryProcessor
# sync_wrapper.py
logger = logging.getLogger(__name__)

class SyncQueryProcessor:
    def __init__(self):
        # Get the shared LLM instance
        self.assistant = BankingAssistant(ModelManager.get_instance().llm)
    
    def process_query(self, user, query: str) -> dict:
        """Process query with user context"""
        try:
            # Store user context
            self.assistant.memory.chat_memory.add_user_message(
                f"User Context: id={user.id}, username={user.phoneNumber}"
            )
            
            # Process the query
            banking_response = self.assistant.process_query(query)
            
            return {
                "query": banking_response.query,
                "type": banking_response.type.value,
                "response": banking_response.response,
                "confidence": float(banking_response.confidence),
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Query processing error: {str(e)}")
            return {
                "query": query,
                "response": "Unable to process your request. Please try again.",
                "status": "error",
                "type": "steps",
                "confidence": 0.0
            }