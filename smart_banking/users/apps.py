from asyncio.log import logger
from django.apps import AppConfig


class UsersConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "users"
    
    
def ready(self):
        """Initialize model when Django starts"""
        from django.conf import settings
        from .model_manager import ModelManager
        
        try:
            ModelManager.get_instance(settings.LLM_MODEL_PATH)
            logger.info("LLM model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load LLM model: {str(e)}")