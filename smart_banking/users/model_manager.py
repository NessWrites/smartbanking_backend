# model_manager.py
from langchain_community.llms import LlamaCpp
import logging

logger = logging.getLogger(__name__)

class ModelManager:
    _instance = None
    
    def __init__(self, model_path: str):
        if ModelManager._instance is not None:
            raise Exception("ModelManager is a singleton!")
        
        try:
            self.llm = LlamaCpp(
                model_path=model_path,
                temperature=0.3,  # More deterministic for classification
                max_tokens=610,  # Limit response length
                n_ctx=2048,
                n_gpu_layers=1,
                n_threads=4,
                verbose=False,
                
            )
            ModelManager._instance = self
            logger.info("LLM model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            raise
    
    @classmethod
    def get_instance(cls, model_path: str = None):
        if cls._instance is None:
            if model_path is None:
                raise ValueError("Model path required for first initialization")
            cls(model_path)
        return cls._instance
    
    @classmethod
    def get_llm(cls):
        if cls._instance is None:
            raise RuntimeError("ModelManager not initialized")
        return cls._instance.llm