from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from config.settings import settings
from crewai import LLM
import logging

logger = logging.getLogger(__name__)


class LLMManager:
    """Manages LLM and embeddings initialization"""
    
    _llm_instance = None
    _embeddings_instance = None
    
    @classmethod
    def get_llm(cls):
        """Get or create LLM instance"""
        if cls._llm_instance is None:
            try:
                cls._llm_instance = Ollama(
                    base_url=settings.ollama_base_url,
                    model=settings.ollama_model,
                    temperature=0.7,
                    num_ctx=4096,
                )
                logger.info(f"Initialized LLM: {settings.ollama_model}")
            except Exception as e:
                logger.error(f"Failed to initialize LLM: {e}")
                raise
        return cls._llm_instance
    
    @classmethod
    def get_embeddings(cls):
        """Get or create embeddings instance"""
        if cls._embeddings_instance is None:
            try:
                cls._embeddings_instance = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                logger.info("Initialized embeddings model")
            except Exception as e:
                logger.error(f"Failed to initialize embeddings: {e}")
                raise
        return cls._embeddings_instance
    
    @classmethod
    def test_connection(cls):
        """Test LLM connection"""
        try:
            llm = cls.get_llm()
            response = llm.invoke("Say 'Hello'")
            logger.info("LLM connection successful")
            return True
        except Exception as e:
            logger.error(f"LLM connection failed: {e}")
            return False
