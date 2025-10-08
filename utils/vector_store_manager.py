import os
import logging
from typing import List, Optional
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from config.settings import settings

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Manages FAISS vector store operations"""

    def __init__(self, embeddings: Embeddings):
        self.embeddings = embeddings
        self.vector_store: Optional[FAISS] = None
        self.index_path = os.path.join(settings.vector_store_path, "faiss_index")

    def create_vector_store(self, documents: List[Document]) -> FAISS:
        """Create a new vector store from documents"""
        try:
            self.vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            logger.info(f"Created vector store with {len(documents)} documents")
            return self.vector_store
        except Exception as e:
            logger.error(f"Failed to create vector store: {e}")
            raise

    def add_documents(self, documents: List[Document]):
        """Add documents to existing vector store"""
        try:
            if self.vector_store is None:
                self.create_vector_store(documents)
            else:
                self.vector_store.add_documents(documents)
            logger.info(f"Added {len(documents)} documents to vector store")
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise

    def save_vector_store(self):
        """Save vector store to disk"""
        try:
            if self.vector_store is not None:
                self.vector_store.save_local(self.index_path)
                logger.info(f"Saved vector store to {self.index_path}")
        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")
            raise

    def load_vector_store(self) -> bool:
        """Load vector store from disk"""
        try:
            if os.path.exists(self.index_path):
                self.vector_store = FAISS.load_local(
                    self.index_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info(f"Loaded vector store from {self.index_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            return False
        
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Perform similarity search"""
        try:
            if self.vector_store is None:
                logger.warning("Vector store not initialized")
                return []
            
            results = self.vector_store.similarity_search(query, k=k)
            logger.info(f"Found {len(results)} similar documents")
            return results
        except Exception as e:
            logger.error(f"Failed to perform similarity search: {e}")
            return []
        
    def get_retriever(self, k: int = 4):
        """Get retriever for RAG"""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        return self.vector_store.as_retriever(
            search_kwargs={"k": k}
        )
    
    def clear_vector_store(self):
        """Clean the vector store"""
        self.vector_store = None
        if os.path.exists(self.index_path):
            import shutil
            shutil.rmtree(self.index_path)
        logger.info("Cleared vector store")