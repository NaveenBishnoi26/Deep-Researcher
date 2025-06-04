"""
Vector store module for Deep Researcher.
Handles document storage and retrieval using a simple in-memory implementation.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    """Handles document storage and retrieval using a simple in-memory implementation."""
    
    def __init__(self):
        """Initialize the vector store."""
        self.documents = []
        self.metadata = {}
        logger.info("Initialized in-memory vector store")
            
    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents to add
        """
        try:
            if not documents:
                logger.warning("No documents to add to vector store")
                return
                
            # Process metadata for each document
            for doc in documents:
                try:
                    processed_metadata = self._process_metadata(doc.metadata)
                    self.documents.append({
                        'content': doc.page_content,
                        'metadata': processed_metadata
                    })
                except Exception as e:
                    logger.error(f"Error processing document metadata: {str(e)}")
                    continue
            
            logger.info(f"Added {len(documents)} documents to vector store")
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise
    
    def similarity_search(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """
        Search for similar documents using simple keyword matching.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of similar documents
        """
        try:
            # Simple keyword-based search
            results = []
            query_terms = query.lower().split()
            
            for doc in self.documents:
                content = doc['content'].lower()
                score = sum(1 for term in query_terms if term in content)
                if score > 0:
                    results.append({
                        'content': doc['content'],
                        'metadata': doc['metadata'],
                        'score': score
                    })
            
            # Sort by score and return top k results
            results.sort(key=lambda x: x['score'], reverse=True)
            return results[:k]
            
        except Exception as e:
            logger.error(f"Error performing similarity search: {str(e)}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary containing vector store statistics
        """
        try:
            return {
                "total_documents": len(self.documents),
                "status": "active"
            }
            
        except Exception as e:
            logger.error(f"Error getting vector store stats: {str(e)}")
            return {
                "total_documents": 0,
                "status": "error",
                "error": str(e)
            }

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Alias for get_stats to maintain compatibility.
        
        Returns:
            Dictionary containing vector store statistics
        """
        return self.get_stats()
    
    def clear(self) -> None:
        """Clear all documents from the vector store."""
        try:
            self.documents = []
            self.metadata = {}
            logger.info("Vector store cleared")
            
        except Exception as e:
            logger.error(f"Error clearing vector store: {str(e)}")
            raise
            
    def _process_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process document metadata.
        
        Args:
            metadata: Document metadata
            
        Returns:
            Processed metadata
        """
        try:
            processed = metadata.copy()
            processed['timestamp'] = datetime.now().isoformat()
            return processed
        except Exception as e:
            logger.error(f"Error processing metadata: {str(e)}")
            return metadata