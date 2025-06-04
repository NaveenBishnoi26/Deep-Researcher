"""
Knowledge base implementation for storing and retrieving research data.
"""

import os
import json
import logging
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

class KnowledgeBase:
    def __init__(self):
        """Initialize the knowledge base."""
        self.documents = []
        self.vector_store = {}
        self.metadata = {}
        
    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        Add documents to the knowledge base.
        
        Args:
            documents: List of documents to add
        """
        try:
            for doc in documents:
                if doc not in self.documents:
                    self.documents.append(doc)
                    # Create a simple vector representation (just using document ID as key)
                    doc_id = len(self.documents) - 1
                    self.vector_store[doc_id] = {
                        'content': doc.get('content', ''),
                        'metadata': doc.get('metadata', {})
                    }
            logger.info(f"Added {len(documents)} documents to knowledge base")
        except Exception as e:
            logger.error(f"Error adding documents to knowledge base: {str(e)}")
            raise
            
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search the knowledge base for relevant documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of relevant documents
        """
        try:
            if not query or not isinstance(query, str):
                logger.warning(f"Invalid search query: {query}")
                return []
                
            if not self.documents or len(self.documents) == 0:
                logger.warning("Knowledge base is empty. No documents to search.")
                return []
                
            # Convert query to lowercase for case-insensitive matching
            query_lower = query.lower()
            query_terms = query_lower.split()
            
            # Filter out stop words and short terms
            stop_words = {'a', 'an', 'the', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about'}
            query_terms = [term for term in query_terms if term not in stop_words and len(term) > 2]
            
            # Prepare results container
            results = []
            
            # Iterate through vector store
            for doc_id, doc_data in self.vector_store.items():
                try:
                    # Get content from document
                    content = doc_data.get('content', '')
                    if not content or not isinstance(content, str):
                        continue
                        
                    content_lower = content.lower()
                    metadata = doc_data.get('metadata', {})
                    
                    # Initialize score
                    score = 0.0
                    
                    # Check exact phrase match (highest score)
                    if query_lower in content_lower:
                        score += 3.0
                    
                    # Check individual term matches
                    for term in query_terms:
                        if term in content_lower:
                            # Add score based on term frequency
                            term_count = content_lower.count(term)
                            score += min(term_count / 5, 1.0)  # Cap at 1.0 per term
                    
                    # Check metadata matches
                    if metadata and isinstance(metadata, dict):
                        # Title match is valuable
                        title = str(metadata.get('title', '')).lower()
                        if title:
                            if query_lower in title:
                                score += 2.0
                            for term in query_terms:
                                if term in title:
                                    score += 0.5
                    
                    # Only include documents with some relevance
                    if score > 0:
                        # Create a document-like object
                        document = {
                            'id': doc_id,
                            'page_content': content,  # Use LangChain format for consistency
                            'content': content,       # Also include original format
                            'metadata': metadata,
                            'score': score
                        }
                        results.append(document)
                
                except Exception as doc_error:
                    logger.error(f"Error processing document {doc_id}: {str(doc_error)}")
                    continue
            
            # Sort by score and return top_k results
            results.sort(key=lambda x: x.get('score', 0), reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Error searching knowledge base: {str(e)}")
            return []
            
    def clear(self):
        """Clear the knowledge base."""
        try:
            self.documents = []
            self.vector_store = {}
            self.metadata = {}
            logger.info("Knowledge base cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing knowledge base: {str(e)}")
            raise
            
    def save(self, path: str):
        """
        Save the knowledge base to disk.
        
        Args:
            path: Path to save the knowledge base
        """
        try:
            data = {
                'documents': self.documents,
                'vector_store': self.vector_store,
                'metadata': self.metadata
            }
            with open(path, 'w') as f:
                json.dump(data, f)
            logger.info(f"Knowledge base saved to {path}")
        except Exception as e:
            logger.error(f"Error saving knowledge base: {str(e)}")
            raise
            
    def load(self, path: str):
        """
        Load the knowledge base from disk.
        
        Args:
            path: Path to load the knowledge base from
        """
        try:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    data = json.load(f)
                self.documents = data.get('documents', [])
                self.vector_store = data.get('vector_store', {})
                self.metadata = data.get('metadata', {})
                logger.info(f"Knowledge base loaded from {path}")
            else:
                logger.warning(f"Knowledge base file not found at {path}")
        except Exception as e:
            logger.error(f"Error loading knowledge base: {str(e)}")
            raise

    def get_all_documents(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get all documents in the knowledge base, up to the specified limit.
        
        Args:
            limit: Maximum number of documents to return
            
        Returns:
            List of documents
        """
        try:
            results = []
            for doc_id, doc_data in self.vector_store.items():
                results.append({
                    'id': doc_id,
                    'content': doc_data['content'],
                    'metadata': doc_data['metadata'],
                    'score': 1.0  # All documents get the same score
                })
                
                if len(results) >= limit:
                    break
                    
            logger.info(f"Retrieved {len(results)} documents from knowledge base")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving all documents: {str(e)}")
            return []

    def get_relevant_documents(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get relevant documents for a given query.
        
        Args:
            query: The search query
            limit: Maximum number of documents to return
            
        Returns:
            List of relevant documents
        """
        try:
            # Use the search method instead of vectorstore
            results = self.search(query, top_k=limit)
            
            # Convert results to the expected format
            documents = []
            for doc in results:
                doc_dict = {
                    'content': doc.get('content', doc.get('page_content', '')),
                    'metadata': doc.get('metadata', {})
                }
                documents.append(doc_dict)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error getting relevant documents: {str(e)}")
            return []

# Example usage
if __name__ == "__main__":
    kb = KnowledgeBase()
    kb.add_documents([{'content': 'This is a test document.', 'metadata': {'source': 'test'}}])
    results = kb.search('test')
    print("Search results:", results)
    kb.clear()
    kb.save('knowledge_base.json')
    kb.load('knowledge_base.json') 