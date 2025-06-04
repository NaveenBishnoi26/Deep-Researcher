"""
Document filtering module for removing irrelevant documents from the knowledge base.
"""

import logging
import math
import re
import string
import json
from typing import List, Dict, Any, Set, Tuple, Optional
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from backend.config import OPENAI_API_KEY, BASE_MODEL, MODEL_TEMPERATURE

logger = logging.getLogger(__name__)

class DocumentFilter:
    """Filters out irrelevant documents from the knowledge base based on expanded research query."""
    
    def __init__(self):
        """Initialize the document filter."""
        self.llm = ChatOpenAI(
            model=BASE_MODEL,
            temperature=MODEL_TEMPERATURE,
            openai_api_key=OPENAI_API_KEY
        )
        
        # Initialize a separate model instance specifically for relevance scoring
        self.scoring_llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.0,
            openai_api_key=OPENAI_API_KEY
        )
    
    def _evaluate_document_relevance(self, document: Dict[str, Any], query: str) -> float:
        """
        Evaluate document relevance using LLM.
        
        Args:
            document: Document to evaluate (as dictionary)
            query: The search query
            
        Returns:
            Float relevance score between 0 and 1
        """
        try:
            # Extract document information
            title = document.get('title', 'No title')
            content = document.get('content', '')
            
            # Limit content length
            max_content_length = 1000
            if len(content) > max_content_length:
                content = content[:max_content_length] + "..."
            
            # Construct prompt
            prompt = f"""Evaluate the relevance of this document to the research query.

Query: {query}

Document Title: {title}
Document Content: {content}

Rate the relevance on a scale of 0.0 to 1.0:
0.0: Completely irrelevant
0.2: Slightly relevant
0.4: Somewhat relevant
0.6: Moderately relevant
0.8: Highly relevant
1.0: Perfectly relevant

Return only a number between 0.0 and 1.0."""
            
            # Get relevance score from LLM
            response = self.llm.invoke(prompt)
            try:
                score = float(response.content.strip())
                return min(max(score, 0.0), 1.0)  # Ensure score is between 0 and 1
            except ValueError:
                logger.error(f"Invalid score format: {response.content}")
                return 0.5
                
        except Exception as e:
            logger.error(f"Error evaluating document relevance: {str(e)}")
            return 0.5
    
    def calculate_relevance_score(self, document: Dict[str, Any], query: str) -> float:
        """
        Calculate relevance score for a document based on its content and the query.
        
        Args:
            document: The document to score
            query: The original research query
            
        Returns:
            Float relevance score between 0 and 1
        """
        try:
            # Get document content and title
            title = document.get('title', '').lower()
            content = document.get('content', '').lower()
            snippet = document.get('snippet', '').lower()
            
            # Normalize query
            query = query.lower()
            
            # Extract query terms (removing common stop words)
            stop_words = {'a', 'an', 'the', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'as'}
            query_terms = [term.strip(string.punctuation) for term in query.split() 
                          if term.strip(string.punctuation) not in stop_words]
            
            # Basic TF-IDF inspired scoring
            score = 0.0
            
            # Check title (highest weight)
            title_score = sum(5.0 for term in query_terms if term in title)
            
            # Check snippet (medium weight)
            snippet_score = sum(3.0 for term in query_terms if term in snippet)
            
            # Check content (lowest weight per occurrence but can add up)
            content_score = 0
            for term in query_terms:
                # Count occurrences
                term_count = content.count(term)
                if term_count > 0:
                    # Log scale to prevent very long documents from dominating
                    content_score += (2.0 + math.log(term_count))
            
            # Add a baseline score to ensure even marginally relevant documents are included
            baseline_score = 0.1
            
            # Combine scores with weights
            total_score = title_score + snippet_score + (content_score * 0.7) + baseline_score
            
            # Normalize score based on number of query terms (to get score between 0-1)
            max_possible_score = len(query_terms) * (5.0 + 3.0 + 0.7) + baseline_score  # maximum possible score
            normalized_score = min(total_score / max_possible_score if max_possible_score > 0 else 0, 1.0)
            
            # Boost the score slightly to ensure more documents pass the threshold
            boosted_score = min(normalized_score * 1.5, 1.0)
            
            return boosted_score
            
        except Exception as e:
            logger.error(f"Error calculating relevance score: {str(e)}")
            return 0.2  # Return a low but passing score in case of error
    
    def filter_documents_by_score(self, documents: List[Dict[str, Any]], 
                                  query: str, 
                                  threshold: float = 0.05) -> List[Dict[str, Any]]:
        """
        Filter documents based on their relevance scores.
        
        Args:
            documents: List of documents to filter
            query: The research query
            threshold: Minimum relevance score to keep a document
            
        Returns:
            List of documents that meet the threshold
        """
        try:
            if not documents:
                logger.warning("No documents to filter")
                return []
                
            logger.info(f"Filtering {len(documents)} documents with threshold {threshold}")
            
            # Calculate scores for documents that don't have them yet
            for doc in documents:
                if 'relevance_score' not in doc:
                    doc['relevance_score'] = self.calculate_relevance_score(doc, query)
            
            # Set a very low minimum threshold to ensure we have enough documents
            effective_threshold = min(threshold, 0.05)
            
            # Filter documents by score
            filtered_docs = [doc for doc in documents if doc.get('relevance_score', 0) >= effective_threshold]
            
            # If we have too few documents, just keep all of them
            if len(filtered_docs) < 10 and len(documents) > 0:
                logger.warning(f"Too few documents meet threshold ({len(filtered_docs)}), keeping all {len(documents)} documents")
                # Assign minimum passing scores to all documents
                for doc in documents:
                    if doc.get('relevance_score', 0) < effective_threshold:
                        doc['relevance_score'] = effective_threshold
                filtered_docs = documents
            
            # Sort by relevance score (highest first)
            filtered_docs.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
            
            # Log filtering stats
            removed_count = len(documents) - len(filtered_docs)
            removed_percentage = round(removed_count / len(documents) * 100, 1) if documents else 0
            logger.info(f"Filtered out {removed_count} documents ({removed_percentage}%)")
            
            return filtered_docs
            
        except Exception as e:
            logger.error(f"Error filtering documents by score: {str(e)}")
            return documents  # Return original documents in case of error
    
    def filter_documents(self, documents: List[Document], expanded_query: Dict[str, Any], knowledge_base) -> List[Document]:
        """
        Filter out irrelevant documents from the knowledge base using LLM evaluation.
        
        Args:
            documents: List of documents to evaluate
            expanded_query: Dictionary containing expanded query details
            knowledge_base: KnowledgeBase instance to update
            
        Returns:
            List of relevant documents
        """
        try:
            if not documents:
                logger.warning("No documents to filter")
                return []
                
            if not knowledge_base or not hasattr(knowledge_base, 'vectorstore'):
                logger.error("Invalid knowledge base instance")
                return documents
                
            logger.info(f"Starting document filtering for {len(documents)} documents")
            
            # CRITICAL: If we have fewer than 50 documents, keep all of them
            if len(documents) < 50:
                logger.warning(f"Only {len(documents)} documents found - keeping all without filtering")
                return documents
            
            # Track relevant documents
            relevant_documents = []
            documents_to_remove = []
            
            # Evaluate each document
            for doc in documents:
                is_relevant, confidence, reasoning = self._evaluate_document_relevance(doc, expanded_query)
                
                # Store evaluation results in document metadata
                doc.metadata['relevance_evaluation'] = {
                    'is_relevant': is_relevant,
                    'confidence_score': confidence,
                    'reasoning': reasoning
                }
                
                if is_relevant:
                    relevant_documents.append(doc)
                else:
                    documents_to_remove.append(doc)
            
            # IMPORTANT: If less than 20% of documents are kept, keep the top 25% by confidence score
            if len(relevant_documents) < len(documents) * 0.2:
                logger.warning(f"Only {len(relevant_documents)} documents evaluated as relevant - below 20% threshold")
                
                # Sort all documents by confidence score
                all_docs_sorted = sorted(documents, 
                                        key=lambda x: x.metadata.get('relevance_evaluation', {}).get('confidence_score', 0), 
                                        reverse=True)
                
                # Keep top 25%
                min_docs_to_keep = max(int(len(documents) * 0.25), 10)
                relevant_documents = all_docs_sorted[:min_docs_to_keep]
                documents_to_remove = all_docs_sorted[min_docs_to_keep:]
                
                logger.info(f"Increased to {len(relevant_documents)} documents by keeping top 25% by confidence")
            
            # Remove irrelevant documents from knowledge base
            if documents_to_remove:
                # Get unique document IDs to remove
                seen_ids = set()
                unique_ids_to_remove = []
                
                for doc in documents_to_remove:
                    doc_id = doc.metadata.get('id')
                    if doc_id and doc_id not in seen_ids:
                        # Verify the document exists in the vector store before adding to removal list
                        try:
                            if hasattr(knowledge_base.vectorstore, '_collection'):
                                # Check if ID exists in collection
                                if doc_id in knowledge_base.vectorstore._collection.get():
                                    seen_ids.add(doc_id)
                                    unique_ids_to_remove.append(doc_id)
                            else:
                                seen_ids.add(doc_id)
                                unique_ids_to_remove.append(doc_id)
                        except Exception as e:
                            logger.warning(f"Error checking document existence: {str(e)}")
                            continue
                
                # Remove documents from vector store
                if unique_ids_to_remove:
                    try:
                        # Check if vector store has delete method
                        if hasattr(knowledge_base.vectorstore, 'delete'):
                            # Delete in batches to handle potential failures
                            batch_size = 10
                            for i in range(0, len(unique_ids_to_remove), batch_size):
                                batch = unique_ids_to_remove[i:i + batch_size]
                                try:
                                    knowledge_base.vectorstore.delete(ids=batch)
                                except Exception as e:
                                    logger.warning(f"Error deleting batch {i//batch_size + 1}: {str(e)}")
                        elif hasattr(knowledge_base.vectorstore, '_collection'):
                            # Try to delete using collection in batches
                            batch_size = 10
                            for i in range(0, len(unique_ids_to_remove), batch_size):
                                batch = unique_ids_to_remove[i:i + batch_size]
                                try:
                                    knowledge_base.vectorstore._collection.delete(ids=batch)
                                except Exception as e:
                                    logger.warning(f"Error deleting batch {i//batch_size + 1} from collection: {str(e)}")
                        else:
                            logger.warning("Vector store does not support document deletion")
                            
                    except Exception as e:
                        logger.error(f"Error removing documents from vector store: {str(e)}")
                    
                logger.info(f"Removed {len(documents_to_remove)} irrelevant documents from knowledge base")
            else:
                logger.info("No irrelevant documents found")
            
            # Get filtering stats
            stats = self.get_filtering_stats(len(documents), len(relevant_documents))
            logger.info("Document filtering stats:")
            logger.info(f"- Original document count: {stats['original_document_count']}")
            logger.info(f"- Final document count: {stats['final_document_count']}")
            logger.info(f"- Documents removed: {stats['documents_removed']}")
            logger.info(f"- Removal percentage: {stats['removal_percentage']}%")
            
            return relevant_documents
            
        except Exception as e:
            logger.error(f"Error filtering documents: {str(e)}")
            return documents  # Return original documents in case of error
    
    def get_filtering_stats(self, original_count: int, final_count: int) -> Dict[str, Any]:
        """
        Get statistics about the filtering process.
        
        Args:
            original_count: Number of documents before filtering
            final_count: Number of documents after filtering
            
        Returns:
            Dictionary containing filtering statistics
        """
        return {
            "original_document_count": original_count,
            "final_document_count": final_count,
            "documents_removed": original_count - final_count,
            "removal_percentage": round((original_count - final_count) / original_count * 100, 2) if original_count > 0 else 0
        }

    def calculate_llm_relevance_score(self, document: Dict[str, Any], query: str) -> float:
        """
        Calculate relevance score for a document using LLM judgment.
        
        Args:
            document: The document to score
            query: The original research query
            
        Returns:
            Float relevance score between 0 and 1
        """
        try:
            # Extract document information
            title = document.get('title', 'No title')
            content = document.get('content', '')
            
            # Limit content length to avoid excessive token usage
            max_content_length = 1000
            if len(content) > max_content_length:
                content = content[:max_content_length] + "..."
            
            # Construct the prompt for LLM relevance scoring
            prompt = f"""Evaluate the relevance of this document to a research query.

Research Query: {query}

Document Title: {title}
Document Content: {content}

Carefully assess how well this document content might be useful for the research query. Consider:

1. Direct relevance: Does the document directly address the query topic?
2. Indirect relevance: Does it provide background, context, or related concepts?
3. Partial relevance: Does it contain at least some information that might be useful?
4. Informational value: Does it provide unique information, regardless of perfect relevance?

VERY IMPORTANT: Be EXTREMELY lenient in your scoring. We prefer to include slightly relevant documents rather than exclude potentially useful ones.

Evaluate the relevance on a scale of 0.0 to 1.0:
- 0.0-0.2: Completely unrelated with no possible connection
- 0.3-0.4: Minimally relevant but has some potentially useful info
- 0.5-0.6: Moderately relevant with some useful information
- 0.7-0.8: Highly relevant with valuable insights
- 0.9-1.0: Perfectly relevant and central to the research query

When in doubt, assign a HIGHER score. Our system needs sufficient document content to generate reports.

Your response must be only a JSON object with:
{{
    "score": [numeric value between 0.0 and 1.0],
    "reasoning": "brief explanation of your assessment"
}}"""

            # Get relevance evaluation from the dedicated scoring LLM
            response = self.scoring_llm.invoke(
                prompt,
                response_format={ "type": "json_object" }  # Force JSON response
            ).content
            
            try:
                evaluation = json.loads(response)
                
                if not isinstance(evaluation, dict):
                    logger.error("Invalid LLM evaluation format")
                    return 0.5  # Default score in case of error
                
                if 'score' not in evaluation:
                    logger.error("Missing score in LLM evaluation")
                    return 0.5
                
                score = float(evaluation.get('score', 0.5))
                reasoning = evaluation.get('reasoning', 'No reasoning provided')
                
                # Boost score to ensure more documents are included
                boosted_score = min(score * 1.2, 1.0)
                
                # Add the LLM reasoning to the document metadata for transparency
                if 'metadata' not in document:
                    document['metadata'] = {}
                document['metadata']['llm_reasoning'] = reasoning
                document['metadata']['original_score'] = score
                document['metadata']['boosted_score'] = boosted_score
                
                logger.info(f"GPT-4o-mini relevance score for '{title[:30]}...': {score:.2f} (boosted to {boosted_score:.2f})")
                return boosted_score
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Error parsing LLM evaluation: {str(e)}")
                return 0.5  # Default score in case of error
                
        except Exception as e:
            logger.error(f"Error in LLM relevance scoring: {str(e)}")
            return 0.5  # Default score in case of error
            
    def filter_documents_with_llm(self, documents: List[Dict[str, Any]], 
                                query: str, 
                                threshold: float = 0.55,
                                use_algorithm_first: bool = True) -> List[Dict[str, Any]]:
        """
        Filter documents using LLM-based relevance scoring with stricter rules.
        
        Args:
            documents: List of documents to filter
            query: The research query
            threshold: Minimum relevance score threshold (default: 0.55)
            use_algorithm_first: Whether to use algorithmic filtering first (default: True)
            
        Returns:
            Filtered list of documents
        """
        try:
            if not documents:
                return []
                
            # If requested, use algorithmic filtering first to reduce LLM calls
            if use_algorithm_first and len(documents) > 10:
                # Use a very low threshold for the algorithmic filter
                algorithmic_threshold = max(0.01, threshold - 0.25)
                pre_filtered = self.filter_documents_by_score(documents, query, algorithmic_threshold)
                logger.info(f"Pre-filtered to {len(pre_filtered)} documents using algorithmic scoring")
                
                # If algorithmic filtering is too aggressive, just use all documents
                if len(pre_filtered) < 10 and len(documents) > 10:
                    logger.warning("Algorithmic filtering too aggressive, using all documents")
                    docs_to_evaluate = documents
                else:
                    docs_to_evaluate = pre_filtered
            else:
                docs_to_evaluate = documents
                
            # Score documents
            scored_docs = []
            for doc in docs_to_evaluate:
                score = self._evaluate_document_relevance(doc, query)
                # Apply boost factor of 1.2
                boosted_score = score * 1.2
                scored_docs.append((doc, boosted_score))
                logger.info(f"GPT-4o-mini relevance score for '{doc.get('title', '')[:50]}...': {score:.2f} (boosted to {boosted_score:.2f})")
            
            # Sort by score
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            # Apply stricter filtering rules
            filtered_docs = []
            removed_count = 0
            
            # Rule 1: Keep only documents above threshold
            for doc, score in scored_docs:
                if score >= threshold:
                    filtered_docs.append(doc)
                else:
                    removed_count += 1
            
            # Rule 2: If we have too many documents, keep only top 30%
            if len(filtered_docs) > 50:
                filtered_docs = [doc for doc, _ in scored_docs[:int(len(scored_docs) * 0.3)]]
                removed_count = len(scored_docs) - len(filtered_docs)
            
            # Rule 3: If we have too few documents after filtering, keep top 20
            if len(filtered_docs) < 20:
                filtered_docs = [doc for doc, _ in scored_docs[:20]]
                removed_count = len(scored_docs) - len(filtered_docs)
            
            # Log filtering results
            logger.info(f"GPT-4o-mini filtering removed {removed_count} documents ({(removed_count/len(documents))*100:.1f}%)")
            if filtered_docs:
                highest_score = scored_docs[0][1]
                highest_doc = scored_docs[0][0]
                logger.info(f"Highest scored document ({highest_score:.2f}): '{highest_doc.get('title', '')[:50]}...'")
                logger.info(f"Reasoning: {highest_doc.get('reasoning', 'No reasoning provided')}")
            
            logger.info(f"LLM-based filtering retained {len(filtered_docs)} of {len(documents)} documents")
            logger.info(f"Filtered out {removed_count} documents ({(removed_count/len(documents))*100:.1f}%)")
            
            # Log top documents
            if filtered_docs:
                logger.info("Top documents by relevance:")
                for i, (doc, score) in enumerate(scored_docs[:5], 1):
                    logger.info(f"{i}. [{score:.2f}] {doc.get('title', '')[:50]}...")
            
            return filtered_docs
            
        except Exception as e:
            logger.error(f"Error in LLM-based document filtering: {str(e)}")
            return documents


# Example usage
if __name__ == "__main__":
    # Test document filtering
    doc_filter = DocumentFilter()
    
    # Sample documents
    test_docs = [
        {"title": "Introduction to Quantum Computing", 
         "content": "Quantum computing is a type of computation that harnesses quantum mechanical phenomena.",
         "snippet": "Learn about the basics of quantum computing technology."},
        {"title": "Weather Forecast for Tomorrow", 
         "content": "Tomorrow will be sunny with a high of 75 degrees.",
         "snippet": "Check out the weather forecast for your area."},
        {"title": "Advanced Quantum Algorithms", 
         "content": "Shor's algorithm is a quantum algorithm for factoring integers in polynomial time.",
         "snippet": "Review of quantum algorithms and their applications."}
    ]
    
    # Test query
    test_query = "Recent advances in quantum computing algorithms"
    
    # Filter documents
    filtered_docs = doc_filter.filter_documents_with_llm(test_docs, test_query, threshold=0.55)
    
    # Print results
    print(f"Original documents: {len(test_docs)}")
    print(f"Filtered documents: {len(filtered_docs)}")
    
    for doc in filtered_docs:
        print(f"\nTitle: {doc['title']}")
        print(f"Relevance score: {doc.get('relevance_score', 0):.2f}")
        print(f"Snippet: {doc['snippet']}") 