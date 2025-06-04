"""
Data gathering module for the research pipeline with enhanced multi-processing capabilities.
"""

import logging
import asyncio
import aiohttp
import os
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from collections import defaultdict
from ratelimit import limits, sleep_and_retry
import re
import string
import math
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

try:
    from arxiv import Client, Search
except ImportError:
    logging.warning("arxiv package not available. arXiv search will be disabled.")

from .web_agents import WebDataGatherer
from .knowledge_base import KnowledgeBase
from .config import OPENAI_API_KEY, BASE_MODEL, MODEL_TEMPERATURE, TAVILY_API_KEY

# Import Tavily client if available
try:
    from tavily import TavilyClient
except ImportError:
    logging.warning("Tavily package not available. Tavily search will be disabled.")
    
# Configure logging
logger = logging.getLogger(__name__)

# Rate limiting configuration
CALLS = 10  # Number of calls
RATE = 60   # Per 60 seconds

@sleep_and_retry
@limits(calls=CALLS, period=RATE)
def rate_limited_call():
    """Rate limiting decorator for API calls."""
    pass

class DataGatherer:
    """Handles data gathering from various sources with enhanced multi-processing."""
    
    def __init__(self):
        """Initialize the data gatherer with multiple scraping capabilities."""
        self.web_gatherer = WebDataGatherer()
        self.session = None
        self.semaphore = asyncio.Semaphore(5)  # Limit concurrent requests
        self.source_stats = defaultdict(int)  # Track documents per source
        self.topic_stats = defaultdict(int)   # Track documents per topic
        
        # Initialize LLM for query elaboration and document scoring
        self.llm = ChatOpenAI(
            model=BASE_MODEL,
            temperature=MODEL_TEMPERATURE,
            openai_api_key=OPENAI_API_KEY
        )
        
        # Initialize document filter for LLM-based filtering
        from backend.document_filter import DocumentFilter
        self.document_filter = DocumentFilter()
        
        # Initialize only Tavily and arXiv search clients
        self.tavily_client = None
        if TAVILY_API_KEY:
            try:
                self.tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
            except Exception as e:
                logger.error(f"Failed to initialize Tavily client: {str(e)}")
                
        self.arxiv_client = None
        try:
            self.arxiv_client = Client()
        except Exception as e:
            logger.error(f"Failed to initialize arXiv client: {str(e)}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def elaborate_query(self, query: str) -> List[str]:
        """
        Elaborate a research query into multiple specific search queries.
        
        Args:
            query: The original research query
            
        Returns:
            List of elaborated search queries
        """
        try:
            # Create prompt template for query elaboration
            elaborate_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a research assistant. Your task is to transform a general research query into 
multiple specific search queries that will yield better results from academic and web sources.

Follow these guidelines:
1. Analyze the original query to identify key concepts, technical terms, and research areas
2. Create 5-8 specific search queries that cover different aspects of the topic
3. IMPORTANT: Create a mix of query types:
   - 2-3 academic/technical queries using domain-specific terminology for academic databases
   - 2-3 simplified queries focusing on core concepts (3-5 terms maximum)
   - 1-2 queries with specific applications or use cases
4. Keep all queries under 50 characters and avoid special characters like quotes or parentheses
5. For academic databases like arXiv, focus on technical keywords and avoid complex phrases

Format your response as a numbered list of search queries, with NO additional explanation.
Keep each query focused and effective for academic search engines."""),
                ("human", "Original research query: {query}")
            ])
            
            # Format the prompt with the query
            prompt = elaborate_prompt.format_messages(query=query)
            
            # Get elaborated queries from LLM
            response = self.llm.invoke(prompt)
            
            # Extract and clean the elaborated queries
            elaborated_text = response.content.strip()
            
            # Extract numbered queries from the response
            elaborated_queries = []
            for line in elaborated_text.split('\n'):
                # Strip any numbers, periods, or other formatting
                cleaned_line = re.sub(r'^\s*\d+\.?\s*', '', line).strip()
                # Remove quotes and other special characters
                cleaned_line = re.sub(r'["\':;,\(\)\[\]\{\}]', ' ', cleaned_line).strip()
                # Remove extra spaces
                cleaned_line = re.sub(r'\s+', ' ', cleaned_line).strip()
                
                if cleaned_line and len(cleaned_line) > 3:  # Only add non-empty lines
                    # Ensure the query doesn't exceed max length for search APIs
                    if len(cleaned_line) > 400:
                        cleaned_line = cleaned_line[:400].rsplit(' ', 1)[0]
                    elaborated_queries.append(cleaned_line)
            
            # Create simple keyword-based queries for academic search engines
            keywords = set()
            # Extract important terms from the original query
            important_terms = re.findall(r'\b[A-Za-z]{4,}\b', query)
            for term in important_terms:
                if len(term) > 3 and term.lower() not in {'with', 'that', 'this', 'from', 'have', 'what', 'when', 'where', 'which', 'their', 'about'}:
                    keywords.add(term)
            
            # Create a few keyword-based queries if we have enough keywords
            if len(keywords) >= 3:
                # Take random samples of 3-4 keywords to create varied queries
                import random
                keyword_list = list(keywords)
                random.shuffle(keyword_list)
                
                # Create 2 keyword-based queries with different combinations
                for i in range(min(2, len(keyword_list) // 3)):
                    start_idx = i * 3
                    if start_idx + 3 <= len(keyword_list):
                        keyword_query = ' '.join(keyword_list[start_idx:start_idx+3])
                        if keyword_query not in elaborated_queries:
                            elaborated_queries.append(keyword_query)
            
            # Always include a shortened version of the original query if it's long
            if len(query) > 400:
                shortened_query = query[:400].rsplit(' ', 1)[0]
                if shortened_query not in elaborated_queries:
                    elaborated_queries.append(shortened_query)
            elif query not in elaborated_queries:
                elaborated_queries.append(query)
                
            logger.info(f"Elaborated {len(elaborated_queries)} search queries from original: {query}")
            for i, eq in enumerate(elaborated_queries):
                logger.info(f"  Query {i+1}: {eq}")
                
            return elaborated_queries
            
        except Exception as e:
            logger.error(f"Error elaborating query: {str(e)}")
            # Return a shortened version of the original query if elaboration fails
            if len(query) > 400:
                return [query[:400].rsplit(' ', 1)[0]]
            return [query]
    
    async def search_tavily(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search using Tavily API."""
        try:
            if not self.tavily_client:
                logger.warning("Tavily client not initialized. Skipping Tavily search.")
                return []
            
            # Truncate query if it exceeds Tavily's max length of 400 characters
            MAX_QUERY_LENGTH = 400
            if len(query) > MAX_QUERY_LENGTH:
                logger.warning(f"Query too long ({len(query)} chars). Truncating to {MAX_QUERY_LENGTH} chars for Tavily search.")
                # Truncate to the last complete word within the limit
                truncated_query = query[:MAX_QUERY_LENGTH].rsplit(' ', 1)[0]
                query = truncated_query
                
            # Use the correct method and parameters for Tavily API
            response = self.tavily_client.search(
                query=query,
                search_depth="advanced",
                max_results=max_results,
                include_domains=["arxiv.org", "scholar.google.com", "ieee.org", "acm.org", 
                                "github.com", "wikipedia.org", "researchgate.net", "semanticscholar.org"]
            )
            
            documents = []
            # Handle both dictionary and object responses
            if isinstance(response, dict):
                results = response.get('results', [])
            else:
                results = response
                
            if isinstance(results, list):
                for result in results:
                    # Handle both dictionary and object responses
                    if isinstance(result, dict):
                        title = result.get('title', '')
                        content = result.get('content', '')
                        url = result.get('url', '')
                        snippet = result.get('snippet', '')
                    else:
                        title = getattr(result, 'title', '')
                        content = getattr(result, 'content', '')
                        url = getattr(result, 'url', '')
                        snippet = getattr(result, 'snippet', '')
                        
                    if title or content:  # Only add if we have content
                        document = {
                            'title': title,
                            'content': content,
                            'source': 'Tavily',
                            'url': url,
                            'snippet': snippet or (content[:200] + '...' if content else '')
                        }
                        documents.append(document)
            
            return documents
        except Exception as e:
            logger.error(f"Error in Tavily search: {str(e)}")
            return []
    
    async def search_arxiv(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search using arXiv API."""
        try:
            if not self.arxiv_client:
                logger.warning("arXiv client not initialized. Skipping arXiv search.")
                return []
                
            # Format query for arXiv - it doesn't handle complex queries well
            # Remove quotes and special characters that might cause problems
            formatted_query = re.sub(r'["\':;,\(\)\[\]\{\}]', ' ', query)
            # Split into terms and reconnect with AND operator for better results
            terms = [term.strip() for term in formatted_query.split() if len(term.strip()) > 2]
            # Keep only the first 6 terms to avoid overly specific queries
            if len(terms) > 6:
                terms = terms[:6]
            # Generate simple keywords search that arXiv can handle
            arxiv_query = ' AND '.join(terms)
            
            logger.info(f"Formatted arXiv query: {arxiv_query}")
                
            # Import required class for sorting if available
            try:
                from arxiv import SortCriterion
                has_sort_criterion = True
            except ImportError:
                has_sort_criterion = False
            
            # Create a search object using the correct API - with version compatibility
            try:
                # Try the newer version syntax with SortCriterion enum if available
                if has_sort_criterion:
                    search = Search(
                        query=arxiv_query,
                        max_results=max_results,
                        sort_by=SortCriterion.Relevance
                    )
                else:
                    # Fall back to string if SortCriterion not available
                    search = Search(
                        query=arxiv_query,
                        max_results=max_results,
                        sort_by="relevance"
                    )
            except TypeError:
                # Fall back to older version format if needed
                search = Search(
                    query=arxiv_query,
                    max_results=max_results
                )
            
            # Get results
            results = list(self.arxiv_client.results(search))
            
            if not results:
                # Fallback to an even simpler query if no results
                simplified_terms = [term for term in terms if len(term) > 3][:3]
                if simplified_terms:
                    fallback_query = ' OR '.join(simplified_terms)
                    logger.info(f"No results, trying fallback arXiv query: {fallback_query}")
                    try:
                        fallback_search = Search(query=fallback_query, max_results=max_results)
                        results = list(self.arxiv_client.results(fallback_search))
                    except Exception as e:
                        logger.error(f"Error in fallback arXiv search: {str(e)}")
            
            documents = []
            for result in results:
                if hasattr(result, 'title') and hasattr(result, 'summary'):
                    document = {
                        'title': result.title,
                        'content': result.summary,
                        'source': 'arXiv',
                        'url': result.entry_id,
                        'snippet': result.summary[:200] + '...'
                    }
                    documents.append(document)
            
            logger.info(f"Found {len(documents)} documents from arXiv")
            return documents
        except Exception as e:
            logger.error(f"Error in arXiv search: {str(e)}")
            return []
            
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
            
            # Combine scores with weights
            total_score = title_score + snippet_score + (content_score * 0.7)
            
            # Normalize score based on number of query terms (to get score between 0-1)
            max_possible_score = len(query_terms) * (5.0 + 3.0 + 0.7)  # maximum possible score
            normalized_score = min(total_score / max_possible_score if max_possible_score > 0 else 0, 1.0)
            
            return normalized_score
            
        except Exception as e:
            logger.error(f"Error calculating relevance score: {str(e)}")
            return 0.0
            
    async def gather_data_with_filtering(self, topics: Dict[str, Any], 
                                  kb: KnowledgeBase, 
                                  relevance_threshold: float = 0.55) -> List[Dict[str, Any]]:
        """
        Gather data from web and academic sources with relevance filtering.
        
        Args:
            topics: Dictionary containing research topics and subtopics
            kb: Knowledge base to check for existing documents
            relevance_threshold: Minimum relevance score (0-1) to keep document
            
        Returns:
            List of relevant gathered documents
        """
        try:
            all_documents = []
            filtered_documents = []
            original_query = topics.get('original_query', '')
            
            if not original_query:
                logger.warning("No original query provided. Document gathering may be less effective.")
                if isinstance(topics, str):
                    original_query = topics
                elif isinstance(topics, dict) and topics:
                    # Try to extract a query from the first key or value
                    original_query = next(iter(topics.values())) if topics.values() else next(iter(topics.keys()))
                else:
                    logger.error("Could not extract a query from topics")
                    return []
            
            # Extract key topic areas from the topics dictionary for fallback searches
            key_topics = []
            if isinstance(topics, dict):
                # Look for focus_areas or key_topics
                if 'focus_areas' in topics and isinstance(topics['focus_areas'], list):
                    key_topics.extend(topics['focus_areas'])
                if 'key_topics' in topics and isinstance(topics['key_topics'], list):
                    key_topics.extend(topics['key_topics'])
            
            # Elaborate the original query into multiple specific search queries
            elaborated_queries = await self.elaborate_query(original_query)
            
            # Generate additional topic-specific queries
            topic_specific_queries = await self.generate_topic_specific_queries(original_query, topics)
            if topic_specific_queries:
                logger.info(f"Adding {len(topic_specific_queries)} topic-specific queries")
                elaborated_queries.extend(topic_specific_queries)
            
            # Ensure all queries are within the appropriate length
            elaborated_queries = [
                query[:400].rsplit(' ', 1)[0] if len(query) > 400 else query 
                for query in elaborated_queries
            ]
            
            # Add key topics as additional search queries if available
            if key_topics:
                for topic in key_topics[:3]:  # Limit to first 3 topics
                    if isinstance(topic, str) and topic not in elaborated_queries:
                        # Truncate if necessary
                        if len(topic) > 400:
                            topic = topic[:400].rsplit(' ', 1)[0]
                        elaborated_queries.append(topic)
            
            # Limit the number of queries to prevent overwhelming search services
            MAX_QUERIES = 8
            if len(elaborated_queries) > MAX_QUERIES:
                logger.info(f"Limiting from {len(elaborated_queries)} to {MAX_QUERIES} elaborated queries")
                elaborated_queries = elaborated_queries[:MAX_QUERIES]
            
            # Track document sources for deduplication
            seen_urls = set()
            
            # Gather data for each elaborated query
            successful_queries = 0
            for query in elaborated_queries:
                try:
                    logger.info(f"Searching with elaborated query: {query} ({len(query)} chars)")
                    
                    # Get documents from Tavily search
                    tavily_docs = await self.search_tavily(query, max_results=8)
                    if tavily_docs:
                        for doc in tavily_docs:
                            url = doc.get('url', '')
                            if url not in seen_urls:
                                seen_urls.add(url)
                                # Calculate relevance score
                                doc['relevance_score'] = self.calculate_relevance_score(doc, original_query)
                                # Add metadata
                                doc['metadata'] = {
                                    'source': doc.get('source', 'Tavily'),
                                    'query': query,
                                    'elaborated_from': original_query,
                                    'gathered_at': str(datetime.now()),
                                    'relevance_score': doc['relevance_score']
                                }
                                all_documents.append(doc)
                        
                        self.source_stats['Tavily'] += len(tavily_docs)
                        logger.info(f"Gathered {len(tavily_docs)} documents from Tavily")
                        successful_queries += 1
                    
                    # Get documents from arXiv search
                    arxiv_docs = await self.search_arxiv(query, max_results=8)
                    if arxiv_docs:
                        for doc in arxiv_docs:
                            url = doc.get('url', '')
                            if url not in seen_urls:
                                seen_urls.add(url)
                                # Calculate relevance score
                                doc['relevance_score'] = self.calculate_relevance_score(doc, original_query)
                                # Add metadata
                                doc['metadata'] = {
                                    'source': doc.get('source', 'arXiv'),
                                    'query': query,
                                    'elaborated_from': original_query,
                                    'gathered_at': str(datetime.now()),
                                    'relevance_score': doc['relevance_score']
                                }
                                all_documents.append(doc)
                        
                        self.source_stats['arXiv'] += len(arxiv_docs)
                        logger.info(f"Gathered {len(arxiv_docs)} documents from arXiv")
                        successful_queries += 1
                    
                    # Track documents by topic
                    self.topic_stats[query] += len(tavily_docs) + len(arxiv_docs)
                        
                except Exception as e:
                    logger.error(f"Error gathering data for query '{query}': {str(e)}")
                    continue
            
            # If we didn't get any successful queries, try again with simpler keywords
            if successful_queries == 0 and len(all_documents) == 0:
                logger.warning("No successful queries - attempting fallback with simpler keywords")
                # Extract single keywords from the original query
                keywords = re.findall(r'\b[a-zA-Z]{4,}\b', original_query)
                relevant_keywords = [k for k in keywords if len(k) > 3 and k.lower() not in {'with', 'that', 'this', 'from', 'have'}][:5]
                
                if relevant_keywords:
                    for keyword in relevant_keywords:
                        try:
                            logger.info(f"Fallback search with keyword: {keyword}")
                            # Try arXiv with individual keywords
                            arxiv_docs = await self.search_arxiv(keyword, max_results=5)
                            if arxiv_docs:
                                for doc in arxiv_docs:
                                    url = doc.get('url', '')
                                    if url not in seen_urls:
                                        seen_urls.add(url)
                                        doc['relevance_score'] = self.calculate_relevance_score(doc, original_query)
                                        doc['metadata'] = {
                                            'source': doc.get('source', 'arXiv'),
                                            'query': keyword,
                                            'elaborated_from': original_query,
                                            'gathered_at': str(datetime.now()),
                                            'relevance_score': doc['relevance_score']
                                        }
                                        all_documents.append(doc)
                                self.source_stats['arXiv (fallback)'] += len(arxiv_docs)
                        except Exception as e:
                            logger.error(f"Error in fallback search: {str(e)}")
            
            # Use LLM-based filtering instead of simple threshold filtering
            logger.info(f"Using LLM-based filtering for {len(all_documents)} documents")
            
            # Fall back to basic filtering if no documents found or very few documents
            if len(all_documents) <= 3:
                # Use a very low threshold when we have few documents
                if len(all_documents) > 0:
                    logger.info(f"Only {len(all_documents)} documents found - using basic filtering with low threshold")
                    relevance_threshold = 0.1
                    filtered_documents = [doc for doc in all_documents if doc.get('relevance_score', 0) >= relevance_threshold]
                else:
                    filtered_documents = []
            else:
                # Adjust threshold based on whether we're in fallback mode
                llm_threshold = 0.3
                if successful_queries == 0:
                    llm_threshold = 0.2  # Lower threshold for fallback results
                
                # Use the LLM-based filtering from DocumentFilter class
                try:
                    # Set use_algorithm_first=True to use algorithmic filtering first when we have lots of documents
                    # This reduces the number of LLM calls while preserving potentially relevant documents
                    filtered_documents = self.document_filter.filter_documents_with_llm(
                        all_documents, 
                        original_query, 
                        threshold=llm_threshold,
                        use_algorithm_first=(len(all_documents) > 10)
                    )
                    logger.info(f"LLM-based filtering retained {len(filtered_documents)} of {len(all_documents)} documents")
                except Exception as e:
                    logger.error(f"Error in LLM-based filtering: {str(e)}")
                    logger.info("Falling back to algorithmic filtering due to error")
                    # Fall back to algorithmic filtering if LLM filtering fails
                    filtered_documents = [doc for doc in all_documents if doc.get('relevance_score', 0) >= relevance_threshold]
            
            # Sort documents by relevance score (highest first)
            filtered_documents.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
            
            # Log filtering stats
            filtered_percentage = round((len(all_documents) - len(filtered_documents)) / max(len(all_documents), 1) * 100, 1)
            logger.info(f"Filtered out {len(all_documents) - len(filtered_documents)} documents ({filtered_percentage}%)")
            
            # Print top documents and their scores
            logger.info("Top documents by relevance:")
            for i, doc in enumerate(filtered_documents[:5], 1):
                logger.info(f"{i}. [{doc.get('relevance_score', 0):.2f}] {doc.get('title', 'Untitled')}")
            
            # Print document gathering stats
            self.print_gathering_stats()
            
            return filtered_documents
            
        except Exception as e:
            logger.error(f"Error in gather_data_with_filtering: {str(e)}")
            return []
    
    def print_gathering_stats(self):
        """Print statistics about gathered documents."""
        logger.info("Document Gathering Statistics:")
        logger.info(f"Documents by source: {dict(self.source_stats)}")
        logger.info(f"Documents by topic: {dict(self.topic_stats)}")
    
    async def gather_data(self, topics: Dict[str, Any], kb: KnowledgeBase) -> List[Dict[str, Any]]:
        """
        Gather data from web and academic sources for the given topics.
        This method is maintained for backward compatibility.
        
        Args:
            topics: Dictionary containing research topics and subtopics
            kb: Knowledge base to check for existing documents
            
        Returns:
            List of gathered documents
        """
        return await self.gather_data_with_filtering(topics, kb)

    async def generate_topic_specific_queries(self, query: str, topics: Dict[str, Any]) -> List[str]:
        """
        Generate topic-specific search queries tailored to the research focus areas.
        
        Args:
            query: The original research query
            topics: Dictionary containing research topics and focus areas
            
        Returns:
            List of topic-specific search queries
        """
        try:
            # Extract key topics and focus areas
            focus_areas = []
            if isinstance(topics, dict):
                if 'focus_areas' in topics and isinstance(topics['focus_areas'], list):
                    focus_areas.extend(topics['focus_areas'])
                if 'key_topics' in topics and isinstance(topics['key_topics'], list):
                    focus_areas.extend(topics['key_topics'])
            
            if not focus_areas:
                logger.warning("No focus areas found for topic-specific queries")
                return []
                
            # Limit to most relevant focus areas
            focus_areas = focus_areas[:5]
            focus_areas_text = "\n".join([f"- {area}" for area in focus_areas])
            
            # Create prompt for generating topic-specific queries
            prompt = f"""You are a research assistant. Generate search queries specifically tailored to each focus area of a research topic.

Original Research Query: {query}

Research Focus Areas:
{focus_areas_text}

For each focus area above, create ONE specific search query that will retrieve the most relevant academic content.
Each query should:
1. Be highly specific to the focus area
2. Include technical terminology relevant to that area
3. Be formulated in a way that would retrieve academic papers/research
4. Be concise (5-10 words maximum)
5. Avoid quotes, parentheses, or special characters

Format your response as a numbered list of search queries ONLY, with NO explanations or additional text."""

            # Get response from LLM
            messages = [
                {"role": "system", "content": "You generate concise, targeted search queries for academic research."},
                {"role": "user", "content": prompt}
            ]
            
            # Format messages for LangChain ChatPromptTemplate
            chat_prompt = ChatPromptTemplate.from_messages(messages)
            formatted_prompt = chat_prompt.format_prompt().to_messages()
            
            # Invoke LLM
            response = self.llm.invoke(formatted_prompt)
            response_text = response.content.strip()
            
            # Parse queries from the response
            queries = []
            for line in response_text.split('\n'):
                # Remove numbering and clean up
                cleaned_line = re.sub(r'^\s*\d+\.?\s*', '', line).strip()
                # Remove quotes and other special characters
                cleaned_line = re.sub(r'["\':;,\(\)\[\]\{\}]', ' ', cleaned_line).strip()
                # Remove extra spaces
                cleaned_line = re.sub(r'\s+', ' ', cleaned_line).strip()
                
                if cleaned_line and len(cleaned_line) > 3:
                    queries.append(cleaned_line)
            
            logger.info(f"Generated {len(queries)} topic-specific queries")
            return queries
            
        except Exception as e:
            logger.error(f"Error generating topic-specific queries: {str(e)}")
            return []

# Entry point for direct testing
if __name__ == "__main__":
    async def test_gatherer():
        gatherer = DataGatherer()
        test_query = "Impact of quantum computing on cryptography"
        kb = KnowledgeBase()
        topics = {"original_query": test_query}
        documents = await gatherer.gather_data_with_filtering(topics, kb)
        print(f"Found {len(documents)} documents")
        for i, doc in enumerate(documents[:3]):
            print(f"\nDocument {i+1}: {doc.get('title', 'No title')}")
            print(f"Source: {doc.get('source', 'Unknown')}")
            print(f"Score: {doc.get('relevance_score', 0):.2f}")
            print(f"Snippet: {doc.get('snippet', 'No snippet')[:100]}...")
    
    import asyncio
    asyncio.run(test_gatherer())