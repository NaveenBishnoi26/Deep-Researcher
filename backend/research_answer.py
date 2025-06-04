"""
Research Answer Generator

This module generates detailed research answers by processing sections and subsections
from gathered research data and using LLM to synthesize comprehensive responses.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from backend.config import OPENAI_API_KEY, BASE_MODEL, MODEL_TEMPERATURE
import openai
import re
import string
from collections import defaultdict
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResearchAnswerGenerator:
    """Generates research answers from knowledge base content."""
    
    def __init__(self):
        """Initialize the research answer generator."""
        self.llm = ChatOpenAI(
            model=BASE_MODEL,
            temperature=MODEL_TEMPERATURE,
            openai_api_key=OPENAI_API_KEY
        )
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        self.chat_history = []  # Track chat history for context
        self.citation_map = {}  # Maps citation text to citation number
        self.used_citations = defaultdict(set)  # Track citations used in each section
        self.technical_terms = set()  # Track technical terms for glossary
        self.max_section_length = 3000  # Maximum length of a section in words
        self.max_citations_per_section = 25  # Increased from 20 to 25
        self.max_citations_per_subsection = 12  # Increased from 10 to 12
        self.max_subsection_length = 1000  # Maximum length of a subsection in words
        self.min_subsection_length = 150  # Minimum length of a subsection in words
        self.extract_terms = False  # Flag to control technical term extraction
        self.cached_documents = {}  # Cache for document retrieval
        logger.info(f"Initialized ResearchAnswerGenerator with model: {BASE_MODEL} and temperature: {MODEL_TEMPERATURE}")

    def get_section_keywords(self, section_title: str, query: str) -> List[str]:
        """
        Extract relevant keywords from section title and query to improve search.
        
        Args:
            section_title: Title of the section
            query: Main research query
            
        Returns:
            List of relevant keywords for document search
        """
        # Use GPT to extract the most relevant keywords for this section
        try:
            prompt = f"""Extract the 5-7 most important search keywords from this research report section title and query.
            The keywords will be used to find relevant documents in a research database.
            
            Research Query: {query}
            Section Title: {section_title}
            
            Extract specific, concrete terms (not generic words like "analysis" or "overview").
            Format the output as a simple comma-separated list with no numbering or bullets.
            For example: "quantum computing, error correction, superconducting qubits, decoherence"
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Use mini model for faster processing
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=100
            )
            
            # Parse the response into a list of keywords
            keywords_text = response.choices[0].message.content.strip()
            keywords = [k.strip() for k in keywords_text.split(',')]
            
            # Add the section title words as individual keywords
            title_words = section_title.split()
            for word in title_words:
                if len(word) > 3 and word.lower() not in ['and', 'the', 'for', 'with', 'from']:
                    keywords.append(word)
            
            # Add important words from the query
            query_words = query.split()
            for word in query_words:
                if len(word) > 4 and word.lower() not in ['and', 'the', 'for', 'with', 'from', 'what', 'when', 'where', 'which', 'how']:
                    keywords.append(word)
            
            # Remove duplicates
            keywords = list(set(keywords))
            logger.info(f"Generated keywords for '{section_title}': {keywords}")
            return keywords
            
        except Exception as e:
            logger.warning(f"Error generating keywords: {str(e)}")
            # Fallback: Extract words from section title and query
            words = set()
            
            # Add words from section title
            for word in section_title.split():
                if len(word) > 3 and word.lower() not in ['and', 'the', 'for', 'with', 'from']:
                    words.add(word.lower())
            
            # Add words from query
            for word in query.split():
                if len(word) > 4 and word.lower() not in ['and', 'the', 'for', 'with', 'from', 'what', 'when', 'where', 'which', 'how']:
                    words.add(word.lower())
                    
            return list(words)

    def generate_answer(self, query: str, documents: List[Document]) -> Dict[str, Any]:
        """Generate a comprehensive answer based on the query and documents."""
        try:
            # Filter out empty documents
            valid_documents = [doc for doc in documents if doc.page_content.strip()]

            # Extract relevant information from documents
            document_texts = [doc.page_content for doc in valid_documents]
            document_sources = [doc.metadata.get("source", "Unknown") for doc in valid_documents]

            # Generate answer using the language model
            prompt = self._create_answer_prompt(query, document_texts)
            response = self.llm.predict(prompt)

            # Process and format the response
            answer = self._process_response(response)
            sources = self._format_sources(valid_documents)
            confidence = self._calculate_confidence(response, valid_documents)

            return {
                "answer": answer,
                "sources": sources,
                "confidence": confidence
            }

        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            raise

    def _create_answer_prompt(self, query: str, document_texts: List[str]) -> str:
        """Create a prompt for answer generation."""
        context = "\n\n".join(document_texts)
        return f"""Based on the following research documents, please provide a comprehensive answer to the question: {query}

Research Documents:
{context}

Please provide a detailed answer that:
1. Directly addresses the question
2. Synthesizes information from multiple sources
3. Highlights key findings and insights
4. Maintains academic rigor and accuracy
5. Cites specific sources when making claims

Answer:"""

    def _process_response(self, response: str) -> str:
        """Process and format the model's response."""
        # Clean up and format the response
        lines = response.strip().split("\n")
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                formatted_lines.append(line)
        
        return "\n".join(formatted_lines)

    def _format_sources(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Format source information from documents."""
        sources = []
        for doc in documents:
            source = {
                "title": doc.metadata.get("title", "Untitled"),
                "url": doc.metadata.get("url", ""),
                "relevance": doc.metadata.get("relevance", 1.0)
            }
            sources.append(source)
        return sources

    def _calculate_confidence(self, response: str, documents: List[Document]) -> float:
        """Calculate confidence score for the generated answer."""
        # Simple confidence calculation based on:
        # 1. Number of sources used
        # 2. Length and detail of the answer
        # 3. Presence of specific citations
        
        confidence = 0.5  # Base confidence
        
        # Adjust based on number of sources
        source_factor = min(len(documents) / 5, 1.0)  # Max out at 5 sources
        confidence += source_factor * 0.2
        
        # Adjust based on answer length
        length_factor = min(len(response) / 1000, 1.0)  # Max out at 1000 chars
        confidence += length_factor * 0.2
        
        # Adjust based on citations
        citation_count = response.count("according to") + response.count("suggests") + response.count("shows")
        citation_factor = min(citation_count / 5, 1.0)  # Max out at 5 citations
        confidence += citation_factor * 0.1
        
        return min(confidence, 1.0)  # Ensure confidence doesn't exceed 1.0

    def _add_to_chat_history(self, role: str, content: str):
        """Add a message to the chat history."""
        self.chat_history.append({"role": role, "content": content})
        
    def _get_chat_history(self) -> List[Dict[str, str]]:
        """Get the current chat history."""
        return self.chat_history.copy()
    
    def _validate_content(self, content: str, is_subsection: bool = False) -> bool:
        """Validate content to ensure it meets standards."""
        if not content or len(content.strip()) < 50:
            logger.warning("Content is too short")
            return False
            
        # Basic checks for all content
        if is_subsection:
            # For subsections, be more lenient with validation
            # Just ensure there's something meaningful
            return True
        else:
            # For main sections, still do some minimal validation
            # but be more lenient compared to before
            
            # Check for forbidden patterns in subsections
            forbidden_patterns = [
                r'will be discussed',
                r'will be explored',
                r'will be examined',
                r'will be analyzed'
            ]
            
            content_lower = content.lower()
            for pattern in forbidden_patterns:
                if re.search(pattern, content_lower):
                    logger.warning(f"Forbidden pattern found in content: {pattern}")
                    return False
                    
        return True
        
    def _extract_technical_terms(self, content: str):
        """Extract technical terms from content."""
        # Skip technical term extraction if flag is False
        if not self.extract_terms:
            return set()
            
        try:
            messages = [
                {"role": "system", "content": "You are a technical term extractor. Identify technical terms and jargon in the given text."},
                {"role": "user", "content": f"Extract technical terms and jargon from the following text:\n\n{content}"}
            ]
            
            response = self.client.chat.completions.create(
                model=BASE_MODEL,
                messages=messages,
                temperature=0.3,
                max_tokens=500
            )
            
            terms = set(response.choices[0].message.content.split('\n'))
            return {term.strip() for term in terms if term.strip()}
            
        except Exception as e:
            logger.error(f"Error extracting technical terms: {str(e)}")
            return set()

    def extract_all_technical_terms(self, report_content: str):
        """Extract technical terms from the entire report at once."""
        self.extract_terms = True  # Enable term extraction temporarily
        all_terms = self._extract_technical_terms(report_content)
        self.technical_terms.update(all_terms)
        self.extract_terms = False  # Disable term extraction again
        return self.technical_terms

    def generate_section_content(
        self,
        section: Dict[str, Any],
        kb,
        query: str,
        is_subsection: bool = False,
        existing_headers: List[str] = [],
        relevant_written_contents: List[str] = []
    ) -> str:
        """
        Generate content for a section or subsection.
        
        Args:
            section: Section information
            kb: Knowledge base instance
            query: The research query
            is_subsection: Whether this is a subsection
            existing_headers: List of existing headers to avoid duplication
            relevant_written_contents: List of relevant written content to avoid duplication
            
        Returns:
            Generated content for the section
        """
        section_title = section.get('section_title', '')
        logger.info(f"Generating content for {'subsection' if is_subsection else 'section'}: {section_title}")
        
        # Create a unique cache key for this section
        cache_key = f"{query}::{section_title}"
        if cache_key in self.cached_documents and len(self.cached_documents[cache_key]) > 0:
            logger.info(f"Using cached documents for {section_title}")
            unique_docs = self.cached_documents[cache_key]
        else:
            # Check if kb has documents before attempting search
            if not hasattr(kb, 'documents') or not kb.documents:
                logger.error(f"Knowledge base does not contain any documents for section: {section_title}")
                return f"This section on {section_title} would typically present information related to {query}. However, sufficient research data could not be found. Please expand your search criteria or provide additional sources about {section_title}."
            
            # Get relevant documents for this section using multiple search strategies
            relevant_docs = []
            
            # Get keywords for better search
            keywords = self.get_section_keywords(section_title, query)
            
            # Strategy 1: Use the section title directly with the original query
            search_query = f"{query} {section_title}"
            try:
                query_docs = kb.search(search_query, top_k=10)
                if query_docs:
                    relevant_docs.extend(query_docs)
                    logger.info(f"Strategy 1: Found {len(query_docs)} documents for {section_title}")
            except Exception as e:
                logger.error(f"Error in search strategy 1: {str(e)}")
            
            # Strategy 2: Use keywords for more targeted search
            if keywords:
                try:
                    # Use all keywords in a single search
                    all_keywords_query = " ".join(keywords)
                    keyword_docs = kb.search(all_keywords_query, top_k=15)
                    if keyword_docs:
                        relevant_docs.extend(keyword_docs)
                        logger.info(f"Strategy 2: Found {len(keyword_docs)} documents using all keywords")
                    
                    # Use pairs of keywords for more focused searches
                    if len(keywords) > 1:
                        for i in range(len(keywords)):
                            for j in range(i+1, len(keywords)):
                                pair_query = f"{keywords[i]} {keywords[j]}"
                                pair_docs = kb.search(pair_query, top_k=5)
                                if pair_docs:
                                    relevant_docs.extend(pair_docs)
                                    logger.info(f"Strategy 2b: Found {len(pair_docs)} documents for keyword pair '{pair_query}'")
                except Exception as e:
                    logger.error(f"Error in search strategy 2: {str(e)}")
            
            # Strategy 3: If section has a description/content field, use that for search
            if section.get('content'):
                try:
                    content_query = f"{section.get('content')} {section_title}"
                    content_docs = kb.search(content_query, top_k=7)
                    if content_docs:
                        relevant_docs.extend(content_docs)
                        logger.info(f"Strategy 3: Found {len(content_docs)} documents using section description")
                except Exception as e:
                    logger.error(f"Error in search strategy 3: {str(e)}")
            
            # Strategy 4: If still no results, use just the section title
            if not relevant_docs:
                try:
                    title_docs = kb.search(section_title, top_k=15)
                    if title_docs:
                        relevant_docs.extend(title_docs)
                        logger.info(f"Strategy 4: Found {len(title_docs)} documents using just section title")
                except Exception as e:
                    logger.error(f"Error in search strategy 4: {str(e)}")
            
            # Strategy 5: If still no results, use a more generic approach - get documents for the main query
            if not relevant_docs:
                logger.warning(f"No relevant documents found for section: {section_title} - using fallback approach")
                try:
                    fallback_docs = kb.search(query, top_k=25)  # Just get many documents related to the main query
                    if fallback_docs:
                        relevant_docs.extend(fallback_docs)
                        logger.info(f"Strategy 5: Found {len(fallback_docs)} documents using fallback search")
                    else:
                        # If we still have nothing, try to get ALL documents
                        logger.error(f"Failed to find any relevant documents for section: {section_title}")
                        try:
                            # Use get_all_documents if it exists
                            if hasattr(kb, 'get_all_documents'):
                                all_docs = kb.get_all_documents(limit=30)
                                if all_docs:
                                    relevant_docs.extend(all_docs)
                                    logger.info(f"Emergency strategy: Retrieved {len(all_docs)} documents as last resort")
                            else:
                                # Fallback if get_all_documents doesn't exist
                                logger.warning("Knowledge base doesn't have get_all_documents method")
                                # Try to access documents directly if possible
                                if hasattr(kb, 'documents') and kb.documents:
                                    relevant_docs.extend(kb.documents[:30])
                                    logger.info(f"Emergency strategy: Retrieved {len(kb.documents[:30])} documents directly from KB")
                        except Exception as inner_e:
                            logger.error(f"Error retrieving all documents: {str(inner_e)}")
                except Exception as e:
                    logger.error(f"Error in search strategy 5: {str(e)}")
            
            # Deduplicate documents by ID
            unique_docs = []
            seen_ids = set()
            
            for doc in relevant_docs:
                # Check if doc is a proper document object or just a dict
                if hasattr(doc, 'metadata'):
                    doc_id = doc.metadata.get('id', '')
                    if doc_id and doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        unique_docs.append(doc)
                    elif not doc_id:  # If no ID, use content as ID
                        content_hash = hash(doc.page_content[:100] if hasattr(doc, 'page_content') else '')
                        if content_hash not in seen_ids and content_hash != hash(''):
                            seen_ids.add(content_hash)
                            unique_docs.append(doc)
                elif isinstance(doc, dict):
                    # Handle dictionary-style documents
                    doc_id = doc.get('id', '')
                    if doc_id and doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        unique_docs.append(doc)
                    elif not doc_id:  # If no ID, use content as ID
                        content_hash = hash(doc.get('content', '')[:100])
                        if content_hash not in seen_ids and content_hash != hash(''):
                            seen_ids.add(content_hash)
                            unique_docs.append(doc)
            
            # Re-rank documents based on relevance to section title if we have document objects
            if unique_docs and len(unique_docs) > 3:
                try:
                    ranked_docs = self._rank_documents_for_section(unique_docs, section_title, query)
                    unique_docs = ranked_docs
                except Exception as e:
                    logger.error(f"Error ranking documents: {str(e)}")
            
            # Cache the documents for future use
            self.cached_documents[cache_key] = unique_docs
                
        logger.info(f"Found {len(unique_docs)} unique documents for section: {section_title}")
            
        # If still no documents after all strategies
        if not unique_docs:
            logger.error(f"Failed to find any usable documents for section: {section_title}")
            return f"This section on {section_title} would typically present information related to {query}. However, sufficient research data could not be found. Please expand your search criteria or provide additional sources about {section_title}."
        
        # Use up to 15 documents for context (to avoid token limits)
        max_docs = 15
        if len(unique_docs) > max_docs:
            unique_docs = unique_docs[:max_docs]
        
        # Prepare context from relevant documents
        context_parts = []
        for doc in unique_docs:
            if hasattr(doc, 'page_content') and hasattr(doc, 'metadata'):
                # Handle Document objects
                context_parts.append(f"Document: {doc.metadata.get('title', 'Untitled')}\nContent: {doc.page_content}")
            elif isinstance(doc, dict):
                # Handle dictionary-style documents
                if 'content' in doc:
                    title = doc.get('metadata', {}).get('title', doc.get('title', 'Untitled'))
                    context_parts.append(f"Document: {title}\nContent: {doc['content']}")
            
        context = "\n\n".join(context_parts)
        
        # Add document sources to the prompt to encourage citation
        sources_text = "\n\nDocument Sources:\n"
        for doc in unique_docs:
            if hasattr(doc, 'metadata'):
                sources_text += f"- {doc.metadata.get('title', 'Untitled')}: {doc.metadata.get('url', 'No URL')}\n"
            elif isinstance(doc, dict):
                metadata = doc.get('metadata', {})
                title = metadata.get('title', doc.get('title', 'Untitled'))
                url = metadata.get('url', 'No URL')
                sources_text += f"- {title}: {url}\n"

        # Prepare system message based on whether this is a subsection
        if is_subsection:
            system_message = """You are a research report subsection generator. Generate content for a subsection of a research report based on the provided documents and query. Follow these rules:

1. Focus on the specific subsection topic - it's important to stay on topic
2. Maintain academic writing style
3. Use clear and concise language
4. Synthesize information from multiple sources 
5. Avoid duplicating content from other sections
6. Make use of ALL provided document content that is relevant
7. Maintain consistency with the main report
8. Use proper formatting and structure
9. Keep the content focused and specific to the subsection
10. Generate SUBSTANTIAL content with specific details, examples, and analysis"""
        else:
            system_message = """You are a research report section generator. Generate content for a main section of a research report based on the provided documents and query. Follow these rules:

1. Focus on the main section topic - it's important to stay on topic
2. Maintain academic writing style
3. Use clear and concise language
4. Synthesize information from multiple sources
5. Avoid duplicating content from other sections
6. Make use of ALL provided document content that is relevant
7. Maintain consistency with the main report
8. Use proper formatting and structure
9. Keep the content focused and specific to the section
10. Generate SUBSTANTIAL content with specific details, examples, and analysis"""
        
        # Prepare user message with context about existing content
        existing_content_context = ""
        if existing_headers:
            existing_content_context += "\n\nExisting headers to avoid duplicating:\n" + "\n".join(f"- {header}" for header in existing_headers)
        if relevant_written_contents:
            existing_content_context += "\n\nRelevant content already written (avoid duplicating this content):\n" + "\n".join(f"- {content[:100]}..." for content in relevant_written_contents)
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"""Generate detailed content for the following {'subsection' if is_subsection else 'section'} of a research report.

Main Query: {query}
Section Title: {section_title}
Section Description: {section.get('content', '')}

IMPORTANT: The documents below contain information you MUST use to generate content for this section.
Research Documents:
{context}

{sources_text}

{existing_content_context}

Please generate content that:
1. Is focused specifically on "{section_title}" 
2. Uses information from the provided documents
3. Includes specific details, facts, examples, and analysis
4. Is comprehensive and thorough
5. Follows academic writing style
6. Avoids vague statements or placeholders
7. Presents a cohesive narrative on the topic
8. Synthesizes information from multiple sources when possible

IMPORTANT: Your task is to GENERATE SPECIFIC CONTENT for this section, not just a placeholder or overview.
You MUST include actual information, details, and analysis from the documents.

Generate the content:"""}
        ]
        
        try:
            # Use a higher token limit to allow for more comprehensive content
            response = self.client.chat.completions.create(
                model=BASE_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=1500  # Increased from 1000 to 1500
            )
            
            content = response.choices[0].message.content.strip()
            
            # If content is very short, try once more with a more specific prompt
            if len(content) < 200:
                logger.warning(f"Generated content for {section_title} is too short ({len(content)} chars), retrying with enhanced prompt")
                
                # Enhanced prompt for retry
                enhanced_messages = [
                    {"role": "system", "content": system_message + "\n\nIMPORTANT: Generate DETAILED and SPECIFIC content with facts, examples, and analysis. Do not generate vague placeholders or short summaries."},
                    {"role": "user", "content": f"""Generate DETAILED content for the following {'subsection' if is_subsection else 'section'} of a research report.

Main Query: {query}
Section Title: {section_title}
Section Description: {section.get('content', '')}

IMPORTANT: The documents below contain information you MUST extract and synthesize for this section.
Research Documents:
{context}

{sources_text}

{existing_content_context}

Please generate EXTENSIVE content that:
1. Is specifically about "{section_title}" in the context of "{query}"
2. Extracts and synthesizes concrete information from the provided documents
3. Includes specific details, facts, figures, examples, and analysis
4. Is comprehensive and thorough (at least 400-500 words)
5. Follows academic writing style
6. Avoids vague statements or placeholders
7. Presents a cohesive narrative on the topic

IMPORTANT: Your task is to GENERATE SPECIFIC CONTENT with actual information, not just a placeholder or overview.
Generate EXTENSIVE content with actual details and substance.

Generate the content:"""}
                ]
                
                # Retry with enhanced prompt
                retry_response = self.client.chat.completions.create(
                    model=BASE_MODEL,
                    messages=enhanced_messages,
                    temperature=0.7,
                    max_tokens=2000  # Even higher token limit for retry
                )
                
                retry_content = retry_response.choices[0].message.content.strip()
                
                # Use the retry content if it's longer than the original
                if len(retry_content) > len(content):
                    content = retry_content
                    logger.info(f"Used enhanced prompt content for {section_title} ({len(content)} chars)")
            
            return content
            
        except Exception as e:
            logger.error(f"Error generating section content: {str(e)}")
            return f"This section on {section_title} would contain information related to {query}. However, an error occurred during content generation: {str(e)}"

    def _rank_documents_for_section(self, documents: List[Document], section_title: str, query: str) -> List[Document]:
        """
        Rank documents based on their relevance to the section title and query.
        
        Args:
            documents: List of documents to rank
            section_title: Section title to use for relevance ranking
            query: Main research query
            
        Returns:
            Ranked list of documents
        """
        try:
            # Add basic error handling to avoid issues with malformed documents
            if not documents:
                return []
                
            # Create a ranking function that works with both Document objects and dict-style documents
            def get_doc_relevance(doc) -> float:
                relevance = 0.0
                
                # Extract content based on document type
                if hasattr(doc, 'page_content'):
                    content = doc.page_content.lower()
                    metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                elif isinstance(doc, dict):
                    content = doc.get('content', '').lower() if isinstance(doc.get('content', ''), str) else ''
                    metadata = doc.get('metadata', {})
                else:
                    # Unknown document type
                    return 0.0
                
                # Use section title words for relevance scoring
                section_words = section_title.lower().split()
                for word in section_words:
                    if len(word) > 3 and word in content:  # Skip short words
                        relevance += content.count(word) * 2  # Double weight for section title matches
                
                # Use query words for additional relevance
                query_words = query.lower().split()
                for word in query_words:
                    if len(word) > 4 and word in content:  # Skip short words
                        relevance += content.count(word)
                
                # Add relevance from metadata
                if isinstance(metadata, dict):
                    title = str(metadata.get('title', '')).lower()
                    if title:
                        for word in section_words:
                            if len(word) > 3 and word in title:
                                relevance += 5  # Title matches are very valuable
                    
                    # Check document type and adjust relevance
                    doc_type = str(metadata.get('type', '')).lower()
                    if doc_type in ['research paper', 'journal article', 'study']:
                        relevance += 3  # Prefer research sources
                    elif doc_type in ['book', 'textbook', 'encyclopedia']:
                        relevance += 2  # Academic sources are good
                    
                    # Consider recency if available
                    if 'date' in metadata or 'year' in metadata:
                        date_str = metadata.get('date', metadata.get('year', ''))
                        try:
                            # Extract year (assuming format like YYYY-MM-DD or just YYYY)
                            year_match = re.search(r'(\d{4})', str(date_str))
                            if year_match:
                                year = int(year_match.group(1))
                                current_year = datetime.now().year
                                if year > current_year - 5:  # If within last 5 years
                                    relevance += 2  # Recent content is valuable
                        except:
                            pass  # Ignore date parsing errors
                            
                # Length penalty for extremely short documents
                if len(content) < 100:
                    relevance *= 0.5
                    
                return relevance
            
            # Calculate relevance scores for each document
            doc_scores = [(doc, get_doc_relevance(doc)) for doc in documents]
            
            # Sort by relevance score (descending) and return documents
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            ranked_docs = [doc for doc, score in doc_scores]
            
            # Ensure at least one document is included regardless of score
            return ranked_docs
            
        except Exception as e:
            logger.error(f"Error ranking documents: {str(e)}")
            # Return original documents if ranking fails
            return documents

    def process_section(self, section: Dict[str, Any], kb, query: str) -> Dict[str, Any]:
        """
        Process a section and its subsections.
        
        Args:
            section: Section information
            kb: Knowledge base instance
            query: The research query
            
        Returns:
            Processed section with content
        """
        # Generate content for main section
        content = self.generate_section_content(section, kb, query)
        
        # Process subsections
        processed_subsections = []
        for subsection in section.get('subsections', []):
            subsection_content = self.generate_section_content(subsection, kb, query, is_subsection=True)
            processed_subsections.append({
                'subsection_title': subsection.get('subsection_title', ''),
                'content': subsection_content
            })
            
        return {
            'section_title': section.get('section_title', ''),
            'content': content,
            'subsections': processed_subsections
        }
    
    def clear_history(self):
        """Clear the chat history and other tracking data."""
        self.chat_history = []
        self.citation_map = {}
        self.used_citations = defaultdict(set)
        self.technical_terms = set()

    def get_relevant_documents_for_section(self, section_title: str, kb, query: str, max_docs: int = 15) -> List[Document]:
        """
        Get the most relevant documents for a specific section.
        
        Args:
            section_title: The section title to find documents for
            kb: Knowledge base instance
            query: The main research query
            max_docs: Maximum number of documents to return
            
        Returns:
            List of relevant documents for the section
        """
        logger.info(f"Finding relevant documents for section: {section_title}")
        
        # Create a unique cache key for this section
        cache_key = f"{query}::{section_title}"
        if cache_key in self.cached_documents and len(self.cached_documents[cache_key]) > 0:
            logger.info(f"Using cached documents for {section_title}")
            unique_docs = self.cached_documents[cache_key]
            return unique_docs[:max_docs] if len(unique_docs) > max_docs else unique_docs
        
        # Check if kb has documents before attempting search
        if not hasattr(kb, 'documents') or not kb.documents:
            logger.error(f"Knowledge base does not contain any documents for section: {section_title}")
            return []
        
        # Get relevant documents for this section using multiple search strategies
        relevant_docs = []
        
        # Get keywords for better search
        keywords = self.get_section_keywords(section_title, query)
        
        # Try multiple search strategies with error handling
        try:
            # Strategy 1: Use the section title directly with the original query
            search_query = f"{query} {section_title}"
            try:
                query_docs = kb.search(search_query, top_k=10)
                if query_docs:
                    relevant_docs.extend(query_docs)
                    logger.info(f"Strategy 1: Found {len(query_docs)} documents for {section_title}")
            except Exception as e:
                logger.error(f"Error in search strategy 1: {str(e)}")
            
            # Strategy 2: Use keywords for more targeted search
            if keywords:
                try:
                    all_keywords_query = " ".join(keywords)
                    keyword_docs = kb.search(all_keywords_query, top_k=15)
                    if keyword_docs:
                        relevant_docs.extend(keyword_docs)
                        logger.info(f"Strategy 2: Found {len(keyword_docs)} documents using keywords")
                except Exception as e:
                    logger.error(f"Error in search strategy 2: {str(e)}")
            
            # Strategy 3: If still no results, use just the section title
            if not relevant_docs:
                try:
                    title_docs = kb.search(section_title, top_k=15)
                    if title_docs:
                        relevant_docs.extend(title_docs)
                        logger.info(f"Strategy 3: Found {len(title_docs)} documents using section title")
                except Exception as e:
                    logger.error(f"Error in search strategy 3: {str(e)}")
            
            # Strategy 4: Last resort, use the main query
            if not relevant_docs:
                try:
                    fallback_docs = kb.search(query, top_k=max_docs)
                    if fallback_docs:
                        relevant_docs.extend(fallback_docs)
                        logger.info(f"Strategy 4: Found {len(fallback_docs)} documents using main query")
                except Exception as e:
                    logger.error(f"Error in search strategy 4: {str(e)}")
            
            # Handle case with no documents
            if not relevant_docs:
                logger.warning(f"No documents found for section: {section_title}")
                return []
            
            # Deduplicate documents by ID
            unique_docs = []
            seen_ids = set()
            
            for doc in relevant_docs:
                # Handle different document formats
                if hasattr(doc, 'metadata'):
                    doc_id = doc.metadata.get('id', '')
                    if doc_id and doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        unique_docs.append(doc)
                    elif not doc_id:  # If no ID, use content as ID
                        content_hash = hash(doc.page_content[:100] if hasattr(doc, 'page_content') else '')
                        if content_hash not in seen_ids and content_hash != hash(''):
                            seen_ids.add(content_hash)
                            unique_docs.append(doc)
                elif isinstance(doc, dict):
                    # Handle dictionary-style documents
                    doc_id = doc.get('id', '')
                    if doc_id and doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        unique_docs.append(doc)
                    elif not doc_id:  # If no ID, use content as ID
                        content_hash = hash(doc.get('content', '')[:100])
                        if content_hash not in seen_ids and content_hash != hash(''):
                            seen_ids.add(content_hash)
                            unique_docs.append(doc)
            
            # Re-rank documents based on relevance to section title
            if unique_docs and len(unique_docs) > 3:
                try:
                    ranked_docs = self._rank_documents_for_section(unique_docs, section_title, query)
                    unique_docs = ranked_docs
                except Exception as e:
                    logger.error(f"Error ranking documents: {str(e)}")
            
            # Cache the documents for future use
            self.cached_documents[cache_key] = unique_docs
            
            # Return top documents
            return unique_docs[:max_docs] if len(unique_docs) > max_docs else unique_docs
            
        except Exception as e:
            logger.error(f"Error getting documents for section {section_title}: {str(e)}")
            return []

def main():
    """Example usage of the ResearchAnswerGenerator."""
    try:
        # Example data
        query = "How effective are transformer-based models in biomedical text analysis?"
        outline = {
            "sections": [
                {
                    "Introduction": {
                        "description": "Overview of transformer models and their applications in biomedicine"
                    }
                },
                {
                    "Methodology": {
                        "description": "Approaches to evaluating transformer models",
                        "subsections": [
                            {
                                "Evaluation Metrics": "Metrics used to assess model performance"
                            },
                            {
                                "Datasets": "Common biomedical datasets used for evaluation"
                            }
                        ]
                    }
                },
                {
                    "Results": {
                        "description": "Performance analysis of transformer models",
                        "subsections": [
                            {
                                "Accuracy": "Quantitative performance metrics"
                            },
                            {
                                "Limitations": "Challenges and limitations identified"
                            }
                        ]
                    }
                }
            ]
        }
        
        # Example documents
        documents = [
            Document(
                page_content="Sample document content",
                metadata={
                    "title": "Sample Paper",
                    "url": "https://example.com",
                    "source": "web"
                }
            )
        ]
        
        # Generate answer
        generator = ResearchAnswerGenerator()
        answer = generator.generate_answer(query, documents)
        
        print("\n=== Research Answer ===")
        print(answer["answer"])
        print("\n=== Sources ===")
        for source in answer["sources"]:
            print(f"[{source['title']}] {source['url']}")
        print(f"\nConfidence: {answer['confidence']:.2f}")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 