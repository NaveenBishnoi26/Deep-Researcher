"""
Report generator module for creating structured research reports.
"""

import logging
from typing import Dict, List, Any, Optional
import openai
from backend.config import OPENAI_API_KEY, BASE_MODEL, MODEL_TEMPERATURE
from backend.research_answer import ResearchAnswerGenerator
import json
import os
from datetime import datetime
import re
from backend.knowledge_base import KnowledgeBase
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

class ReportGenerator:
    """Generates structured research reports from knowledge base content."""
    
    def __init__(self):
        """Initialize the report generator."""
        self.model = ChatOpenAI(
            model=BASE_MODEL,
            temperature=MODEL_TEMPERATURE,
            openai_api_key=OPENAI_API_KEY
        )
        self.research_generator = ResearchAnswerGenerator()
        self.max_sections = 4  # Tighter than before (was 5)
        self.max_subsections = 2  # Tighter than before (was 3)
        self.min_section_length = 200  # Minimum words per section
        self.max_section_length = 1000  # Maximum words per section
        self.min_citations = 2  # Minimum citations per section
        self.max_citations = 5  # Maximum citations per section
        self.existing_headers = []  # Track existing headers
        self.relevant_written_contents = []  # Track written content
        self.temperature = 0.3  # Lower temperature for more focused output
        self.max_tokens = 2000  # Maximum tokens per section
        self.max_words_per_section = 500  # Tighter than before (was 800)
        self.max_words_per_subsection = 250  # Tighter than before (was 400)
        self.total_max_words = 8000  # Absolute hard limit (≈ 20 pages)
        
    def generate_outline(self, query: str, kb) -> Dict[str, Any]:
        """
        Generate a structured outline for the research report.
        
        Args:
            query: The research query
            kb: Knowledge base instance
            
        Returns:
            Dictionary containing the report outline
        """
        # Get relevant documents for outline generation
        relevant_docs = kb.search(query, top_k=10)
        if not relevant_docs:
            logger.warning("No relevant documents found for outline generation")
            return {}
            
        # Prepare context from relevant documents
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Generate outline using LLM
        messages = [
            {"role": "system", "content": """You are a research report outline generator. Create a structured outline for a research report based on the provided documents and query. Follow IMRaD structure and include relevant subsections. Your response must be valid JSON.

IMPORTANT: The outline should follow these rules:
1. Use clear, descriptive section titles
2. Include relevant subsections for each main section
3. Maintain logical flow and organization
4. Cover all key aspects of the research topic
5. Follow IMRaD structure (Introduction, Methods, Results, and Discussion)
6. Include a clear introduction and conclusion
7. Use proper section hierarchy
8. Keep subsections focused and specific
9. Avoid redundant or overlapping sections
10. Ensure each section has a clear purpose"""},
            {"role": "user", "content": f"""Based on the following research documents and query, generate a structured outline for a research report.

Query: {query}

Research Documents:
{context}

Please create an outline that:
1. Follows IMRaD structure (Introduction, Methods, Results, and Discussion)
2. Includes relevant subsections for each main section
3. Maintains logical flow and organization
4. Covers all key aspects of the research topic
5. Is well-structured and hierarchical
6. Includes a clear introduction and conclusion
7. Has at most {self.max_sections} main sections
8. Has at most {self.max_subsections} subsections per section
9. Uses clear and descriptive section titles
10. Maintains academic writing style
11. Should be properly formatted
IMPORTANT: Your response must be valid JSON with the following structure:
{{
    "title": "Report Title",
    "sections": [
        {{
            "section_title": "Section Title",
            "content": "Brief description of section content",
            "subsections": [
                {{
                    "section_title": "Subsection Title",
                    "content": "Brief description of subsection content"
                }}
            ]
        }}
    ]
}}

Note: Use "section_title" consistently for both main sections and subsections to maintain proper hierarchy.

Generate the outline:"""}
        ]
        
        try:
            response = self.model.chat.completions.create(
                model=BASE_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            
            outline = json.loads(response.choices[0].message.content)
            return outline
            
        except Exception as e:
            logger.error(f"Error generating outline: {str(e)}")
            return {}
    
    def generate_report(self, outline: Dict[str, Any], kb) -> Dict[str, Any]:
        """Generate a complete research report from the outline."""
        try:
            processed_sections = []
            all_content = []
            all_citations = []  # Track all citations across sections
            
            # Reset tracking variables
            self.existing_headers = []
            self.relevant_written_contents = []
            
            # Process each section
            for section in outline.get('sections', []):
                # Ensure section has required fields
                if 'section_title' not in section:
                    section['section_title'] = section.get('title', 'Untitled Section')
                if 'content' not in section:
                    section['content'] = ''
                    
                # Process subsections
                if 'subsections' in section:
                    for subsection in section['subsections']:
                        if 'subsection_title' not in subsection:
                            subsection['subsection_title'] = subsection.get('title', 'Untitled Subsection')
                        if 'content' not in subsection:
                            subsection['content'] = ''
                
                try:
                    # Process section with context of existing content
                    processed_section = self.process_section(
                        section=section,
                        kb=kb,
                        query=outline.get('title', ''),
                        existing_headers=self.existing_headers,
                        relevant_written_contents=self.relevant_written_contents
                    )
                    processed_sections.append(processed_section)
                    
                    # Update tracking variables
                    self.existing_headers.append(processed_section.get('section_title', ''))
                    if processed_section.get('content'):
                        self.relevant_written_contents.append(processed_section['content'])
                    
                    # Collect content for technical term extraction
                    all_content.append(processed_section.get('content', ''))
                    for subsection in processed_section.get('subsections', []):
                        self.existing_headers.append(subsection.get('subsection_title', ''))
                        if subsection.get('content'):
                            self.relevant_written_contents.append(subsection['content'])
                            all_content.append(subsection['content'])
                        
                    # Collect citations from this section
                    if 'citations' in processed_section:
                        all_citations.extend(processed_section['citations'])
                        
                except Exception as e:
                    logger.error(f"Error processing section {section.get('section_title')}: {str(e)}")
                    continue
            
            # Extract technical terms from all content at once
            combined_content = "\n\n".join(all_content)
            technical_terms = self.research_generator.extract_all_technical_terms(combined_content)
            
            # Generate glossary from collected terms
            glossary = self._generate_glossary(technical_terms)
            
            # Remove duplicate citations while maintaining order
            unique_citations = []
            for citation in all_citations:
                if citation not in unique_citations:
                    unique_citations.append(citation)
            
            # Generate conclusion
            conclusion = self._generate_conclusion(combined_content)
            
            # Compile final report
            report = {
                'title': outline.get('title', 'Untitled Report'),
                'date': datetime.now().strftime('%Y-%m-%d'),
                'query': outline.get('title', ''),
                'sections': processed_sections,
                'glossary': glossary,
                'conclusion': conclusion,
                'references': unique_citations
            }
            
            # Enforce total word limit before returning
            report = self.enforce_total_word_limit(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            return {}
    
    def process_section(self, section: Dict[str, Any], kb: KnowledgeBase, report_title: str) -> Dict[str, Any]:
        """Process a section of the report."""
        try:
            section_title = section.get('section_title', 'Untitled Section')
            logger.info(f"Processing section: {section_title}")
            
            # Get relevant documents for this section
            relevant_docs = kb.get_relevant_documents(section_title, limit=5)
            
            # Convert documents to the correct format if needed
            processed_docs = []
            for doc in relevant_docs:
                if isinstance(doc, dict):
                    # If it's already a dictionary, ensure it has the right fields
                    if 'content' in doc:
                        processed_docs.append(doc)
                    elif 'page_content' in doc:
                        doc['content'] = doc['page_content']
                        processed_docs.append(doc)
                else:
                    # If it's a Document object, convert to dictionary
                    processed_docs.append({
                        'content': doc.page_content if hasattr(doc, 'page_content') else str(doc),
                        'metadata': doc.metadata if hasattr(doc, 'metadata') else {}
                    })
            
            # Generate section content
            section_content = self.generate_section_content(section_title, processed_docs, max_words=self.max_words_per_section)
            
            # Process subsections if they exist
            subsections = []
            if 'subsections' in section:
                for subsection in section['subsections']:
                    try:
                        subsection_title = subsection.get('subsection_title', 'Untitled Subsection')
                        subsection_content = self.generate_section_content(
                            f"{section_title} - {subsection_title}",
                            processed_docs,
                            max_words=self.max_words_per_subsection
                        )
                        subsections.append({
                            'subsection_title': subsection_title,
                            'content': subsection_content
                        })
                    except Exception as e:
                        logger.error(f"Error processing subsection {subsection_title}: {str(e)}")
                        subsections.append({
                            'subsection_title': subsection_title,
                            'content': f"Error generating content for this subsection: {str(e)}"
                        })
            
            return {
                'section_title': section_title,
                'content': section_content,
                'subsections': subsections
            }
            
        except Exception as e:
            logger.error(f"Error processing section {section_title}: {str(e)}")
            return {
                'section_title': section_title,
                'content': f"Error generating content for this section: {str(e)}",
                'subsections': []
            }
    
    def generate_section_content(self, section_title: str, documents: List[Dict[str, Any]], max_words: int = 500) -> str:
        """Generate content for a section based on relevant documents."""
        try:
            if not documents:
                return f"No specific information found for {section_title}."
            
            # Prepare context from documents
            context = "\n\n".join([doc.get('content', '') for doc in documents])
            
            # Create prompt for content generation
            prompt = f"""Generate a concise and informative section about {section_title} based on the following research:

{context}

Guidelines:
1. Write in an academic style
2. Focus on key findings and insights
3. Include relevant statistics and data points
4. Maintain objectivity and clarity
5. Keep the content under {max_words} words
6. Use proper citations when referencing specific information
7. Should be properly formatted
8. If section name is "Introduction", write the introduction for the report, don't include introduction if tile is not introduction..
9. If section name is "Conclusion", write the conclusion for the report, don't include conlusion if tile is not conclusion.
10. Remember point 6
Write the section content:"""
            
            # Generate content using LLM
            response = self.model.chat.completions.create(
                model=BASE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1000
            )
            content = response.choices[0].message.content.strip()
            
            # Ensure content doesn't exceed word limit
            words = content.split()
            if len(words) > max_words:
                content = ' '.join(words[:max_words]) + '...'
            
            return content
            
        except Exception as e:
            logger.error(f"Error generating content for section {section_title}: {str(e)}")
            return f"Error generating content for {section_title}: {str(e)}"
    
    def _extract_citations(self, content: str) -> List[Dict[str, str]]:
        """
        Extract citations from section content.
        
        Args:
            content: Section content
            
        Returns:
            List of citation dictionaries
        """
        citations = []
        citation_pattern = r'\(([^)]+\d{4}[^)]*)\)'
        matches = re.findall(citation_pattern, content)
        
        for match in matches:
            if match and len(match) > 5:  # Basic validation
                citations.append({"text": match})
        
        return citations
    
    def _generate_glossary(self, technical_terms: List[str]) -> List[Dict[str, str]]:
        """
        Generate a glossary from collected technical terms.
        
        Args:
            technical_terms: List of technical terms
            
        Returns:
            List of glossary entries
        """
        glossary = []
        for term in technical_terms:
            # Generate definition using LLM
            messages = [
                {"role": "system", "content": "You are a technical term definition generator. Generate clear and concise definitions for technical terms."},
                {"role": "user", "content": f"Generate a clear and concise definition for the technical term: {term}"}
            ]
            
            try:
                response = self.model.chat.completions.create(
                    model=BASE_MODEL,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=100
                )
                
                definition = response.choices[0].message.content.strip()
                glossary.append({
                    'term': term,
                    'definition': definition
                })
                
            except Exception as e:
                logger.error(f"Error generating definition for term {term}: {str(e)}")
                continue
                
        return glossary
    
    def _generate_references(self) -> List[Dict[str, str]]:
        """
        Generate a references list from collected citations.
        
        Returns:
            List of references
        """
        references = []
        for citation_text, citation_number in self.research_generator.citation_map.items():
            # Extract reference information using LLM
            messages = [
                {"role": "system", "content": "You are a reference formatter. Extract and format reference information from citation text."},
                {"role": "user", "content": f"Extract and format reference information from the following citation text:\n\n{citation_text}"}
            ]
            
            try:
                response = self.model.chat.completions.create(
                    model=BASE_MODEL,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=200
                )
                
                reference = response.choices[0].message.content.strip()
                references.append({
                    'number': citation_number,
                    'text': reference
                })
                
            except Exception as e:
                logger.error(f"Error generating reference for citation {citation_number}: {str(e)}")
                continue
                
        return sorted(references, key=lambda x: x['number'])
    
    def _generate_conclusion(self, content: str) -> str:
        """Generate a conclusion based on the report content."""
        try:
            response = self.model.chat.completions.create(
                model=BASE_MODEL,
                messages=[
                    {"role": "system", "content": """You are a research report generator. Generate a comprehensive conclusion that:
1. Summarizes the key findings
2. Synthesizes the main points
3. Provides final thoughts
4. Avoids introducing new information
5. Maintains academic tone
6. Is concise and focused
7. Should be properly formatted"""},
                    {"role": "user", "content": f"Generate a conclusion for the following research content:\n\n{content}"}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating conclusion: {str(e)}")
            return ""
    
    def export_report(self, report: Dict[str, Any], format: str = 'json', output_dir: str = 'reports') -> Optional[str]:
        """
        Export the report in the specified format.
        
        Args:
            report: The report to export
            format: Export format ('json' or 'docx')
            output_dir: Directory to save the exported file
            
        Returns:
            Path to the exported file if successful, None otherwise
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"report_{timestamp}"
        
        try:
            if format.lower() == 'json':
                output_path = os.path.join(output_dir, f"{filename}.json")
                with open(output_path, 'w') as f:
                    json.dump(report, f, indent=2)
                    
            elif format.lower() == 'docx':
                output_path = os.path.join(output_dir, f"{filename}.docx")
                # TODO: Implement DOCX export
                logger.warning("DOCX export not yet implemented")
                return None
                
            else:
                logger.error(f"Unsupported export format: {format}")
                return None
                
            return output_path
            
        except Exception as e:
            logger.error(f"Error exporting report: {str(e)}")
            return None
    
    def enforce_total_word_limit(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enforce the total word limit for the entire report.
        
        Args:
            report: The complete report dictionary
            
        Returns:
            Report with enforced word limits
        """
        try:
            total_words = 0
            sections = report.get('sections', [])
            
            # Count words in all sections and subsections
            for i, section in enumerate(sections):
                section_words = len(section.get('content', '').split())
                total_words += section_words
                
                # Count words in subsections
                subsections = section.get('subsections', [])
                for j, subsection in enumerate(subsections):
                    subsection_words = len(subsection.get('content', '').split())
                    total_words += subsection_words
            
            # If total words exceeds limit, trim content proportionally
            if total_words > self.total_max_words:
                # Calculate the ratio to scale all content
                scale_ratio = self.total_max_words / total_words
                
                logger.info(f"Report exceeds word limit ({total_words} > {self.total_max_words}). Scaling by {scale_ratio:.2f}")
                
                # Scale each section and subsection
                for i, section in enumerate(sections):
                    content = section.get('content', '')
                    words = content.split()
                    new_word_count = int(len(words) * scale_ratio)
                    sections[i]['content'] = ' '.join(words[:new_word_count])
                    
                    # Scale subsections
                    subsections = section.get('subsections', [])
                    for j, subsection in enumerate(subsections):
                        content = subsection.get('content', '')
                        words = content.split()
                        new_word_count = int(len(words) * scale_ratio)
                        sections[i]['subsections'][j]['content'] = ' '.join(words[:new_word_count])
            
            # Update the report with scaled sections
            report['sections'] = sections
            
            return report
            
        except Exception as e:
            logger.error(f"Error enforcing word limit: {str(e)}")
            return report

def main():
    """Example usage of the ReportGenerator."""
    from backend.knowledge_base import KnowledgeBase
    
    # Initialize knowledge base
    kb = KnowledgeBase()
    
    # Initialize report generator
    generator = ReportGenerator()
    
    # Generate report
    query = "What are the latest developments in quantum computing?"
    outline = generator.generate_outline(query, kb)
    report = generator.generate_report(outline, kb)
    
    # Export report
    if report:
        output_path = generator.export_report(report, format='json')
        if output_path:
            print(f"Report exported to: {output_path}")

if __name__ == "__main__":
    main() 