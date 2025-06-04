"""
Streamlit application for the research pipeline with enhanced capabilities.
"""

import os
import logging
import warnings
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import time
import concurrent.futures
import functools
import re

# Import our custom TF configuration
from backend.tensorflow_utils import configure_tensorflow_environment, initialize_tensorflow

# Configure TensorFlow environment before any other imports
configure_tensorflow_environment()

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from backend.config import (
    OPENAI_API_KEY,
    VECTORSTORE_DIR,
    VECTORSTORE_CONFIGS,
    EXPORTS_DIR
)
from backend.planner import ResearchPlanner
from backend.gather import DataGatherer
from backend.knowledge_base import KnowledgeBase
from backend.report_generator import ReportGenerator
from backend.export import ReportExporter
from backend.clarify import QueryClarifier
from backend.agents import AgentOrchestrator
from backend.context_manager import ContextManager
from backend.quality_control import QualityController

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Reduce logging level for specific loggers that produce warnings
logging.getLogger("googleapiclient").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("streamlit").setLevel(logging.ERROR)

# Initialize TensorFlow with completely disabled TensorFlow Lite
initialize_tensorflow()

# Set up performance monitoring
def timing_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time
        logger.info(f"Function {func.__name__} took {duration:.2f} seconds to run")
        return result
    return wrapper

class EnhancedResearchPipeline:
    """Enhanced research pipeline with multi-agent capabilities."""
    
    def __init__(self, model_name="gpt-4o-mini", temperature=0.7):
        """Initialize the enhanced research pipeline components."""
        try:
            # Initialize LLM
            self.llm = ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                streaming=True
            )
            
            # Initialize traditional components
            self.planner = ResearchPlanner()
            self.gatherer = DataGatherer()
            self.kb = KnowledgeBase()
            self.report_generator = ReportGenerator()
            self.exporter = ReportExporter()
            self.clarifier = QueryClarifier()
            
            # Initialize new components
            self.agent_orchestrator = AgentOrchestrator()
            self.context_manager = ContextManager()
            self.quality_controller = QualityController()
            
            # Initialize cache dictionaries
            self.outline_cache = {}
            self.topics_cache = {}
            self.documents_cache = {}
        except Exception as e:
            logger.error(f"Error initializing enhanced research pipeline: {str(e)}")
            raise

    async def process_report_sections(self, outline, kb):
        """Process report sections in parallel for better performance."""
        # Create base report structure
        report = {
            'title': outline['title'],
            'date': datetime.now().isoformat(),
            'sections': [],
            'citations': []
        }
        
        # Process sections in parallel if possible
        sections_to_process = outline.get('sections', [])
        processed_sections = []
        
        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(sections_to_process))) as executor:
            # Create a list of future tasks
            future_to_section = {
                executor.submit(self._process_section_with_llm, section, kb, outline['title']): section
                for section in sections_to_process
            }
            
            # Process completed tasks as they finish
            for future in concurrent.futures.as_completed(future_to_section):
                section = future_to_section[future]
                try:
                    processed_section = future.result()
                    processed_sections.append(processed_section)
                    if 'citations' in processed_section:
                        report['citations'].extend(processed_section.get('citations', []))
                except Exception as e:
                    logger.error(f"Error processing section {section.get('section_title')}: {str(e)}")
                    # Add a default section with placeholder content when processing fails
                    section_title = section.get('section_title', 'Untitled Section')
                    placeholder_section = {
                        "section_title": section_title,
                        "content": f"This section about {section_title} is being processed. Content will be available in the final report.",
                        "subsections": []
                    }
                    processed_sections.append(placeholder_section)
        
        # Sort sections to maintain original order
        section_map = {section.get('section_title'): i for i, section in enumerate(sections_to_process)}
        processed_sections.sort(key=lambda s: section_map.get(s.get('section_title'), 999))
        
        # Add processed sections to report
        report['sections'] = processed_sections
        
        # Ensure we have at least one section
        if not processed_sections:
            default_section = {
                "section_title": "Report Overview",
                "content": "This report is currently being generated. More detailed content will be available in the final version.",
                "subsections": []
            }
            report['sections'].append(default_section)
            
        return report

    def _process_section_with_llm(self, section, kb, report_title):
        """Process a single section using the LLM."""
        try:
            section_title = section.get('section_title', 'Untitled Section')
            is_conclusion = section_title.lower() == 'conclusion'
            
            system_prompt = f"""You are an expert research assistant. Generate focused content for the section '{section_title}' 
            in a research report titled '{report_title}'. Follow these strict guidelines:

            1. Focus ONLY on the specific topic of this section. Do not include general introductions or conclusions unless this is the Introduction or Conclusion section.
            2. Each paragraph should directly address the section's topic.
            3. Include specific references and citations for every claim or statement.
            4. Format references as [Author, Year] within the text.
            5. At the end of the section, include a "References" subsection with full citations in this format:
               Author, A. (Year). Title of the source. Journal/Publisher. URL: [full URL if available]
            6. Keep the content concise and focused on the section's specific topic.
            7. Use the provided knowledge base to inform your response.
            8. Format the content in markdown.
            
            {'For the Conclusion section, ensure to:' if is_conclusion else ''}
            {'- Summarize key findings from all sections' if is_conclusion else ''}
            {'- Provide clear, actionable conclusions' if is_conclusion else ''}
            {'- Include implications and future directions' if is_conclusion else ''}
            {'- Maintain proper citations for all claims' if is_conclusion else ''}
            
            Remember: Stay strictly focused on the section's topic while ensuring all references include complete information including URLs when available."""
            
            # Get relevant context from knowledge base using existing search method
            context = kb.search(section_title, top_k=5)
            context_text = "\n\n".join([doc.get('content', '') for doc in context])
            
            # Create messages for the LLM
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Generate focused content for the section '{section_title}' using this context: {context_text}")
            ]
            
            # Get response from LLM
            response = self.llm.invoke(messages)
            
            # Process the response
            processed_section = {
                "section_title": section_title,
                "content": response.content,
                "subsections": []
            }
            
            # Process subsections if they exist
            if 'subsections' in section:
                for subsection in section['subsections']:
                    subsection_title = subsection.get('subsection_title', 'Untitled Subsection')
                    is_conclusion_subsection = subsection_title.lower() == 'conclusion'
                    
                    subsection_prompt = f"""Generate focused content for the subsection '{subsection_title}' 
                    under the section '{section_title}' in the report '{report_title}'. Follow these strict guidelines:

                    1. Focus ONLY on the specific topic of this subsection.
                    2. Each paragraph should directly address the subsection's topic.
                    3. Include specific references and citations for every claim or statement.
                    4. Format references as [Author, Year] within the text.
                    5. At the end of the subsection, include a "References" section with full citations in this format:
                       Author, A. (Year). Title of the source. Journal/Publisher. URL: [full URL if available]
                    6. Keep the content concise and focused on the subsection's specific topic.
                    7. Use the provided knowledge base to inform your response.
                    8. Format the content in markdown.
                    
                    {'For the Conclusion subsection, ensure to:' if is_conclusion_subsection else ''}
                    {'- Summarize key findings from all sections' if is_conclusion_subsection else ''}
                    {'- Provide clear, actionable conclusions' if is_conclusion_subsection else ''}
                    {'- Include implications and future directions' if is_conclusion_subsection else ''}
                    {'- Maintain proper citations for all claims' if is_conclusion_subsection else ''}
                    
                    Remember: Stay strictly focused on the subsection's topic while ensuring all references include complete information including URLs when available."""
                    
                    # Get relevant context for subsection using search
                    subsection_context = kb.search(subsection_title, top_k=3)
                    subsection_context_text = "\n\n".join([doc.get('content', '') for doc in subsection_context])
                    
                    subsection_messages = [
                        SystemMessage(content=subsection_prompt),
                        HumanMessage(content=f"Generate focused content for the subsection '{subsection_title}' using this context: {subsection_context_text}")
                    ]
                    
                    subsection_response = self.llm.invoke(subsection_messages)
                    
                    processed_section['subsections'].append({
                        "subsection_title": subsection_title,
                        "content": subsection_response.content
                    })
            
            return processed_section
            
        except Exception as e:
            logger.error(f"Error generating content for section {section.get('section_title')}: {str(e)}")
            raise

    async def process_query(self, query: str, progress_bar) -> Dict[str, Any]:
        """
        Process a research query through the enhanced pipeline.
        
        Args:
            query: The research query to process
            progress_bar: Streamlit progress bar object
            
        Returns:
            Dictionary containing the results of the pipeline
        """
        start_time = time.time()
        
        try:
            results = {
                "query": query,
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "outputs": {},
                "performance_metrics": {}
            }
            
            # Initialize context
            self.context_manager.update_context(query, {"status": "started"})
            
            # Step 1: Clarify query (20% of progress)
            clarify_start = time.time()
            progress_bar.progress(0)
            st.write("🔍 Clarifying your query...")
            clarified_query, is_complete, questions = self.clarifier.clarify_query(query)
            
            if not is_complete:
                st.warning("Query needs more clarification. Please provide more details about:")
                for q in questions:
                    st.write(f"- {q}")
                return {
                    "query": query,
                    "status": "needs_clarification",
                    "timestamp": datetime.now().isoformat(),
                    "questions": questions
                }
            
            results["outputs"]["clarified_query"] = clarified_query
            clarify_time = time.time() - clarify_start
            results["performance_metrics"]["clarify_time"] = clarify_time
            progress_bar.progress(20)
            
            # Step 2: Generate research outline (40% of progress)
            outline_start = time.time()
            st.write("📝 Generating research outline...")
            
            # Check cache for outline
            cache_key = f"outline_{hash(clarified_query)}"
            if cache_key in self.outline_cache:
                st.info("Using cached outline...")
                outline = self.outline_cache[cache_key]
            else:
                outline = self.planner.generate_outline(clarified_query)
                # Store in cache
                self.outline_cache[cache_key] = outline
                
            results["outputs"]["outline"] = outline
            outline_time = time.time() - outline_start
            results["performance_metrics"]["outline_time"] = outline_time
            progress_bar.progress(40)
            
            # Step 3: Generate research topics (60% of progress)
            topics_start = time.time()
            st.write("🔍 Generating research topics...")
            
            # Check cache for topics
            cache_key = f"topics_{hash(str(outline))}"
            if cache_key in self.topics_cache:
                st.info("Using cached research topics...")
                topics = self.topics_cache[cache_key]
            else:
                topics = self.planner.generate_research_topics(outline)
                # Store in cache
                self.topics_cache[cache_key] = topics
                
            # Add original query to topics for document filtering
            topics["original_query"] = clarified_query
            topics_time = time.time() - topics_start
            results["performance_metrics"]["topics_time"] = topics_time
            progress_bar.progress(60)
            
            # Step 4: Gather and filter documents (80% of progress)
            gather_start = time.time()
            st.write("📚 Gathering and filtering documents...")
            try:
                # Use the enhanced data gathering method with filtering
                relevance_threshold = 0.65  # Increased threshold for better quality
                
                documents = await self.gatherer.gather_data_with_filtering(
                    topics, 
                    self.kb, 
                    relevance_threshold
                )
                
                # Print document statistics
                if documents:
                    # Calculate average relevance score
                    avg_score = sum(doc.get('relevance_score', 0) for doc in documents) / len(documents)
                    st.info(f"Average document relevance score: {avg_score:.2f}")
                    
                    # Count documents by source
                    sources = {}
                    for doc in documents:
                        source = doc.get('metadata', {}).get('source', 'Unknown')
                        sources[source] = sources.get(source, 0) + 1
                    st.info(f"Documents by source: {sources}")
                    
                    # Store documents in knowledge base
                    self.kb.add_documents(documents)
                    st.info(f"Added {len(documents)} filtered documents to knowledge base")
                else:
                    st.warning("No relevant documents found")
            except Exception as e:
                logger.error(f"Error during document gathering: {str(e)}")
                st.warning("Continuing with available documents...")
                
            gather_time = time.time() - gather_start
            results["performance_metrics"]["gather_time"] = gather_time
            progress_bar.progress(80)
            
            # Step 5: Generate final report (100% of progress)
            report_start = time.time()
            st.write("📊 Generating comprehensive report...")
            
            # Process sections in parallel
            report = await self.process_report_sections(outline, self.kb)
            
            results["outputs"]["report"] = report
            report_time = time.time() - report_start
            results["performance_metrics"]["report_time"] = report_time
            progress_bar.progress(100)
            
            # Export report to multiple formats
            try:
                # Create a safe filename
                unsafe_filename = report['title'].lower()
                safe_filename = re.sub(r'[^\w\s-]', '', unsafe_filename)
                safe_filename = re.sub(r'[-\s]+', '_', safe_filename)
                safe_filename = safe_filename[:50]  # Maximum 50 characters
                
                # Format report content for export
                report_content = f"# {report['title']}\n\n"
                for section in report['sections']:
                    report_content += f"## {section['section_title']}\n\n"
                    if 'content' in section:
                        report_content += f"{section['content']}\n\n"
                    if 'subsections' in section:
                        for subsection in section['subsections']:
                            report_content += f"### {subsection.get('subsection_title', 'Untitled Subsection')}\n\n"
                            if 'content' in subsection:
                                report_content += f"{subsection['content']}\n\n"
                
                # Ensure citations are properly formatted
                citations = report.get('citations', [])
                valid_citations = []
                
                for citation in citations:
                    if not isinstance(citation, dict):
                        valid_citations.append({"text": str(citation) if citation else "Unknown citation"})
                    else:
                        if not citation.get('text'):
                            text = citation.get('title', 'Untitled')
                            if 'authors' in citation:
                                text = f"{text} by {citation['authors']}"
                            if 'url' in citation:
                                text = f"{text} - {citation['url']}"
                            citation['text'] = text
                        valid_citations.append(citation)
                
                # Export to different formats
                pdf_path = self.exporter.export_pdf(report_content, valid_citations, safe_filename)
                docx_path = self.exporter.export_docx(report, safe_filename)
                html_path = self.exporter.export_html(report, safe_filename)
                
                results["outputs"]["export_paths"] = {
                    "pdf": pdf_path,
                    "docx": docx_path,
                    "html": html_path
                }
                
            except Exception as e:
                logger.error(f"Error exporting report: {str(e)}")
                st.error(f"Error exporting report: {str(e)}")
            
            # Update final context
            self.context_manager.update_context(
                query,
                {"final_results": results},
                {"stage": "completed"}
            )
            
            # Calculate and display total time
            total_time = time.time() - start_time
            results["performance_metrics"]["total_time"] = total_time
            
            st.success(f"Research completed successfully in {total_time:.2f} seconds!")
            st.write("\nPerformance breakdown:")
            st.write(f"- Query clarification: {clarify_time:.2f}s ({(clarify_time/total_time)*100:.1f}%)")
            st.write(f"- Outline generation: {outline_time:.2f}s ({(outline_time/total_time)*100:.1f}%)")
            st.write(f"- Topics generation: {topics_time:.2f}s ({(topics_time/total_time)*100:.1f}%)")
            st.write(f"- Document gathering: {gather_time:.2f}s ({(gather_time/total_time)*100:.1f}%)")
            st.write(f"- Report generation: {report_time:.2f}s ({(report_time/total_time)*100:.1f}%)")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in enhanced research pipeline: {str(e)}")
            return {
                "query": query,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

@st.cache_resource
def get_pipeline(model_name="gpt-4o-mini", temperature=0.7):
    """Get or create the research pipeline instance with specified model settings."""
    return EnhancedResearchPipeline(model_name=model_name, temperature=temperature)

def main():
    """Main application entry point."""
    st.title("Enhanced Research Pipeline")
    
    # Initialize pipeline
    pipeline = get_pipeline()
    
    # Query input
    query = st.text_input("Enter your research query:")
    
    if query:
        # Create progress bar
        progress_bar = st.progress(0)
        
        # Process query
        results = asyncio.run(pipeline.process_query(query, progress_bar))
        
        # Display results
        if results["status"] == "success":
            st.success("Research completed successfully!")
            
            # Display report
            if "report" in results["outputs"]:
                st.subheader("Research Report")
                report = results["outputs"]["report"]
                st.write(f"# {report['title']}")
                
                for section in report["sections"]:
                    st.write(f"## {section['section_title']}")
                    if 'content' in section:
                        st.write(section['content'])
                    if 'subsections' in section:
                        for subsection in section['subsections']:
                            st.write(f"### {subsection.get('subsection_title', 'Untitled Subsection')}")
                            if 'content' in subsection:
                                st.write(subsection['content'])
            
            # Display export links
            if "export_paths" in results["outputs"]:
                st.subheader("Download Report")
                export_paths = results["outputs"]["export_paths"]
                for format, path in export_paths.items():
                    if os.path.exists(path):
                        with open(path, "rb") as f:
                            st.download_button(
                                f"Download {format.upper()}",
                                f,
                                file_name=os.path.basename(path),
                                mime=f"application/{format}"
                            )
            
            # Display performance metrics
            st.subheader("Performance Metrics")
            metrics = results["performance_metrics"]
            for metric, value in metrics.items():
                st.write(f"{metric}: {value:.2f} seconds")
        else:
            st.error(f"Error: {results.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main() 