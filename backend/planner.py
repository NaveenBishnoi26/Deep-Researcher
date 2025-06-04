"""
Research planning module for the Deep Researcher application.
"""

from typing import List, Dict, Any
import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from .config import BASE_MODEL, MODEL_TEMPERATURE, OPENAI_API_KEY
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResearchOutline(BaseModel):
    """Schema for research outline output."""
    title: str = Field(description="Clear and focused title of the research")
    sections: List[Dict[str, Any]] = Field(
        description="List of research sections, each containing a title and subsections",
        min_items=3,
        max_items=12  # Increased to accommodate all required sections plus content
    )
    key_questions: List[str] = Field(
        description="3-5 key questions that guide the research",
        min_items=3,
        max_items=5
    )
    required_sources: List[str] = Field(
        description="2-3 types of sources needed for the research",
        min_items=2,
        max_items=3
    )

    class Config:
        json_schema_extra = {
            "example": {
                "title": "Impact of Quantum Computing on Cryptography",
                "sections": [
                    {
                        "section_title": "Introduction",
                        "subsections": [
                            {"subsection_title": "Basic Principles", "content": "Overview of quantum computing fundamentals"},
                            {"subsection_title": "Current State", "content": "Recent developments and capabilities"}
                        ]
                    },
                    {
                        "section_title": "Applications",
                        "subsections": [
                            {"content": "Practical applications and use cases"}
                        ]
                    }
                ],
                "key_questions": [
                    "How does quantum computing threaten current cryptographic systems?",
                    "What are the most promising post-quantum cryptographic solutions?"
                ],
                "required_sources": [
                    "Academic papers",
                    "Industry reports"
                ]
            }
        }

class ResearchTopics(BaseModel):
    """Schema for research topics output."""
    topics_by_section: Dict[str, List[str]] = Field(description="Dictionary mapping outline sections to their specific research topics")
    search_queries: Dict[str, str] = Field(description="Dictionary mapping topics to their optimized search queries")

class ResearchPlanner:
    """Handles research planning and outline generation."""
    
    def __init__(self):
        """
        Initialize the research planner.
        Uses the dynamically selected research model and temperature.
        """
        self.model = ChatOpenAI(
            model=BASE_MODEL,
            temperature=MODEL_TEMPERATURE,
            openai_api_key=OPENAI_API_KEY
        )
        self.outline_parser = PydanticOutputParser(pydantic_object=ResearchOutline)
        self.topics_parser = PydanticOutputParser(pydantic_object=ResearchTopics)
        
        self.outline_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a research assistant helping to create a comprehensive research outline.
            Create a detailed outline that includes:
            1. A clear, concise title
            2. Well-structured sections and subsections (maximum 12 sections total)
            3. Key questions to answer
            4. Types of sources needed
            
            Guidelines:
            - Keep the title focused and specific
            - Organize sections in a logical flow
            - Each section should have a clear purpose
            - Avoid repeating similar content across sections
            - Make subsections specific and actionable
            - Ensure each section builds upon previous ones
            - Include 3-5 key questions that guide the research
            - Specify 2-3 types of sources needed
            
            Required sections (must be included in this order):
            1. Introduction (at the beginning)
            2. [Content sections - add as needed]
            3. Applications (if relevant)
            4. Conclusion (must be the final section)
            
            The Conclusion section must include:
            - Summary of key findings
            - Synthesis of main points
            - Implications and future directions
            - Final thoughts and recommendations
            
            Note: The total number of sections should not exceed 12, including the required sections.
            Plan your content sections accordingly to stay within this limit.
            
            {format_instructions}
            """),
            ("human", "Research topic: {query}")
        ])

        self.topics_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a research assistant helping to identify specific topics for data gathering.
            Based on the research query and outline, create a list of specific topics for each section that need to be researched.
            For each topic, create an optimized search query that will yield the most relevant results.
            
            The output should be organized by sections from the outline, with each section containing:
            1. A list of specific research topics (maximum 3 per section)
            2. For each topic, a comprehensive search query that combines multiple aspects
            
            Guidelines:
            - Focus on creating 3 comprehensive topics per section that cover all important aspects
            - Each search query should be comprehensive and combine multiple related aspects
            - Make sure topics align with the section's focus and purpose
            - Include key technical terms, concepts, and recent developments in search queries
            - Combine related subtopics into broader, more comprehensive topics
            - Ensure each topic has a corresponding search query with the exact same name
            - Keep topic names concise but descriptive
            - For sections with subsections, combine related subsection topics into comprehensive topics
            - Do not create nested structures in the topics_by_section dictionary
            - Each search query should be comprehensive enough to cover multiple related aspects
            
            Example of a comprehensive search query:
            "Recent advances in quantum computing hardware, algorithms, and applications in cryptography and machine learning"
            
            {format_instructions}
            """),
            ("human", """Research query: {query}
            Research outline: {outline_json}
            """)
        ])
    
    def generate_outline(self, query: str) -> Dict[str, Any]:
        """
        Generate a research outline for the given query.
        
        Args:
            query: Research query to generate outline for
            
        Returns:
            Dictionary containing the research outline
        """
        try:
            # Format the prompt with the query and parser instructions
            prompt = self.outline_prompt.format_messages(
                query=query,
                format_instructions=self.outline_parser.get_format_instructions()
            )
            
            logger.info("Sending prompt to LLM for outline generation...")
            logger.info(f"Using model: {self.model.model_name} with temperature: {self.model.temperature}")
            logger.info(f"Prompt: {prompt}")
            
            # Get response from LLM
            response = self.model.invoke(prompt)
            logger.info("Received response from LLM:")
            logger.info(f"Raw response: {response.content}")
            
            # Parse the response into structured format
            outline = self.outline_parser.parse(response.content)
            logger.info("Parsed outline:")
            logger.info(f"Title: {outline.title}")
            logger.info(f"Sections: {outline.sections}")
            logger.info(f"Key questions: {outline.key_questions}")
            logger.info(f"Required sources: {outline.required_sources}")
            
            # Convert to dictionary for easier handling
            return outline.model_dump()
            
        except Exception as e:
            logger.error(f"Error generating outline: {str(e)}")
            raise

    def generate_research_topics(self, outline: dict) -> dict:
        """
        Generate research topics and search queries based on the outline.
        
        Args:
            outline: Research outline dictionary
            
        Returns:
            Dictionary containing topics and search queries
        """
        try:
            # Format the prompt with the query, outline, and parser instructions
            prompt = self.topics_prompt.format_messages(
                query=outline.get('title', ''),
                outline_json=json.dumps(outline),
                format_instructions=self.topics_parser.get_format_instructions()
            )
            
            logger.info("Sending prompt to LLM for topic generation...")
            logger.info(f"Using model: {self.model.model_name} with temperature: {self.model.temperature}")
            
            # Get response from LLM
            response = self.model.invoke(prompt)
            logger.info("Received response from LLM")
            
            # Parse the response into structured format
            topics = self.topics_parser.parse(response.content)
            logger.info("Parsed topics:")
            logger.info(f"Topics by section: {topics.topics_by_section}")
            logger.info(f"Search queries: {topics.search_queries}")
            
            # Convert to dictionary and add topics list
            topics_dict = topics.model_dump()
            topics_dict['topics'] = []
            
            # Create a flat list of topics with their queries
            for section, section_topics in topics_dict['topics_by_section'].items():
                for topic in section_topics:
                    if topic in topics_dict['search_queries']:
                        topics_dict['topics'].append({
                            'title': topic,
                            'query': topics_dict['search_queries'][topic],
                            'section': section
                        })
            
            return topics_dict
            
        except Exception as e:
            logger.error(f"Error generating research topics: {str(e)}")
            raise

    def clarify_query(self, query: str) -> str:
        """
        Clarify and expand the research query.
        
        Args:
            query: The original research query
            
        Returns:
            Clarified and expanded query
        """
        try:
            # Create prompt template for query clarification
            clarify_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a research assistant helping to clarify and expand research queries.
Consider:
1. Specific aspects or subtopics that should be explored
2. Any temporal constraints (e.g., recent developments, historical context)
3. Technical depth required
4. Relevant applications or use cases
5. Current state of research in this area

Provide a detailed, well-structured query that will guide the research process."""),
                ("human", "Original query: {query}")
            ])
            
            # Format the prompt with the query
            prompt = clarify_prompt.format_messages(query=query)
            
            # Get clarified query from LLM
            response = self.model.invoke(prompt)
            
            # Extract and clean the clarified query
            clarified_query = response.content.strip()
            logger.info(f"Clarified query: {clarified_query}")
            
            return clarified_query
            
        except Exception as e:
            logger.error(f"Error clarifying query: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    planner = ResearchPlanner()
    outline = planner.generate_outline("What's the latest in quantum computing?")
    topics = planner.generate_research_topics(outline)
    print("Outline:", outline)
    print("Topics:", topics) 