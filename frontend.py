"""
Frontend for the Enhanced Research Pipeline with a beautiful, responsive UI.
"""

import os
import sys
import time
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

# Configure import paths correctly
# Get the absolute path of the current file and add the final-ver directory to the Python path
file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(file_path)
final_ver_dir = os.path.join(parent_dir, "final-ver")

# Add both the current directory and final-ver directory to the Python path
sys.path.insert(0, parent_dir)
sys.path.insert(0, final_ver_dir)

# Try importing from app, if it fails use dummy implementations
try:
    import streamlit as st
    from streamlit.runtime.scriptrunner import add_script_run_ctx
    from langchain_openai import ChatOpenAI  # Updated import
    from app import get_pipeline, EnhancedResearchPipeline
    logger = logging.getLogger(__name__)
    logger.info("Successfully imported get_pipeline from app")
except ImportError as e:
    import streamlit as st
    from streamlit.runtime.scriptrunner import add_script_run_ctx
    # Fall back to dummy implementation
    class DummyEnhancedResearchPipeline:
        def __init__(self):
            self.planner = type('obj', (object,), {
                'llm': None,
                'generate_outline': lambda x: {'title': 'Sample Outline', 'sections': []},
                'generate_research_topics': lambda x: {'topics': []}
            })
            self.gatherer = type('obj', (object,), {
                'llm': None,
                'gather_data_with_filtering': lambda x, y, z: []
            })
            self.clarifier = type('obj', (object,), {
                'llm': None,
                'clarify_query': lambda x: (x, True, [])
            })
            self.kb = type('obj', (object,), {
                'add_documents': lambda x: None
            })
            self.report_generator = type('obj', (object,), {
                'process_section': lambda x, y, z: {'section_title': x, 'content': 'Sample content'}
            })
            self.exporter = type('obj', (object,), {
                'export_pdf': lambda x, y, z: 'dummy.pdf',
                'export_docx': lambda x, y: 'dummy.docx',
                'export_html': lambda x, y: 'dummy.html'
            })

        async def process_query(self, query, progress_bar):
            # Simulate a real pipeline with realistic timing
            await asyncio.sleep(1.5)
            progress_bar.progress(20)
            
            await asyncio.sleep(2)
            progress_bar.progress(40)
            
            await asyncio.sleep(1.5)
            progress_bar.progress(60)
            
            await asyncio.sleep(3)
            progress_bar.progress(80)
            
            await asyncio.sleep(2.5)
            progress_bar.progress(90)
            
            await asyncio.sleep(1.5)
            progress_bar.progress(100)
            
            return {
                "query": query,
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "outputs": {
                    "documents_count": 42,
                    "avg_relevance": 0.78,
                    "report": {
                        "title": "Analysis of " + query,
                        "sections": [
                            {
                                "section_title": "Introduction",
                                "content": "This is a sample introduction to " + query,
                                "subsections": []
                            }
                        ]
                    },
                    "export_paths": {
                        "pdf": "demo_report.pdf",
                        "docx": "demo_report.docx",
                        "html": "demo_report.html"
                    }
                },
                "performance_metrics": {
                    "clarify_time": 1.5,
                    "outline_time": 2.0,
                    "topics_time": 1.5,
                    "gather_time": 3.0,
                    "report_time": 2.5,
                    "export_time": 1.5,
                    "total_time": 12.0
                }
            }

    get_pipeline = lambda model_name="gpt-4o-mini", temperature=0.7: DummyEnhancedResearchPipeline()
    EnhancedResearchPipeline = DummyEnhancedResearchPipeline
    logger = logging.getLogger(__name__)
    logger.warning(f"Error importing from app.py: {e}. Using dummy implementation.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Streamlit page configuration
st.set_page_config(
    page_title="Deep Researcher",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Custom theme variables */
    :root {
        --primary-color: #7B5CFF;
        --secondary-color: #5B3FFF;
        --accent-color: #9F85FF;
        --background-color: #F9F9FB;
        --secondary-bg: #FFFFFF;
        --text-color: #0F0F0F;
        --font: 'Inter', sans-serif;
        --card-shadow: 0 4px 12px rgba(15,23,42,0.08);
        --border-radius: 12px;
    }
    
    /* Base styling */
    .main {
        background-color: var(--background-color);
        color: var(--text-color);
        font-family: var(--font);
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: var(--secondary-bg);
    }
    
    /* Card styling */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #ECECF1;
        border-radius: var(--border-radius);
        padding: 18px;
        box-shadow: var(--card-shadow);
        height: 100%;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(15,23,42,0.12);
    }
    .metric-label {
        font-size: 0.85rem;
        color: #6B7280;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 8px;
        color: var(--primary-color);
    }
    
    /* Status indicators */
    .status-indicator {
        display: flex;
        align-items: center;
        margin-bottom: 14px;
        padding: 8px 12px;
        border-radius: 8px;
        transition: background-color 0.2s ease;
    }
    .status-indicator:hover {
        background-color: rgba(0,0,0,0.03);
    }
    .status-bullet {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 10px;
    }
    .status-grey {
        background-color: #D1D5DB;
    }
    .status-blue {
        background-color: #3B82F6;
        box-shadow: 0 0 0 4px rgba(59,130,246,0.2);
    }
    .status-green {
        background-color: #10B981;
        box-shadow: 0 0 0 4px rgba(16,185,129,0.2);
    }
    .status-red {
        background-color: #EF4444;
        box-shadow: 0 0 0 4px rgba(239,68,68,0.2);
    }
    .status-text {
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    /* Button styling */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    .primary-button {
        background-color: var(--primary-color);
        color: white;
    }
    
    /* Report preview styling */
    .report-section {
        margin-bottom: 24px;
        padding: 20px;
        background: white;
        border-radius: var(--border-radius);
        box-shadow: var(--card-shadow);
    }
    .report-section h1 {
        font-size: 1.8rem;
        margin-bottom: 16px;
        color: var(--primary-color);
        border-bottom: 2px solid var(--accent-color);
        padding-bottom: 8px;
    }
    .report-section h2 {
        font-size: 1.5rem;
        margin-bottom: 12px;
        color: var(--secondary-color);
    }
    .report-section h3 {
        font-size: 1.2rem;
        margin-bottom: 8px;
        color: var(--text-color);
    }
    
    /* Download buttons */
    .download-button {
        margin-bottom: 12px;
    }
    
    /* Custom tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 8px 8px 0 0;
        font-weight: 500;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: var(--primary-color);
    }
    
    /* Logo styling */
    .app-logo {
        display: flex;
        align-items: center;
        margin-bottom: 20px;
    }
    .app-logo-text {
        font-size: 1.5rem;
        font-weight: 700;
        margin-left: 10px;
        background: linear-gradient(120deg, var(--primary-color), var(--secondary-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Hero section */
    .hero-section {
        text-align: center;
        padding: 20px;
        margin-bottom: 30px;
        background: linear-gradient(120deg, rgba(123, 92, 255, 0.1), rgba(159, 133, 255, 0.1));
        border-radius: var(--border-radius);
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 10px;
        color: var(--primary-color);
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: #6B7280;
        margin-bottom: 20px;
    }
    
    /* Query input styling */
    .query-container {
        background: white;
        padding: 24px;
        border-radius: var(--border-radius);
        box-shadow: var(--card-shadow);
        margin-bottom: 30px;
    }
    
    /* Progress bar styling */
    .stProgress > div > div {
        background-color: var(--primary-color);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    if "query" not in st.session_state:
        st.session_state.query = ""
    if "research_started" not in st.session_state:
        st.session_state.research_started = False
    if "research_completed" not in st.session_state:
        st.session_state.research_completed = False
    if "current_stage" not in st.session_state:
        st.session_state.current_stage = "idle"
    if "progress" not in st.session_state:
        st.session_state.progress = 0
    if "start_time" not in st.session_state:
        st.session_state.start_time = None
    if "end_time" not in st.session_state:
        st.session_state.end_time = None
    if "log_messages" not in st.session_state:
        st.session_state.log_messages = []
    if "results" not in st.session_state:
        st.session_state.results = None
    if "error" not in st.session_state:
        st.session_state.error = None
    if "metrics" not in st.session_state:
        st.session_state.metrics = {}
    if "documents_count" not in st.session_state:
        st.session_state.documents_count = 0
    if "avg_relevance" not in st.session_state:
        st.session_state.avg_relevance = 0.0
    if "model_name" not in st.session_state:
        st.session_state.model_name = "gpt-4o-mini"
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.7
    # Add new state variables for query clarification
    if "clarification_questions" not in st.session_state:
        st.session_state.clarification_questions = []
    if "needs_clarification" not in st.session_state:
        st.session_state.needs_clarification = False
    if "clarified_query" not in st.session_state:
        st.session_state.clarified_query = None

# Custom progress bar callback
class StreamlitProgressBar:
    def __init__(self, progress_placeholder):
        self.progress_placeholder = progress_placeholder
        
    def progress(self, value):
        st.session_state.progress = value
        self.progress_placeholder.progress(value / 100)

# Log message handler
def add_log_message(message: str):
    """Add a log message to the session state"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.log_messages.append(f"[{timestamp}] {message}")

# Clear session state
def clear_session():
    """Clear the session state"""
    for key in list(st.session_state.keys()):
        if key != "pipeline":  # Keep the pipeline instance
            del st.session_state[key]
    init_session_state()
    st.rerun()  # Using st.rerun() instead of experimental_rerun

# Update stage
def update_stage(stage: str, error: bool = False):
    """Update the current stage"""
    st.session_state.current_stage = stage
    if error:
        st.session_state.error = True
    add_log_message(f"Stage changed to: {stage}")

# Process the research query
async def process_research_query(pipeline: EnhancedResearchPipeline, query: str, progress_bar, 
                                relevance_threshold: float, output_formats: List[str]):
    """Process the research query asynchronously"""
    try:
        # Update state
        st.session_state.research_started = True
        st.session_state.start_time = datetime.now()
        st.session_state.research_completed = False
        st.session_state.error = False
        
        # Stage 1: Clarification (20%)
        update_stage("Clarifying")
        progress_bar.progress(10)
        await asyncio.sleep(0.5)  # Small delay for UI responsiveness
        
        # Check if query needs clarification
        clarified_query, is_complete, questions = pipeline.clarifier.clarify_query(query)
        
        if not is_complete:
            st.session_state.needs_clarification = True
            st.session_state.clarification_questions = questions
            st.session_state.clarified_query = None
            progress_bar.progress(0)  # Reset progress
            add_log_message("Query needs clarification. Please provide more details.")
            return {
                "query": query,
                "status": "needs_clarification",
                "questions": questions,
                "timestamp": datetime.now().isoformat()
            }
        
        # If query is clarified, update the query and log it
        st.session_state.clarified_query = clarified_query
        st.session_state.needs_clarification = False
        st.session_state.clarification_questions = []
        
        # Log the clarification process
        if "Original Query:" in query:
            # This is a combined context from clarification
            original_query = query.split("Original Query:")[1].split("\n\n")[0].strip()
            add_log_message(f"Original query: {original_query}")
            add_log_message(f"Query clarified with additional context: {clarified_query}")
        else:
            add_log_message(f"Query clarified: {clarified_query}")
            
        progress_bar.progress(20)
        
        # Continue with the rest of the pipeline
        try:
            # Stage 2: Outlining (40%)
            update_stage("Outlining")
            progress_bar.progress(30)
            await asyncio.sleep(0.5)
            progress_bar.progress(40)
            
            # Stage 3: Topic Mining (60%)
            update_stage("Topic Mining")
            progress_bar.progress(50)
            await asyncio.sleep(0.5)
            progress_bar.progress(60)
            
            # Stage 4: Gathering Docs (70%)
            update_stage("Gathering Docs")
            progress_bar.progress(65)
            await asyncio.sleep(0.5)
            progress_bar.progress(70)
            
            # Stage 5: Writing Report (90%)
            update_stage("Writing Report")
            progress_bar.progress(80)
            await asyncio.sleep(0.5)
            progress_bar.progress(90)
            
            # Stage 6: Exporting (100%)
            update_stage("Exporting Files")
            progress_bar.progress(95)
            
            # Get actual result from pipeline using clarified query
            add_log_message("Starting research process with clarified query...")
            results = await pipeline.process_query(clarified_query, progress_bar)
            
            # Update state with results
            st.session_state.results = results
            st.session_state.research_completed = True
            st.session_state.end_time = datetime.now()
            
            # Extract metrics
            if "performance_metrics" in results:
                st.session_state.metrics = results["performance_metrics"]
            
            # Extract document info if available
            outputs = results.get("outputs", {})
            if "documents_count" in outputs:
                st.session_state.documents_count = outputs["documents_count"]
            if "avg_relevance" in outputs:
                st.session_state.avg_relevance = outputs["avg_relevance"]
            
            # Final progress update
            progress_bar.progress(100)
            
            # Update the final stage
            if results["status"] == "success":
                update_stage("Done")
                add_log_message("Research completed successfully!")
            else:
                update_stage("Error", error=True)
                add_log_message(f"Error during research: {results.get('error', 'Unknown error')}")
            
            return results
            
        except Exception as e:
            # Handle stage-specific exceptions
            logger.error(f"Error during pipeline stage: {str(e)}")
            st.session_state.error = True
            add_log_message(f"Error during research: {str(e)}")
            update_stage("Error", error=True)
            return {
                "query": query,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            
    except Exception as e:
        # Handle overall exceptions
        logger.error(f"Error processing research query: {str(e)}")
        st.session_state.error = True
        add_log_message(f"Error: {str(e)}")
        update_stage("Error", error=True)
        return {
            "query": query,
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

def start_research_task():
    """Start the research task in a non-blocking way"""
    if not st.session_state.query:
        st.warning("Please enter a research query")
        return
    
    # Create a new progress bar placeholder
    progress_placeholder = st.session_state.progress_container
    progress_bar = StreamlitProgressBar(progress_placeholder)
    
    # Get the pipeline with current model and temperature settings
    pipeline = get_pipeline(
        model_name=st.session_state.model_name,
        temperature=st.session_state.temperature
    )
    
    # Reset state for new research
    st.session_state.research_completed = False
    st.session_state.progress = 0
    st.session_state.log_messages = []
    st.session_state.results = None
    st.session_state.error = None
    
    # Add log message
    add_log_message(f"Research started for query: {st.session_state.query}")
    
    # Create a new event loop for the current thread if there isn't one
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # No event loop exists, so create a new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    # Define an async function to wrap our async process
    async def run_async_task():
        try:
            update_stage("Clarifying")
            result = await process_research_query(
                pipeline, 
                st.session_state.query, 
                progress_bar, 
                st.session_state.relevance_threshold, 
                st.session_state.output_formats
            )
            return result
        except Exception as e:
            logger.error(f"Error in async task: {str(e)}")
            st.session_state.error = True
            add_log_message(f"Error: {str(e)}")
            update_stage("Error", error=True)
            return None
    
    # Run the async task
    with st.spinner("Processing your research query..."):
        try:
            result = loop.run_until_complete(run_async_task())
            st.rerun()  # Use st.rerun() instead of experimental_rerun
        except Exception as e:
            logger.error(f"Error running async task: {str(e)}")
            st.session_state.error = True
            add_log_message(f"Error: {str(e)}")
            update_stage("Error", error=True)

def render_status_stage(stage_name, emoji, current_stage):
    """Render a status stage with the appropriate color"""
    status_class = "status-grey"  # Default
    
    if current_stage == stage_name:
        status_class = "status-blue"  # Current stage
    elif current_stage == "Done" and stage_name != "Error":
        status_class = "status-green"  # Completed stage
    elif current_stage == "Error" and stage_name == "Error":
        status_class = "status-red"  # Error stage
    elif current_stage != "idle" and stage_list.index(current_stage) > stage_list.index(stage_name):
        status_class = "status-green"  # Completed stage
    
    st.markdown(
        f"""
        <div class="status-indicator">
            <div class="status-bullet {status_class}"></div>
            <div class="status-text">{emoji} {stage_name}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_metric_card(label, value, col):
    """Render a metric card with the given label and value"""
    with col:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

def format_time_delta(start_time, end_time=None):
    """Format time delta between start and end time"""
    if not start_time:
        return "N/A"
    
    if not end_time:
        end_time = datetime.now()
    
    delta = end_time - start_time
    total_seconds = delta.total_seconds()
    
    if total_seconds < 60:
        return f"{total_seconds:.1f} seconds"
    elif total_seconds < 3600:
        minutes = total_seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = total_seconds / 3600
        return f"{hours:.1f} hours"

def render_report_preview(report):
    """Render the report preview"""
    if not report:
        st.info("No report generated yet")
        return
    
    st.markdown(f"# {report['title']}")
    
    for section in report["sections"]:
        # st.markdown(f"## {section['section_title']}")
        
        # Display the main section content
        if 'content' in section and section['content']:
            with st.container():
                st.markdown(section['content'])
        
        # Display subsections without using nested expanders
        if 'subsections' in section and section['subsections']:
            # st.markdown("### Subsections")
            
            for i, subsection in enumerate(section['subsections']):
                subsection_title = subsection.get('subsection_title', f'Subsection {i+1}')
                with st.container():
                    # st.markdown(f"**{subsection_title}**")
                    if 'content' in subsection and subsection['content']:
                        st.markdown(subsection['content'])
                # Add a divider between subsections unless it's the last one
                if i < len(section['subsections']) - 1:
                    st.divider()
        
        # Add separation between sections
        st.markdown("---")

def render_downloads(export_paths):
    """Render download buttons for exported files"""
    if not export_paths:
        st.info("No exports available yet")
        return
    
    for format_name, path in export_paths.items():
        if os.path.exists(path):
            with open(path, "rb") as f:
                format_display = format_name.upper()
                st.download_button(
                    f"📥 Download {format_display}",
                    f,
                    file_name=os.path.basename(path),
                    mime=f"application/{format_name}",
                    use_container_width=True,
                    key=f"download_{format_name}"
                )
        else:
            st.warning(f"{format_name.upper()} file not found at {path}")

def render_metrics_table(metrics):
    """Render a table of performance metrics"""
    if not metrics:
        st.info("No metrics available yet")
        return
    
    # Calculate total time
    total_time = metrics.get("total_time", 0)
    
    # Prepare metrics data
    metrics_data = []
    for metric_name, value in metrics.items():
        if metric_name != "total_time":  # Skip total time as it's displayed separately
            percentage = (value / total_time * 100) if total_time > 0 else 0
            step_name = metric_name.replace("_time", "").replace("_", " ").title()
            metrics_data.append({
                "Step": step_name,
                "Seconds": f"{value:.2f}",
                "% of Total": f"{percentage:.1f}%",
                "Cache Used": "No"  # Placeholder for future cache info
            })
    
    # Add total time
    metrics_data.append({
        "Step": "Total",
        "Seconds": f"{total_time:.2f}",
        "% of Total": "100.0%",
        "Cache Used": "-"
    })
    
    # Display as dataframe
    st.dataframe(metrics_data, use_container_width=True)

# List of stages for the status sidebar
stage_list = [
    "Clarifying", "Outlining", "Topic Mining", "Gathering Docs", 
    "Writing Report", "Exporting Files", "Done", "Error"
]

# Main application
def main():
    """Main application entry point"""
    # Initialize session state
    init_session_state()
    
    # App header with logo
    st.markdown(
        """
        <div class="app-logo">
            <span>🔬</span>
            <span class="app-logo-text">Deep Researcher</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Hero section
    st.markdown(
        """
        <div class="hero-section">
            <div class="hero-title">Advanced Research Assistant</div>
            <div class="hero-subtitle">Discover insights, analyze data, and generate comprehensive reports</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Center the query input
    with st.container():
        st.markdown('<div class="query-container">', unsafe_allow_html=True)
        st.subheader("What would you like to research today?")
        
        # Show clarification questions if needed
        if st.session_state.needs_clarification:
            st.warning("Your query needs more details. Please answer the following questions:")
            
            # Create a dictionary to store answers for each question
            if "clarification_answers" not in st.session_state:
                st.session_state.clarification_answers = {}
            
            # Display each question with its own text input
            for i, question in enumerate(st.session_state.clarification_questions, 1):
                st.markdown(f"{i}. {question}")
                # Create a unique key for each question
                answer_key = f"answer_{i}"
                # Store the answer in session state
                st.session_state.clarification_answers[answer_key] = st.text_area(
                    f"Answer for question {i}",
                    key=answer_key,
                    height=80
                )
            
            # Add a button to submit the clarification
            if st.button("Submit Clarification", key="submit_clarification"):
                # Check if all questions have been answered
                all_answered = all(
                    st.session_state.clarification_answers.get(f"answer_{i+1}")
                    for i in range(len(st.session_state.clarification_questions))
                )
                
                if all_answered:
                    # Create a combined context for the LLM
                    original_query = st.session_state.query
                    questions = st.session_state.clarification_questions
                    answers = [
                        st.session_state.clarification_answers[f"answer_{i+1}"]
                        for i in range(len(questions))
                    ]
                    
                    # Format the context for the LLM
                    context = f"""Original Query: {original_query}

Clarification Questions and Answers:
"""
                    for i, (question, answer) in enumerate(zip(questions, answers), 1):
                        context += f"{i}. Q: {question}\n   A: {answer}\n\n"
                    
                    # Update the query with the combined context
                    st.session_state.query = context
                    st.session_state.needs_clarification = False
                    st.session_state.clarification_questions = []
                    st.session_state.clarification_answers = {}  # Clear the answers
                    
                    # Reset research state to start fresh
                    st.session_state.research_started = False
                    st.session_state.research_completed = False
                    st.session_state.current_stage = "idle"
                    st.session_state.progress = 0
                    st.session_state.start_time = None
                    st.session_state.end_time = None
                    st.session_state.results = None
                    st.session_state.error = None
                    st.session_state.metrics = {}
                    st.session_state.documents_count = 0
                    st.session_state.avg_relevance = 0.0
                    st.rerun()
                else:
                    st.error("Please answer all questions before submitting.")
        
        # Research query input with larger size
        st.text_area(
            "Enter your research query",
            key="query",
            placeholder="E.g., 'The impact of artificial intelligence on healthcare systems' or 'Advancements in renewable energy technologies since 2010'...",
            height=120,
            disabled=st.session_state.needs_clarification
        )
        
        # Create two columns for settings and buttons
        col1, col2 = st.columns(2)
        
        with col1:
            # Model selection
            st.selectbox(
                "Model",
                ["gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"],
                index=0,
                key="model_name",
                help="Select the language model to use for research"
            )
            
            # Output formats
            st.multiselect(
                "Output Formats",
                ["PDF", "DOCX", "HTML", "Markdown"],
                default=["PDF", "DOCX", "HTML"],
                key="output_formats"
            )
        
        with col2:
            # Relevance threshold
            st.slider(
                "Relevance Threshold",
                min_value=0.3,
                max_value=0.95,
                value=0.65,
                step=0.05,
                key="relevance_threshold",
                help="Higher values prioritize quality over quantity of results"
            )
            
            # Temperature control
            st.slider(
                "Temperature",
                min_value=0.1,
                max_value=1.0,
                value=0.7,
                step=0.1,
                key="temperature",
                help="Higher values make the output more creative but less focused"
            )
        
        # Run button with primary styling
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            run_button = st.button(
                "🚀 Start Research Process",
                disabled=not st.session_state.query or st.session_state.needs_clarification,
                use_container_width=True,
                key="run_button"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Create the 3-panel layout for the rest of the UI
    status_col, main_col = st.columns([1, 3])
    
    # Right sidebar (Run Status)
    with status_col:
        st.markdown(
            """
            <div style="background: white; padding: 18px; border-radius: var(--border-radius); box-shadow: var(--card-shadow);">
            """, 
            unsafe_allow_html=True
        )
        
        st.subheader("Research Status")
        
        # Display stages with improved styling
        render_status_stage("Clarifying", "🔍", st.session_state.current_stage)
        render_status_stage("Outlining", "📝", st.session_state.current_stage)
        render_status_stage("Topic Mining", "🔬", st.session_state.current_stage)
        render_status_stage("Gathering Docs", "📚", st.session_state.current_stage)
        render_status_stage("Writing Report", "✍️", st.session_state.current_stage)
        render_status_stage("Exporting Files", "📤", st.session_state.current_stage)
        render_status_stage("Done", "✅", st.session_state.current_stage)
        render_status_stage("Error", "❌", st.session_state.current_stage)
        
        # Progress bar
        st.subheader("Progress")
        progress_container = st.empty()
        st.session_state.progress_container = progress_container
        
        # Show current progress
        progress_container.progress(st.session_state.progress / 100)
        
        # Clear session button
        clear_button = st.button(
            "🗑️ Clear Session",
            use_container_width=True,
            key="clear_button"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main stage (tabs)
    with main_col:
        # Create tabs with improved styling
        tabs = st.tabs(["📊 Overview", "📝 Live Logs", "📄 Report Preview", "📥 Downloads", "⏱️ Metrics"])
        
        # Overview tab with enhanced layout
        with tabs[0]:
            st.subheader("Research Overview")
            
            # Create metric cards with improved layout
            overview_cols = st.columns(3)
            
            # Determine values for cards
            query_display = st.session_state.query[:120] + "..." if len(st.session_state.query) > 120 else st.session_state.query or "N/A"
            start_time_display = st.session_state.start_time.strftime("%H:%M:%S") if st.session_state.start_time else "N/A"
            docs_count = st.session_state.documents_count or "N/A"
            avg_relevance = f"{st.session_state.avg_relevance:.2f}" if st.session_state.avg_relevance else "N/A"
            
            # If research completed, calculate runtime
            runtime = format_time_delta(st.session_state.start_time, st.session_state.end_time) if st.session_state.research_completed else "In progress..."
            
            # Render metric cards in first row
            render_metric_card("Research Query", query_display, overview_cols[0])
            render_metric_card("Start Time", start_time_display, overview_cols[1])
            render_metric_card("Runtime", runtime, overview_cols[2])
            
            # Second row of metrics
            overview_cols2 = st.columns(3)
            render_metric_card("Documents Analyzed", docs_count, overview_cols2[0])
            render_metric_card("Average Relevance", avg_relevance, overview_cols2[1])
            
            # Add a visual indicator for research status
            status = "Complete ✅" if st.session_state.research_completed else "In Progress 🔄"
            if st.session_state.error:
                status = "Error ❌"
            render_metric_card("Status", status, overview_cols2[2])
            
            # Display any errors with improved styling
            if st.session_state.error:
                st.error("An error occurred during the research process. Check the Live Logs tab for details.")
            
            # If research is completed, show a summary
            if st.session_state.research_completed and st.session_state.results:
                st.markdown("""
                <div style="background: rgba(123, 92, 255, 0.1); padding: 16px; border-radius: 8px; margin-top: 20px;">
                    <h3 style="color: var(--primary-color);">Research Summary</h3>
                    <p>Your research has been completed. You can view the full report in the Report Preview tab and download the results in your preferred format from the Downloads tab.</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Live Logs tab
        with tabs[1]:
            st.subheader("Live Logs")
            
            # Create an empty container for logs with improved styling
            log_container = st.empty()
            
            # Display logs
            log_text = "\n".join(st.session_state.log_messages)
            log_container.text_area(
                "Process Logs",
                value=log_text,
                height=400,
                disabled=True
            )
        
        # Report Preview tab
        with tabs[2]:
            st.subheader("Report Preview")
            
            if st.session_state.research_completed and st.session_state.results:
                # Get report from results
                report = st.session_state.results.get("outputs", {}).get("report")
                render_report_preview(report)
            else:
                st.info("Run a research query to generate a report")
        
        # Downloads tab
        with tabs[3]:
            st.subheader("Downloads")
            
            if st.session_state.research_completed and st.session_state.results:
                # Get export paths from results
                export_paths = st.session_state.results.get("outputs", {}).get("export_paths", {})
                render_downloads(export_paths)
            else:
                st.info("Run a research query to generate downloadable reports")
        
        # Metrics tab
        with tabs[4]:
            st.subheader("Performance Metrics")
            
            render_metrics_table(st.session_state.metrics)
    
    # Handle button clicks with if statements to avoid auto-execution
    if clear_button:
        clear_session()
    
    if run_button:
        if not st.session_state.research_started:
            st.session_state.research_started = True
            start_research_task()
            
    # Footer attribution
    st.markdown(
        """
        <div style="text-align: center; margin-top: 30px; padding: 10px; font-size: 0.8rem; color: #6B7280;">
            Made with ❤️ by BITS Deep Research team
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Relevance threshold explanation
    st.markdown(
        """
        <div style="margin-top: 20px; padding: 15px; background-color: #f8f9fa; border-radius: 8px; font-size: 0.9rem;">
            <h4 style="color: #4B5563; margin-bottom: 10px;">About Relevance Threshold</h4>
            <p style="color: #6B7280; line-height: 1.5;">
                The relevance threshold (0.3 - 0.95) determines how strictly the system filters research results:
                <br><br>
                • <strong>Lower values (0.3-0.5):</strong> More inclusive, returns a broader range of results but may include less relevant content
                <br>
                • <strong>Medium values (0.5-0.7):</strong> Balanced approach, good for general research topics
                <br>
                • <strong>Higher values (0.7-0.95):</strong> More selective, returns only highly relevant results but may miss some related content
                <br><br>
                Recommended: Start with 0.65 for most research topics. Adjust based on your needs for breadth vs. precision.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

@st.cache_resource
def get_pipeline(model_name="gpt-4o-mini", temperature=0.7):
    """Get or create the research pipeline instance with specified model settings."""
    try:
        pipeline = EnhancedResearchPipeline()
        # Update model settings
        pipeline.planner.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        pipeline.gatherer.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        pipeline.clarifier.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        return pipeline
    except Exception as e:
        logger.error(f"Error creating pipeline: {e}")
        return DummyEnhancedResearchPipeline()

if __name__ == "__main__":
    main() 