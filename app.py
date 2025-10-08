"""
Financial Document Analyzer - Main Streamlit Application
Multi-Agent System with CrewAI and MCP Servers
"""
import streamlit as st
import os
import sys
import logging
import json
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.settings import settings
from utils.llm_manager import LLMManager
from utils.document_loader import DocumentLoader
from utils.vector_store_manager import VectorStoreManager
from flows.qa_flow import QAFlow
from flows.mcq_flow import MCQFlow
from flows.summary_flow import SummaryFlow

# Monitoring
from langsmith.integrations.otel import OtelSpanProcessor
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from openinference.instrumentation.crewai import CrewAIInstrumentor

# Load environment variables from .env file
load_dotenv()

# Get or create tracer provider
tracer_provider = trace.get_tracer_provider()
if not isinstance(tracer_provider, TracerProvider):
    tracer_provider = TracerProvider()
    trace.set_tracer_provider(tracer_provider)
# Add OtelSpanProcessor to the tracer provider
tracer_provider.add_span_processor(OtelSpanProcessor())
# Instrument CrewAI and OpenAI
CrewAIInstrumentor().instrument()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Financial Document Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 1rem;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ffc107;
    }
    .agent-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
        st.session_state.documents_loaded = False
        st.session_state.chat_history = []
        st.session_state.uploaded_files = []
        st.session_state.current_documents = []
        st.session_state.vector_manager = None
        st.session_state.agents_ready = False
        st.session_state.mcq_questions = []
        st.session_state.mcq_answers = {}
        st.session_state.mcq_sources = []


def initialize_system():
    """Initialize LLM and agents"""
    try:
        with st.spinner("Initializing AI system..."):
            # Test LLM connection
            if not LLMManager.test_connection():
                st.error("Failed to connect to Ollama. Please ensure Ollama is running.")
                st.info("Run: `ollama serve` in a terminal")
                return False
            
            # Initialize embeddings
            embeddings = LLMManager.get_embeddings()
            
            # Initialize vector store manager
            if st.session_state.vector_manager is None:
                st.session_state.vector_manager = VectorStoreManager(embeddings)
                st.session_state.vector_manager.load_vector_store()
            
            # Initialize agents
            if not st.session_state.agents_ready:
                st.session_state.qa_agent = QAFlow(st.session_state.vector_manager)
                st.session_state.qa_agent.plot("QAFlow")
                st.session_state.summary_agent = SummaryFlow()
                st.session_state.summary_agent.plot("SummaryFLow")
                st.session_state.mcq_agent = MCQFlow()
                st.session_state.mcq_agent.plot("MCQFlow")
                st.session_state.agents_ready = True
            
            st.session_state.initialized = True
            return True
            
    except Exception as e:
        st.error(f"Initialization error: {str(e)}")
        logger.error(f"Initialization error: {e}")
        return False


def save_chat_history():
    """Save chat history to file"""
    try:
        history_file = os.path.join(settings.logs_dir, "chat_history.json")
        with open(history_file, 'w') as f:
            json.dump(st.session_state.chat_history, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving chat history: {e}")


def load_chat_history():
    """Load chat history from file"""
    try:
        history_file = os.path.join(settings.logs_dir, "chat_history.json")
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                st.session_state.chat_history = json.load(f)
    except Exception as e:
        logger.error(f"Error loading chat history: {e}")


def process_uploaded_files(uploaded_files):
    """Process uploaded files and create vector store"""
    try:
        doc_loader = DocumentLoader()
        all_documents = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing {uploaded_file.name}...")
            
            # Save file
            file_path = os.path.join(settings.upload_dir, uploaded_file.name)
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            # Load and process
            documents = doc_loader.load_file(file_path)
            all_documents.extend(documents)
            
            progress_bar.progress((idx + 1) / len(uploaded_files))
        
        status_text.text("Creating vector embeddings...")
        
        # Add to vector store
        st.session_state.vector_manager.add_documents(all_documents)
        st.session_state.vector_manager.save_vector_store()
        
        st.session_state.current_documents = all_documents
        st.session_state.documents_loaded = True
        st.session_state.uploaded_files = [f.name for f in uploaded_files]
        
        progress_bar.empty()
        status_text.empty()
        
        return True, len(all_documents)
        
    except Exception as e:
        logger.error(f"Error processing files: {e}")
        return False, str(e)
    
def deduplicate_dicts(dicts_list):
    seen = set()
    unique_dicts = []
    for d in dicts_list:
        tup = tuple(sorted(d.items()))
        if tup not in seen:
            seen.add(tup)
            unique_dicts.append(d)
    return unique_dicts


# Main application
def main():
    """Main application function"""
    
    # Initialize session state
    init_session_state()
    
    # Header
    st.markdown('<div class="main-header">📊 Financial Document Analyzer</div>', unsafe_allow_html=True)
    st.markdown("### Multi-Agent AI System for Document Analysis")
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/artificial-intelligence.png", width=80)
        st.title("🤖 Control Panel")
        
        # System status
        st.markdown("---")
        st.subheader("System Status")
        
        if not st.session_state.initialized:
            if st.button("🚀 Initialize System", type="primary"):
                if initialize_system():
                    st.success("✅ System initialized!")
                    st.rerun()
        else:
            st.success("✅ System Online")
            st.info(f"📁 Model: {settings.ollama_model}")
            
            if st.session_state.documents_loaded:
                st.success(f"✅ {len(st.session_state.uploaded_files)} files loaded")
                st.success(f"✅ {len(st.session_state.current_documents)} chunks indexed")
        
        # File upload
        st.markdown("---")
        st.subheader("📤 Upload Documents")
        
        uploaded_files = st.file_uploader(
            "Upload PDF or CSV files",
            type=['pdf', 'csv'],
            accept_multiple_files=True,
            help="Upload financial documents for analysis"
        )
        
        if uploaded_files and st.button("Process Files", type="primary"):
            if not st.session_state.initialized:
                st.warning("Please initialize the system first!")
            else:
                success, result = process_uploaded_files(uploaded_files)
                if success:
                    st.success(f"✅ Processed {result} document chunks!")
                    st.rerun()
                else:
                    st.error(f"❌ Error: {result}")
        
        # Clear data
        st.markdown("---")
        if st.button("🗑️ Clear All Data"):
            if st.session_state.vector_manager:
                st.session_state.vector_manager.clear_vector_store()
            st.session_state.documents_loaded = False
            st.session_state.uploaded_files = []
            st.session_state.current_documents = []
            st.session_state.chat_history = []
            st.success("Data cleared!")
            st.rerun()
    
    # Main content
    if not st.session_state.initialized:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.info("👈 Click 'Initialize System' in the sidebar to get started!")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Show system requirements
        st.markdown("### 📋 System Requirements")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Before starting:**
            1. ✅ Ollama installed and running
            2. ✅ Gemma2 model downloaded
            3. ✅ Python environment configured
            """)
        
        with col2:
            st.markdown("""
            **Quick Start:**
            ```bash
            # Start Ollama
            ollama serve
            
            # Pull model
            ollama pull gemma2:2b
            ```
            """)
        
        return
    
    # Agent tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Q&A Agent", "📝 Summary Agent", "❓ MCQ Agent", "📊 Chat History"])
    
    # Tab 1: Q&A Agent
    with tab1:
        st.markdown('<div class="sub-header">Question & Answer Agent</div>', unsafe_allow_html=True)
        
        if not st.session_state.documents_loaded:
            st.warning("⚠️ Please upload documents first!")
        else:
            st.markdown('<div class="agent-card">', unsafe_allow_html=True)
            st.markdown("**Ask questions about your financial documents**")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Question input
            question = st.text_area(
                "Enter your question:",
                height=100,
                placeholder="e.g., What is the total revenue mentioned in the document?"
            )
            
            col1, col2 = st.columns([1, 4])
            with col1:
                ask_button = st.button("🔍 Ask Question", type="primary")
            
            if ask_button and question:
                with st.spinner("🤔 Analyzing documents..."):
                    result = st.session_state.qa_agent.kickoff(inputs={"question": question})
                    
                    if result['success']:
                        st.markdown('<div class="success-box">', unsafe_allow_html=True)
                        st.markdown("**Answer:**")
                        st.write(result['answer'])
                        st.markdown('</div>', unsafe_allow_html=True)
                        

                        keywords = result.get("keywords", "")
                        if keywords:
                            keywords_arr = [kw.strip() for kw in keywords.split(',')]
                            st.markdown("**Keywords:**")
                            st.markdown(
                                " ".join([f'<span style="background-color:#e0e0e0; padding:4px 8px; margin:2px; border-radius:5px; display:inline-block;">{kw}</span>' for kw in keywords_arr]),
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown("_No keywords found._")

                        # Show sources
                        with st.expander("📚 View Sources"):
                            original_sources = result['sources']
                            clean_sources = deduplicate_dicts(original_sources)

                            for idx, source in enumerate(clean_sources, 1):
                                st.markdown(f"**Source {idx}:** {source['source']} (Type: {source['type']})")
                                if 'page' in source:
                                    st.markdown(f"- Page: {source['page']}")
                        
                        # Add to chat history
                        st.session_state.chat_history.append({
                            "timestamp": datetime.now().isoformat(),
                            "type": "qa",
                            "question": question,
                            "answer": result['answer'],
                            "sources": result['sources']
                        })
                        save_chat_history()
                    else:
                        st.error(f"❌ {result['answer']}")
    
    # Tab 2: Summary Agent
    with tab2:
        st.markdown('<div class="sub-header">Document Summary Agent</div>', unsafe_allow_html=True)
        
        if not st.session_state.documents_loaded:
            st.warning("⚠️ Please upload documents first!")
        else:
            st.markdown('<div class="agent-card">', unsafe_allow_html=True)
            st.markdown("**Generate comprehensive summaries of your financial documents**")
            st.markdown('</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                summary_type = st.selectbox(
                    "Summary Type:",
                    ["comprehensive", "brief", "executive"],
                    help="Choose the level of detail for the summary"
                )
            
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                generate_summary_btn = st.button("📝 Generate Summary", type="primary")
            
            if generate_summary_btn:
                with st.spinner(f"✍️ Generating {summary_type} summary..."):
                    result = st.session_state.summary_agent.kickoff(inputs={
                        "documents" : st.session_state.current_documents,
                        "summary_type" : summary_type
                    })
                    
                    if result['success']:
                        st.markdown('<div class="success-box">', unsafe_allow_html=True)
                        st.markdown(f"**{summary_type.capitalize()} Summary:**")
                        st.write(result['summary'])
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Metadata
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Documents", result['num_documents'])
                        with col2:
                            st.metric("Word Count", result['word_count'])
                        with col3:
                            st.metric("Sources", len(result['sources']))
                        
                        # Add to chat history
                        st.session_state.chat_history.append({
                            "timestamp": datetime.now().isoformat(),
                            "type": "summary",
                            "summary_type": summary_type,
                            "summary": result['summary'],
                            "metadata": {
                                "num_documents": result['num_documents'],
                                "word_count": result['word_count']
                            }
                        })
                        save_chat_history()
                    else:
                        st.error(f"❌ {result['summary']}")
    
    # Tab 3: MCQ Agent
    with tab3:
        st.markdown('<div class="sub-header">MCQ Generation Agent</div>', unsafe_allow_html=True)
        
        if not st.session_state.documents_loaded:
            st.warning("⚠️ Please upload documents first!")
        else:
            st.markdown('<div class="agent-card">', unsafe_allow_html=True)
            st.markdown("**Generate multiple choice questions to test understanding**")
            st.markdown('</div>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                num_questions = st.number_input(
                    "Number of Questions:",
                    min_value=1,
                    max_value=10,
                    value=5
                )
            
            with col2:
                difficulty = st.selectbox(
                    "Difficulty Level:",
                    ["easy", "medium", "hard"],
                    help="Choose the level of difficulty level"
                )
            
            with col3:
                st.markdown("<br>", unsafe_allow_html=True)
                generate_mcq_btn = st.button("❓ Generate MCQs", type="primary")
            
            if generate_mcq_btn:
                with st.spinner(f"🎯 Generating {num_questions} {difficulty} questions..."):
                    result = st.session_state.mcq_agent.kickoff(inputs={
                        "documents" : st.session_state.current_documents,
                        "num_questions" : num_questions,
                        "difficulty": difficulty
                    })
                    
                    if result['success']:
                        st.session_state.mcq_questions = result['questions']
                        st.success(f"✅ Generated {len(result['questions'])} questions!")

                        st.session_state.mcq_sources = result['sources']
                        
                        # Add to chat history
                        st.session_state.chat_history.append({
                            "timestamp": datetime.now().isoformat(),
                            "type": "mcq",
                            "difficulty": difficulty,
                            "num_questions": len(result['questions']),
                            "questions": result['questions']
                        })
                        save_chat_history()
                    else:
                        st.error(f"❌ {result.get('message', 'Error generating questions')}")
            
            # Display questions
            if st.session_state.mcq_questions:
                st.markdown("---")
                st.markdown("### 📝 Generated Questions")
                
                for idx, q in enumerate(st.session_state.mcq_questions, 1):
                    with st.expander(f"Question {idx}: {q['question'][:80]}...", expanded=idx==1):
                        st.markdown(f"**{q['question']}**")
                        st.markdown("")
                        
                        # Display options
                        for opt_key, opt_value in q['options'].items():
                            st.markdown(f"**{opt_key})** {opt_value}")
                        
                        # Answer selection
                        user_answer = st.radio(
                            "Your answer:",
                            options=list(q['options'].keys()),
                            key=f"q_{idx}",
                            horizontal=True
                        )
                        
                        if st.button(f"Check Answer", key=f"check_{idx}"):
                            if user_answer == q['correct_answer']:
                                st.success(f"✅ Correct! {q['explanation']}")
                            else:
                                st.error(f"❌ Incorrect. The correct answer is {q['correct_answer']}")
                                st.info(f"💡 {q['explanation']}")

                with st.expander("📚 View Sources"):
                    for src in st.session_state.mcq_sources:
                        st.markdown(f"- `{src}`")
    
    # Tab 4: Chat History
    with tab4:
        st.markdown('<div class="sub-header">Chat History & Traceability</div>', unsafe_allow_html=True)
        
        if not st.session_state.chat_history:
            st.info("No chat history yet. Start by asking questions or generating summaries!")
        else:
            st.markdown(f"**Total Interactions:** {len(st.session_state.chat_history)}")
            
            # Filter options
            interaction_types = list(set([item['type'] for item in st.session_state.chat_history]))
            filter_type = st.multiselect("Filter by type:", interaction_types, default=interaction_types)
            
            st.markdown("---")
            
            # Display history
            for idx, item in enumerate(reversed(st.session_state.chat_history), 1):
                if item['type'] not in filter_type:
                    continue
                
                with st.expander(f"#{len(st.session_state.chat_history) - idx + 1} - {item['type'].upper()} - {item['timestamp'][:19]}"):
                    
                    if item['type'] == 'qa':
                        st.markdown(f"**Question:** {item['question']}")
                        st.markdown(f"**Answer:** {item['answer']}")
                        st.markdown(f"**Sources:** {len(item.get('sources', []))}")
                    
                    elif item['type'] == 'summary':
                        st.markdown(f"**Type:** {item['summary_type']}")
                        st.markdown(f"**Summary:**")
                        st.write(item['summary'][:500] + "..." if len(item['summary']) > 500 else item['summary'])
                    
                    elif item['type'] == 'mcq':
                        st.markdown(f"**Difficulty:** {item['difficulty']}")
                        st.markdown(f"**Questions Generated:** {item['num_questions']}")
            
            # Export history
            if st.button("💾 Export History (JSON)"):
                history_json = json.dumps(st.session_state.chat_history, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=history_json,
                    file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )


if __name__ == "__main__":
    main()