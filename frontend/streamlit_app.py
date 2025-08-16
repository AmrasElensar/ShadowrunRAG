"""Streamlit web UI for Shadowrun RAG system."""
import os

import streamlit as st
import requests
from pathlib import Path
import json
import logging

@st.cache_data(ttl=30)  # Cache for 30 seconds
def get_available_models():
    """Fetch available models with caching."""
    try:
        response = api_request("/models", timeout=5)  # Add timeout
        return response.get("models", ["llama3"]) if response else ["llama3"]
    except Exception as e:
        logger.warning(f"Failed to fetch models: {e}")
        return []  # Fallback

@st.cache_data(ttl=30)  # Cache for 30 seconds
def get_available_documents():
    """Fetch available documents with caching."""
    try:
        response = api_request("/documents", timeout=5)  # Add timeout
        return response.get("documents", []) if response else []
    except Exception as e:
        logger.warning(f"Failed to fetch documents: {e}")
        return []  # Fallback


# API configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Shadowrun RAG Assistant",
    page_icon="🎲",
    layout="wide"
)

# Initialize session state
if 'query_input' not in st.session_state:
    st.session_state.query_input = ""

# Custom CSS
st.markdown("""
<style>
/* Theme-aware styling */
:root {
    --bg-primary: #f0f2f6;
    --bg-secondary: #f8f9fa;
    --border-color: #0066cc;
    --text-color: #111;
}

[data-testid="stApp"] [data-baseweb="tab-list"] {
    gap: 24px;
}

/* Override in dark mode */
@media (prefers-color-scheme: dark) {
    :root {
        --bg-primary: #262730;
        --bg-secondary: #2f3136;
        --border-color: #0066cc;
        --text-color: #e0e0e0;
    }
}

.source-box {
    background-color: var(--bg-primary);
    color: var(--text-color);
    padding: 10px;
    border-radius: 5px;
    margin: 5px 0;
    font-family: monospace;
    font-size: 0.9em;
}

.chunk-box {
    background-color: var(--bg-secondary);
    color: var(--text-color);
    padding: 15px;
    border-left: 3px solid var(--border-color);
    margin: 10px 0;
    border-radius: 0 4px 4px 0;
}

/* Ensure markdown text is readable */
.stMarkdown {
    color: var(--text-color);
}
</style>
""", unsafe_allow_html=True)

def api_request(endpoint: str, method: str = "GET", timeout: int = 10, **kwargs):
    """Make API request with error handling and timeout."""
    try:
        url = f"{API_URL}{endpoint}"
        response = requests.request(method, url, timeout=timeout, **kwargs)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return None

# Title and description
st.title("🎲 Shadowrun RAG Assistant")
st.markdown("*Your AI-powered guide to the Sixth World*")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")

    # Cached model selection - only hits API every 30 seconds
    available_models = get_available_models()
    selected_model = st.selectbox(
        "LLM Model",
        available_models,
        index=0
    )

    # Query settings
    st.subheader("Query Settings")
    n_results = st.slider("Number of sources", 1, 10, 5)

    query_type = st.selectbox(
        "Query Type",
        ["general", "rules", "session"],
        format_func=lambda x: {
            "general": "General Query",
            "rules": "Rules Question",
            "session": "Session History"
        }[x]
    )

    st.subheader("👤 Character Context")
    character_role = st.selectbox(
        "Character Role (optional)",
        ["None", "Decker", "Mage", "Street Samurai", "Rigger", "Adept", "Technomancer", "Face"],
        index=0
    )
    st.caption("*Character role auto-filters to relevant section if no manual filter is selected*")
    character_role = None if character_role == "None" else character_role.lower().replace(" ", "_")

    character_stats = st.text_input(
        "Character Stats (optional)",
        placeholder="e.g., Logic 6, Hacking 5, Sleaze 4"
    )

    edition = st.selectbox(
        "Preferred Edition (optional)",
        ["None", "SR5", "SR6", "SR4"],
        index=0
    )
    edition = None if edition == "None" else edition

    # Cached document filter - only hits API every 30 seconds
    documents = get_available_documents()
    filter_source = None
    if documents:
        filter_source = st.selectbox(
            "Filter by Source (optional)",
            ["All"] + documents
        )
        filter_source = None if filter_source == "All" else filter_source

    # Section filter (static - no API call)
    section_options = ["All", "Combat", "Matrix", "Magic", "Gear", "Character Creation", "Riggers", "Technomancy"]
    filter_section = st.selectbox(
        "Filter by Section (optional)",
        section_options,
        index=0
    )
    filter_section = None if filter_section == "All" else filter_section

    # Subsection filter (static - no API call)
    filter_subsection = st.text_input(
        "Filter by Subsection (optional)",
        placeholder="e.g., Hacking, Spellcasting, Initiative"
    )
    filter_subsection = None if filter_subsection.strip() == "" else filter_subsection

# Main interface tabs
tab1, tab2, tab3, tab4 = st.tabs(["💬 Query", "📤 Upload", "📚 Documents", "📝 Session Notes"])

with tab1:
    st.header("Ask a Question")
    
    # Query input
    col1, col2 = st.columns([5, 1])
    with col1:
        query = st.text_area(
            "Your question:",
            value=st.session_state.query_input,
            key="query_input",  # This syncs with session state
            placeholder="e.g., 'How do recoil penalties work in Shadowrun 5e?' or 'What happened in our last session?'",
            height=100
        )
    
    with col2:
        st.write("")  # Spacer
        st.write("")
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)

    # Handle query

    if search_button and query:
        with st.spinner("Streaming from the shadows..."):
            try:
                # Build where_filter for metadata filtering
                where_filter = {}

                # Role-based section filter
                role_to_section = {
                    "decker": "Matrix", "hacker": "Matrix", "mage": "Magic", "adept": "Magic",
                    "street_samurai": "Combat", "rigger": "Riggers", "technomancer": "Technomancy"
                }

                if filter_source:
                    where_filter["source"] = {"$contains": filter_source}
                if filter_section:
                    where_filter["Section"] = filter_section
                else:
                    # No manual section → use role as convenience
                    if character_role and character_role in role_to_section:
                        where_filter["Section"] = role_to_section[character_role]
                if filter_subsection:
                    where_filter["Subsection"] = filter_subsection

                response = requests.post(
                    f"{API_URL}/query_stream",
                    json={
                        "question": query,
                        "n_results": n_results,
                        "query_type": query_type,
                        "where_filter": where_filter,
                        "character_role": character_role,
                        "character_stats": character_stats,
                        "edition": edition,
                        "model": selected_model  # ADD THIS LINE
                    },
                    stream=True
                )
                response.raise_for_status()

                # Stream the answer
                st.markdown("### 🎯 Answer")
                message_placeholder = st.empty()
                full_response = ""
                metadata = None
                collecting_metadata = False
                metadata_buffer = ""

                for chunk in response.iter_content(chunk_size=32, decode_unicode=True):
                    if chunk:
                        # Check for metadata start
                        if "__METADATA_START__" in chunk:
                            # Split at metadata start
                            parts = chunk.split("__METADATA_START__")
                            if parts[0]:
                                full_response += parts[0]
                                message_placeholder.markdown(full_response + "▌")
                            
                            collecting_metadata = True
                            if len(parts) > 1:
                                metadata_buffer = parts[1]
                            continue
                        
                        # If collecting metadata
                        if collecting_metadata:
                            metadata_buffer += chunk
                            
                            # Check for metadata end
                            if "__METADATA_END__" in metadata_buffer:
                                # Extract JSON between markers
                                json_part = metadata_buffer.split("__METADATA_END__")[0].strip()
                                try:
                                    metadata = json.loads(json_part)
                                    logger.info("Successfully parsed metadata")
                                except json.JSONDecodeError as e:
                                    logger.error(f"Metadata parse failed: {e}")
                                    logger.error(f"Raw metadata: {json_part[:200]}...")
                                break
                        else:
                            # Normal content streaming
                            full_response += chunk
                            message_placeholder.markdown(full_response + "▌")

                # Final update without cursor
                message_placeholder.markdown(full_response)

                # Display sources from metadata
                if metadata and metadata.get('sources'):
                    st.markdown("### 📖 Sources")
                    cols = st.columns(min(len(metadata['sources']), 3))
                    for i, source in enumerate(metadata['sources']):
                        with cols[i % 3]:
                            source_name = Path(source).name
                            st.markdown(f"""
                            <div class="source-box">
                                📄 <strong>{source_name}</strong>
                            </div>
                            """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Stream failed: {e}")
                # Optionally fall back to non-streaming
    
    # Example queries
    st.markdown("---")
    st.markdown("#### 💡 Example Queries")
    
    example_cols = st.columns(3)
    examples = [
        ("Rules", "How does the Matrix Initiative work?"),
        ("Combat", "What are the modifiers for called shots?"),
        ("Magic", "How do you learn new spells as a mage?"),
        ("Session", "Who was the Johnson from our last run?"),
        ("Character", "What cyberware did we find in the corp facility?"),
        ("Lore", "Tell me about the history of Aztechnology")
    ]
    
    for i, (category, example) in enumerate(examples):
        with example_cols[i % 3]:
            if st.button(f"{category}: {example[:30]}...", key=f"ex_{i}"):
                st.session_state.query_input = example
                st.rerun()

with tab2:
    st.header("Upload PDFs")

    # CSS to hide file uploader file list
    st.markdown("""
    <style>
    .uploadedFile { display: none !important; }
    div[data-testid="fileUploader"] section[data-testid="fileDropzoneInstructions"] + div { display: none !important; }
    .uploadedFileData { display: none !important; }
    </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'ready_files' not in st.session_state:
        st.session_state.ready_files = []
    if 'processing_files' not in st.session_state:
        st.session_state.processing_files = {}
    if 'completed_files' not in st.session_state:
        st.session_state.completed_files = {}
    if 'uploader_key' not in st.session_state:
        st.session_state.uploader_key = 0

    # Process pending actions ONCE at the start
    if 'action' in st.session_state:
        action = st.session_state.action

        if action['type'] == 'clear_all':
            st.session_state.ready_files = []
            st.session_state.uploader_key += 1
        elif action['type'] == 'process_all':
            for file_info in st.session_state.ready_files:
                st.session_state.processing_files[file_info['id']] = {
                    'name': file_info['name'],
                    'size': file_info['size'],
                    'file': file_info['file'],
                    'stage': 'uploading'
                }
            st.session_state.ready_files = []
            st.session_state.uploader_key += 1
        elif action['type'] == 'remove_file':
            file_id = action['file_id']
            st.session_state.ready_files = [f for f in st.session_state.ready_files if f['id'] != file_id]
            if not st.session_state.ready_files:
                st.session_state.uploader_key += 1
        elif action['type'] == 'process_file':
            file_id = action['file_id']
            file_info = next((f for f in st.session_state.ready_files if f['id'] == file_id), None)
            if file_info:
                st.session_state.processing_files[file_id] = {
                    'name': file_info['name'],
                    'size': file_info['size'],
                    'file': file_info['file'],
                    'stage': 'uploading'
                }
                st.session_state.ready_files = [f for f in st.session_state.ready_files if f['id'] != file_id]
                if not st.session_state.ready_files:
                    st.session_state.uploader_key += 1

        del st.session_state.action
        st.rerun()  # Single rerun after all actions

    # File uploader
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=['pdf'],
        help="Upload Shadowrun rulebooks, session notes, or any other PDF documents",
        accept_multiple_files=True,
        key=f"pdf_uploader_{st.session_state.uploader_key}"
    )

    # Add uploaded files (minimize processing)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_id = f"{uploaded_file.name}_{uploaded_file.size}"
            if not any(f['id'] == file_id for f in st.session_state.ready_files):
                st.session_state.ready_files.append({
                    'id': file_id,
                    'name': uploaded_file.name,
                    'size': uploaded_file.size,
                    'file': uploaded_file
                })

    # Ready to Process Section
    if st.session_state.ready_files:
        st.markdown("---")
        st.subheader("📁 Ready to Process")

        # Batch buttons - simplified layout
        if st.button("📤 Process All", type="primary"):
            st.session_state.action = {'type': 'process_all'}
            st.rerun()

        if st.button("🗑️ Clear All"):
            st.session_state.action = {'type': 'clear_all'}
            st.rerun()

        # Individual files - simpler layout for speed
        for file_info in st.session_state.ready_files:
            # Single row layout - faster than columns
            st.write(f"📄 **{file_info['name']}** ({file_info['size'] / 1024:.1f} KB)")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("📤 Process", key=f"proc_{file_info['id']}"):
                    st.session_state.action = {'type': 'process_file', 'file_id': file_info['id']}
                    st.rerun()
            with col2:
                if st.button("❌ Remove", key=f"rem_{file_info['id']}"):
                    st.session_state.action = {'type': 'remove_file', 'file_id': file_info['id']}
                    st.rerun()

    # Processing section - REMOVED AUTO-PROGRESS for speed
    if st.session_state.processing_files:
        st.markdown("---")
        st.subheader("🔄 Processing")

        for file_id, file_info in st.session_state.processing_files.items():
            st.write(f"🔄 **{file_info['name']}** - Processing in background...")

            # Manual progress update button instead of auto-updates
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🔄 Check Progress", key=f"progress_{file_id}"):
                    # Simulate progress check
                    current_stage = file_info.get('stage', 'uploading')
                    stages = ['uploading', 'extracting', 'chunking', 'embedding', 'indexing', 'complete']
                    if current_stage in stages:
                        current_idx = stages.index(current_stage)
                        if current_idx < len(stages) - 1:
                            st.session_state.processing_files[file_id]['stage'] = stages[current_idx + 1]
                        else:
                            st.session_state.completed_files[file_id] = file_info
                            del st.session_state.processing_files[file_id]
                    st.rerun()

            with col2:
                if st.button("✅ Mark Complete", key=f"complete_{file_id}"):
                    st.session_state.completed_files[file_id] = file_info
                    del st.session_state.processing_files[file_id]
                    st.rerun()

            with col3:
                if st.button("❌ Cancel", key=f"cancel_{file_id}"):
                    del st.session_state.processing_files[file_id]
                    st.rerun()

    # Completed section
    if st.session_state.completed_files:
        st.markdown("---")
        st.subheader("✅ Completed")

        for file_id, file_info in st.session_state.completed_files.items():
            st.success(f"✅ **{file_info['name']}** - Successfully processed!")

    # Manual indexing
    st.markdown("---")
    st.subheader("Manual Indexing")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Re-index All Documents"):
            with st.spinner("Re-indexing all documents..."):
                response = api_request("/index", method="POST", json={"force_reindex": True})
                if response:
                    st.success("✅ Re-indexing complete!")

    with col2:
        if st.button("📊 Index New Documents Only"):
            with st.spinner("Indexing new documents..."):
                response = api_request("/index", method="POST", json={"force_reindex": False})
                if response:
                    st.success("✅ Indexing complete!")

with tab3:
    st.header("Indexed Documents")
    
    # Refresh button
    if st.button("🔄 Refresh List"):
        st.rerun()
    
    # Get document list
    docs_response = api_request("/documents")
    
    if docs_response and docs_response.get("documents"):
        documents = docs_response["documents"]
        
        # Group by source type
        rulebooks = [d for d in documents if "rulebook" in d.lower() or "core" in d.lower()]
        sessions = [d for d in documents if "session" in d.lower()]
        other = [d for d in documents if d not in rulebooks and d not in sessions]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 📘 Rulebooks")
            for doc in rulebooks:
                doc_name = Path(doc).stem
                st.markdown(f"• **{doc_name}**")
            if not rulebooks:
                st.markdown("*No rulebooks indexed*")
        
        with col2:
            st.markdown("### 📝 Session Logs")
            for doc in sessions:
                doc_name = Path(doc).stem
                st.markdown(f"• **{doc_name}**")
            if not sessions:
                st.markdown("*No session logs indexed*")
        
        with col3:
            st.markdown("### 📄 Other Documents")
            for doc in other:
                doc_name = Path(doc).stem
                st.markdown(f"• **{doc_name}**")
            if not other:
                st.markdown("*No other documents*")
        
        # Statistics
        st.markdown("---")
        st.metric("Total Documents", len(documents))
    else:
        st.info("No documents indexed yet. Upload PDFs to get started!")

with tab4:
    st.header("Session Notes")
    
    # Session note creator
    st.subheader("Create New Session Note")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        session_number = st.number_input("Session Number", min_value=1, value=1)
    with col2:
        session_date = st.date_input("Date")
    
    session_title = st.text_input(
        "Session Title",
        placeholder="The Time We Blew Up the Renraku Lab"
    )
    
    session_content = st.text_area(
        "Session Summary",
        placeholder="Write your session summary here...\n\nKey Events:\n- \n\nNPCs Met:\n- \n\nLoot Found:\n- ",
        height=300
    )
    
    if st.button("💾 Save Session Note", type="primary"):
        if session_title and session_content:
            # Create markdown file
            filename = f"Session_{session_number:02d}_{session_title.replace(' ', '_')}.md"
            filepath = Path("data/processed_markdown/SessionLogs") / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Format content
            content = f"""# Session {session_number}: {session_title}
Date: {session_date}

## Summary
{session_content}

---
*Generated by Shadowrun RAG Assistant*
"""
            
            filepath.write_text(content, encoding='utf-8')
            st.success(f"✅ Session note saved as {filename}")
            
            # Trigger indexing
            api_request("/index", method="POST", json={"force_reindex": False})
        else:
            st.error("Please fill in both title and content")
    
    # Quick templates
    st.markdown("---")
    st.subheader("📋 Quick Templates")
    
    template_cols = st.columns(2)
    
    with template_cols[0]:
        if st.button("🎯 Combat Encounter Template"):
            st.session_state['template'] = """**Location:** 
**Enemies:** 
**Tactics Used:** 
**Outcome:** 
**Loot:** """
    
    with template_cols[1]:
        if st.button("🤝 Johnson Meeting Template"):
            st.session_state['template'] = """**Johnson Name:** 
**Meeting Location:** 
**Job Offered:** 
**Payment:** 
**Special Conditions:** """

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <small>🎲 Shadowrun RAG Assistant v1.0 | Powered by Ollama & ChromaDB</small>
    </div>
    """,
    unsafe_allow_html=True
)

# Initialize session state for query
if 'query_text' in st.session_state:
    st.session_state['query'] = st.session_state['query_text']
    del st.session_state['query_text']