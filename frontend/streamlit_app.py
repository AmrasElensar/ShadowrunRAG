"""Streamlit web UI for Shadowrun RAG system."""
import os

import streamlit as st
import requests
from pathlib import Path
import json
import logging

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

def api_request(endpoint: str, method: str = "GET", **kwargs):
    """Make API request with error handling."""
    try:
        url = f"{API_URL}{endpoint}"
        response = requests.request(method, url, **kwargs)
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
    
    # Model selection
    models_response = api_request("/models")
    available_models = models_response.get("models", ["llama3"]) if models_response else ["llama3"]
    
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

    # Document filter
    docs_response = api_request("/documents")
    if docs_response:
        documents = docs_response.get("documents", [])
        if documents:
            filter_source = st.selectbox(
                "Filter by Source (optional)",
                ["All"] + documents
            )
            filter_source = None if filter_source == "All" else filter_source

    # Section filter
    section_options = ["All", "Combat", "Matrix", "Magic", "Gear", "Character Creation", "Riggers", "Technomancy"]
    filter_section = st.selectbox(
        "Filter by Section (optional)",
        section_options,
        index=0
    )
    filter_section = None if filter_section == "All" else filter_section

    # Subsection filter
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
                        "edition": edition
                    },
                    stream=True
                )
                response.raise_for_status()

                # Stream the answer
                st.markdown("### 🎯 Answer")
                message_placeholder = st.empty()
                full_response = ""
                metadata = None

                for chunk in response.iter_content(chunk_size=16, decode_unicode=True):
                    if chunk:
                        # Check if this chunk contains metadata
                        if "__METADATA__" in chunk:
                            parts = chunk.split("__METADATA__")
                            # Add text before metadata
                            if parts[0]:
                                full_response += parts[0]
                            # Parse metadata
                            try:
                                metadata = json.loads(parts[1])
                            except Exception as e:
                                logger.error(f"Metadata parse failed: {e}")
                            break  # Metadata is last
                        else:
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
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload Shadowrun rulebooks, session notes, or any other PDF documents"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.info(f"📄 **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)")
        
        with col2:
            if st.button("📤 Upload & Process", type="primary"):
                with st.spinner("Uploading and processing..."):
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    response = requests.post(f"{API_URL}/upload", files=files)
                    
                    if response.status_code == 200:
                        st.success("✅ PDF uploaded successfully! Processing in background...")
                        st.balloons()
                    else:
                        st.error(f"Upload failed: {response.text}")
    
    # Manual indexing
    st.markdown("---")
    st.subheader("Manual Indexing")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Re-index All Documents"):
            with st.spinner("Re-indexing all documents..."):
                response = api_request(
                    "/index",
                    method="POST",
                    json={"force_reindex": True}
                )
                if response:
                    st.success("✅ Re-indexing complete!")
    
    with col2:
        if st.button("📊 Index New Documents Only"):
            with st.spinner("Indexing new documents..."):
                response = api_request(
                    "/index",
                    method="POST",
                    json={"force_reindex": False}
                )
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