"""Streamlit web UI for Shadowrun RAG system with WebSocket progress tracking."""
import os
import time
import threading
import json
from datetime import datetime

import streamlit as st
import requests
from pathlib import Path
import logging
import websocket

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
if 'job_progress' not in st.session_state:
    st.session_state.job_progress = {}
if 'ws_connected' not in st.session_state:
    st.session_state.ws_connected = False

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

.progress-stage {
    font-size: 0.9em;
    color: #666;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# WebSocket functions
def on_progress_message(ws, message):
    """Handle incoming WebSocket progress messages."""
    try:
        data = json.loads(message)
        job_id = data['job_id']

        # Check if session state exists and is accessible
        if hasattr(st, 'session_state') and hasattr(st.session_state, 'job_progress'):
            st.session_state.job_progress[job_id] = data
            logger.info(f"Progress update: {job_id} - {data['stage']} ({data['progress']}%)")
        else:
            logger.warning(f"Session state not accessible for progress update: {job_id}")

    except Exception as e:
        logger.error(f"Progress message error: {e}")

def on_ws_error(ws, error):
    logger.error(f"WebSocket error: {error}")
    st.session_state.ws_connected = False

def on_ws_close(ws, close_status_code, close_msg):
    logger.info("WebSocket connection closed")
    st.session_state.ws_connected = False

def connect_progress_websocket():
    """Connect to the progress WebSocket."""
    if st.session_state.ws_connected:
        return

    try:
        # Convert HTTP URL to WebSocket URL
        ws_url = API_URL.replace('http://', 'ws://').replace('https://', 'wss://') + '/ws/progress'
        logger.info(f"Connecting to WebSocket: {ws_url}")

        ws = websocket.WebSocketApp(
            ws_url,
            on_message=on_progress_message,
            on_error=on_ws_error,
            on_close=on_ws_close
        )

        def run_websocket():
            ws.run_forever()

        # Run WebSocket in background thread
        ws_thread = threading.Thread(target=run_websocket, daemon=True)
        ws_thread.start()
        st.session_state.ws_connected = True
        logger.info("WebSocket connection started")

    except Exception as e:
        logger.error(f"Failed to connect WebSocket: {e}")

# Auto-connect WebSocket on app start
if not st.session_state.ws_connected:
    connect_progress_websocket()

@st.cache_data(ttl=30)  # Cache for 30 seconds
def get_available_models():
    """Fetch available models with caching."""
    try:
        response = api_request("/models", timeout=5)  # Add timeout
        models = response.get("models", ["llama3"]) if response else ["llama3"]
        # Ensure we always return at least one model to prevent empty selectbox
        return models if models else ["llama3"]
    except Exception as e:
        logger.warning(f"Failed to fetch models: {e}")
        return ["llama3"]  # Always return a list with at least one item

@st.cache_data(ttl=30)  # Cache for 30 seconds
def get_available_documents():
    """Fetch available documents with caching."""
    try:
        # Longer timeout for documents as it might need to query ChromaDB
        response = api_request("/documents", timeout=15)  # Increased from 5 to 15 seconds
        docs = response.get("documents", []) if response else []
        return docs if isinstance(docs, list) else []  # Ensure it's always a list
    except Exception as e:
        logger.warning(f"Failed to fetch documents: {e}")
        return []  # Always return a list

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

    # Initialize the selectbox with a key to prevent state errors
    selected_model = st.selectbox(
        "LLM Model",
        available_models,
        index=0,
        key="model_selectbox"  # Add explicit key
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

    # Note: Document source filter moved to Documents tab
    filter_source = None

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
                        "model": selected_model
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

    # CSS to hide file uploader file list - Using stable selectors
    st.markdown("""
    <style>
    /* Target the file list container directly - more reliable approach */
    div[data-testid="stFileUploader"] > div:last-child {
        display: none !important;
    }
    
    /* Alternative: Hide any element containing uploaded file data */
    div[data-testid="stFileUploader"] ul {
        display: none !important;
    }
    
    /* Hide individual file items */
    .stFileUploaderFile {
        display: none !important;
    }
    
    /* Hide file data sections */
    div[data-testid="stFileUploaderFile"] {
        display: none !important;
    }
    
    /* Target by structure: file uploader > last div (which contains the file list) */
    .stFileUploader > div:last-of-type {
        display: none !important;
    }
    
    /* Hide elements with file names */
    div[data-testid="stFileUploaderFileName"] {
        display: none !important;
    }
    
    /* Hide delete buttons */
    div[data-testid="stFileUploaderDeleteBtn"] {
        display: none !important;
    }
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
            # IMMEDIATELY move all files to processing view
            st.session_state.upload_queue = st.session_state.get('upload_queue', [])

            for file_info in st.session_state.ready_files:
                st.session_state.processing_files[file_info['id']] = {
                    'name': file_info['name'],
                    'size': file_info['size'],
                    'job_id': 'pending',  # Special marker
                    'stage': 'preparing',
                    'progress': 0,
                    'details': 'Preparing to upload...',
                    'start_time': datetime.now()
                }

                # Queue for upload
                st.session_state.upload_queue.append({
                    'file_id': file_info['id'],
                    'name': file_info['name'],
                    'file_data': file_info['file'].getvalue(),
                    'attempt': 0
                })

            # Clear ready files
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
                # IMMEDIATELY move to processing view (no upload yet)
                st.session_state.processing_files[file_id] = {
                    'name': file_info['name'],
                    'size': file_info['size'],
                    'job_id': 'pending',  # Special marker
                    'stage': 'preparing',
                    'progress': 0,
                    'details': 'Preparing to upload...',
                    'start_time': datetime.now()
                }

                # Queue the upload for background processing
                st.session_state.upload_queue = st.session_state.get('upload_queue', [])
                st.session_state.upload_queue.append({
                    'file_id': file_id,
                    'name': file_info['name'],
                    'file_data': file_info['file'].getvalue(),
                    'attempt': 0
                })

                # Remove from ready files
                st.session_state.ready_files = [f for f in st.session_state.ready_files if f['id'] != file_id]
                if not st.session_state.ready_files:
                    st.session_state.uploader_key += 1

        del st.session_state.action
        st.rerun()  # Single rerun after all actions

    # Process upload queue (one file per page load to avoid blocking)
    if 'upload_queue' in st.session_state and st.session_state.upload_queue:
        # Only process ONE upload per page load to keep UI responsive
        upload_item = st.session_state.upload_queue.pop(0)
        file_id = upload_item['file_id']

        if file_id in st.session_state.processing_files:
            try:
                # Update status
                st.session_state.processing_files[file_id]['stage'] = 'uploading'
                st.session_state.processing_files[file_id]['details'] = f"Uploading {upload_item['name']}..."

                # Perform upload with longer timeout since we're only uploading, not processing
                files = {"file": (upload_item['name'], upload_item['file_data'], "application/pdf")}

                # Upload should be fast now - only file transfer, processing happens async
                response = requests.post(f"{API_URL}/upload", files=files, timeout=30)

                if response.status_code == 200:
                    result = response.json()
                    job_id = result['job_id']

                    # Update with real job ID
                    st.session_state.processing_files[file_id].update({
                        'job_id': job_id,
                        'stage': 'uploaded',
                        'progress': 5,
                        'details': 'Upload complete, processing started...'
                    })

                    st.success(f"✅ {upload_item['name']} uploaded! Job ID: {job_id}")
                else:
                    st.error(f"❌ Upload failed for {upload_item['name']}: {response.text}")
                    del st.session_state.processing_files[file_id]

            except Exception as e:
                st.error(f"❌ Upload error for {upload_item['name']}: {str(e)}")
                del st.session_state.processing_files[file_id]

        # If there are more uploads in queue, trigger another rerun
        if st.session_state.upload_queue:
            st.rerun()

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

    # Enhanced processing display with manual progress polling instead of WebSocket session state
    if st.session_state.processing_files:
        st.markdown("---")
        st.subheader("🔄 Processing Files")

        for file_id, file_info in list(st.session_state.processing_files.items()):
            job_id = file_info.get('job_id')

            # Poll backend for progress instead of relying on WebSocket session state
            if job_id and job_id != 'pending':
                try:
                    # Get current job status from backend
                    status_response = api_request(f"/job/{job_id}", timeout=5)
                    if status_response and status_response.get('stage'):
                        # Update file info with backend data
                        file_info.update({
                            'stage': status_response.get('stage', 'unknown'),
                            'progress': status_response.get('progress', 0),
                            'details': status_response.get('details', ''),
                            'elapsed_time': status_response.get('elapsed_time', 0)
                        })
                except Exception as e:
                    logger.warning(f"Failed to poll job status: {e}")

            # Display file header
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"📄 **{file_info['name']}** ({file_info['size'] / 1024:.1f} KB)")
            with col2:
                if job_id and job_id != 'pending':
                    st.caption(f"Job: {job_id[-8:]}")  # Show last 8 chars of job ID

            # Progress bar
            progress_val = max(0, min(100, file_info.get('progress', 0))) / 100.0

            if file_info.get('progress', 0) < 0:  # Error case
                st.error("❌ Processing failed!")
            else:
                st.progress(progress_val)

            # Stage indicator with detailed icons
            stage_info = {
                'starting': ('🚀', 'Initializing'),
                'reading': ('📖', 'Reading PDF'),
                'extraction': ('📝', 'Extracting Elements'),
                'table_detection': ('📊', 'Table Detection'),
                'cleaning': ('🧹', 'Cleaning Content'),
                'chunking': ('✂️', 'Creating Chunks'),
                'saving': ('💾', 'Saving Files'),
                'indexing': ('🔍', 'Indexing for Search'),
                'complete': ('🎉', 'Complete'),
                'error': ('❌', 'Error'),
                'uploaded': ('📤', 'Uploaded - Ready'),
                'pending': ('⏳', 'Pending Upload')
            }

            current_stage = file_info.get('stage', 'unknown')
            stage_icon, stage_label = stage_info.get(current_stage, ('⚙️', current_stage.title()))

            col1, col2, col3 = st.columns([2, 1, 2])
            with col1:
                st.write(f"{stage_icon} **{stage_label}**")
            with col2:
                st.write(f"**{file_info.get('progress', 0):.0f}%**")
            with col3:
                elapsed = file_info.get('elapsed_time', 0)
                if elapsed > 0:
                    st.caption(f"⏱️ {elapsed:.1f}s elapsed")

            # Details and special messaging
            details = file_info.get('details', '')
            if details:
                st.caption(f"💬 {details}")

            # Special alerts for long-running stages
            if current_stage in ['extraction', 'table_detection']:
                st.info("📊 **Table detection in progress** - This can take several minutes for complex documents with many tables...")
            elif current_stage == 'chunking':
                st.info("✂️ **Creating semantic chunks** - Breaking document into searchable sections...")
            elif current_stage == 'indexing':
                st.info("🔍 **Indexing for search** - Making content searchable...")

            # Handle completion
            if current_stage == 'complete':
                # Move to completed
                st.session_state.completed_files[file_id] = file_info
                del st.session_state.processing_files[file_id]
                st.rerun()

            # Handle errors
            elif current_stage == 'error':
                st.error(f"❌ Processing failed: {details}")
                # Move to completed with error status
                file_info['status'] = 'error'
                st.session_state.completed_files[file_id] = file_info
                del st.session_state.processing_files[file_id]
                st.rerun()

            st.markdown("---")

        # Auto-refresh for real-time updates every 2 seconds (instead of WebSocket)
        if st.session_state.processing_files:
            time.sleep(2)
            st.rerun()

    # Completed section
    if st.session_state.completed_files:
        st.markdown("---")
        st.subheader("✅ Completed")

        for file_id, file_info in st.session_state.completed_files.items():
            if file_info.get('status') == 'error':
                st.error(f"❌ **{file_info['name']}** - Processing failed!")
            else:
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
    st.header("📚 Document Library")

    # Only load documents when this tab is active - no more sidebar performance hit!
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🔄 Refresh", type="primary"):
            st.cache_data.clear()
            st.rerun()
    with col2:
        search_docs = st.text_input(
            "🔍 Search documents",
            placeholder="Type to filter documents...",
            key="doc_search"
        )

    # Get document list (only when this tab is used)
    with st.spinner("Loading document library..."):
        docs_response = api_request("/documents", timeout=20)

    if docs_response and docs_response.get("documents"):
        all_file_paths = docs_response["documents"]

        # Group files by their parent document (folder name)
        documents_by_source = {}
        for file_path in all_file_paths:
            path_obj = Path(file_path)
            # Get the parent folder name (which should be the PDF name)
            parent_folder = path_obj.parent.name

            # Skip if it's just "processed_markdown" (top level files)
            if parent_folder == "processed_markdown":
                parent_folder = "Uncategorized"

            if parent_folder not in documents_by_source:
                documents_by_source[parent_folder] = []

            documents_by_source[parent_folder].append(file_path)

        # Filter by search term (search both document names and file content)
        if search_docs:
            filtered_docs = {}
            search_lower = search_docs.lower()
            for doc_name, files in documents_by_source.items():
                if search_lower in doc_name.lower():
                    # Document name matches
                    filtered_docs[doc_name] = files
                else:
                    # Check if any files match
                    matching_files = [f for f in files if search_lower in Path(f).name.lower()]
                    if matching_files:
                        filtered_docs[doc_name] = matching_files
            documents_by_source = filtered_docs

        # Categorize documents by type
        doc_categories = {
            "📘 Core Rulebooks": {},
            "📖 Supplements & Expansions": {},
            "📝 Session Logs": {},
            "📄 Other Documents": {}
        }

        for doc_name, files in documents_by_source.items():
            doc_lower = doc_name.lower()
            if any(keyword in doc_lower for keyword in ["core", "basic", "main", "rulebook"]):
                doc_categories["📘 Core Rulebooks"][doc_name] = files
            elif "session" in doc_lower:
                doc_categories["📝 Session Logs"][doc_name] = files
            elif any(keyword in doc_lower for keyword in ["supplement", "expansion", "addon", "chrome", "arsenal", "matrix", "magic"]):
                doc_categories["📖 Supplements & Expansions"][doc_name] = files
            else:
                doc_categories["📄 Other Documents"][doc_name] = files

        # Display statistics
        total_documents = len(documents_by_source)
        total_files = sum(len(files) for files in documents_by_source.values())

        col1, col2 = st.columns(2)
        with col1:
            st.metric("📚 Documents", total_documents)
        with col2:
            st.metric("📄 Total Chunks", total_files)

        if search_docs:
            st.info(f"🔍 Showing {total_documents} documents matching '{search_docs}'")

        # Display each category
        for category_name, category_docs in doc_categories.items():
            if category_docs:
                st.markdown(f"### {category_name}")

                for doc_name, files in sorted(category_docs.items()):
                    with st.expander(f"📖 **{doc_name}** ({len(files)} chunks)"):

                        # Document-level actions
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            if st.button(f"🎯 Search This Doc", key=f"search_doc_{doc_name}"):
                                # Set filter and switch to query tab
                                st.session_state.doc_filter = doc_name
                                st.info(f"💡 Switch to Query tab - document filter is set to '{doc_name}'")

                        with col2:
                            if st.button(f"📊 Show Stats", key=f"stats_{doc_name}"):
                                st.session_state.show_stats = doc_name

                        with col3:
                            if st.button(f"📄 List Chunks", key=f"chunks_{doc_name}"):
                                st.session_state.show_chunks = doc_name

                        with col4:
                            # Calculate estimated pages (rough estimate: 20 chunks per page)
                            estimated_pages = max(1, len(files) // 20)
                            st.caption(f"📄 ~{estimated_pages} pages")

                        # Show document statistics if requested
                        if st.session_state.get('show_stats') == doc_name:
                            st.markdown("#### 📊 Document Statistics")

                            # Basic stats
                            chunk_count = len(files)
                            avg_chunk_size = "~500 words"  # Rough estimate

                            stat_col1, stat_col2, stat_col3 = st.columns(3)
                            with stat_col1:
                                st.metric("Chunks", chunk_count)
                            with stat_col2:
                                st.metric("Avg Chunk", avg_chunk_size)
                            with stat_col3:
                                st.metric("Est. Pages", estimated_pages)

                            # Top-level sections (based on chunk file names)
                            section_names = []
                            for file_path in files[:10]:  # Sample first 10
                                chunk_name = Path(file_path).stem
                                # Clean up chunk names for section detection
                                clean_name = chunk_name.replace('_', ' ').title()
                                section_names.append(clean_name[:30] + "..." if len(clean_name) > 30 else clean_name)

                            if section_names:
                                st.markdown("**Sample Sections:**")
                                for section in section_names[:5]:
                                    st.caption(f"• {section}")
                                if len(files) > 5:
                                    st.caption(f"... and {len(files) - 5} more sections")

                        # Show chunk list if requested
                        if st.session_state.get('show_chunks') == doc_name:
                            st.markdown("#### 📄 Document Chunks")

                            # Show chunks in a more organized way
                            for i, file_path in enumerate(files[:20]):  # Limit to first 20
                                chunk_name = Path(file_path).stem
                                clean_name = chunk_name.replace('_', ' ').title()

                                chunk_col1, chunk_col2 = st.columns([3, 1])
                                with chunk_col1:
                                    st.caption(f"{i+1}. {clean_name}")
                                with chunk_col2:
                                    if st.button("🔍", help="Search this chunk", key=f"search_chunk_{i}_{doc_name}"):
                                        st.session_state.chunk_filter = clean_name
                                        st.info(f"💡 Switch to Query tab - searching for content from '{clean_name}'")

                            if len(files) > 20:
                                st.caption(f"... and {len(files) - 20} more chunks")

                        # Clear buttons for opened sections
                        if st.session_state.get('show_stats') == doc_name or st.session_state.get('show_chunks') == doc_name:
                            if st.button("🔼 Collapse", key=f"collapse_{doc_name}"):
                                if 'show_stats' in st.session_state:
                                    del st.session_state.show_stats
                                if 'show_chunks' in st.session_state:
                                    del st.session_state.show_chunks
                                st.rerun()

    else:
        st.info("📭 No documents indexed yet. Upload PDFs in the Upload tab to get started!")

        # Show sample structure
        st.markdown("### 💡 How it works")
        st.markdown("""
        When you upload PDFs, they get processed into searchable chunks:
        - **📚 Each PDF** becomes a document group
        - **📄 Each page/section** becomes a searchable chunk  
        - **🔍 Search** can find content across all documents
        - **🎯 Filter** lets you search within specific documents
        """)

        if st.button("📤 Upload Documents", type="primary"):
            st.info("👆 Click the 'Upload' tab above to add your first documents!")

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