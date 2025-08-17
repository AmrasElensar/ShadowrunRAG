"""Streamlit web UI with FIXED WebSocket progress tracking + intelligent polling fallback."""
import os
import time
import threading
import json
from datetime import datetime
from typing import Dict, Optional

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

# FIXED: Global progress storage (not tied to session state)
if 'global_progress' not in st.session_state:
    st.session_state.global_progress = {}
if 'websocket_connected' not in st.session_state:
    st.session_state.websocket_connected = False
if 'websocket_thread' not in st.session_state:
    st.session_state.websocket_thread = None

# Initialize other session state
if 'query_input' not in st.session_state:
    st.session_state.query_input = ""
if 'ready_files' not in st.session_state:
    st.session_state.ready_files = []
if 'processing_files' not in st.session_state:
    st.session_state.processing_files = {}
if 'completed_files' not in st.session_state:
    st.session_state.completed_files = {}
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0

# FIXED: Thread-safe WebSocket handler
class ProgressWebSocketHandler:
    """Thread-safe WebSocket handler for real-time progress updates."""

    def __init__(self):
        self.ws = None
        self.running = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.connected = False  # Thread-safe connection status
        self.global_progress = {}  # Thread-safe progress storage
        self.lock = threading.Lock()

    def on_message(self, ws, message):
        """Handle incoming progress messages."""
        try:
            data = json.loads(message)
            job_id = data.get('job_id')

            if job_id:
                # Store in thread-safe global progress
                with self.lock:
                    self.global_progress[job_id] = data

                logger.info(f"WebSocket progress update: {job_id} - {data.get('stage')} ({data.get('progress')}%)")

        except Exception as e:
            logger.error(f"WebSocket message error: {e}")

    def on_error(self, ws, error):
        """Handle WebSocket errors."""
        logger.warning(f"WebSocket error: {error}")
        with self.lock:
            self.connected = False

    def on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close."""
        logger.info(f"WebSocket closed: {close_status_code} - {close_msg}")
        with self.lock:
            self.connected = False

        # Only attempt reconnection if we were deliberately running
        if self.running and self.reconnect_attempts < self.max_reconnect_attempts:
            self.reconnect_attempts += 1
            logger.info(f"Attempting WebSocket reconnection {self.reconnect_attempts}/{self.max_reconnect_attempts}")
            time.sleep(min(2 ** self.reconnect_attempts, 30))  # Cap at 30 seconds

            # Only reconnect if we're still supposed to be running
            if self.running:
                self.connect()

    def on_open(self, ws):
        """Handle WebSocket open."""
        logger.info("WebSocket connected successfully")
        with self.lock:
            self.connected = True
        self.reconnect_attempts = 0

    def connect(self):
        """Connect to WebSocket."""
        with self.lock:
            if self.connected:
                return

        try:
            ws_url = API_URL.replace('http://', 'ws://').replace('https://', 'wss://') + '/ws/progress'
            logger.info(f"Attempting WebSocket connection to: {ws_url}")

            # Test if the backend is reachable first
            try:
                import requests
                health_check = requests.get(f"{API_URL}/", timeout=5)
                logger.info(f"Backend health check: {health_check.status_code}")
            except Exception as e:
                logger.error(f"Backend not reachable: {e}")
                with self.lock:
                    self.connected = False
                return

            self.ws = websocket.WebSocketApp(
                ws_url,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close,
                on_open=self.on_open
            )

            self.running = True
            logger.info("Starting WebSocket run_forever...")
            self.ws.run_forever()

        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            with self.lock:
                self.connected = False

    def disconnect(self):
        """Disconnect WebSocket."""
        self.running = False
        if self.ws:
            self.ws.close()

    def is_connected(self):
        """Thread-safe connection status check."""
        with self.lock:
            return self.connected

    def get_progress(self, job_id):
        """Thread-safe progress retrieval."""
        with self.lock:
            return self.global_progress.get(job_id)

    def get_all_progress(self):
        """Get all progress data (thread-safe)."""
        with self.lock:
            return self.global_progress.copy()

# Global WebSocket handler
websocket_handler = ProgressWebSocketHandler()

def start_websocket_connection():
    """Start WebSocket connection in background thread."""
    if not st.session_state.websocket_connected and not st.session_state.websocket_thread:
        def run_websocket():
            websocket_handler.connect()

        thread = threading.Thread(target=run_websocket, daemon=True)
        thread.start()
        st.session_state.websocket_thread = thread
        logger.info("WebSocket thread started")

def poll_job_status(job_id: str) -> Optional[Dict]:
    """Poll job status as fallback when WebSocket is unavailable."""
    try:
        response = api_request(f"/job/{job_id}", timeout=3)
        if response and response.get('job_id'):
            # Store in global progress
            st.session_state.global_progress[job_id] = response
            return response
    except Exception as e:
        logger.warning(f"Polling failed for {job_id}: {e}")
    return None

def get_job_progress(job_id: str) -> Optional[Dict]:
    """Get job progress from global storage or polling."""
    # First check global progress storage
    if job_id in st.session_state.global_progress:
        return st.session_state.global_progress[job_id]

    # Fallback to polling if WebSocket not connected
    if not st.session_state.websocket_connected:
        return poll_job_status(job_id)

    return None

# Custom CSS (same as before)
st.markdown("""
<style>
:root {
    --bg-primary: #f0f2f6;
    --bg-secondary: #f8f9fa;
    --border-color: #0066cc;
    --text-color: #111;
}

[data-testid="stApp"] [data-baseweb="tab-list"] {
    gap: 24px;
}

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

.stMarkdown {
    color: var(--text-color);
}

.progress-stage {
    font-size: 0.9em;
    color: #666;
    margin-bottom: 10px;
}

.status-indicator {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    margin-right: 8px;
}

.status-connected { background-color: #28a745; }
.status-disconnected { background-color: #dc3545; }
.status-connecting { background-color: #ffc107; }
</style>
""", unsafe_allow_html=True)

# Start WebSocket connection
start_websocket_connection()

# Auto-refresh for connection status (only if not actively processing and not too frequent)
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()

# Only refresh every 30 seconds and only when not processing
if not st.session_state.processing_files and (time.time() - st.session_state.last_refresh) > 30:
    st.session_state.last_refresh = time.time()
    st.rerun()

@st.cache_data(ttl=30)
def get_available_models():
    """Fetch available models with caching."""
    try:
        response = api_request("/models", timeout=5)
        models = response.get("models", ["llama3"]) if response else ["llama3"]
        return models if models else ["llama3"]
    except Exception as e:
        logger.warning(f"Failed to fetch models: {e}")
        return ["llama3"]

@st.cache_data(ttl=30)
def get_available_documents():
    """Fetch available documents with caching."""
    try:
        response = api_request("/documents", timeout=15)
        docs = response.get("documents", []) if response else []
        return docs if isinstance(docs, list) else []
    except Exception as e:
        logger.warning(f"Failed to fetch documents: {e}")
        return []

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

# Start WebSocket connection immediately after initialization
start_websocket_connection()

# Title and description
st.title("🎲 Shadowrun RAG Assistant")
st.markdown("*Your AI-powered guide to the Sixth World*")

# Connection status indicator with debugging
col1, col2 = st.columns([4, 1])
with col2:
    # Update session state from thread-safe WebSocket handler
    st.session_state.websocket_connected = websocket_handler.is_connected()

    # Try to restart connection if it's been disconnected for a while
    if not st.session_state.websocket_connected and st.session_state.websocket_thread is None:
        logger.info("WebSocket not connected, attempting to restart...")
        start_websocket_connection()

    if st.session_state.websocket_connected:
        st.markdown('<span class="status-indicator status-connected"></span>**WebSocket Connected**',
                   unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-indicator status-disconnected"></span>**WebSocket Disconnected** (using polling)',
                   unsafe_allow_html=True)

# Debug info (remove this after testing)
with st.expander("🔧 WebSocket Debug Info"):
    st.write(f"**WebSocket URL:** {API_URL.replace('http://', 'ws://').replace('https://', 'wss://')}/ws/progress")
    st.write(f"**Thread alive:** {st.session_state.websocket_thread and st.session_state.websocket_thread.is_alive() if st.session_state.websocket_thread else 'No thread'}")
    st.write(f"**Handler connected:** {websocket_handler.is_connected()}")
    st.write(f"**Reconnect attempts:** {websocket_handler.reconnect_attempts}")

    if st.button("🔄 Force Reconnect"):
        # Reset the connection
        st.session_state.websocket_thread = None
        websocket_handler.disconnect()
        websocket_handler.reconnect_attempts = 0
        start_websocket_connection()
        st.rerun()

# Sidebar (same as before, shortened for brevity)
with st.sidebar:
    st.header("⚙️ Configuration")

    available_models = get_available_models()
    selected_model = st.selectbox("LLM Model", available_models, index=0, key="model_selectbox")

    st.subheader("Query Settings")
    n_results = st.slider("Number of sources", 1, 10, 5)
    query_type = st.selectbox(
        "Query Type",
        ["general", "rules", "session"],
        format_func=lambda x: {"general": "General Query", "rules": "Rules Question", "session": "Session History"}[x]
    )

    st.subheader("👤 Character Context")
    character_role = st.selectbox(
        "Character Role (optional)",
        ["None", "Decker", "Mage", "Street Samurai", "Rigger", "Adept", "Technomancer", "Face"],
        index=0
    )
    character_role = None if character_role == "None" else character_role.lower().replace(" ", "_")

    character_stats = st.text_input("Character Stats (optional)", placeholder="e.g., Logic 6, Hacking 5, Sleaze 4")

    edition = st.selectbox("Preferred Edition (optional)", ["None", "SR5", "SR6", "SR4"], index=0)
    edition = None if edition == "None" else edition

    # Filters
    section_options = ["All", "Combat", "Matrix", "Magic", "Gear", "Character Creation", "Riggers", "Technomancy"]
    filter_section = st.selectbox("Filter by Section (optional)", section_options, index=0)
    filter_section = None if filter_section == "All" else filter_section

    filter_subsection = st.text_input("Filter by Subsection (optional)", placeholder="e.g., Hacking, Spellcasting, Initiative")
    filter_subsection = None if filter_subsection.strip() == "" else filter_subsection

# Main interface tabs
tab1, tab2, tab3, tab4 = st.tabs(["💬 Query", "📤 Upload", "📚 Documents", "📝 Session Notes"])

with tab1:
    st.header("Ask a Question")

    col1, col2 = st.columns([5, 1])
    with col1:
        query = st.text_area(
            "Your question:",
            value=st.session_state.query_input,
            key="query_input",
            placeholder="e.g., 'How do recoil penalties work in Shadowrun 5e?' or 'What happened in our last session?'",
            height=100
        )

    with col2:
        st.write("")
        st.write("")
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)

    # Handle query (same as before - query logic unchanged)
    if search_button and query:
        with st.spinner("Streaming from the shadows..."):
            try:
                where_filter = {}

                role_to_section = {
                    "decker": "Matrix", "hacker": "Matrix", "mage": "Magic", "adept": "Magic",
                    "street_samurai": "Combat", "rigger": "Riggers", "technomancer": "Technomancy"
                }

                if filter_section:
                    where_filter["Section"] = filter_section
                else:
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

                st.markdown("### 🎯 Answer")
                message_placeholder = st.empty()
                full_response = ""
                metadata = None
                collecting_metadata = False
                metadata_buffer = ""

                for chunk in response.iter_content(chunk_size=32, decode_unicode=True):
                    if chunk:
                        if "__METADATA_START__" in chunk:
                            parts = chunk.split("__METADATA_START__")
                            if parts[0]:
                                full_response += parts[0]
                                message_placeholder.markdown(full_response + "▌")

                            collecting_metadata = True
                            if len(parts) > 1:
                                metadata_buffer = parts[1]
                            continue

                        if collecting_metadata:
                            metadata_buffer += chunk

                            if "__METADATA_END__" in metadata_buffer:
                                json_part = metadata_buffer.split("__METADATA_END__")[0].strip()
                                try:
                                    metadata = json.loads(json_part)
                                    logger.info("Successfully parsed metadata")
                                except json.JSONDecodeError as e:
                                    logger.error(f"Metadata parse failed: {e}")
                                break
                        else:
                            full_response += chunk
                            message_placeholder.markdown(full_response + "▌")

                message_placeholder.markdown(full_response)

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

def sync_websocket_progress():
    """Sync progress from WebSocket handler to session state for UI updates."""
    # Get all progress from thread-safe WebSocket handler
    all_progress = websocket_handler.get_all_progress()

    # Update session state for UI (only in main thread)
    for job_id, progress_data in all_progress.items():
        # Find matching processing files and update them
        for file_id, file_info in st.session_state.processing_files.items():
            if file_info.get('job_id') == job_id:
                file_info.update({
                    'stage': progress_data.get('stage', 'unknown'),
                    'progress': progress_data.get('progress', 0),
                    'details': progress_data.get('details', ''),
                    'timestamp': progress_data.get('timestamp', time.time())
                })

with tab2:
    st.header("Upload PDFs")

    # CSS to hide file uploader file list
    st.markdown("""
    <style>
    div[data-testid="stFileUploader"] > div:last-child {
        display: none !important;
    }
    
    div[data-testid="stFileUploader"] ul {
        display: none !important;
    }
    
    .stFileUploaderFile {
        display: none !important;
    }
    
    div[data-testid="stFileUploaderFile"] {
        display: none !important;
    }
    
    .stFileUploader > div:last-of-type {
        display: none !important;
    }
    
    div[data-testid="stFileUploaderFileName"] {
        display: none !important;
    }
    
    div[data-testid="stFileUploaderDeleteBtn"] {
        display: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

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
                    'job_id': 'pending',
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
                    'job_id': 'pending',
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
            st.rerun()

        for file_info in st.session_state.ready_files:
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

    # ENHANCED processing display with real progress tracking
    if st.session_state.processing_files:
        st.markdown("---")
        st.subheader("🔄 Processing Files")

        # Sync progress from WebSocket handler to session state
        sync_websocket_progress()

        for file_id, file_info in list(st.session_state.processing_files.items()):
            job_id = file_info.get('job_id')

            # Get REAL progress from WebSocket handler or polling
            if job_id and job_id != 'pending':
                progress_data = get_job_progress(job_id)
                if progress_data:
                    # Update file info with REAL progress
                    file_info.update({
                        'stage': progress_data.get('stage', 'unknown'),
                        'progress': progress_data.get('progress', 0),
                        'details': progress_data.get('details', ''),
                        'timestamp': progress_data.get('timestamp', time.time())
                    })

            # Display file header
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.write(f"📄 **{file_info['name']}** ({file_info['size'] / 1024:.1f} KB)")
            with col2:
                if job_id and job_id != 'pending':
                    st.caption(f"Job: {job_id[-8:]}")
            with col3:
                # Connection status for this job
                if websocket_handler.is_connected():
                    st.caption("🟢 Real-time")
                else:
                    st.caption("🟡 Polling")

            # Progress bar with real values
            progress_val = max(0, min(100, file_info.get('progress', 0))) / 100.0

            if file_info.get('progress', 0) < 0:  # Error case
                st.error("❌ Processing failed!")
            else:
                st.progress(progress_val)

            # ENHANCED stage indicator with real stages from unstructured
            stage_info = {
                'starting': ('🚀', 'Initializing'),
                'reading': ('📖', 'Reading PDF'),
                'extraction': ('🔍', 'Extracting Elements'),
                'table_detection': ('📊', 'Table Detection'),
                'extraction_complete': ('✅', 'Extraction Complete'),
                'analyzing': ('🔬', 'Analyzing Structure'),
                'cleaning': ('🧹', 'Cleaning Content'),
                'chunking': ('✂️', 'Creating Chunks'),
                'chunking_complete': ('✅', 'Chunking Complete'),
                'saving': ('💾', 'Saving Files'),
                'complete': ('🎉', 'Complete'),
                'error': ('❌', 'Error'),
                'uploaded': ('📤', 'Uploaded'),
                'pending': ('⏳', 'Pending')
            }

            current_stage = file_info.get('stage', 'unknown')
            stage_icon, stage_label = stage_info.get(current_stage, ('⚙️', current_stage.title()))

            col1, col2, col3 = st.columns([2, 1, 2])
            with col1:
                st.write(f"{stage_icon} **{stage_label}**")
            with col2:
                st.write(f"**{file_info.get('progress', 0):.0f}%**")
            with col3:
                # Show elapsed time if available
                start_time = file_info.get('start_time')
                if start_time:
                    if isinstance(start_time, str):
                        # Convert string timestamp if needed
                        try:
                            start_time = datetime.fromisoformat(start_time)
                        except:
                            start_time = datetime.now()
                    elapsed = (datetime.now() - start_time).total_seconds()
                    st.caption(f"⏱️ {elapsed:.1f}s elapsed")

            # REAL details from unstructured logging
            details = file_info.get('details', '')
            if details:
                st.caption(f"💬 {details}")

            # Special alerts for long-running stages
            if current_stage == 'table_detection':
                st.info("📊 **Table detection in progress** - This uses AI models and can take several minutes for complex documents...")
            elif current_stage == 'extraction':
                st.info("🔍 **High-resolution extraction** - Processing document layout and structure...")
            elif current_stage == 'chunking':
                st.info("✂️ **Creating semantic chunks** - Breaking document into searchable sections...")

            # Handle completion and errors
            if current_stage == 'complete':
                st.session_state.completed_files[file_id] = file_info
                del st.session_state.processing_files[file_id]
                st.rerun()
            elif current_stage == 'error':
                st.error(f"❌ Processing failed: {details}")
                file_info['status'] = 'error'
                st.session_state.completed_files[file_id] = file_info
                del st.session_state.processing_files[file_id]
                st.rerun()

            st.markdown("---")

        # Smart refresh logic: Real-time via WebSocket or polling fallback
        if st.session_state.processing_files:
            if websocket_handler.is_connected():
                # WebSocket connected: minimal refresh rate
                time.sleep(1)
                st.rerun()
            else:
                # WebSocket disconnected: use polling
                time.sleep(3)  # Slightly longer interval for polling
                st.rerun()

    # Completed section
    if st.session_state.completed_files:
        st.markdown("---")
        st.subheader("✅ Completed")

        for file_id, file_info in st.session_state.completed_files.items():
            if file_info.get('status') == 'error':
                st.error(f"❌ **{file_info['name']}** - Processing failed!")
                details = file_info.get('details', 'Unknown error')
                st.caption(f"Error: {details}")
            else:
                st.success(f"✅ **{file_info['name']}** - Successfully processed!")
                # Show final stats if available
                if file_info.get('details'):
                    st.caption(f"Result: {file_info['details']}")

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

    # Get document list
    with st.spinner("Loading document library..."):
        docs_response = api_request("/documents", timeout=20)

    if docs_response and docs_response.get("documents"):
        all_file_paths = docs_response["documents"]

        # Group files by their parent document
        documents_by_source = {}
        for file_path in all_file_paths:
            path_obj = Path(file_path)
            parent_folder = path_obj.parent.name

            if parent_folder == "processed_markdown":
                parent_folder = "Uncategorized"

            if parent_folder not in documents_by_source:
                documents_by_source[parent_folder] = []

            documents_by_source[parent_folder].append(file_path)

        # Filter by search term
        if search_docs:
            filtered_docs = {}
            search_lower = search_docs.lower()
            for doc_name, files in documents_by_source.items():
                if search_lower in doc_name.lower():
                    filtered_docs[doc_name] = files
                else:
                    matching_files = [f for f in files if search_lower in Path(f).name.lower()]
                    if matching_files:
                        filtered_docs[doc_name] = matching_files
            documents_by_source = filtered_docs

        # Categorize documents
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

        # Display each category (same as before)
        for category_name, category_docs in doc_categories.items():
            if category_docs:
                st.markdown(f"### {category_name}")

                for doc_name, files in sorted(category_docs.items()):
                    with st.expander(f"📖 **{doc_name}** ({len(files)} chunks)"):
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            if st.button(f"🎯 Search This Doc", key=f"search_doc_{doc_name}"):
                                st.session_state.doc_filter = doc_name
                                st.info(f"💡 Switch to Query tab - document filter is set to '{doc_name}'")

                        with col2:
                            if st.button(f"📊 Show Stats", key=f"stats_{doc_name}"):
                                st.session_state.show_stats = doc_name

                        with col3:
                            if st.button(f"📄 List Chunks", key=f"chunks_{doc_name}"):
                                st.session_state.show_chunks = doc_name

                        with col4:
                            estimated_pages = max(1, len(files) // 20)
                            st.caption(f"📄 ~{estimated_pages} pages")

                        # Show document statistics if requested
                        if st.session_state.get('show_stats') == doc_name:
                            st.markdown("#### 📊 Document Statistics")

                            chunk_count = len(files)
                            avg_chunk_size = "~500 words"

                            stat_col1, stat_col2, stat_col3 = st.columns(3)
                            with stat_col1:
                                st.metric("Chunks", chunk_count)
                            with stat_col2:
                                st.metric("Avg Chunk", avg_chunk_size)
                            with stat_col3:
                                st.metric("Est. Pages", estimated_pages)

                            section_names = []
                            for file_path in files[:10]:
                                chunk_name = Path(file_path).stem
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

                            for i, file_path in enumerate(files[:20]):
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
        <small>🎲 Shadowrun RAG Assistant v2.0 | Real-time Progress Tracking | Powered by Ollama & ChromaDB</small>
    </div>
    """,
    unsafe_allow_html=True
)