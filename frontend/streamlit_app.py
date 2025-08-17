"""Streamlit web UI with FIXED persistent WebSocket connection."""
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

# FIXED: Persistent WebSocket connection across reruns
class PersistentWebSocketHandler:
    """Persistent WebSocket handler that survives Streamlit reruns."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.ws = None
        self.running = False
        self.connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.global_progress = {}
        self.progress_lock = threading.Lock()
        self._initialized = True
        logger.info("🔧 PersistentWebSocketHandler initialized (singleton)")

    def on_message(self, ws, message):
        """Handle incoming progress messages."""
        try:
            data = json.loads(message)
            job_id = data.get('job_id')

            if job_id:
                with self.progress_lock:
                    self.global_progress[job_id] = data
                logger.info(f"WebSocket progress: {job_id} - {data.get('stage')} ({data.get('progress')}%)")

        except Exception as e:
            logger.error(f"WebSocket message error: {e}")

    def on_error(self, ws, error):
        """Handle WebSocket errors."""
        logger.warning(f"WebSocket error: {error}")
        self.connected = False

    def on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close."""
        logger.info(f"WebSocket closed: {close_status_code} - {close_msg}")
        self.connected = False

        # Attempt reconnection if we were running
        if self.running and self.reconnect_attempts < self.max_reconnect_attempts:
            self.reconnect_attempts += 1
            logger.info(f"Reconnecting WebSocket {self.reconnect_attempts}/{self.max_reconnect_attempts}")
            time.sleep(min(2 ** self.reconnect_attempts, 30))

            if self.running:
                self.connect()

    def on_open(self, ws):
        """Handle WebSocket open."""
        logger.info("WebSocket connected successfully")
        self.connected = True
        self.reconnect_attempts = 0

    def connect(self):
        """Connect to WebSocket (only if not already connected)."""
        with self.progress_lock:
            if self.connected:
                return

        try:
            ws_url = API_URL.replace('http://', 'ws://').replace('https://', 'wss://') + '/ws/progress'
            logger.info(f"Connecting to WebSocket: {ws_url}")

            # Test backend reachability first
            try:
                health_check = requests.get(f"{API_URL}/", timeout=5)
                logger.info(f"Backend health check: {health_check.status_code}")
            except Exception as e:
                logger.error(f"Backend not reachable: {e}")
                return

            self.ws = websocket.WebSocketApp(
                ws_url,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close,
                on_open=self.on_open
            )

            self.running = True
            self.ws.run_forever()

        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            with self.progress_lock:
                self.connected = False

    def disconnect(self):
        """Disconnect WebSocket."""
        self.running = False
        if self.ws:
            self.ws.close()
        with self.progress_lock:
            self.connected = False

    def is_connected(self):
        """Check if WebSocket is connected."""
        return self.connected

    def get_progress(self, job_id):
        """Get progress for specific job."""
        with self.progress_lock:
            return self.global_progress.get(job_id)

    def get_all_progress(self):
        """Get all progress data."""
        with self.progress_lock:
            return self.global_progress.copy()

# Get the singleton WebSocket handler
websocket_handler = PersistentWebSocketHandler()

# FIXED: Session state initialization (only initialize once)
if 'session_initialized' not in st.session_state:
    st.session_state.session_initialized = True
    st.session_state.websocket_connected = False
    st.session_state.websocket_thread = None
    st.session_state.query_input = ""
    st.session_state.ready_files = []
    st.session_state.processing_files = {}
    st.session_state.completed_files = {}
    st.session_state.uploader_key = 0
    st.session_state.last_refresh = time.time()

def start_websocket_connection():
    """Start WebSocket connection (only if not already running)."""
    if not websocket_handler.is_connected() and not st.session_state.websocket_thread:
        def run_websocket():
            try:
                logger.info("Starting WebSocket connection thread...")
                websocket_handler.connect()
            except Exception as e:
                logger.error(f"WebSocket thread error: {e}")

        thread = threading.Thread(target=run_websocket, daemon=True)
        thread.start()
        st.session_state.websocket_thread = thread
        logger.info("WebSocket thread started")

def poll_job_status(job_id: str) -> Optional[Dict]:
    """Poll job status as fallback when WebSocket is unavailable."""
    try:
        response = api_request(f"/job/{job_id}", timeout=3)
        if response and response.get('job_id'):
            return response
    except Exception as e:
        logger.warning(f"Polling failed for {job_id}: {e}")
    return None

def get_job_progress(job_id: str) -> Optional[Dict]:
    """Get job progress from WebSocket handler or polling."""
    progress_data = websocket_handler.get_progress(job_id)
    if progress_data:
        return progress_data

    if not websocket_handler.is_connected():
        return poll_job_status(job_id)

    return None

def sync_websocket_progress():
    """Sync progress from WebSocket handler to session state for UI updates."""
    all_progress = websocket_handler.get_all_progress()

    for job_id, progress_data in all_progress.items():
        for file_id, file_info in st.session_state.processing_files.items():
            if file_info.get('job_id') == job_id:
                file_info.update({
                    'stage': progress_data.get('stage', 'unknown'),
                    'progress': progress_data.get('progress', 0),
                    'details': progress_data.get('details', ''),
                    'timestamp': progress_data.get('timestamp', time.time())
                })

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

# Custom CSS
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

# Start WebSocket connection ONCE
start_websocket_connection()

# Title and description
st.title("🎲 Shadowrun RAG Assistant")
st.markdown("*Your AI-powered guide to the Sixth World*")

# Connection status indicator
col1, col2 = st.columns([4, 1])
with col2:
    # Update session state from persistent WebSocket handler
    st.session_state.websocket_connected = websocket_handler.is_connected()

    if st.session_state.websocket_connected:
        st.markdown('<span class="status-indicator status-connected"></span>**WebSocket Connected**',
                   unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-indicator status-disconnected"></span>**WebSocket Disconnected** (using polling)',
                   unsafe_allow_html=True)

# Debug info
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

# Auto-refresh (only when needed)
if not st.session_state.processing_files and (time.time() - st.session_state.last_refresh) > 30:
    st.session_state.last_refresh = time.time()
    st.rerun()

# Sidebar
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

    # Handle query
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

with tab2:
    st.header("Upload PDFs")

    # Process pending actions
    if 'action' in st.session_state:
        action = st.session_state.action

        if action['type'] == 'clear_all':
            st.session_state.ready_files = []
            st.session_state.uploader_key += 1

        elif action['type'] == 'process_all':
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

                st.session_state.upload_queue.append({
                    'file_id': file_info['id'],
                    'name': file_info['name'],
                    'file_data': file_info['file'].getvalue(),
                    'attempt': 0
                })

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
                    'job_id': 'pending',
                    'stage': 'preparing',
                    'progress': 0,
                    'details': 'Preparing to upload...',
                    'start_time': datetime.now()
                }

                st.session_state.upload_queue = st.session_state.get('upload_queue', [])
                st.session_state.upload_queue.append({
                    'file_id': file_id,
                    'name': file_info['name'],
                    'file_data': file_info['file'].getvalue(),
                    'attempt': 0
                })

                st.session_state.ready_files = [f for f in st.session_state.ready_files if f['id'] != file_id]
                if not st.session_state.ready_files:
                    st.session_state.uploader_key += 1

        del st.session_state.action
        st.rerun()

    # Process upload queue
    if 'upload_queue' in st.session_state and st.session_state.upload_queue:
        upload_item = st.session_state.upload_queue.pop(0)
        file_id = upload_item['file_id']

        if file_id in st.session_state.processing_files:
            try:
                st.session_state.processing_files[file_id]['stage'] = 'uploading'
                st.session_state.processing_files[file_id]['details'] = f"Uploading {upload_item['name']}..."

                files = {"file": (upload_item['name'], upload_item['file_data'], "application/pdf")}
                response = requests.post(f"{API_URL}/upload", files=files, timeout=30)

                if response.status_code == 200:
                    result = response.json()
                    job_id = result['job_id']

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

    # Add uploaded files
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

        if st.button("📤 Process All", type="primary"):
            st.session_state.action = {'type': 'process_all'}
            st.rerun()

        if st.button("🗑️ Clear All"):
            st.session_state.action = {'type': 'clear_all'}
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

    # Processing display
    if st.session_state.processing_files:
        st.markdown("---")
        st.subheader("🔄 Processing Files")

        sync_websocket_progress()

        for file_id, file_info in list(st.session_state.processing_files.items()):
            job_id = file_info.get('job_id')

            if job_id and job_id != 'pending':
                progress_data = get_job_progress(job_id)
                if progress_data:
                    file_info.update({
                        'stage': progress_data.get('stage', 'unknown'),
                        'progress': progress_data.get('progress', 0),
                        'details': progress_data.get('details', ''),
                        'timestamp': progress_data.get('timestamp', time.time())
                    })

            # Display file progress
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.write(f"📄 **{file_info['name']}** ({file_info['size'] / 1024:.1f} KB)")
            with col2:
                if job_id and job_id != 'pending':
                    st.caption(f"Job: {job_id[-8:]}")
            with col3:
                if websocket_handler.is_connected():
                    st.caption("🟢 Real-time")
                else:
                    st.caption("🟡 Polling")

            progress_val = max(0, min(100, file_info.get('progress', 0))) / 100.0

            if file_info.get('progress', 0) < 0:
                st.error("❌ Processing failed!")
            else:
                st.progress(progress_val)

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
                start_time = file_info.get('start_time')
                if start_time:
                    if isinstance(start_time, str):
                        try:
                            start_time = datetime.fromisoformat(start_time)
                        except:
                            start_time = datetime.now()
                    elapsed = (datetime.now() - start_time).total_seconds()
                    st.caption(f"⏱️ {elapsed:.1f}s elapsed")

            details = file_info.get('details', '')
            if details:
                st.caption(f"💬 {details}")

            if current_stage == 'table_detection':
                st.info("📊 **Table detection in progress** - This uses AI models and can take several minutes...")
            elif current_stage == 'extraction':
                st.info("🔍 **High-resolution extraction** - Processing document layout and structure...")
            elif current_stage == 'chunking':
                st.info("✂️ **Creating semantic chunks** - Breaking document into searchable sections...")

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

        # Smart refresh logic
        if st.session_state.processing_files:
            if websocket_handler.is_connected():
                time.sleep(1)
                st.rerun()
            else:
                time.sleep(3)
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
    st.info("📭 No documents indexed yet. Upload PDFs in the Upload tab to get started!")

with tab4:
    st.header("📝 Session Notes")
    st.info("🔧 Session notes functionality coming soon!")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <small>🎲 Shadowrun RAG Assistant v2.0 | Persistent WebSocket | Powered by Ollama & ChromaDB</small>
    </div>
    """,
    unsafe_allow_html=True
)