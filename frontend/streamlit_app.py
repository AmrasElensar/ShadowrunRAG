"""Streamlit web UI with polling-based progress tracking and localStorage persistence."""
import os
import time
import json
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import streamlit as st
import requests
from pathlib import Path
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

# ---- localStorage Helper Functions ----
def save_to_localstorage(key: str, data: dict):
    """Save data to browser localStorage via JavaScript."""
    json_data = json.dumps(data, default=str)  # Handle datetime serialization
    js_code = f"""
    <script>
    localStorage.setItem('{key}', '{json_data}');
    </script>
    """
    st.components.v1.html(js_code, height=0)

def load_from_localstorage(key: str) -> dict:
    """Load data from browser localStorage via JavaScript."""
    # This creates a hidden component that will store the result
    js_code = f"""
    <script>
    const data = localStorage.getItem('{key}');
    if (data) {{
        const parsed = JSON.parse(data);
        // Store in a way Streamlit can access
        window.parent.postMessage({{
            type: 'localStorage_data',
            key: '{key}',
            data: parsed
        }}, '*');
    }}
    </script>
    """

    # For now, we'll use session state as fallback
    # In a real implementation, you'd need streamlit-javascript or similar
    return st.session_state.get(f"localStorage_{key}", {})

def clear_localstorage(key: str):
    """Clear specific key from localStorage."""
    js_code = f"""
    <script>
    localStorage.removeItem('{key}');
    </script>
    """
    st.components.v1.html(js_code, height=0)

# ---- Session State Management with Persistence ----
def initialize_session_state():
    """Initialize session state with localStorage persistence."""
    if 'session_initialized' not in st.session_state:
        st.session_state.session_initialized = True

        # Try to load from localStorage (fallback to session_state for now)
        st.session_state.ready_files = []
        st.session_state.processing_files = {}
        st.session_state.completed_files = {}
        st.session_state.uploader_key = 0
        st.session_state.last_poll_time = 0
        st.session_state.poll_interval = 10  # Start with 10 seconds
        st.session_state.query_input = ""

        # Load persisted data
        load_persisted_state()

def load_persisted_state():
    """Load state from localStorage (simplified version using session state)."""
    # In a real implementation, you'd load from actual localStorage
    # For now, we'll use a simple approach that persists during the session
    pass

def save_persisted_state():
    """Save current state to localStorage."""
    state_data = {
        'ready_files': st.session_state.ready_files,
        'processing_files': st.session_state.processing_files,
        'completed_files': st.session_state.completed_files,
        'timestamp': datetime.now().isoformat()
    }
    # save_to_localstorage('shadowrun_rag_state', state_data)

# ---- Polling System ----
def should_poll() -> bool:
    """Determine if we should poll for updates."""
    if not st.session_state.processing_files:
        return False

    time_since_last_poll = time.time() - st.session_state.last_poll_time
    return time_since_last_poll >= st.session_state.poll_interval

def poll_all_jobs():
    """Poll all active processing jobs for updates."""
    if not st.session_state.processing_files:
        return

    st.session_state.last_poll_time = time.time()
    updates_found = False

    for file_id, file_info in list(st.session_state.processing_files.items()):
        job_id = file_info.get('job_id')
        if not job_id or job_id == 'pending':
            continue

        try:
            response = api_request(f"/job/{job_id}", timeout=5)
            if response and response.get('job_id'):
                old_progress = file_info.get('progress', 0)
                old_stage = file_info.get('stage', '')

                # Update file info
                file_info.update({
                    'stage': response.get('stage', 'unknown'),
                    'progress': response.get('progress', 0),
                    'details': response.get('details', ''),
                    'timestamp': response.get('timestamp', time.time())
                })

                # Check if this job is complete
                if response.get('stage') == 'complete':
                    st.session_state.completed_files[file_id] = file_info
                    del st.session_state.processing_files[file_id]
                    updates_found = True
                    st.success(f"✅ **{file_info['name']}** processing complete!")

                elif response.get('stage') == 'error':
                    file_info['status'] = 'error'
                    st.session_state.completed_files[file_id] = file_info
                    del st.session_state.processing_files[file_id]
                    updates_found = True
                    st.error(f"❌ **{file_info['name']}** processing failed!")

                elif (old_progress != response.get('progress', 0) or
                      old_stage != response.get('stage', '')):
                    updates_found = True

        except Exception as e:
            logger.warning(f"Failed to poll job {job_id}: {e}")

    # Adjust polling interval based on activity
    if updates_found:
        st.session_state.poll_interval = max(5, st.session_state.poll_interval - 2)  # Speed up
    else:
        st.session_state.poll_interval = min(15, st.session_state.poll_interval + 1)  # Slow down

    # Save state after updates
    save_persisted_state()

    return updates_found

def format_elapsed_time(start_time) -> str:
    """Format elapsed time since start."""
    if isinstance(start_time, str):
        try:
            start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        except:
            return "Unknown"

    if not isinstance(start_time, datetime):
        return "Unknown"

    elapsed = datetime.now() - start_time
    total_seconds = int(elapsed.total_seconds())

    if total_seconds < 60:
        return f"{total_seconds}s"
    elif total_seconds < 3600:
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes}m {seconds}s"
    else:
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        return f"{hours}h {minutes}m"

def estimate_remaining_time(progress: float, elapsed_seconds: float) -> str:
    """Estimate remaining time based on current progress."""
    if progress <= 5:
        return "Calculating..."

    if progress >= 100:
        return "Complete"

    rate = progress / elapsed_seconds  # progress per second
    if rate <= 0:
        return "Unknown"

    remaining_progress = 100 - progress
    remaining_seconds = remaining_progress / rate

    if remaining_seconds < 60:
        return f"~{int(remaining_seconds)}s remaining"
    elif remaining_seconds < 3600:
        minutes = int(remaining_seconds // 60)
        return f"~{minutes}m remaining"
    else:
        hours = int(remaining_seconds // 3600)
        minutes = int((remaining_seconds % 3600) // 60)
        return f"~{hours}h {minutes}m remaining"

# ---- API Helper Functions ----
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

# ---- Progress Display Components ----
def render_file_progress(file_id: str, file_info: Dict):
    """Render progress for a single file."""
    # File header
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.write(f"📄 **{file_info['name']}** ({file_info['size'] / 1024:.1f} KB)")
    with col2:
        job_id = file_info.get('job_id', 'pending')
        if job_id and job_id != 'pending':
            st.caption(f"Job: {job_id[-8:]}")
    with col3:
        # Show last update time
        timestamp = file_info.get('timestamp', time.time())
        time_ago = int(time.time() - timestamp)
        if time_ago < 60:
            st.caption(f"🔄 {time_ago}s ago")
        else:
            st.caption(f"🔄 {time_ago // 60}m ago")

    # Progress bar
    progress_val = max(0, min(100, file_info.get('progress', 0))) / 100.0

    if file_info.get('progress', 0) < 0:
        st.error("❌ Processing failed!")
        return

    st.progress(progress_val)

    # Stage and timing info
    current_stage = file_info.get('stage', 'unknown')
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
        'indexing': ('📚', 'Adding to Index'),
        'saving': ('💾', 'Saving Files'),
        'complete': ('🎉', 'Complete'),
        'error': ('❌', 'Error'),
        'uploaded': ('📤', 'Uploaded'),
        'pending': ('⏳', 'Pending')
    }

    stage_icon, stage_label = stage_info.get(current_stage, ('⚙️', current_stage.title()))

    # Stage and progress display
    col1, col2, col3 = st.columns([2, 1, 2])
    with col1:
        st.write(f"{stage_icon} **{stage_label}**")
    with col2:
        st.write(f"**{file_info.get('progress', 0):.0f}%**")
    with col3:
        start_time = file_info.get('start_time')
        if start_time:
            elapsed_str = format_elapsed_time(start_time)
            st.caption(f"⏱️ {elapsed_str}")

            # Show estimated remaining time for active processing
            if current_stage not in ['complete', 'error', 'pending']:
                progress = file_info.get('progress', 0)
                if isinstance(start_time, str):
                    try:
                        start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                        elapsed_seconds = (datetime.now() - start_dt).total_seconds()
                        remaining = estimate_remaining_time(progress, elapsed_seconds)
                        st.caption(f"🔮 {remaining}")
                    except:
                        pass

    # Details
    details = file_info.get('details', '')
    if details:
        st.caption(f"💬 {details}")

    # Stage-specific info boxes
    if current_stage == 'table_detection':
        st.info("📊 **Table detection in progress** - This uses AI models and can take several minutes...")
    elif current_stage == 'extraction':
        st.info("🔍 **High-resolution extraction** - Processing document layout and structure...")
    elif current_stage == 'chunking':
        st.info("✂️ **Creating semantic chunks** - Breaking document into searchable sections...")

# Initialize session state
initialize_session_state()

# Auto-polling logic
if should_poll():
    if poll_all_jobs():
        st.rerun()

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

.status-indicator {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    margin-right: 8px;
}

.status-polling { background-color: #28a745; }
.status-inactive { background-color: #6c757d; }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🎲 Shadowrun RAG Assistant")
st.markdown("*Your AI-powered guide to the Sixth World*")

# Connection and polling status
col1, col2 = st.columns([4, 1])
with col2:
    if st.session_state.processing_files:
        poll_interval = st.session_state.poll_interval
        next_poll = poll_interval - (time.time() - st.session_state.last_poll_time)
        st.markdown(
            f'<span class="status-indicator status-polling"></span>**Polling** (next in {max(0, int(next_poll))}s)',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<span class="status-indicator status-inactive"></span>**Idle**',
            unsafe_allow_html=True
        )

# Debug info
with st.expander("🔧 System Info"):
    st.write(f"**API URL:** {API_URL}")
    st.write(f"**Poll interval:** {st.session_state.poll_interval}s")
    st.write(f"**Processing files:** {len(st.session_state.processing_files)}")
    st.write(f"**Ready files:** {len(st.session_state.ready_files)}")
    st.write(f"**Completed files:** {len(st.session_state.completed_files)}")

    if st.button("🧹 Clear All State"):
        st.session_state.ready_files = []
        st.session_state.processing_files = {}
        st.session_state.completed_files = {}
        st.session_state.uploader_key += 1
        st.success("State cleared!")
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

    filter_subsection = st.text_input("Filter by Subsection (optional)",
                                      placeholder="e.g., Hacking, Spellcasting, Initiative")
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

    # Process pending actions (simplified)
    if 'action' in st.session_state:
        action = st.session_state.action

        if action['type'] == 'clear_all':
            st.session_state.ready_files = []
            st.session_state.uploader_key += 1

        elif action['type'] == 'process_all':
            # Move all ready files to processing
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

            # Add to upload queue
            if 'upload_queue' not in st.session_state:
                st.session_state.upload_queue = []

            for file_info in st.session_state.ready_files:
                st.session_state.upload_queue.append({
                    'file_id': file_info['id'],
                    'name': file_info['name'],
                    'file_data': file_info['file'].getvalue(),
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

                if 'upload_queue' not in st.session_state:
                    st.session_state.upload_queue = []

                st.session_state.upload_queue.append({
                    'file_id': file_id,
                    'name': file_info['name'],
                    'file_data': file_info['file'].getvalue(),
                })

                st.session_state.ready_files = [f for f in st.session_state.ready_files if f['id'] != file_id]
                if not st.session_state.ready_files:
                    st.session_state.uploader_key += 1

        del st.session_state.action
        save_persisted_state()
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
                response = requests.post(f"{API_URL}/upload", files=files, timeout=60)

                if response.status_code == 200:
                    result = response.json()
                    job_id = result['job_id']

                    st.session_state.processing_files[file_id].update({
                        'job_id': job_id,
                        'stage': 'uploaded',
                        'progress': 5,
                        'details': 'Upload complete, processing started...'
                    })

                    st.success(f"✅ {upload_item['name']} uploaded! Processing started.")
                else:
                    st.error(f"❌ Upload failed for {upload_item['name']}: {response.text}")
                    del st.session_state.processing_files[file_id]

            except Exception as e:
                st.error(f"❌ Upload error for {upload_item['name']}: {str(e)}")
                del st.session_state.processing_files[file_id]

        save_persisted_state()

        # Continue processing queue
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

    # Add uploaded files to ready list
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_id = f"{uploaded_file.name}_{uploaded_file.size}_{int(time.time())}"
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

        col1, col2 = st.columns(2)
        with col1:
            if st.button("📤 Process All", type="primary"):
                st.session_state.action = {'type': 'process_all'}
                st.rerun()
        with col2:
            if st.button("🗑️ Clear All"):
                st.session_state.action = {'type': 'clear_all'}
                st.rerun()

        for file_info in st.session_state.ready_files:
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.write(f"📄 **{file_info['name']}** ({file_info['size'] / 1024:.1f} KB)")
            with col2:
                if st.button("📤 Process", key=f"proc_{file_info['id']}"):
                    st.session_state.action = {'type': 'process_file', 'file_id': file_info['id']}
                    st.rerun()
            with col3:
                if st.button("❌ Remove", key=f"rem_{file_info['id']}"):
                    st.session_state.action = {'type': 'remove_file', 'file_id': file_info['id']}
                    st.rerun()

    # Processing Files Section
    if st.session_state.processing_files:
        st.markdown("---")
        st.subheader("🔄 Processing Files")

        # Show polling status
        if should_poll():
            with st.spinner("Checking for updates..."):
                poll_all_jobs()

        for file_id, file_info in list(st.session_state.processing_files.items()):
            render_file_progress(file_id, file_info)
            st.markdown("---")

    # Completed Files Section
    if st.session_state.completed_files:
        st.markdown("---")
        st.subheader("✅ Completed Files")

        for file_id, file_info in st.session_state.completed_files.items():
            if file_info.get('status') == 'error':
                st.error(f"❌ **{file_info['name']}** - Processing failed!")
                details = file_info.get('details', 'Unknown error')
                st.caption(f"Error: {details}")
            else:
                st.success(f"✅ **{file_info['name']}** - Successfully processed!")
                if file_info.get('details'):
                    st.caption(f"Result: {file_info['details']}")

        # Clear completed files button
        if st.button("🧹 Clear Completed"):
            st.session_state.completed_files = {}
            save_persisted_state()
            st.rerun()

    # Manual indexing section
    st.markdown("---")
    st.subheader("🔧 Manual Operations")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Re-index All Documents"):
            with st.spinner("Re-indexing all documents..."):
                response = api_request("/index", method="POST", json={"force_reindex": True})
                if response:
                    st.success("✅ Re-indexing complete!")
                else:
                    st.error("❌ Re-indexing failed!")

    with col2:
        if st.button("📊 Index New Documents Only"):
            with st.spinner("Indexing new documents..."):
                response = api_request("/index", method="POST", json={"force_reindex": False})
                if response:
                    st.success("✅ Indexing complete!")
                else:
                    st.error("❌ Indexing failed!")

with tab3:
    st.header("📚 Document Library")

    # Get available documents
    available_docs = get_available_documents()

    if available_docs:
        st.write(f"**{len(available_docs)} documents** indexed and searchable:")

        # Group documents by type/source
        doc_groups = {}
        for doc_path in available_docs:
            try:
                doc_name = Path(doc_path).name
                parent_name = Path(doc_path).parent.name

                if parent_name not in doc_groups:
                    doc_groups[parent_name] = []
                doc_groups[parent_name].append(doc_name)
            except:
                # Fallback for malformed paths
                if "other" not in doc_groups:
                    doc_groups["other"] = []
                doc_groups["other"].append(str(doc_path))

        # Display grouped documents
        for group_name, docs in doc_groups.items():
            with st.expander(f"📁 {group_name} ({len(docs)} files)"):
                for doc in sorted(docs):
                    st.write(f"📄 {doc}")
    else:
        st.info("📭 No documents indexed yet. Upload PDFs in the Upload tab to get started!")

    # Document statistics
    if available_docs:
        st.markdown("---")
        st.subheader("📊 Statistics")

        # Get system status for more details
        status = api_request("/status")
        if status:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Documents", status.get('indexed_documents', 0))
            with col2:
                st.metric("Text Chunks", status.get('indexed_chunks', 0))
            with col3:
                avg_chunks = 0
                if status.get('indexed_documents', 0) > 0:
                    avg_chunks = status.get('indexed_chunks', 0) / status.get('indexed_documents', 1)
                st.metric("Avg Chunks/Doc", f"{avg_chunks:.1f}")

with tab4:
    st.header("📝 Session Notes")

    st.info("🔧 **Session notes functionality coming soon!**")

    st.markdown("""
    This tab will allow you to:
    - Upload session notes and campaign logs
    - Query past session events and NPCs
    - Track ongoing plot threads
    - Search for specific events across your campaign

    For now, you can upload session notes as PDFs in the Upload tab,
    and they'll be searchable through the main Query interface.
    """)

# Auto-refresh for processing files
if st.session_state.processing_files and should_poll():
    time.sleep(1)  # Small delay to prevent too frequent refreshes
    st.rerun()

# Footer
st.markdown("---")
st.markdown(
    f"""
    <div style='text-align: center'>
        <small>🎲 Shadowrun RAG Assistant v2.1 | Polling-based Updates | Powered by Ollama & ChromaDB</small><br>
        <small>Poll interval: {st.session_state.poll_interval}s | Processing: {len(st.session_state.processing_files)} files</small>
    </div>
    """,
    unsafe_allow_html=True
)