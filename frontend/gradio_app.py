"""
Gradio Frontend for Shadowrun RAG System
Replaces Streamlit with better state management and streaming support
"""

import gradio as gr
import requests
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import threading
import logging
from datetime import datetime
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Configuration
import os
API_URL = os.getenv("API_URL", "http://localhost:8000")

class RAGClient:
    """Client for interacting with the FastAPI backend."""

    def __init__(self, api_url: str = API_URL):
        self.api_url = api_url
        self.active_jobs = {}
        self.lock = threading.Lock()

    def upload_pdf(self, file_path: str) -> Dict:
        """Upload a PDF and get job ID."""
        try:
            with open(file_path, 'rb') as f:
                files = {'file': (Path(file_path).name, f, 'application/pdf')}
                response = requests.post(f"{self.api_url}/upload", files=files, timeout=60)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return {"error": str(e)}

    def get_job_status(self, job_id: str) -> Dict:
        """Poll job status."""
        try:
            response = requests.get(f"{self.api_url}/job/{job_id}", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Status check failed: {e}")
            return {"error": str(e)}

    def query_stream(self, question: str, **params) -> tuple:
        """Stream query response with metadata."""
        try:
            response = requests.post(
                f"{self.api_url}/query_stream",
                json={"question": question, **params},
                stream=True,
                timeout=30
            )
            response.raise_for_status()

            full_response = ""
            metadata = None
            metadata_buffer = ""
            collecting_metadata = False

            for chunk in response.iter_content(chunk_size=32, decode_unicode=True):
                if chunk:
                    if "__METADATA_START__" in chunk:
                        parts = chunk.split("__METADATA_START__")
                        if parts[0]:
                            full_response += parts[0]
                            yield full_response, None, "generating"

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
                            except json.JSONDecodeError as e:
                                logger.error(f"Metadata parse failed: {e}")
                            break
                    else:
                        full_response += chunk
                        yield full_response, None, "generating"

            # Final yield with metadata
            yield full_response, metadata, "complete"

        except Exception as e:
            logger.error(f"Query failed: {e}")
            yield f"Error: {str(e)}", None, "error"

    def get_models(self) -> List[str]:
        """Get available models."""
        try:
            response = requests.get(f"{self.api_url}/models", timeout=5)
            response.raise_for_status()
            return response.json().get("models", ["llama3"])
        except Exception as e:
            logger.error(f"Failed to fetch models: {e}")
            return ["llama3"]

    def get_documents(self) -> List[str]:
        """Get indexed documents."""
        try:
            response = requests.get(f"{self.api_url}/documents", timeout=10)
            response.raise_for_status()
            return response.json().get("documents", [])
        except Exception as e:
            logger.error(f"Failed to fetch documents: {e}")
            return []

    def get_status(self) -> Dict:
        """Get system status."""
        try:
            response = requests.get(f"{self.api_url}/status", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch status: {e}")
            return {"status": "offline", "error": str(e)}

    def reindex(self, force: bool = False) -> Dict:
        """Trigger reindexing."""
        try:
            response = requests.post(
                f"{self.api_url}/index",
                json={"force_reindex": force},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Reindex failed: {e}")
            return {"error": str(e)}

# Initialize client
client = RAGClient()

# ===== UPLOAD TAB FUNCTIONS =====

def process_uploads(files) -> str:
    """Process uploaded PDFs with progress tracking."""
    if not files:
        return "No files selected"

    results = []
    for file in files:
        file_name = Path(file.name).name

        # Upload file
        result = client.upload_pdf(file.name)
        if "error" in result:
            results.append(f"❌ {file_name}: {result['error']}")
            continue

        job_id = result.get("job_id")
        if not job_id:
            results.append(f"❌ {file_name}: No job ID returned")
            continue

        results.append(f"✅ {file_name}: Processing started (Job: {job_id[-8:]})")

    return "\n".join(results)

def poll_progress():
    """Poll all active jobs and return progress info."""
    try:
        response = requests.get(f"{client.api_url}/jobs", timeout=5)
        response.raise_for_status()
        jobs = response.json().get("active_jobs", {})

        if not jobs:
            return "No active processing jobs", gr.update(visible=False), 0

        # Format progress information
        progress_lines = []
        progress_values = []

        for job_id, job_info in jobs.items():
            stage = job_info.get("stage", "unknown")
            progress = job_info.get("progress", 0)
            details = job_info.get("details", "")

            # Extract filename from job_id
            filename = job_id.rsplit("_", 1)[0] if "_" in job_id else job_id

            progress_lines.append(
                f"📄 **{filename}**\n"
                f"   Stage: {stage} | Progress: {progress:.0f}%\n"
                f"   {details}"
            )

            progress_values.append(progress)

        progress_text = "\n\n".join(progress_lines)

        # Calculate average progress
        avg_progress = sum(progress_values) / len(progress_values) if progress_values else 0

        return progress_text, gr.update(visible=True, value=avg_progress), avg_progress

    except Exception as e:
        return f"Error checking progress: {str(e)}", gr.update(visible=False), 0

# ===== QUERY TAB FUNCTIONS =====

def submit_query(
    question: str,
    model: str,
    n_results: int,
    query_type: str,
    character_role: str,
    character_stats: str,
    edition: str,
    filter_section: str,
    filter_subsection: str
):
    """Submit query and stream response."""
    if not question:
        yield "Please enter a question", "", []
        return

    # Prepare parameters
    params = {
        "n_results": n_results,
        "query_type": query_type.lower(),
        "model": model
    }

    # Add optional parameters
    if character_role != "None":
        params["character_role"] = character_role.lower().replace(" ", "_")
    if character_stats:
        params["character_stats"] = character_stats
    if edition != "None":
        params["edition"] = edition
    if filter_section != "All":
        params["filter_section"] = filter_section
    if filter_subsection:
        params["filter_subsection"] = filter_subsection

    # Stream response
    for response, metadata, status in client.query_stream(question, **params):
        if status == "error":
            yield response, "", []
        elif status == "complete" and metadata:
            # Format sources
            sources_text = ""
            if metadata.get("sources"):
                sources_list = [Path(s).name for s in metadata["sources"]]
                sources_text = "**Sources:**\n" + "\n".join([f"📄 {s}" for s in sources_list])

            # Create chunks dataframe for display
            chunks_data = []
            if metadata.get("chunks"):
                for i, (chunk, dist) in enumerate(zip(
                    metadata.get("chunks", []),
                    metadata.get("distances", [])
                )):
                    chunks_data.append({
                        "Relevance": f"{(1 - dist):.2%}" if dist else "N/A",
                        "Content": chunk[:200] + "..." if len(chunk) > 200 else chunk
                    })

            yield response, sources_text, chunks_data
        else:
            # Still generating
            yield response + "▌", "", []

# ===== DOCUMENT TAB FUNCTIONS =====

def refresh_documents():
    """Refresh document library display."""
    docs = client.get_documents()
    status = client.get_status()

    if not docs:
        return "No documents indexed yet", "", {}

    # Group documents
    doc_groups = {}
    for doc_path in docs:
        try:
            doc_name = Path(doc_path).name
            parent_name = Path(doc_path).parent.name

            if parent_name not in doc_groups:
                doc_groups[parent_name] = []
            doc_groups[parent_name].append(doc_name)
        except:
            if "other" not in doc_groups:
                doc_groups["other"] = []
            doc_groups["other"].append(str(doc_path))

    # Format document list
    doc_text = f"**Total: {len(docs)} documents indexed**\n\n"
    for group, files in doc_groups.items():
        doc_text += f"📁 **{group}** ({len(files)} files)\n"
        for file in sorted(files)[:5]:  # Show first 5
            doc_text += f"   📄 {file}\n"
        if len(files) > 5:
            doc_text += f"   ... and {len(files) - 5} more\n"
        doc_text += "\n"

    # Format statistics
    stats_text = f"""
**System Statistics:**
- Documents: {status.get('indexed_documents', 0)}
- Text Chunks: {status.get('indexed_chunks', 0)}
- Active Jobs: {status.get('active_jobs', 0)}
- Status: {status.get('status', 'unknown')}
"""

    # Create dataframe for better display
    stats_df = pd.DataFrame([
        {"Metric": "Documents", "Value": status.get('indexed_documents', 0)},
        {"Metric": "Text Chunks", "Value": status.get('indexed_chunks', 0)},
        {"Metric": "Avg Chunks/Doc", "Value": f"{status.get('indexed_chunks', 0) / max(status.get('indexed_documents', 1), 1):.1f}"},
        {"Metric": "Active Jobs", "Value": status.get('active_jobs', 0)},
    ])

    return doc_text, stats_text, stats_df

def reindex_documents(force_reindex: bool):
    """Trigger document reindexing."""
    result = client.reindex(force=force_reindex)
    if "error" in result:
        return f"❌ Reindexing failed: {result['error']}"
    return "✅ Reindexing complete!"

# ===== BUILD GRADIO INTERFACE =====

def build_interface():
    """Build the complete Gradio interface."""

    with gr.Blocks(title="🎲 Shadowrun RAG Assistant", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🎲 Shadowrun RAG Assistant")
        gr.Markdown("*Your AI-powered guide to the Sixth World*")

        with gr.Tabs():
            # ===== QUERY TAB =====
            with gr.Tab("💬 Query"):
                with gr.Row():
                    with gr.Column(scale=3):
                        question_input = gr.Textbox(
                            label="Your Question",
                            placeholder="e.g., 'How do recoil penalties work in Shadowrun 5e?'",
                            lines=3
                        )

                        with gr.Row():
                            submit_btn = gr.Button("🔍 Search", variant="primary")
                            clear_btn = gr.ClearButton(components=[question_input], value="Clear")

                        answer_output = gr.Markdown(label="Answer")
                        sources_output = gr.Markdown(label="Sources")

                        with gr.Accordion("📊 Retrieved Chunks", open=False):
                            chunks_output = gr.Dataframe(
                                headers=["Relevance", "Content"],
                                label="Context Chunks"
                            )

                    with gr.Column(scale=1):
                        gr.Markdown("### ⚙️ Configuration")

                        model_select = gr.Dropdown(
                            choices=client.get_models() or ["llama3:8b-instruct-q4_K_M"],
                            value="llama3:8b-instruct-q4_K_M",
                            label="LLM Model",
                            allow_custom_value=True
                        )

                        n_results_slider = gr.Slider(
                            minimum=1, maximum=10, value=5, step=1,
                            label="Number of Sources"
                        )

                        query_type_select = gr.Dropdown(
                            choices=["General", "Rules", "Session"],
                            value="General",
                            label="Query Type"
                        )

                        with gr.Accordion("👤 Character Context", open=False):
                            character_role_select = gr.Dropdown(
                                choices=["None", "Decker", "Mage", "Street Samurai",
                                        "Rigger", "Adept", "Technomancer", "Face"],
                                value="None",
                                label="Character Role"
                            )

                            character_stats_input = gr.Textbox(
                                label="Character Stats",
                                placeholder="e.g., Logic 6, Hacking 5"
                            )

                            edition_select = gr.Dropdown(
                                choices=["None", "SR5", "SR6", "SR4"],
                                value="None",
                                label="Preferred Edition"
                            )

                        with gr.Accordion("🔍 Filters", open=False):
                            section_filter = gr.Dropdown(
                                choices=["All", "Combat", "Matrix", "Magic", "Gear",
                                        "Character Creation", "Riggers", "Technomancy"],
                                value="All",
                                label="Filter by Section"
                            )

                            subsection_filter = gr.Textbox(
                                label="Filter by Subsection",
                                placeholder="e.g., Hacking, Spellcasting"
                            )

                # Wire up query submission
                submit_btn.click(
                    fn=submit_query,
                    inputs=[
                        question_input, model_select, n_results_slider,
                        query_type_select, character_role_select, character_stats_input,
                        edition_select, section_filter, subsection_filter
                    ],
                    outputs=[answer_output, sources_output, chunks_output]
                )

            # ===== UPLOAD TAB =====
            with gr.Tab("📤 Upload"):
                with gr.Row():
                    with gr.Column():
                        file_upload = gr.File(
                            label="Upload PDFs",
                            file_types=[".pdf"],
                            file_count="multiple"
                        )

                        upload_btn = gr.Button("📤 Process Files", variant="primary")

                        upload_status = gr.Textbox(
                            label="Upload Status",
                            lines=5,
                            interactive=False
                        )

                        # Progress indicator (not a component, just for display)
                        progress_display = gr.Number(
                            label="Overall Progress (%)",
                            value=0,
                            interactive=False,
                            visible=False
                        )

                        with gr.Row():
                            check_progress_btn = gr.Button("🔄 Check Progress")
                            auto_refresh = gr.Checkbox(label="Auto-refresh progress", value=False)

                        progress_status = gr.Markdown(label="Processing Status")

                    with gr.Column():
                        gr.Markdown("### 🔧 Manual Operations")

                        reindex_btn = gr.Button("🔄 Re-index All Documents")
                        index_new_btn = gr.Button("📊 Index New Documents Only")

                        reindex_output = gr.Textbox(label="Operation Result", lines=2)

                # Wire up upload functionality
                upload_btn.click(
                    fn=process_uploads,
                    inputs=[file_upload],
                    outputs=[upload_status]
                )

                check_progress_btn.click(
                    fn=poll_progress,
                    outputs=[progress_status, progress_display, gr.State()]
                )

                reindex_btn.click(
                    fn=lambda: reindex_documents(True),
                    outputs=[reindex_output]
                )

                index_new_btn.click(
                    fn=lambda: reindex_documents(False),
                    outputs=[reindex_output]
                )

                # Auto-refresh timer
                def auto_refresh_progress(enable):
                    if enable:
                        return poll_progress()
                    return "", gr.update(visible=False), 0

                timer = gr.Timer(5.0, active=False)
                auto_refresh.change(
                    fn=lambda x: gr.update(active=x),
                    inputs=[auto_refresh],
                    outputs=[timer]
                )
                timer.tick(
                    fn=poll_progress,
                    outputs=[progress_status, progress_display, gr.State()]
                )

            # ===== DOCUMENTS TAB =====
            with gr.Tab("📚 Documents"):
                refresh_docs_btn = gr.Button("🔄 Refresh Document Library")

                with gr.Row():
                    with gr.Column():
                        docs_display = gr.Markdown(label="Document Library")

                    with gr.Column():
                        stats_display = gr.Markdown(label="Statistics")
                        stats_table = gr.Dataframe(label="Metrics")

                # Wire up document refresh
                refresh_docs_btn.click(
                    fn=refresh_documents,
                    outputs=[docs_display, stats_display, stats_table]
                )

                # Load documents on tab load
                app.load(
                    fn=refresh_documents,
                    outputs=[docs_display, stats_display, stats_table]
                )

            # ===== SESSION NOTES TAB =====
            with gr.Tab("📝 Session Notes"):
                gr.Markdown("""
                ### 🔧 Session Notes - Coming Soon!
                
                This feature will allow you to:
                - Upload and manage session notes
                - Query past events and NPCs
                - Track ongoing plot threads
                - Search across your entire campaign
                
                For now, upload session notes as PDFs in the Upload tab.
                """)

        # Footer
        gr.Markdown("---")
        gr.Markdown(
            """
            <center>
            <small>🎲 Shadowrun RAG Assistant v3.0 | Powered by Gradio, Ollama & ChromaDB</small>
            </center>
            """,
            elem_classes=["footer"]
        )

    return app

# ===== MAIN EXECUTION =====

if __name__ == "__main__":
    print("🚀 Starting Gradio frontend for Shadowrun RAG...")
    print(f"📡 Connecting to API at: {API_URL}")

    # Check API connection
    try:
        status = client.get_status()
        print(f"✅ API Status: {status.get('status', 'unknown')}")
        print(f"📚 Indexed documents: {status.get('indexed_documents', 0)}")
        print(f"📊 Indexed chunks: {status.get('indexed_chunks', 0)}")
    except Exception as e:
        print(f"⚠️ Warning: Could not connect to API: {e}")
        print("Make sure the FastAPI backend is running!")

    # Build and launch interface
    app = build_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )