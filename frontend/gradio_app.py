"""
Enhanced Gradio Frontend for Shadowrun RAG System
Includes document type selection, <think> tag support, and improved filtering
"""

import gradio as gr
import requests
import json
import re

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import threading
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Configuration
import os
API_URL = os.getenv("API_URL", "http://localhost:8000")

class RAGClient:
    """Enhanced client for interacting with the FastAPI backend."""

    def __init__(self, api_url: str = API_URL):
        self.api_url = api_url
        self.active_jobs = {}
        self.lock = threading.Lock()

    def upload_pdf(self, file_path: str, document_type: str = "rulebook") -> Dict:
        """Upload a PDF with document type specification."""
        try:
            with open(file_path, 'rb') as f:
                files = {'file': (Path(file_path).name, f, 'application/pdf')}
                data = {'document_type': document_type}
                response = requests.post(
                    f"{self.api_url}/upload",
                    files=files,
                    data=data,
                    timeout=60
                )
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
        """Enhanced stream query response with <think> tag detection."""
        try:
            response = requests.post(
                f"{self.api_url}/query_stream",
                json={"question": question, **params},
                stream=True,
                timeout=30
            )
            response.raise_for_status()

            full_response = ""
            thinking_content = ""
            metadata = None
            metadata_buffer = ""
            collecting_metadata = False
            in_thinking = False

            for chunk in response.iter_content(chunk_size=32, decode_unicode=True):
                if chunk:
                    # Handle thinking tags
                    if "__THINKING_START__" in chunk:
                        in_thinking = True
                        parts = chunk.split("__THINKING_START__")
                        if parts[0]:
                            full_response += parts[0]
                            yield full_response, thinking_content, None, "generating"
                        if len(parts) > 1:
                            thinking_content += parts[1]
                            yield full_response, thinking_content, None, "thinking"
                        continue

                    if "__THINKING_END__" in chunk and in_thinking:
                        in_thinking = False
                        parts = chunk.split("__THINKING_END__")
                        if parts[0]:
                            thinking_content += parts[0]
                            yield full_response, thinking_content, None, "thinking"
                        if len(parts) > 1:
                            full_response += parts[1]
                            yield full_response, thinking_content, None, "generating"
                        continue

                    # Handle metadata
                    if "__METADATA_START__" in chunk:
                        parts = chunk.split("__METADATA_START__")
                        if parts[0]:
                            if in_thinking:
                                thinking_content += parts[0]
                            else:
                                full_response += parts[0]
                            yield full_response, thinking_content, None, "generating"

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
                        # Regular content
                        if in_thinking:
                            thinking_content += chunk
                            yield full_response, thinking_content, None, "thinking"
                        else:
                            full_response += chunk
                            yield full_response, thinking_content, None, "generating"

            # Final yield with metadata
            yield full_response, thinking_content, metadata, "complete"

        except Exception as e:
            logger.error(f"Query failed: {e}")
            yield f"Error: {str(e)}", "", None, "error"

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

    def get_document_stats(self) -> Dict:
        """Get enhanced document statistics."""
        try:
            response = requests.get(f"{self.api_url}/document_stats", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch document stats: {e}")
            return {"error": str(e)}

    def get_document_content(self, file_path: str) -> Dict:
        """Get content of a specific document."""
        try:
            response = requests.get(f"{self.api_url}/document/{file_path}", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch document content: {e}")
            return {"error": str(e)}

    def search_documents(self, query: str, group: str = None, document_type: str = None,
                        edition: str = None, page: int = 1, page_size: int = 20) -> Dict:
        """Enhanced document search with metadata filtering."""
        try:
            search_params = {
                "query": query,
                "page": page,
                "page_size": page_size
            }
            if group and group != "All Groups":
                search_params["group"] = group
            if document_type and document_type != "All Types":
                search_params["document_type"] = document_type
            if edition and edition != "All Editions":
                search_params["edition"] = edition

            response = requests.post(
                f"{self.api_url}/search_documents",
                json=search_params,
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {"error": str(e)}

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

# Global variables for pagination and state
search_file_paths = {}
current_search_params = {}
current_page = 1

# ===== UPLOAD TAB FUNCTIONS =====

def process_uploads(files, document_type: str) -> tuple:
    """Process uploaded PDFs with document type specification."""
    if not files:
        return "No files selected", gr.update(value=[])

    if not document_type:
        document_type = "rulebook"  # Default fallback

    results = []
    for file in files:
        file_name = Path(file.name).name

        # Upload file with document type
        result = client.upload_pdf(file.name, document_type)
        if "error" in result:
            results.append(f"❌ {file_name}: {result['error']}")
            continue

        job_id = result.get("job_id")
        if not job_id:
            results.append(f"❌ {file_name}: No job ID returned")
            continue

        results.append(f"✅ {file_name}: Processing started as {document_type} (Job: {job_id[-8:]})")

    return "\n".join(results), gr.update(value=[])

def poll_progress():
    """Enhanced progress polling with document type information."""
    try:
        response = requests.get(f"{client.api_url}/jobs", timeout=5)
        response.raise_for_status()
        jobs = response.json().get("active_jobs", {})

        if not jobs:
            return "No active processing jobs", gr.update(visible=False), 0, "✅ All processing complete!"

        # Format progress information
        progress_lines = []
        progress_values = []

        for job_id, job_info in jobs.items():
            stage = job_info.get("stage", "unknown")
            progress = job_info.get("progress", 0)
            details = job_info.get("details", "")

            # Extract document type and filename from job_id
            parts = job_id.split("_")
            if len(parts) >= 2:
                doc_type = parts[0]
                filename = "_".join(parts[1:-1])  # Exclude timestamp
            else:
                doc_type = "unknown"
                filename = job_id

            # Enhanced progress display with document type
            progress_lines.append(
                f"📄 **{filename}** ({doc_type})\n"
                f"   Stage: {stage} | Progress: {progress:.0f}%\n"
                f"   {details}"
            )

            progress_values.append(progress)

        progress_text = "\n\n".join(progress_lines)
        avg_progress = sum(progress_values) / len(progress_values) if progress_values else 0

        # Create upload status summary
        upload_summary = f"📄 Processing {len(jobs)} file(s):\n" + "\n".join([
            f"• {parts[1] if len(parts) >= 2 else job_id}: {stage} ({progress:.0f}%)"
            for job_id, job_info in jobs.items()
            for stage, progress in [(job_info.get('stage', 'unknown'), job_info.get('progress', 0))]
            for parts in [job_id.split('_')]
        ])

        return progress_text, gr.update(visible=True, value=avg_progress), avg_progress, upload_summary

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
    filter_subsection: str,
    filter_document_type: str,
    filter_edition: str
):
    """Enhanced query submission with improved filtering and <think> tag support."""
    if not question:
        yield "Please enter a question", "", "", [], gr.update(visible=False)
        return

    # Prepare enhanced parameters
    params = {
        "n_results": n_results,
        "query_type": query_type.lower(),
        "model": model,
        "edition": edition if edition != "None" else "SR5"  # Default to SR5
    }

    # Add optional parameters
    if character_role != "None":
        params["character_role"] = character_role.lower().replace(" ", "_")
    if character_stats:
        params["character_stats"] = character_stats
    if filter_section != "All":
        params["filter_section"] = filter_section
    if filter_subsection:
        params["filter_subsection"] = filter_subsection
    if filter_document_type != "All Types":
        params["filter_document_type"] = filter_document_type
    if filter_edition != "All Editions":
        params["filter_edition"] = filter_edition

    # Stream response with thinking support
    for response, thinking, metadata, status in client.query_stream(question, **params):
        if status == "error":
            yield response, "", "", [], gr.update(visible=False)
        elif status == "complete" and metadata:
            # Format sources
            sources_text = ""
            if metadata.get("sources"):
                sources_list = [Path(s).name for s in metadata["sources"]]
                sources_text = "**Sources:**\n" + "\n".join([f"📄 {s}" for s in sources_list])

            # Show applied filters for debugging
            if metadata.get("applied_filters"):
                filters_text = f"\n\n**Applied Filters:** {metadata['applied_filters']}"
                sources_text += filters_text

            # Create chunks dataframe
            chunks_data = []
            if metadata.get("chunks"):
                for i, (chunk, dist) in enumerate(zip(
                        metadata.get("chunks", []),
                        metadata.get("distances", [])
                )):
                    relevance = f"{(1 - dist):.2%}" if dist else "N/A"
                    content = chunk[:200] + "..." if len(chunk) > 200 else chunk
                    chunks_data.append([relevance, content])

            # Show thinking accordion if there's thinking content
            thinking_visible = bool(thinking and thinking.strip())

            yield response, thinking, sources_text, chunks_data, gr.update(visible=thinking_visible)
        else:
            # Still generating - show thinking accordion if content exists
            thinking_visible = bool(thinking and thinking.strip())
            cursor = "▌" if status == "generating" else "🤔" if status == "thinking" else ""
            yield response + cursor, thinking, "", [], gr.update(visible=thinking_visible)


# ===== DOCUMENT TAB FUNCTIONS =====

def refresh_documents():
    """Enhanced document library refresh with metadata statistics."""
    docs = client.get_documents()
    stats = client.get_document_stats()

    if "error" in stats:
        return f"Error loading stats: {stats['error']}", {}, pd.DataFrame(), "", gr.update(choices=[]), gr.update(
            choices=[]), gr.update(choices=[]), gr.update(choices=[])

    if not docs:
        return "No documents indexed yet", {}, pd.DataFrame(), "", gr.update(choices=[]), gr.update(
            choices=[]), gr.update(choices=[]), gr.update(choices=[])

    # Group documents by parent directory
    doc_groups = {}
    for doc_path in docs:
        try:
            parent_name = Path(doc_path).parent.name
            relative_path = str(Path(doc_path).relative_to(Path("data/processed_markdown")))

            if parent_name not in doc_groups:
                doc_groups[parent_name] = []
            doc_groups[parent_name].append({
                "filename": Path(doc_path).name,
                "path": relative_path
            })
        except:
            if "other" not in doc_groups:
                doc_groups["other"] = []
            doc_groups["other"].append({
                "filename": str(doc_path),
                "path": str(doc_path)
            })

    # Enhanced statistics with metadata breakdown
    total_chunks = stats.get('total_chunks', 0)
    unique_docs = stats.get('unique_documents', 0)
    active_jobs = 0  # Will be updated from status

    try:
        status = client.get_status()
        active_jobs = status.get('active_jobs', 0)
    except:
        pass

    stats_text = f"""
**Enhanced System Statistics:**
- Unique Documents: {unique_docs}
- Total Text Chunks: {total_chunks}
- Active Jobs: {active_jobs}
- Document Types: {len(stats.get('document_types', {}))}
- Editions Found: {len(stats.get('editions', {}))}
- Sections Detected: {len(stats.get('sections', {}))}
"""

    # Create detailed stats dataframe
    stats_rows = [
        {"Category": "Documents", "Count": unique_docs},
        {"Category": "Text Chunks", "Count": total_chunks},
        {"Category": "Active Jobs", "Count": active_jobs},
    ]

    # Add document type breakdown
    doc_types = stats.get('document_types', {})
    for doc_type, count in doc_types.items():
        stats_rows.append({"Category": f"Type: {doc_type}", "Count": count})

    # Add edition breakdown
    editions = stats.get('editions', {})
    for edition, count in editions.items():
        stats_rows.append({"Category": f"Edition: {edition}", "Count": count})

    stats_df = pd.DataFrame(stats_rows)

    # Update dropdown choices
    group_choices = ["All Groups"] + list(doc_groups.keys())

    # Document type choices from stats
    type_choices = ["All Types"] + list(doc_types.keys())

    # Edition choices from stats
    edition_choices = ["All Editions"] + list(editions.keys())

    print(f"DEBUG: doc_groups keys: {list(doc_groups.keys()) if doc_groups else 'None'}")
    print(f"DEBUG: group_choices: {group_choices}")

    return (
        f"Total: {len(docs)} documents in {len(doc_groups)} groups",
        doc_groups,
        stats_df,
        stats_text,
        gr.update(choices=group_choices),
        gr.update(choices=type_choices),
        gr.update(choices=edition_choices),
        gr.update(choices=group_choices)
    )


def load_document_group(group_name, doc_groups):
    """Load all files for a specific document group."""
    if not group_name or group_name not in doc_groups:
        return gr.update(choices=[], visible=False)

    files = doc_groups[group_name]
    choices = []

    for file_info in files:
        display_name = f"📄 {file_info['filename']}"
        choices.append(display_name)

    return gr.update(choices=choices, visible=True, value=None)


def handle_library_file_selection(selected_file, group_name, doc_groups):
    """Handle file selection from document library."""
    if not selected_file or not group_name or group_name not in doc_groups:
        return "No file selected"

    # Find the file path
    filename = selected_file.replace("📄 ", "")
    for file_info in doc_groups[group_name]:
        if file_info['filename'] == filename:
            return load_document_content(file_info['path'])

    return "File not found"


def search_docs_fn(query: str, selected_group: str, selected_type: str, selected_edition: str, page: int = 1):
    """Enhanced document search with metadata filtering and pagination."""
    global search_file_paths, current_search_params, current_page

    # Store search parameters for pagination
    current_search_params = {
        "query": query,
        "selected_group": selected_group,
        "selected_type": selected_type,
        "selected_edition": selected_edition
    }
    current_page = page

    search_file_paths = {}

    if not query.strip():
        return (
            "Enter a search query to find documents",
            gr.update(choices=[], visible=False),
            "",
            gr.update(visible=False),
            gr.update(value="Page 0 of 0")
        )

    group_filter = None if selected_group == "All Groups" else selected_group
    type_filter = None if selected_type == "All Types" else selected_type
    edition_filter = None if selected_edition == "All Editions" else selected_edition

    results = client.search_documents(
        query, group_filter, type_filter, edition_filter, page, 20
    )

    if "error" in results:
        return (
            f"Search error: {results['error']}",
            gr.update(choices=[], visible=False),
            "",
            gr.update(visible=False),
            gr.update(value="Page 0 of 0")
        )

    files = results.get("results", [])
    total = results.get("total", 0)
    current_page_num = results.get("page", 1)
    total_pages = results.get("total_pages", 1)

    if not files:
        return (
            "No files found matching your search",
            gr.update(choices=[], visible=False),
            "",
            gr.update(visible=False),
            gr.update(value="Page 0 of 0")
        )

    # Create choices for radio buttons and store file paths
    choices = []
    for file_info in files:
        match_type = file_info.get("match_type", "")
        match_indicator = {"filename": "📝", "content": "🔍", "all": "📄"}.get(match_type, "📄")

        display_name = f"{match_indicator} {file_info['filename']} | 📂 {file_info['group']}"
        choices.append(display_name)
        search_file_paths[display_name] = file_info['file_path']

    # Enhanced summary with filter info
    filter_info = []
    if group_filter:
        filter_info.append(f"Group: {group_filter}")
    if type_filter:
        filter_info.append(f"Type: {type_filter}")
    if edition_filter:
        filter_info.append(f"Edition: {edition_filter}")

    filter_text = f" | Filters: {', '.join(filter_info)}" if filter_info else ""
    summary_text = f"**Found {total} files**{filter_text} - Click to view:"

    # Auto-load first result
    first_content = load_document_content(files[0]['file_path'])

    # Update pagination info
    page_text = f"Page {current_page_num} of {total_pages}"
    pagination_visible = total_pages > 1

    return (
        summary_text,
        gr.update(choices=choices, visible=True, value=choices[0]),
        first_content,
        gr.update(visible=pagination_visible),
        gr.update(value=page_text)
    )


def navigate_search_results(direction: int):
    """Navigate search results pagination."""
    global current_search_params, current_page

    if not current_search_params:
        return search_docs_fn("", "All Groups", "All Types", "All Editions", 1)

    new_page = max(1, current_page + direction)

    return search_docs_fn(
        current_search_params["query"],
        current_search_params["selected_group"],
        current_search_params["selected_type"],
        current_search_params["selected_edition"],
        new_page
    )


def handle_file_selection(selected_choice):
    """Handle radio button selection."""
    global search_file_paths

    if not selected_choice or selected_choice not in search_file_paths:
        return "No file selected or file not found"

    file_path = search_file_paths[selected_choice]
    return load_document_content(file_path)


def load_document_content(file_path: str):
    """Load and display document content with enhanced metadata display."""
    if not file_path:
        return "No file selected"

    result = client.get_document_content(file_path)

    if "error" in result:
        return f"❌ **Error loading file**\n\n{result['error']}"

    content = result.get("content", "")
    filename = Path(file_path).name

    # Extract and format YAML metadata if present
    metadata_display = ""
    if content.startswith('---'):
        yaml_end = content.find('---', 3)
        if yaml_end > 0:
            yaml_content = content[3:yaml_end].strip()
            main_content = content[yaml_end + 3:].strip()

            # Parse key metadata for display
            metadata_lines = []
            for line in yaml_content.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip().strip('"')
                    if key in ['document_type', 'edition', 'primary_focus']:
                        metadata_lines.append(f"**{key.title()}:** {value}")

            if metadata_lines:
                metadata_display = f"\n\n📊 **Metadata:**  \n" + "  \n".join(metadata_lines) + "\n\n---\n\n"
        else:
            main_content = content
    else:
        main_content = content

    # Format with enhanced header
    formatted_content = f"""# 📄 {filename}

**Path:** `{file_path}`{metadata_display}

{main_content}"""

    return formatted_content


def reindex_documents(force_reindex: bool):
    """Trigger enhanced document reindexing."""
    result = client.reindex(force=force_reindex)
    if "error" in result:
        return f"❌ Reindexing failed: {result['error']}"
    return "✅ Enhanced reindexing complete with metadata extraction!"


# ===== BUILD ENHANCED GRADIO INTERFACE =====

def build_interface():
    """Build the enhanced Gradio interface with document types and thinking support."""

    with gr.Blocks(title="🎲 Shadowrun RAG Assistant", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🎲 Shadowrun RAG Assistant")
        gr.Markdown("*Your Enhanced AI-powered Guide to the Sixth World*")

        with gr.Tabs():
            # ===== ENHANCED QUERY TAB =====
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

                        # Enhanced thinking accordion with visual indicator
                        with gr.Accordion("🤔 Model Thinking Process", open=False, visible=False) as thinking_accordion:
                            thinking_output = gr.Markdown(
                                label="AI Reasoning",
                                value="*The model's reasoning process will appear here...*"
                            )

                        sources_output = gr.Markdown(label="Sources & Filters")

                        with gr.Accordion("📊 Retrieved Chunks", open=False):
                            chunks_output = gr.Dataframe(
                                headers=["Relevance", "Content"],
                                label="Context Chunks"
                            )

                    with gr.Column(scale=1):
                        gr.Markdown("### ⚙️ Enhanced Configuration")

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
                                label="Character Role (overrides section filter)"
                            )

                            character_stats_input = gr.Textbox(
                                label="Character Stats",
                                placeholder="e.g., Logic 6, Hacking 5"
                            )

                            edition_select = gr.Dropdown(
                                choices=["SR5", "SR6", "SR4", "SR3", "None"],  # SR5 first as default
                                value="SR5",  # Default to SR5
                                label="Preferred Edition"
                            )

                        with gr.Accordion("🔍 Enhanced Filters", open=False):
                            gr.Markdown("*Character role selection overrides section filter*")

                            section_filter = gr.Dropdown(
                                choices=["All", "Combat", "Matrix", "Magic", "Gear",
                                         "Character Creation", "Riggers", "Technomancy", "Social"],
                                value="All",
                                label="Filter by Section"
                            )

                            subsection_filter = gr.Textbox(
                                label="Filter by Subsection",
                                placeholder="e.g., Hacking, Spellcasting"
                            )

                            document_type_filter = gr.Dropdown(
                                choices=["All Types", "rulebook", "character_sheet", "universe_info", "adventure"],
                                value="All Types",
                                label="Filter by Document Type"
                            )

                            edition_filter = gr.Dropdown(
                                choices=["All Editions", "SR5", "SR6", "SR4", "SR3"],
                                value="All Editions",
                                label="Filter by Edition"
                            )

                # Wire up enhanced query submission
                submit_btn.click(
                    fn=submit_query,
                    inputs=[
                        question_input, model_select, n_results_slider,
                        query_type_select, character_role_select, character_stats_input,
                        edition_select, section_filter, subsection_filter,
                        document_type_filter, edition_filter
                    ],
                    outputs=[answer_output, thinking_output, sources_output, chunks_output, thinking_accordion]
                )

            # ===== ENHANCED UPLOAD TAB =====
            with gr.Tab("📤 Upload"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 📄 Enhanced Document Upload")

                        document_type_select = gr.Dropdown(
                            choices=["rulebook", "character_sheet", "universe_info", "adventure"],
                            value="rulebook",
                            label="Document Type",
                            info="Affects processing strategy and metadata"
                        )

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

                        progress_display = gr.Slider(
                            minimum=0,
                            maximum=100,
                            value=0,
                            label="Overall Progress (%)",
                            interactive=False,
                            visible=False
                        )

                        with gr.Row():
                            check_progress_btn = gr.Button("🔄 Check Progress")
                            auto_refresh = gr.Checkbox(label="Auto-refresh progress", value=False)

                        progress_status = gr.Markdown(label="Enhanced Processing Status")

                    with gr.Column():
                        gr.Markdown("### 📋 Document Type Guide")

                        gr.Markdown("""
                        **📚 Rulebook**: Core rules, supplements, source books
                        - Optimized for rule extraction and semantic chunking
                        - Detects sections: Combat, Magic, Matrix, etc.

                        **👤 Character Sheet**: PC/NPC sheets, character data
                        - Faster processing with table detection
                        - Smaller chunks for precise character info

                        **🌍 Universe Info**: Lore, setting, background material
                        - Focuses on narrative content and world-building
                        - Detects corporations, locations, timeline events

                        **🎯 Adventure**: Scenarios, missions, campaigns
                        - Optimized for plot hooks and encounter data
                        - Extracts NPCs, locations, and story elements
                        """)

                        gr.Markdown("### 🔧 Manual Operations")

                        reindex_btn = gr.Button("🔄 Re-index All Documents")
                        index_new_btn = gr.Button("📊 Index New Documents Only")

                        reindex_output = gr.Textbox(label="Operation Result", lines=2)

                        # Wire up enhanced upload functionality
                        upload_btn.click(
                            fn=process_uploads,
                            inputs=[file_upload, document_type_select],
                            outputs=[upload_status, file_upload]
                        )

                        check_progress_btn.click(
                            fn=poll_progress,
                            outputs=[progress_status, progress_display, gr.State(), upload_status]
                        )

                        reindex_btn.click(
                            fn=lambda: reindex_documents(True),
                            outputs=[reindex_output]
                        )

                        index_new_btn.click(
                            fn=lambda: reindex_documents(False),
                            outputs=[reindex_output]
                        )

                        # Enhanced auto-refresh timer
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
                    outputs=[progress_status, progress_display, gr.State(), upload_status]
                )

            # ===== ENHANCED DOCUMENTS TAB =====
            with gr.Tab("📚 Documents"):
                with gr.Row():
                    # Left Panel - Enhanced Search & Library (40% width)
                    with gr.Column(scale=40):
                        refresh_docs_btn = gr.Button("🔄 Refresh Document Library")

                        # Enhanced Search Results Accordion
                        with gr.Accordion("🔍 Enhanced Document Search", open=True):
                            search_query = gr.Textbox(
                                label="Search Query",
                                placeholder="Search filenames and content...",
                                lines=1
                            )

                            with gr.Row():
                                group_filter = gr.Dropdown(
                                    label="Group",
                                    choices=["All Groups"],
                                    value="All Groups",
                                    scale=1
                                )
                                type_filter = gr.Dropdown(
                                    label="Type",
                                    choices=["All Types"],
                                    value="All Types",
                                    scale=1
                                )

                            edition_filter = gr.Dropdown(
                                label="Edition",
                                choices=["All Editions"],
                                value="All Editions"
                            )

                            search_btn = gr.Button("🔍 Search", variant="primary")

                            # Search results with pagination
                            search_summary = gr.Markdown(
                                value="Enter a search query to find documents"
                            )

                            # Pagination controls
                            with gr.Row(visible=False) as pagination_row:
                                prev_btn = gr.Button("◀ Previous", size="sm", scale=1)
                                page_info = gr.Markdown("Page 1 of 1")
                                next_btn = gr.Button("Next ▶", size="sm", scale=1)

                            file_selector = gr.Radio(
                                label="Select File",
                                choices=[],
                                visible=False,
                                interactive=True
                            )

                        # Document Library Accordion
                        with gr.Accordion("📚 Document Library", open=False):
                            library_summary = gr.Markdown(value="Loading...")
                            doc_groups_state = gr.State({})

                            # Group selector
                            group_selector = gr.Radio(
                                label="Select Document Group",
                                choices=[],
                                interactive=True
                            )

                            # File selector for selected group
                            library_file_selector = gr.Radio(
                                label="Select File",
                                choices=[],
                                visible=False,
                                interactive=True
                            )

                        # Enhanced Statistics Accordion
                        with gr.Accordion("📊 Enhanced Statistics", open=False):
                            stats_display = gr.Markdown(label="System Stats")
                            stats_table = gr.Dataframe(
                                label="Detailed Metrics",
                                headers=["Category", "Count"],
                                datatype=["str", "number"]
                            )

                    # Right Panel - Enhanced Document Viewer (60% width)
                    with gr.Column(scale=60):
                        with gr.Group():
                            gr.Markdown("### 📖 Enhanced Document Viewer")
                            document_content = gr.Markdown(
                                value="🔍 Search for documents or browse the library to view content with metadata",
                                label="Content",
                                show_label=False
                            )

                # Wire up enhanced functionality
                refresh_docs_btn.click(
                    fn=refresh_documents,
                    outputs=[
                        library_summary, doc_groups_state, stats_table, stats_display,
                        group_filter, type_filter, edition_filter, group_selector
                    ]
                )

                search_btn.click(
                    fn=search_docs_fn,
                    inputs=[search_query, group_filter, type_filter, edition_filter],
                    outputs=[search_summary, file_selector, document_content, pagination_row, page_info]
                )

                # Pagination navigation
                prev_btn.click(
                    fn=lambda: navigate_search_results(-1),
                    outputs=[search_summary, file_selector, document_content, pagination_row, page_info]
                )

                next_btn.click(
                    fn=lambda: navigate_search_results(1),
                    outputs=[search_summary, file_selector, document_content, pagination_row, page_info]
                )

                file_selector.change(
                    fn=handle_file_selection,
                    inputs=[file_selector],
                    outputs=[document_content]
                )

                # Library group selection
                group_selector.change(
                    fn=load_document_group,
                    inputs=[group_selector, doc_groups_state],
                    outputs=[library_file_selector]
                )

                # Library file selection
                library_file_selector.change(
                    fn=handle_library_file_selection,
                    inputs=[library_file_selector, group_selector, doc_groups_state],
                    outputs=[document_content]
                )

                # Enhanced auto-load on startup
                app.load(
                    fn=refresh_documents,
                    outputs=[
                        library_summary, doc_groups_state, stats_table, stats_display,
                        group_filter, type_filter, edition_filter, group_selector
                    ]
                )

            # ===== SESSION NOTES TAB =====
            with gr.Tab("📝 Session Notes"):
                gr.Markdown("""
                ### 🔧 Session Notes - Enhanced Features Coming Soon!

                **Enhanced features will include:**
                - **Campaign Management**: Upload session notes as "adventure" type documents
                - **NPC Tracking**: Automatic extraction of character names and relationships  
                - **Plot Thread Analysis**: AI-powered detection of ongoing story elements
                - **Timeline Integration**: Chronological organization of campaign events
                - **Cross-Reference Search**: Find connections between sessions and rulebook content

                **For now:**
                - Upload session notes as PDFs using **"adventure"** document type in Upload tab
                - Search and query them through the main Query interface
                - They'll be processed with adventure-optimized extraction
                """)

        # Enhanced Footer
        gr.Markdown("---")
        gr.Markdown(
            """
            <center>
            <small>🎲 Shadowrun RAG Assistant v4.0 | Enhanced with Document Types, Think Tags & SR5 Defaults</small><br>
            <small>Powered by Gradio, Ollama & ChromaDB | Character Role Precedence Active</small>
            </center>
            """,
            elem_classes=["footer"]
        )

        # Enhanced Custom CSS with better document viewer
        gr.HTML("""
        <style>
        .prose {
            padding-left: 1rem !important;
        }
        .label {
            padding-left: 1rem !important;
        }
        .group { 
            padding: 1rem !important; 
            margin: 0.5rem 0 !important; 
            border-radius: 8px !important;
        }
        
        /* Enhanced document viewer - remove nested scrolling */
        .gradio-markdown { 
            padding: 0.75rem !important;
            overflow: visible !important;
            height: auto !important;
            max-height: none !important;
        }
        
        /* Improve container heights for document viewer */
        .gradio-column:has(.gradio-markdown) {
            min-height: 70vh !important;
        }
        
        .gradio-dataframe { 
            margin: 0.5rem 0 !important; 
        }
        .gradio-textbox, .gradio-dropdown { 
            margin-bottom: 0.5rem !important; 
        }

        /* Enhanced thinking indicator */
        .thinking-active {
            border-left: 4px solid #ff6b6b !important;
            background: rgba(255, 107, 107, 0.1) !important;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0% { border-left-color: #ff6b6b; }
            50% { border-left-color: #ffd93d; }
            100% { border-left-color: #ff6b6b; }
        }

        /* Document type indicators */
        .doc-type-rulebook { border-left: 3px solid #4CAF50; }
        .doc-type-character { border-left: 3px solid #2196F3; }
        .doc-type-universe { border-left: 3px solid #FF9800; }
        .doc-type-adventure { border-left: 3px solid #9C27B0; }
        
        /* Pagination styling */
        .pagination-row {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            margin: 0.5rem 0;
        }
        </style>
        """)
    return app


# ===== MAIN EXECUTION =====

if __name__ == "__main__":
    print("🚀 Starting Enhanced Gradio frontend for Shadowrun RAG...")
    print(f"📡 Connecting to API at: {API_URL}")

    # Check API connection with enhanced info
    try:
        status = client.get_status()
        stats = client.get_document_stats()

        print(f"✅ API Status: {status.get('status', 'unknown')}")
        print(f"📚 Indexed documents: {status.get('indexed_documents', 0)}")
        print(f"📊 Indexed chunks: {status.get('indexed_chunks', 0)}")

        if not isinstance(stats, dict) or "error" not in stats:
            doc_types = stats.get('document_types', {})
            editions = stats.get('editions', {})
            print(f"📋 Document types: {list(doc_types.keys())}")
            print(f"🎲 Editions found: {list(editions.keys())}")

    except Exception as e:
        print(f"⚠️ Warning: Could not connect to API: {e}")
        print("Make sure the enhanced FastAPI backend is running!")

    # Build and launch enhanced interface
    app = build_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )