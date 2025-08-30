"""
Upload UI Components for Shadowrun RAG System
Clean extraction of upload functionality with progress tracking.
"""

import gradio as gr
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from frontend.ui_helpers.ui_helpers import UIErrorHandler

logger = logging.getLogger(__name__)


class UploadUI:
    """Upload interface with progress tracking and document type selection."""

    def __init__(self, rag_client):
        self.rag_client = rag_client
        self.active_jobs = {}

    def build_upload_tab(self):
        """Build the complete upload tab."""
        with gr.Tab("📤 Upload"):
            return self._build_upload_interface()

    def _build_upload_interface(self):
        """Build the main upload interface."""
        components = {}

        with gr.Row():
            # Left Panel - Upload Controls (50% width)
            with gr.Column(scale=50):
                components.update(self._build_upload_section())
                components.update(self._build_progress_section())

            # Right Panel - Documentation & Manual Operations (50% width)
            with gr.Column(scale=50):
                components.update(self._build_documentation_section())
                components.update(self._build_manual_operations_section())

        return components

    def _build_upload_section(self):
        """Build the file upload section."""
        components = {}

        gr.Markdown("### 📄 Enhanced Document Upload")

        components["document_type_select"] = gr.Dropdown(
            choices=["rulebook", "character_sheet", "universe_info", "adventure"],
            value="rulebook",
            label="Document Type",
            info="Affects processing strategy and metadata"
        )

        components["extraction_method_select"] = gr.Radio(
            choices=[
                ("Hybrid Pipeline", "hybrid"),
                ("Vision Analysis", "vision")
            ],
            value="hybrid",
            label="🔧 Extraction Strategy",
            info="Hybrid: Fast Marker+LLM → fallbacks | Vision: Direct visual analysis for complex tables"
        )

        # NEW: Add vision model selection (hidden by default)
        components["vision_model_select"] = gr.Dropdown(
            choices=["qwen2.5vl:7b", "llama3.2-vision:11b", "granite3.2-vision"],
            value="qwen2.5vl:7b",
            label="👁️ Vision Model",
            visible=False,
            info="Vision model for complex table extraction"
        )

        components["file_upload"] = gr.File(
            label="Upload PDFs",
            file_types=[".pdf"],
            file_count="multiple"
        )

        components["upload_btn"] = gr.Button("📤 Process Files", variant="primary")

        components["upload_status"] = gr.Textbox(
            label="Upload Status",
            lines=5,
            interactive=False
        )

        return components

    def _build_progress_section(self):
        """Build the progress monitoring section."""
        components = {}

        components["progress_display"] = gr.Slider(
            minimum=0,
            maximum=100,
            value=0,
            label="Overall Progress (%)",
            interactive=False,
            visible=False
        )

        with gr.Row():
            components["check_progress_btn"] = gr.Button("🔄 Check Progress")
            components["auto_refresh"] = gr.Checkbox(label="Auto-refresh progress", value=False)

        components["progress_status"] = gr.Markdown("### 📊 Processing Status")

        # Timer for auto-refresh
        components["timer"] = gr.Timer(5.0, active=False)

        return components

    def _build_documentation_section(self):
        """Build the documentation section."""
        components = {}

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

        return components

    def _build_manual_operations_section(self):
        """Build the manual operations section."""
        components = {}

        gr.Markdown("### 🔧 Manual Operations")

        components["reindex_btn"] = gr.Button("🔄 Re-index All Documents")
        components["index_new_btn"] = gr.Button("📊 Index New Documents Only")

        components["reindex_output"] = gr.Textbox(label="Operation Result", lines=2)

        return components


class UploadHandlers:
    """Event handlers for upload functionality."""

    def __init__(self, rag_client):
        self.rag_client = rag_client
        self.active_jobs = {}

    def process_uploads(self, files: List, document_type: str, extraction_method: str = "hybrid", vision_model: str = "qwen2.5vl:7b") -> Tuple[str, gr.update]:
        """Process multiple uploaded files."""
        try:
            if not files:
                return "❌ No files selected for upload.", gr.update()

            status_messages = []
            job_ids = []

            for file in files:
                if file is None:
                    continue

                try:
                    # Upload the file
                    result = self.rag_client.upload_pdf(file.name,
                                                        document_type = document_type,
                                                        extraction_method = extraction_method,
                                                        vision_model = vision_model,
                                                        )

                    if "error" in result:
                        status_messages.append(f"❌ **{file.name}**: {result['error']}")
                    else:
                        job_id = result.get("job_id")
                        if job_id:
                            self.active_jobs[job_id] = {
                                "filename": result.get("filename", file.name),
                                "document_type": document_type,
                                "status": "processing"
                            }
                            job_ids.append(job_id)
                            status_messages.append(
                                f"✅ **{file.name}**: Upload started\n"
                                f"   📋 Type: {document_type}\n"
                                f"   🔑 Job ID: {job_id}"
                            )
                        else:
                            status_messages.append(f"❌ **{file.name}**: No job ID returned")

                except Exception as e:
                    error_msg = UIErrorHandler.handle_exception(e, f"uploading {file.name}")
                    status_messages.append(f"❌ **{file.name}**: {str(e)}")

            # Combine all status messages
            combined_status = "\n\n".join(status_messages)

            if job_ids:
                combined_status += f"\n\n📊 **Tracking {len(job_ids)} job(s)**\nUse 'Check Progress' to monitor status."

            return combined_status, gr.update(value=None)  # Clear file upload

        except Exception as e:
            error_msg = UIErrorHandler.handle_exception(e, "processing uploads")
            return error_msg, gr.update()

    def poll_progress(self) -> Tuple[str, gr.update, gr.update]:
        """Poll progress for all active jobs."""
        try:
            if not self.active_jobs:
                return (
                    "### 📊 Processing Status\n\nNo active uploads to track.",
                    gr.update(visible=False),
                    "No active jobs"
                )

            status_lines = ["### 📊 Processing Status\n"]
            overall_progress = 0
            completed_jobs = []
            total_progress = 0
            job_count = 0

            for job_id, job_info in self.active_jobs.items():
                try:
                    result = self.rag_client.get_job_status(job_id)

                    if "error" in result:
                        status_lines.append(f"❌ **{job_info['filename']}** (Error): {result['error']}")
                        completed_jobs.append(job_id)
                        continue

                    stage = result.get("stage", "unknown")
                    progress = result.get("progress", 0)
                    details = result.get("details", "")

                    # Update job info
                    job_info["status"] = stage
                    job_info["progress"] = progress

                    if stage == "complete":
                        status_lines.append(
                            f"✅ **{job_info['filename']}** ({job_info['document_type']}): Complete!"
                        )
                        completed_jobs.append(job_id)
                    elif stage == "error":
                        status_lines.append(
                            f"❌ **{job_info['filename']}** ({job_info['document_type']}): {details}"
                        )
                        completed_jobs.append(job_id)
                    else:
                        progress_bar = "🟩" * int(progress // 10) + "⬜" * (10 - int(progress // 10))
                        status_lines.append(
                            f"⏳ **{job_info['filename']}** ({job_info['document_type']})\n"
                            f"   📍 Stage: {stage}\n"
                            f"   📊 Progress: {progress:.1f}% {progress_bar}\n"
                            f"   ℹ️ {details}"
                        )
                        total_progress += progress
                        job_count += 1

                except Exception as e:
                    status_lines.append(f"❌ **{job_info['filename']}**: Status check failed - {str(e)}")
                    completed_jobs.append(job_id)

            # Clean up completed jobs
            for job_id in completed_jobs:
                del self.active_jobs[job_id]

            # Calculate overall progress
            if job_count > 0:
                overall_progress = total_progress / job_count
                progress_update = gr.update(value=overall_progress, visible=True)
            else:
                progress_update = gr.update(visible=False)

            status_markdown = "\n\n".join(status_lines)

            # Add summary
            if self.active_jobs:
                status_markdown += f"\n\n**Active Jobs**: {len(self.active_jobs)}"
            else:
                status_markdown += "\n\n**All jobs completed!**"

            return status_markdown, progress_update, "Progress updated"

        except Exception as e:
            error_msg = UIErrorHandler.handle_exception(e, "polling progress")
            return f"### 📊 Processing Status\n\n{error_msg}", gr.update(visible=False), "Error occurred"

    def reindex_documents(self, force_reindex: bool) -> str:
        """Trigger document reindexing through existing backend API."""
        try:
            operation = "full reindex" if force_reindex else "incremental index"

            # Show immediate feedback
            status_msg = f"🔄 Starting {operation}...\n\n"

            # Call your existing backend API
            result = self.rag_client.reindex(force=force_reindex)

            if "error" in result:
                return f"❌ **{operation.title()} Failed**\n\n" \
                       f"Error: {result['error']}"

            # Handle your existing response format
            if result.get("status") == "success":
                status_msg += f"✅ **{result.get('message', 'Indexing completed')}**\n\n"

                if force_reindex:
                    status_msg += f"🔥 **Full reindex completed** - all documents re-processed with improved classification!\n\n"
                else:
                    status_msg += f"📈 **Incremental index completed** - new/changed files processed.\n\n"

                status_msg += f"💡 **Next steps:**\n"
                status_msg += f"   • Test your taser query in the Query tab\n"
                status_msg += f"   • Run chunk analyzer to see improvements\n"
                status_msg += f"   • Check classification diversity increased\n"
            else:
                status_msg += f"✅ **{operation.title()} Triggered**\n\n{result}"

            return status_msg

        except Exception as e:
            error_msg = UIErrorHandler.handle_exception(e, f"reindexing ({'full' if force_reindex else 'incremental'})")
            return error_msg


def wire_upload_events(components: Dict, handlers: UploadHandlers):
    """Wire up upload event handlers."""

    def toggle_vision_model_visibility(extraction_method):
        """Show/hide vision model selector based on extraction method."""
        return gr.update(visible=(extraction_method == "vision"))

    # Main upload functionality
    components["upload_btn"].click(
        fn=handlers.process_uploads,
        inputs=[components["file_upload"],
                components["document_type_select"],
                components["extraction_method_select"],
                components["vision_model_select"]
        ],
        outputs=[components["upload_status"],
                 components["file_upload"]
        ]
    )

    # Progress checking
    components["check_progress_btn"].click(
        fn=handlers.poll_progress,
        outputs=[components["progress_status"], components["progress_display"], components["upload_status"]]
    )

    # Manual operations
    components["reindex_btn"].click(
        fn=lambda: handlers.reindex_documents(True),
        outputs=[components["reindex_output"]]
    )

    components["index_new_btn"].click(
        fn=lambda: handlers.reindex_documents(False),
        outputs=[components["reindex_output"]]
    )

    # Auto-refresh functionality
    components["auto_refresh"].change(
        fn=lambda x: gr.update(active=x),
        inputs=[components["auto_refresh"]],
        outputs=[components["timer"]]
    )

    components["timer"].tick(
        fn=handlers.poll_progress,
        outputs=[components["progress_status"], components["progress_display"], components["upload_status"]]
    )

    components["extraction_method_select"].change(
        fn=toggle_vision_model_visibility,
        inputs=[components["extraction_method_select"]],
        outputs=[components["vision_model_select"]]
    )

    return components