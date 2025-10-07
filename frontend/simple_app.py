"""Simplified Gradio Application for Shadowrun RAG System."""

import gradio as gr
import requests
import logging
import os
import json
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

class SimpleRAGClient:
    """Simple API client for RAG operations."""
    
    def __init__(self, api_url: str):
        self.api_url = api_url
    
    def query(self, question: str, section: str = None, model: str = None, n_results: int = 5):
        """Send query to API."""
        try:
            payload = {
                "question": question,
                "n_results": n_results
            }
            if section and section != "All":
                payload["section"] = section
            if model:
                payload["model"] = model
                
            response = requests.post(f"{self.api_url}/query", json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {"answer": f"Error: {str(e)}", "sources": [], "chunks": []}
    
    def upload_file(self, file_path: str):
        """Upload PDF file."""
        try:
            with open(file_path, 'rb') as f:
                files = {'file': (Path(file_path).name, f, 'application/pdf')}
                response = requests.post(f"{self.api_url}/upload", files=files)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return {"error": str(e)}
    
    def get_job_status(self, job_id: str):
        """Get job status."""
        try:
            response = requests.get(f"{self.api_url}/job/{job_id}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Status check failed: {e}")
            return {"stage": "error", "details": str(e)}
    
    def trigger_indexing(self, force_reindex: bool = False):
        """Trigger manual indexing."""
        try:
            payload = {
                "directory": "data/processed_markdown",
                "force_reindex": force_reindex
            }
            response = requests.post(f"{self.api_url}/index", json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Indexing failed: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_stats(self):
        """Get document statistics."""
        try:
            response = requests.get(f"{self.api_url}/stats")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Stats failed: {e}")
            return {"error": str(e)}
    
    def get_models(self):
        """Get available models."""
        try:
            response = requests.get(f"{self.api_url}/models")
            response.raise_for_status()
            return response.json().get("models", [])
        except Exception as e:
            logger.error(f"Models failed: {e}")
            return []

    def query_stream(self, question: str, section: str = None, model: str = None, n_results: int = 5):
        """Send streaming query to API with proper thinking tag handling."""
        try:
            payload = {
                "question": question,
                "n_results": n_results
            }
            if section and section != "All":
                payload["section"] = section
            if model:
                payload["model"] = model
                
            response = requests.post(f"{self.api_url}/query_stream", json=payload, stream=True)
            response.raise_for_status()
            
            full_response = ""
            thinking_content = ""
            metadata = None
            metadata_buffer = ""
            collecting_metadata = False
            in_thinking = False
            think_depth = 0  # Track nested <think> tags
            buffer = ""  # Buffer for handling split tags
            
            for chunk in response.iter_content(chunk_size=32, decode_unicode=True):
                if chunk:
                    buffer += chunk
                    
                    # Process buffer for complete tags
                    while buffer:
                        processed = False
                        
                        # Handle backend markers first (if they exist)
                        if "__THINKING_START__" in buffer:
                            in_thinking = True
                            think_depth = 1
                            parts = buffer.split("__THINKING_START__", 1)
                            if parts[0]:
                                full_response += parts[0]
                                yield full_response, thinking_content, None, "generating"
                            buffer = parts[1] if len(parts) > 1 else ""
                            processed = True
                            continue

                        if "__THINKING_END__" in buffer and in_thinking:
                            in_thinking = False
                            think_depth = 0
                            parts = buffer.split("__THINKING_END__", 1)
                            if parts[0]:
                                thinking_content += parts[0]
                                yield full_response, thinking_content, None, "thinking"
                            buffer = parts[1] if len(parts) > 1 else ""
                            processed = True
                            continue
                        
                        # Handle XML think tags - opening
                        if "<think>" in buffer:
                            if not in_thinking:
                                in_thinking = True
                                think_depth = 1
                            else:
                                think_depth += 1  # Handle nested <think> tags
                            
                            parts = buffer.split("<think>", 1)
                            if parts[0]:
                                if in_thinking and think_depth > 1:
                                    # This was content inside thinking, add <think> back for context
                                    thinking_content += parts[0] + "<think>"
                                else:
                                    # This was regular content before thinking started
                                    full_response += parts[0]
                                    yield full_response, thinking_content, None, "generating"
                            buffer = parts[1] if len(parts) > 1 else ""
                            processed = True
                            continue
                            
                        # Handle XML think tags - closing
                        if "</think>" in buffer and in_thinking:
                            think_depth -= 1
                            if think_depth <= 0:
                                in_thinking = False
                                think_depth = 0
                            
                            parts = buffer.split("</think>", 1)
                            if parts[0]:
                                if in_thinking:
                                    # Still inside nested thinking, add </think> back for context
                                    thinking_content += parts[0] + "</think>"
                                else:
                                    # End of thinking block
                                    thinking_content += parts[0]
                                    yield full_response, thinking_content, None, "thinking"
                            buffer = parts[1] if len(parts) > 1 else ""
                            processed = True
                            continue

                        # Handle metadata
                        if "__METADATA_START__" in buffer:
                            parts = buffer.split("__METADATA_START__", 1)
                            if parts[0]:
                                if in_thinking:
                                    thinking_content += parts[0]
                                else:
                                    full_response += parts[0]
                                yield full_response, thinking_content, None, "generating"

                            collecting_metadata = True
                            buffer = parts[1] if len(parts) > 1 else ""
                            processed = True
                            continue

                        if collecting_metadata:
                            if "__METADATA_END__" in buffer:
                                json_part = buffer.split("__METADATA_END__")[0].strip()
                                try:
                                    metadata = json.loads(json_part)
                                except json.JSONDecodeError as e:
                                    logger.error(f"Metadata parse failed: {e}")
                                break
                            # Continue collecting metadata
                            break
                        
                        # No special tags found, process content
                        if not processed:
                            if in_thinking:
                                thinking_content += buffer
                                yield full_response, thinking_content, None, "thinking"
                            else:
                                full_response += buffer
                                yield full_response, thinking_content, None, "generating"
                            buffer = ""

            # Final yield with metadata
            yield full_response, thinking_content, metadata, "complete"
                
        except Exception as e:
            logger.error(f"Stream query failed: {e}")
            yield f"Error: {str(e)}", "", None, "error"

class SimpleShadowrunApp:
    """Main simplified application."""
    
    def __init__(self):
        self.client = SimpleRAGClient(API_URL)
        self.sections = ["All", "Matrix", "Combat", "Magic", "Riggers", "Social", "Skills", "Character Creation"]
    
    def query_handler(self, question: str, section: str, model: str):
        """Handle streaming query requests with thinking tag support."""
        if not question.strip():
            return "Please enter a question.", "", "", "", gr.update(visible=False)
        
        # Initialize response parts
        full_answer = ""
        thinking_content = ""
        sources_text = ""
        chunks_text = ""
        thinking_visible = False
        
        try:
            # Stream the response
            for response, thinking, metadata, status in self.client.query_stream(question, section, model):
                full_answer = response
                thinking_content = thinking or ""
                
                # Update thinking visibility
                if thinking_content and thinking_content.strip():
                    thinking_visible = True
                
                # Check if this is the final response (has metadata)
                if metadata and status == "complete":
                    # Format sources
                    sources = metadata.get("sources", [])
                    sources_text = "\n".join([f"• {source}" for source in sources]) if sources else "No sources found"
                    
                    # Format chunks (first 3 chunks for debugging)
                    chunks = metadata.get("chunks", [])
                    if chunks:
                        chunks_text = "\n\n---\n\n".join(chunks[:3])
                        if len(chunks) > 3:
                            chunks_text += f"\n\n... and {len(chunks) - 3} more chunks"
                    
                    # Final yield with all data
                    yield (full_answer, thinking_content, sources_text, chunks_text, 
                           gr.update(visible=thinking_visible))
                    break
                else:
                    # Intermediate yield for smooth streaming
                    cursor = "▌" if status == "generating" else "🤔" if status == "thinking" else ""
                    # For markdown, only add cursor if we're not in the middle of markdown syntax
                    display_answer = full_answer
                    if cursor and not full_answer.endswith(('*', '_', '`', '#', '|', '[', ']')):
                        display_answer += cursor
                    
                    yield (display_answer, thinking_content, "", "", 
                           gr.update(visible=thinking_visible))
            
        except Exception as e:
            yield f"Error: {str(e)}", "", "", "", gr.update(visible=False)
    
    def upload_handler(self, file):
        """Handle file uploads."""
        if file is None:
            return "No file selected", "", ""
        
        try:
            # Upload file
            result = self.client.upload_file(file.name)
            
            if "error" in result:
                return f"Upload failed: {result['error']}", "", ""
            
            job_id = result.get("job_id", "")
            message = result.get("message", "Upload started")
            
            return f"✅ {message}", job_id, "Upload initiated..."
            
        except Exception as e:
            return f"Upload error: {str(e)}", "", ""
    
    def check_job_status(self, job_id: str):
        """Check job processing status."""
        if not job_id:
            return "No active job"
        
        try:
            status = self.client.get_job_status(job_id)
            stage = status.get("stage", "unknown")
            progress = status.get("progress", 0)
            details = status.get("details", "")
            
            if stage == "complete":
                return f"✅ Complete: {details}"
            elif stage == "error":
                return f"❌ Error: {details}"
            elif stage == "not_found":
                return "Job not found or completed"
            else:
                return f"⏳ {stage}: {progress}% - {details}"
                
        except Exception as e:
            return f"Status check error: {str(e)}"
    
    def index_handler(self, force_reindex: bool):
        """Handle manual indexing."""
        try:
            result = self.client.trigger_indexing(force_reindex)
            
            if result.get("status") == "success":
                return f"✅ {result.get('message', 'Indexing completed')}"
            else:
                return f"❌ {result.get('message', 'Indexing failed')}"
                
        except Exception as e:
            return f"Indexing error: {str(e)}"
    
    def get_stats_display(self):
        """Get formatted statistics display."""
        try:
            stats = self.client.get_stats()
            
            if "error" in stats:
                return f"Error getting stats: {stats['error']}"
            
            total_chunks = stats.get("total_chunks", 0)
            unique_docs = stats.get("unique_documents", 0)
            sections = stats.get("sections", {})
            sources = stats.get("sources", [])
            
            # Format display
            display = f"📊 **Database Statistics**\n\n"
            display += f"• **Total Chunks**: {total_chunks}\n"
            display += f"• **Unique Documents**: {unique_docs}\n\n"
            
            if sections:
                display += "**Chunks by Section**:\n"
                for section, count in sorted(sections.items()):
                    display += f"  • {section}: {count}\n"
                display += "\n"
            
            if sources:
                display += "**Available Documents**:\n"
                for source in sorted(sources):
                    display += f"  • {source}\n"
            
            return display
            
        except Exception as e:
            return f"Error getting statistics: {str(e)}"
    
    def create_interface(self):
        """Create the Gradio interface."""
        
        # Get available models
        models = self.client.get_models()
        default_model = models[0] if models else "qwen2.5:14b-instruct-q6_K"
        
        with gr.Blocks(title="Shadowrun RAG - Simplified", theme=gr.themes.Soft()) as interface:
            
            gr.Markdown("# 🎲 Shadowrun RAG System - Simplified")
            gr.Markdown("Ask questions about Shadowrun 5th Edition rules and lore.")
            
            with gr.Tabs():
                
                # Query Tab
                with gr.TabItem("Query"):
                    with gr.Row():
                        with gr.Column(scale=3):
                            question_input = gr.Textbox(
                                label="Question",
                                placeholder="Ask about Shadowrun rules, e.g., 'How does matrix combat work?'",
                                lines=2
                            )
                            
                            with gr.Row():
                                section_dropdown = gr.Dropdown(
                                    choices=self.sections,
                                    value="All",
                                    label="Section Filter",
                                    info="Filter results by rulebook section"
                                )
                                
                                model_dropdown = gr.Dropdown(
                                    choices=models,
                                    value=default_model,
                                    label="Model",
                                    info="Select AI model"
                                )
                            
                            query_btn = gr.Button("Ask Question", variant="primary")
                        
                        with gr.Column(scale=1):
                            gr.Markdown("### Quick Examples")
                            gr.Markdown("""
                            • How do I hack a device?
                            • What are the combat rules?
                            • How does spell casting work?
                            • What are the different metatypes?
                            • How do rigger drones work?
                            """)
                    
                    # Results
                    answer_output = gr.Markdown(
                        label="Answer",
                        value="",
                        show_copy_button=True
                    )
                    
                    # Thinking process (collapsible)
                    thinking_accordion = gr.Accordion("🤔 AI Thinking Process", open=False, visible=False)
                    with thinking_accordion:
                        thinking_output = gr.Markdown(
                            label="Thinking",
                            value="",
                            show_copy_button=True
                        )
                    
                    with gr.Accordion("Sources & Debug Info", open=False):
                        sources_output = gr.Textbox(
                            label="Sources",
                            lines=3
                        )
                        
                        chunks_output = gr.Textbox(
                            label="Retrieved Chunks (Debug)",
                            lines=5,
                            max_lines=10
                        )
                
                # Upload Tab
                with gr.TabItem("Upload & Process"):
                    gr.Markdown("### Upload PDF Rulebooks")
                    gr.Markdown("Upload Shadowrun PDF files for processing. Only TOC-guided extraction is used.")
                    
                    with gr.Row():
                        with gr.Column():
                            file_input = gr.File(
                                label="Select PDF File",
                                file_types=[".pdf"]
                            )
                            upload_btn = gr.Button("Upload & Process", variant="primary")
                            
                            upload_status = gr.Textbox(
                                label="Upload Status",
                                lines=2
                            )
                        
                        with gr.Column():
                            job_id_display = gr.Textbox(
                                label="Job ID",
                                interactive=False
                            )
                            
                            status_check_btn = gr.Button("Check Status")
                            
                            job_status = gr.Textbox(
                                label="Processing Status",
                                lines=3
                            )
                
                # Indexing Tab
                with gr.TabItem("Indexing"):
                    gr.Markdown("### Manual Indexing")
                    gr.Markdown("After files are processed, manually trigger indexing to add them to the search database.")
                    
                    with gr.Row():
                        with gr.Column():
                            force_reindex_checkbox = gr.Checkbox(
                                label="Force Re-index",
                                info="Re-index all documents even if already indexed"
                            )
                            
                            index_btn = gr.Button("Start Indexing", variant="primary")
                            
                            index_status = gr.Textbox(
                                label="Indexing Status",
                                lines=2
                            )
                        
                        with gr.Column():
                            gr.Markdown("### Process")
                            gr.Markdown("""
                            1. **Upload** PDF files in the Upload tab
                            2. **Wait** for processing to complete
                            3. **Index** the processed files here
                            4. **Query** the indexed content
                            """)
                
                # Statistics Tab
                with gr.TabItem("Statistics"):
                    gr.Markdown("### Database Statistics")
                    
                    with gr.Row():
                        with gr.Column():
                            refresh_stats_btn = gr.Button("Refresh Statistics")
                            
                        with gr.Column():
                            pass
                    
                    stats_display = gr.Markdown("Click 'Refresh Statistics' to see database info.")
            
            # Event handlers
            query_btn.click(
                fn=self.query_handler,
                inputs=[question_input, section_dropdown, model_dropdown],
                outputs=[answer_output, thinking_output, sources_output, chunks_output, thinking_accordion]
            )
            
            question_input.submit(
                fn=self.query_handler,
                inputs=[question_input, section_dropdown, model_dropdown],
                outputs=[answer_output, thinking_output, sources_output, chunks_output, thinking_accordion]
            )
            
            upload_btn.click(
                fn=self.upload_handler,
                inputs=[file_input],
                outputs=[upload_status, job_id_display, job_status]
            )
            
            status_check_btn.click(
                fn=self.check_job_status,
                inputs=[job_id_display],
                outputs=[job_status]
            )
            
            index_btn.click(
                fn=self.index_handler,
                inputs=[force_reindex_checkbox],
                outputs=[index_status]
            )
            
            refresh_stats_btn.click(
                fn=self.get_stats_display,
                outputs=[stats_display]
            )
        
        return interface

def main():
    """Main function to run the application."""
    app = SimpleShadowrunApp()
    interface = app.create_interface()
    
    # Launch the interface
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,  # Use Docker internal port
        share=False,
        debug=False
    )

if __name__ == "__main__":
    main()
