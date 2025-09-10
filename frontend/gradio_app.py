"""
Main Gradio Application for Shadowrun RAG System
Clean, modular application that brings all components together.
"""

import gradio as gr
import logging
import os

# Import our modular components
from api_clients.rag_client import RAGClient
from api_clients.character_client import CharacterAPIClient
from components.rag_ui import RAGQueryUI, RAGQueryHandlers, wire_query_events, get_initial_character_choices
from components.character_ui import CharacterUI, CharacterEventHandlers, wire_character_events
from components.document_ui import DocumentUI, DocumentHandlers, wire_document_events
from components.upload_ui import UploadUI, UploadHandlers, wire_upload_events
from ui_helpers.ui_helpers import setup_ui_logging, get_custom_css

logger = logging.getLogger(__name__)

# API Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")


class ShadowrunRAGApp:
    """Main application class that orchestrates all components."""

    def __init__(self):
        self.api_url = API_URL
        self.rag_client = RAGClient(self.api_url)
        self.char_api = CharacterAPIClient(self.api_url)

        # Initialize UI components
        self.rag_ui = RAGQueryUI(self.rag_client, self.char_api)
        self.character_ui = CharacterUI(self.char_api)
        self.document_ui = DocumentUI(self.rag_client)
        self.upload_ui = UploadUI(self.rag_client)

        # Initialize event handlers
        self.rag_handlers = RAGQueryHandlers(self.rag_client, self.char_api)
        self.character_handlers = CharacterEventHandlers(self.char_api)
        self.document_handlers = DocumentHandlers(self.rag_client)
        self.upload_handlers = UploadHandlers(self.rag_client)

        # Store components for event wiring
        self.rag_components = {}
        self.document_components = {}
        self.character_components = {}
        self.upload_components = {}

    def build_interface(self):
        """Build the complete Gradio interface."""
        with gr.Blocks(title="🎲 Shadowrun RAG Assistant", theme=gr.themes.Soft(), css=get_custom_css()) as app:
            gr.Markdown("# 🎲 Shadowrun RAG Assistant")
            gr.Markdown("*Your Enhanced AI-powered Guide to the Sixth World*")

            with gr.Tabs():
                # Build each tab and store components
                self.rag_components = self._build_query_tab()
                self.upload_components = self._build_upload_tab()
                self.document_components = self._build_documents_tab()
                self.character_components = self._build_character_tab()
                self._build_session_notes_tab()

            # Add footer
            self._build_footer()

            # Wire up all events
            self._wire_all_events()

            # Set up auto-load functionality
            self._setup_auto_load(app)

        return app

    def _build_query_tab(self):
        """Build the query tab."""
        return self.rag_ui.build_query_tab()

    def _build_upload_tab(self):
        """Build the upload tab."""
        return self.upload_ui.build_upload_tab()

    def _build_documents_tab(self):
        """Build the documents tab."""
        return self.document_ui.build_documents_tab()

    def _build_character_tab(self):
        """Build the character tab."""
        return self.character_ui.build_complete_character_tab()

    def _build_session_notes_tab(self):
        """Build the session notes tab."""
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
            return {}

    def _build_footer(self):
        """Build the application footer."""
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

    def _wire_all_events(self):
        """Wire up all event handlers across components."""
        # Wire RAG query events
        wire_query_events(self.rag_components, self.rag_handlers)

        # Wire upload events
        wire_upload_events(self.upload_components, self.upload_handlers)

        # Wire document events
        wire_document_events(self.document_components, self.document_handlers)

        # Wire character events
        wire_character_events(self.character_components, self.character_handlers)

    def _setup_auto_load(self, app):
        """Set up auto-load functionality for the application."""
        # Auto-load character choices on app load
        app.load(
            fn=lambda: get_initial_character_choices(self.char_api),
            outputs=[self.rag_components["character_query_selector"]]
        )

        # Auto-load models on app load
        app.load(
            fn=self.rag_handlers.refresh_models,
            outputs=[self.rag_components["model_select"]]
        )

        # Auto-load document library on app load
        app.load(
            fn=self.document_handlers.refresh_documents,
            outputs=[
                self.document_components["library_summary"],
                self.document_components["doc_groups_state"],
                self.document_components["stats_table"],
                self.document_components["stats_display"],
                self.document_components["group_filter"],
                self.document_components["type_filter"],
                self.document_components["edition_filter"],
                self.document_components["group_selector"]
            ]
        )

    def run(self):
        """Run the application."""
        print("🚀 Starting Enhanced Gradio frontend for Shadowrun RAG...")
        print(f"📡 Connecting to API at: {self.api_url}")

        # Check API connection
        self._check_api_connection()

        # Build and launch interface
        app = self.build_interface()
        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True
        )

    def _check_api_connection(self):
        """Check API connection status."""
        try:
            status = self.rag_client.get_status()
            stats = self.rag_client.get_document_stats()

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


def main():
    """Main entry point for the application."""
    # Set up logging
    setup_ui_logging()

    # Create and run the application
    app = ShadowrunRAGApp()
    app.run()


if __name__ == "__main__":
    main()