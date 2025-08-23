"""
Document Management UI Components for Shadowrun RAG System
Clean extraction of document library and search functionality.
"""

import gradio as gr
import pandas as pd
import logging
from typing import Dict
from pathlib import Path
from frontend.ui_helpers.ui_helpers import UIErrorHandler

logger = logging.getLogger(__name__)


class DocumentUI:
    """Document management interface with search and library functionality."""

    def __init__(self, rag_client):
        self.rag_client = rag_client
        self.search_file_paths = {}
        self.current_search_params = {}
        self.current_page = 1

    def build_documents_tab(self):
        """Build the complete documents tab."""
        with gr.Tab("📚 Documents"):
            return self._build_documents_interface()

    def _build_documents_interface(self):
        """Build the main documents interface."""
        components = {}

        with gr.Row():
            # Left Panel - Enhanced Search & Library (40% width)
            with gr.Column(scale=40):
                components.update(self._build_search_section())
                components.update(self._build_library_section())
                components.update(self._build_stats_section())

            # Right Panel - Enhanced Document Viewer (60% width)
            with gr.Column(scale=60):
                components.update(self._build_document_viewer_section())

        return components

    def _build_search_section(self):
        """Build document search section."""
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

        return {
            "search_query": search_query,
            "group_filter": group_filter,
            "type_filter": type_filter,
            "edition_filter": edition_filter,
            "search_btn": search_btn,
            "search_summary": search_summary,
            "pagination_row": pagination_row,
            "prev_btn": prev_btn,
            "page_info": page_info,
            "next_btn": next_btn,
            "file_selector": file_selector
        }

    def _build_library_section(self):
        """Build document library section."""
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

        return {
            "library_summary": library_summary,
            "doc_groups_state": doc_groups_state,
            "group_selector": group_selector,
            "library_file_selector": library_file_selector
        }

    def _build_stats_section(self):
        """Build statistics section."""
        with gr.Accordion("📊 Enhanced Statistics", open=False):
            stats_display = gr.Markdown(label="System Stats")
            stats_table = gr.Dataframe(
                label="Detailed Metrics",
                headers=["Category", "Count"],
                datatype=["str", "number"]
            )

        return {
            "stats_display": stats_display,
            "stats_table": stats_table
        }

    def _build_document_viewer_section(self):
        """Build document viewer section."""
        with gr.Group():
            gr.Markdown("### 📖 Enhanced Document Viewer")
            document_content = gr.Markdown(
                value="🔍 Search for documents or browse the library to view content with metadata",
                label="Content",
                show_label=False
            )

        return {
            "document_content": document_content
        }


class DocumentHandlers:
    """Event handlers for document management operations."""

    def __init__(self, rag_client):
        self.rag_client = rag_client
        self.search_file_paths = {}
        self.current_search_params = {}
        self.current_page = 1

    def refresh_documents(self):
        """Enhanced document library refresh with metadata statistics."""
        try:
            docs = self.rag_client.get_documents()
            stats = self.rag_client.get_document_stats()

            if "error" in stats:
                return (
                    f"Error loading stats: {stats['error']}",
                    {},
                    pd.DataFrame(),
                    "",
                    gr.update(choices=[]),
                    gr.update(choices=[]),
                    gr.update(choices=[]),
                    gr.update(choices=[])
                )

            if not docs:
                return (
                    "No documents indexed yet",
                    {},
                    pd.DataFrame(),
                    "",
                    gr.update(choices=[]),
                    gr.update(choices=[]),
                    gr.update(choices=[]),
                    gr.update(choices=[])
                )

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
            active_jobs = 0

            try:
                status = self.rag_client.get_status()
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
            type_choices = ["All Types"] + list(doc_types.keys())
            edition_choices = ["All Editions"] + list(editions.keys())

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

        except Exception as e:
            error_msg = UIErrorHandler.handle_exception(e, "document refresh")
            return (
                error_msg,
                {},
                pd.DataFrame(),
                "",
                gr.update(choices=[]),
                gr.update(choices=[]),
                gr.update(choices=[]),
                gr.update(choices=[])
            )

    def search_docs(self, query: str, selected_group: str, selected_type: str, selected_edition: str, page: int = 1):
        """Enhanced document search with metadata filtering and pagination."""
        try:
            # Store search parameters for pagination
            self.current_search_params = {
                "query": query,
                "selected_group": selected_group,
                "selected_type": selected_type,
                "selected_edition": selected_edition
            }
            self.current_page = page

            self.search_file_paths = {}

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

            results = self.rag_client.search_documents(
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
                self.search_file_paths[display_name] = file_info['file_path']

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
            first_content = self.load_document_content(files[0]['file_path'])

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

        except Exception as e:
            error_msg = UIErrorHandler.handle_exception(e, "document search")
            return (
                error_msg,
                gr.update(choices=[], visible=False),
                "",
                gr.update(visible=False),
                gr.update(value="Page 0 of 0")
            )

    def navigate_search_results(self, direction: int):
        """Navigate search results pagination."""
        if not self.current_search_params:
            return self.search_docs("", "All Groups", "All Types", "All Editions", 1)

        new_page = max(1, self.current_page + direction)

        return self.search_docs(
            self.current_search_params["query"],
            self.current_search_params["selected_group"],
            self.current_search_params["selected_type"],
            self.current_search_params["selected_edition"],
            new_page
        )

    def handle_file_selection(self, selected_choice: str):
        """Handle radio button selection."""
        if not selected_choice or selected_choice not in self.search_file_paths:
            return "No file selected or file not found"

        file_path = self.search_file_paths[selected_choice]
        return self.load_document_content(file_path)

    def load_document_content(self, file_path: str):
        """Load and display document content with enhanced metadata display."""
        try:
            if not file_path:
                return "No file selected"

            result = self.rag_client.get_document_content(file_path)

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

        except Exception as e:
            error_msg = UIErrorHandler.handle_exception(e, "document loading")
            return error_msg

    def load_document_group(self, group_name: str, doc_groups: Dict):
        """Load all files for a specific document group."""
        try:
            if not group_name or group_name not in doc_groups:
                return gr.update(choices=[], visible=False)

            files = doc_groups[group_name]
            choices = []

            for file_info in files:
                display_name = f"📄 {file_info['filename']}"
                choices.append(display_name)

            return gr.update(choices=choices, visible=True, value=None)

        except Exception as e:
            logger.error(f"Failed to load document group: {e}")
            return gr.update(choices=[], visible=False)

    def handle_library_file_selection(self, selected_file: str, group_name: str, doc_groups: Dict):
        """Handle file selection from document library."""
        try:
            if not selected_file or not group_name or group_name not in doc_groups:
                return "No file selected"

            # Find the file path
            filename = selected_file.replace("📄 ", "")
            for file_info in doc_groups[group_name]:
                if file_info['filename'] == filename:
                    return self.load_document_content(file_info['path'])

            return "File not found"

        except Exception as e:
            error_msg = UIErrorHandler.handle_exception(e, "library file selection")
            return error_msg


def wire_document_events(components: Dict, handlers: DocumentHandlers):
    """Wire up document management event handlers."""

    # Document refresh
    components["search_btn"].click(
        fn=handlers.refresh_documents,
        outputs=[
            components["library_summary"],
            components["doc_groups_state"],
            components["stats_table"],
            components["stats_display"],
            components["group_filter"],
            components["type_filter"],
            components["edition_filter"],
            components["group_selector"]
        ]
    )

    # Search functionality
    components["search_btn"].click(
        fn=handlers.search_docs,
        inputs=[
            components["search_query"],
            components["group_filter"],
            components["type_filter"],
            components["edition_filter"]
        ],
        outputs=[
            components["search_summary"],
            components["file_selector"],
            components["document_content"],
            components["pagination_row"],
            components["page_info"]
        ]
    )

    # Pagination navigation
    components["prev_btn"].click(
        fn=lambda: handlers.navigate_search_results(-1),
        outputs=[
            components["search_summary"],
            components["file_selector"],
            components["document_content"],
            components["pagination_row"],
            components["page_info"]
        ]
    )

    components["next_btn"].click(
        fn=lambda: handlers.navigate_search_results(1),
        outputs=[
            components["search_summary"],
            components["file_selector"],
            components["document_content"],
            components["pagination_row"],
            components["page_info"]
        ]
    )

    # File selection
    components["file_selector"].change(
        fn=handlers.handle_file_selection,
        inputs=[components["file_selector"]],
        outputs=[components["document_content"]]
    )

    # Library group selection
    components["group_selector"].change(
        fn=handlers.load_document_group,
        inputs=[components["group_selector"], components["doc_groups_state"]],
        outputs=[components["library_file_selector"]]
    )

    # Library file selection
    components["library_file_selector"].change(
        fn=handlers.handle_library_file_selection,
        inputs=[components["library_file_selector"], components["group_selector"], components["doc_groups_state"]],
        outputs=[components["document_content"]]
    )

    return components