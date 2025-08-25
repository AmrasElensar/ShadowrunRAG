"""
RAG Query UI Components for Shadowrun RAG System
Clean extraction of query interface with character integration.
"""


import gradio as gr
import logging
from typing import Dict
from pathlib import Path
from frontend.api_clients.character_client import CharacterAPIClient
from frontend.ui_helpers.ui_helpers import format_dice_result, create_info_message, UIErrorHandler

logger = logging.getLogger(__name__)


class RAGQueryUI:
    """RAG query interface with character integration."""

    def __init__(self, rag_client, char_api: CharacterAPIClient):
        self.rag_client = rag_client
        self.char_api = char_api

    def build_query_tab(self):
        """Build the complete query tab with character integration."""
        with gr.Tab("💬 Query"):
            return self._build_query_interface()

    def _build_query_interface(self):
        """Build the main query interface components."""
        components = {}

        with gr.Row():
            with gr.Column(scale=3):
                # Query input area
                components.update(self._build_query_input_section())

                # Answer and thinking area
                components.update(self._build_output_section())

            with gr.Column(scale=1):
                # Configuration and character integration
                components.update(self._build_configuration_section())

        return components

    def _build_query_input_section(self):
        """Build query input section."""
        question_input = gr.Textbox(
            label="Your Question",
            placeholder="e.g., 'How do recoil penalties work in Shadowrun 5e?'",
            lines=3
        )

        with gr.Row():
            submit_btn = gr.Button("🔍 Search", variant="primary")
            clear_btn = gr.ClearButton(value="Clear")

        clear_btn.add(question_input)

        return {
            "question_input": question_input,
            "submit_btn": submit_btn,
            "clear_btn": clear_btn
        }

    def _build_output_section(self):
        """Build answer and thinking output section."""
        answer_output = gr.Markdown(label="Answer")

        # Enhanced thinking accordion
        thinking_accordion = gr.Accordion("🤔 Model Thinking Process", open=False, visible=False)
        with thinking_accordion:
            thinking_output = gr.Markdown(
                label="AI Reasoning",
                value="*The model's reasoning process will appear here...*"
            )

        sources_output = gr.Markdown(label="Sources & Filters")

        chunks_accordion = gr.Accordion("📊 Retrieved Chunks", open=False)
        with chunks_accordion:
            chunks_output = gr.Dataframe(
                headers=["Relevance", "Content"],
                label="Context Chunks"
            )

        return {
            "answer_output": answer_output,
            "thinking_accordion": thinking_accordion,
            "thinking_output": thinking_output,
            "sources_output": sources_output,
            "chunks_accordion": chunks_accordion,
            "chunks_output": chunks_output
        }

    def _build_configuration_section(self):
        """Build configuration section with character integration."""
        with gr.Column():
            gr.Markdown("### ⚙️ Enhanced Configuration")

            # Model selection
            model_select = gr.Dropdown(
                choices=self.rag_client.get_models() or ["llama3:8b-instruct-q4_K_M"],
                value="llama3:8b-instruct-q4_K_M",
                label="LLM Model",
                allow_custom_value=True
            )

            refresh_models_btn = gr.Button("🔄 Refresh models", size="sm")

            # Results configuration
            n_results_slider = gr.Slider(
                minimum=1, maximum=10, value=5, step=1,
                label="Number of Sources"
            )

            query_type_select = gr.Dropdown(
                choices=["General", "Rules", "Session"],
                value="General",
                label="Query Type"
            )

            # Character context integration
            character_components = self._build_character_context_section()

            # Enhanced filters
            filter_components = self._build_filter_section()

            return {
                "model_select": model_select,
                "refresh_models_btn": refresh_models_btn,
                "n_results_slider": n_results_slider,
                "query_type_select": query_type_select,
                **character_components,
                **filter_components
            }

    def _build_character_context_section(self):
        """Build character context section for queries."""
        with gr.Accordion("👤 Character Context", open=True):
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
                choices=["SR5", "SR6", "SR4", "SR3", "None"],
                value="SR5",
                label="Preferred Edition"
            )

            # Active character selector
            character_query_selector = gr.Dropdown(
                label="Active Character",
                choices=["None"],
                value="None",
                info="Select character for context-aware queries"
            )

            character_context_display = gr.Textbox(
                label="Character Status",
                value="No character selected",
                interactive=False,
                lines=2
            )

            refresh_char_btn = gr.Button("🔄 Refresh Characters", size="sm")

        return {
            "character_role_select": character_role_select,
            "character_stats_input": character_stats_input,
            "edition_select": edition_select,
            "character_query_selector": character_query_selector,
            "character_context_display": character_context_display,
            "refresh_char_btn": refresh_char_btn
        }

    def _build_filter_section(self):
        """Build enhanced filter section."""
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

        return {
            "section_filter": section_filter,
            "subsection_filter": subsection_filter,
            "document_type_filter": document_type_filter,
            "edition_filter": edition_filter
        }


class RAGQueryHandlers:
    """Event handlers for RAG query operations."""

    def __init__(self, rag_client, char_api: CharacterAPIClient):
        self.rag_client = rag_client
        self.char_api = char_api

    def submit_query(self, question: str, model: str, n_results: int, query_type: str,
                     character_role: str, character_stats: str, edition: str,
                     filter_section: str, filter_subsection: str, filter_document_type: str,
                     filter_edition: str, character_selector: str = "None"):
        """Enhanced query submission with character context integration."""
        if not question:
            yield "Please enter a question", "", "", [], gr.update(visible=False)
            return

        # Check for active character and dice pool queries
        try:
            active_char = self.char_api.get_active_character()

            # Check if this is a dice pool question first
            if active_char and any(keyword in question.lower() for keyword in
                                   ['dice', 'roll', 'pool', 'test', 'check']):
                # Try to resolve as dice pool query
                dice_result = self.char_api.get_dice_pool(active_char['id'], question)

                if not dice_result.get('error') and dice_result.get('dice_pool', 0) > 0:
                    # This was successfully resolved as a dice pool query
                    explanation = dice_result.get('explanation', '')
                    dice_pool = dice_result.get('dice_pool', 0)

                    answer = format_dice_result(dice_pool, explanation, active_char['name'])
                    yield answer, "", f"**Character:** {active_char['name']}", [], gr.update(visible=False)
                    return

        except Exception as e:
            logger.warning(f"Character context check failed: {e}")
            # Continue with normal query if character check fails

        # Prepare enhanced parameters
        params = {
            "n_results": n_results,
            "query_type": query_type.lower(),
            "model": model,
            "edition": edition if edition != "None" else "SR5"
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

        # Add character context if available
        try:
            if active_char:
                # Get character context for the query
                context_response = self.char_api.get_character_query_context(active_char['id'])
                if not context_response.get('error'):
                    character_context = context_response.get('context', '')
                    if character_context:
                        params["character_context"] = character_context
                        logger.info(f"Added character context: {character_context}")
        except Exception as e:
            logger.warning(f"Failed to get character context: {e}")
            # Continue without character context

        # Stream response with thinking support - FIXED STREAMING LOGIC
        try:
            # Initialize streaming state
            current_answer = ""
            current_thinking = ""
            thinking_visible = False

            for response, thinking, metadata, status in self.rag_client.query_stream(question, **params):
                if status == "error":
                    yield response, "", "", [], gr.update(visible=False)
                    return

                # Update current content
                current_answer = response
                current_thinking = thinking

                # Check if we have thinking content
                has_thinking = bool(thinking and thinking.strip())
                if has_thinking and not thinking_visible:
                    thinking_visible = True

                if status == "complete" and metadata:
                    # Format sources
                    sources_text = ""
                    if metadata.get("sources"):
                        sources_list = [Path(s).name for s in metadata["sources"]]
                        sources_text = "**Sources:**\n" + "\n".join([f"📄 {s}" for s in sources_list])

                    # Add character info to sources if used
                    try:
                        if active_char and params.get("character_context"):
                            sources_text += f"\n\n**Character:** {active_char['name']} ({active_char['metatype']})"
                    except:
                        pass

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

                    # Final yield with complete data
                    yield current_answer, current_thinking, sources_text, chunks_data, gr.update(
                        visible=thinking_visible)
                else:
                    # Intermediate yield for smooth streaming
                    # Show cursor while generating
                    cursor = "▌" if status == "generating" else "🤔" if status == "thinking" else ""
                    display_answer = current_answer + cursor

                    yield display_answer, current_thinking, "", [], gr.update(visible=thinking_visible)

        except Exception as e:
            error_msg = UIErrorHandler.handle_exception(e, "query submission")
            yield error_msg, "", "", [], gr.update(visible=False)

    def get_character_selector_choices(self):
        """Get character choices for query tab dropdown."""
        try:
            characters = self.char_api.list_characters()
            if not characters:
                return gr.update(choices=["None"], value="None")

            choices = ["None"] + [f"{char['name']} ({char['metatype']})" for char in characters]

            # Set active character as default
            active_char = self.char_api.get_active_character()
            default_value = "None"
            if active_char:
                default_value = f"{active_char['name']} ({active_char['metatype']})"

            return gr.update(choices=choices, value=default_value)

        except Exception as e:
            logger.error(f"Failed to get character choices: {e}")
            return gr.update(choices=["None"], value="None")

    def handle_character_selection_for_query(self, character_choice: str):
        """Handle character selection in query tab."""
        if character_choice == "None":
            return create_info_message("No character selected for queries")

        # Find character by name
        characters = self.char_api.list_characters()
        selected_char = None
        for char in characters:
            if f"{char['name']} ({char['metatype']})" == character_choice:
                selected_char = char
                break

        if not selected_char:
            return UIErrorHandler.handle_api_error({"error": "Character not found"}, "character selection")

        # Set as active
        result = self.char_api.set_active_character(selected_char['id'])
        if "error" in result:
            return UIErrorHandler.handle_api_error(result, "set active character")

        return create_info_message(
            f"Active character: {selected_char['name']} - queries will include character context")

    def refresh_models(self):
        """Refresh available models from the backend."""
        try:
            models = self.rag_client.get_models()
            return gr.update(choices=models, value=models[0] if models else "llama3:8b-instruct-q4_K_M")
        except Exception as e:
            logger.error(f"Failed to refresh models: {e}")
            return gr.update(choices=["llama3:8b-instruct-q4_K_M"], value="llama3:8b-instruct-q4_K_M")


def wire_query_events(components: Dict, handlers: RAGQueryHandlers):
    """Wire up RAG query event handlers."""

    # Query submission
    components["submit_btn"].click(
        fn=handlers.submit_query,
        inputs=[
            components["question_input"],
            components["model_select"],
            components["n_results_slider"],
            components["query_type_select"],
            components["character_role_select"],
            components["character_stats_input"],
            components["edition_select"],
            components["section_filter"],
            components["subsection_filter"],
            components["document_type_filter"],
            components["edition_filter"],
            components["character_query_selector"]
        ],
        outputs=[
            components["answer_output"],
            components["thinking_output"],
            components["sources_output"],
            components["chunks_output"],
            components["thinking_accordion"]
        ]
    )

    # Character selection refresh
    components["refresh_char_btn"].click(
        fn=handlers.get_character_selector_choices,
        outputs=[components["character_query_selector"]]
    )

    # Character selection change
    components["character_query_selector"].change(
        fn=handlers.handle_character_selection_for_query,
        inputs=[components["character_query_selector"]],
        outputs=[components["character_context_display"]]
    )

    components["refresh_models_btn"].click(
        fn=handlers.refresh_models,
        outputs=[components["model_select"]]
    )

    return components


def get_initial_character_choices(char_api: CharacterAPIClient):
    """Get initial character choices for app load."""
    try:
        return RAGQueryHandlers(None, char_api).get_character_selector_choices()
    except Exception as e:
        logger.error(f"Failed to get initial character choices: {e}")
        return gr.update(choices=["None"], value="None")