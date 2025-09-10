"""
RAG Query UI Components for Shadowrun RAG System
Fixed version with correct Gradio event handling and output formats.
"""

import gradio as gr
import logging
from typing import Dict
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

        # CRITICAL: Define conversation_history as a state component
        conversation_history = gr.State([])

        with gr.Row():
            with gr.Column(scale=3):
                # Query input area
                components.update(self._build_query_input_section())
                # Answer and thinking area
                components.update(self._build_output_section())

            with gr.Column(scale=1):
                # Configuration and character integration
                components.update(self._build_configuration_section())

        # Add conversation_history to components so event handlers can access it
        components["conversation_history"] = conversation_history

        return components

    def _build_query_input_section(self):
        """Build query input section with conversation support."""
        question_input = gr.Textbox(
            label="Your Question",
            placeholder="e.g., 'How do recoil penalties work in Shadowrun 5e?'",
            lines=3
        )

        with gr.Row():
            submit_btn = gr.Button("🔍 Ask", variant="primary")
            continue_btn = gr.Button("💬 Follow Up", variant="secondary")
            new_chat_btn = gr.Button("🆕 New Chat", variant="outline")

        return {
            "question_input": question_input,
            "submit_btn": submit_btn,
            "continue_btn": continue_btn,
            "new_chat_btn": new_chat_btn
        }

    def _build_output_section(self):
        """Build answer and conversation output section."""
        # Current streaming answer display
        current_answer = gr.Textbox(
            label="🤖 Current Answer",
            lines=8,
            show_copy_button=True,
            visible=True
        )

        # Conversation history display - CRITICAL: Must receive list of [user, bot] pairs
        conversation_display = gr.Chatbot(
            label="💬 Conversation History",
            height=400,
            show_label=True
        )

        # Current thinking process
        thinking_output = gr.Textbox(
            label="🤔 AI Thinking Process",
            lines=8,
            max_lines=15,
            show_copy_button=True,
            visible=False
        )

        # Sources and metadata
        sources_output = gr.Textbox(
            label="📚 Sources Used",
            lines=3,
            show_copy_button=True
        )

        with gr.Accordion("🔍 Retrieved Chunks", open=False):
            chunks_output = gr.Textbox(
                label="Raw Retrieved Content",
                lines=10,
                max_lines=20,
                show_copy_button=True
            )

        return {
            "current_answer": current_answer,
            "conversation_display": conversation_display,
            "thinking_output": thinking_output,
            "sources_output": sources_output,
            "chunks_output": chunks_output
        }

    def _build_configuration_section(self):
        """Build configuration section with character integration."""
        # Model selection
        model_select = gr.Dropdown(
            choices=["llama3:8b-instruct-q4_K_M"],
            value="llama3:8b-instruct-q4_K_M",
            label="🧠 Model"
        )

        refresh_models_btn = gr.Button("🔄 Refresh Models", size="sm")

        # Basic query parameters
        n_results_slider = gr.Slider(
            minimum=1, maximum=20, value=5, step=1,
            label="📄 Number of Results"
        )

        query_type_select = gr.Dropdown(
            choices=["general", "specific", "technical", "rules", "character"],
            value="general",
            label="🎯 Query Type"
        )

        edition_select = gr.Dropdown(
            choices=["SR5", "SR6", "SR4", "SR3"],
            value="SR5",
            label="📚 Edition"
        )

        # Character integration
        with gr.Accordion("🎭 Character Integration", open=True):
            character_query_selector = gr.Dropdown(
                choices=["None"],
                value="None",
                label="Active Character",
                interactive=True
            )

            refresh_char_btn = gr.Button("🔄 Refresh Characters", size="sm")

            character_context_display = gr.Textbox(
                label="Character Context",
                lines=2,
                interactive=False
            )

            character_role_select = gr.Dropdown(
                choices=["None", "Street Samurai", "Decker", "Mage", "Adept",
                        "Rigger", "Face", "Technomancer"],
                value="None",
                label="Character Role"
            )

            character_stats_input = gr.Textbox(
                label="Character Stats",
                placeholder="e.g., Body 5, Agility 6, Firearms 8"
            )

        # Enhanced filters
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
            "model_select": model_select,
            "refresh_models_btn": refresh_models_btn,
            "n_results_slider": n_results_slider,
            "query_type_select": query_type_select,
            "edition_select": edition_select,
            "character_query_selector": character_query_selector,
            "refresh_char_btn": refresh_char_btn,
            "character_context_display": character_context_display,
            "character_role_select": character_role_select,
            "character_stats_input": character_stats_input,
            "section_filter": section_filter,
            "subsection_filter": subsection_filter,
            "document_type_filter": document_type_filter,
            "edition_filter": edition_filter
        }

class RAGQueryHandlers:
    """Event handlers for RAG query operations with fixed output formats."""

    def __init__(self, rag_client, char_api: CharacterAPIClient):
        self.rag_client = rag_client
        self.char_api = char_api

    def submit_query(self, question: str, conversation_history, model: str, n_results: int,
                     query_type: str, character_role: str, character_stats: str, edition: str,
                     filter_section: str, filter_subsection: str, filter_document_type: str,
                     filter_edition: str, character_selector: str = "None",
                     is_follow_up: bool = False):
        """FIXED: Submit query with correct Gradio output format."""

        if not question:
            current_conversation = conversation_history or []
            # Return 6 outputs to match event handler expectations
            yield (
                current_conversation,  # conversation_history (preserve state)
                current_conversation,  # conversation_display (preserve state)
                "",  # current_answer (empty)
                "",  # thinking_output (empty)
                "",  # sources_output (empty)
                "",  # chunks_output (empty)
                gr.update(visible=False),  # thinking_output visibility (hide)
                gr.update()  # question_input (don't clear when empty)
            )
            return

        # Build conversation context if this is a follow-up
        conversation_context = None
        if is_follow_up and conversation_history:
            raw_context = self._build_conversation_context(conversation_history)
            conversation_context = self._manage_conversation_context(raw_context, max_tokens=1500)

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

                    # Update conversation and return with correct format (6 outputs)
                    updated_conversation = conversation_history + [[question, answer]]
                    yield (
                        updated_conversation,  # conversation_history (updated)
                        updated_conversation,  # conversation_display (updated)
                        "",  # current_answer (clear after dice roll)
                        f"**Character:** {active_char['name']}",  # thinking_output (show character info)
                        "",  # sources_output (empty for dice rolls)
                        "",  # chunks_output (empty for dice rolls)
                        gr.update(visible=False),  # thinking_output visibility (hide)
                        gr.update(value="")  # question_input (clear after successful query)
                    )
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

        if conversation_context:
            params["conversation_context"] = conversation_context

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
            logger.warning(f"Character context addition failed: {e}")

        # Stream the response
        try:
            current_answer = ""
            current_thinking = ""
            thinking_visible = False

            for response, thinking_content, metadata, status in self.rag_client.query_stream(question, **params):
                current_answer = response
                current_thinking = thinking_content or ""

                # Update thinking visibility
                if thinking_content and thinking_content.strip():
                    thinking_visible = True

                # Check if this is the final response (has metadata)
                if metadata and status == "complete":
                    # Process final metadata
                    sources_text = ""
                    if metadata.get("sources"):
                        sources_text = f"**Sources:** {', '.join(metadata['sources'])}"

                    if metadata.get("applied_filters"):
                        filters_text = f"\n\n**Applied Filters:** {metadata['applied_filters']}"
                        sources_text += filters_text

                    # Create chunks data
                    chunks_text = ""
                    if metadata.get("chunks"):
                        chunk_summaries = []
                        for i, (chunk, dist) in enumerate(zip(
                                metadata.get("chunks", []),
                                metadata.get("distances", [])
                        )):
                            relevance = f"{(1 - dist):.2%}" if dist else "N/A"
                            content = chunk[:200] + "..." if len(chunk) > 200 else chunk
                            chunk_summaries.append(f"**Chunk {i+1}** (Relevance: {relevance}):\n{content}")
                        chunks_text = "\n\n".join(chunk_summaries)

                    # Final yield with complete data - FIXED: 6 outputs
                    updated_conversation = conversation_history + [[question, current_answer]]
                    yield (
                        updated_conversation,  # conversation_history
                        updated_conversation,  # conversation_display
                        "",  # current_answer (cleared)
                        current_thinking,  # thinking_output
                        sources_text,  # sources_output
                        chunks_text,  # chunks_output
                        gr.update(visible=thinking_visible),  # thinking_output visibility
                        gr.update(value="")  # question_input (cleared)
                    )

                else:
                    # Intermediate yield for smooth streaming - FIXED: 6 outputs
                    cursor = "▌" if status == "generating" else "🤔" if status == "thinking" else ""
                    display_answer = current_answer + cursor

                    yield (
                        conversation_history,  # conversation_history (unchanged during streaming)
                        conversation_history,  # conversation_display (unchanged during streaming)
                        display_answer,  # current_answer (with cursor for streaming effect)
                        current_thinking,  # thinking_output
                        "",  # sources_output (empty during streaming)
                        "",  # chunks_output (empty during streaming)
                        gr.update(visible=thinking_visible),  # thinking_output visibility
                        gr.update()  # question_input (no change during streaming)
                    )

        except Exception as e:
            error_msg = UIErrorHandler.handle_exception(e, "query submission")
            # FIXED: 6 outputs
            yield (
                conversation_history,  # conversation_history (preserve state)
                conversation_history,  # conversation_display (preserve state)
                error_msg,  # current_answer (show error)
                "",  # thinking_output (clear)
                "",  # sources_output (clear)
                "",  # chunks_output (clear)
                gr.update(visible=False),  # thinking_output visibility (hide)
                gr.update()  # question_input (don't clear on error)
            )

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

    def submit_new_query(self, question: str, conversation_history, model: str, n_results: int,
                         query_type: str, character_role: str, character_stats: str, edition: str,
                         filter_section: str, filter_subsection: str, filter_document_type: str,
                         filter_edition: str, character_selector: str = "None"):
        """Start new conversation (reset history)."""
        conversation_history = []  # Reset conversation
        for result in self.submit_query(question, conversation_history, model, n_results,
                                        query_type, character_role, character_stats, edition,
                                        filter_section, filter_subsection, filter_document_type,
                                        filter_edition, character_selector, is_follow_up=False):
            yield result

    def submit_follow_up(self, question: str, conversation_history, model: str, n_results: int,
                         query_type: str, character_role: str, character_stats: str, edition: str,
                         filter_section: str, filter_subsection: str, filter_document_type: str,
                         filter_edition: str, character_selector: str = "None"):
        """Continue existing conversation."""
        for result in self.submit_query(question, conversation_history, model, n_results,
                                        query_type, character_role, character_stats, edition,
                                        filter_section, filter_subsection, filter_document_type,
                                        filter_edition, character_selector, is_follow_up=True):
            yield result

    def new_chat(self):
        """Reset conversation - FIXED: Return 6 outputs."""
        return (
            [],  # conversation_history (reset to empty)
            [],  # conversation_display (reset to empty)
            "",  # current_answer (clear)
            "",  # thinking_output (clear)
            "",  # sources_output (clear)
            gr.update(value="")  # question_input (clear)
        )

    def _build_conversation_context(self, conversation_history):
        """Build conversation context for backend."""
        if not conversation_history:
            return ""

        context_parts = []
        for i, (user_msg, bot_msg) in enumerate(conversation_history[-3:]):  # Last 3 exchanges
            context_parts.append(f"Previous Q{i + 1}: {user_msg}")
            context_parts.append(f"Previous A{i + 1}: {bot_msg}")

        return "\n".join(context_parts)

    def _manage_conversation_context(self, conversation_context: str, max_tokens: int = 1500):
        """Smart conversation context trimming."""
        if not conversation_context:
            return ""

        # Keep only last 3 exchanges if too long
        lines = conversation_context.split('\n')
        if len(conversation_context) > max_tokens * 4:  # Rough token estimate
            # Keep only recent exchanges
            recent_lines = lines[-12:]  # Last 3 Q&A pairs (4 lines each)
            return '\n'.join(recent_lines)

        return conversation_context


def wire_query_events(components: Dict, handlers: RAGQueryHandlers):
    """FIXED: Wire up RAG query event handlers with conversation history persistence and input clearing."""

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

    # Model refresh
    components["refresh_models_btn"].click(
        fn=handlers.refresh_models,
        outputs=[components["model_select"]]
    )

    # FIXED: New query (resets conversation) - 8 outputs including conversation_history + input clearing
    components["submit_btn"].click(
        fn=handlers.submit_new_query,
        inputs=[
            components["question_input"],
            components["conversation_history"],  # CRITICAL: Include conversation state
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
            components["conversation_history"],    # CRITICAL: First output - updated conversation state
            components["conversation_display"],
            components["current_answer"],
            components["thinking_output"],
            components["sources_output"],
            components["chunks_output"],
            components["thinking_output"],         # Visibility update for thinking
            components["question_input"]           # CRITICAL: Clear input after submit
        ]
    )

    # FIXED: Follow-up query (continues conversation) - 8 outputs including conversation_history + input clearing
    components["continue_btn"].click(
        fn=handlers.submit_follow_up,
        inputs=[
            components["question_input"],
            components["conversation_history"],  # CRITICAL: Include conversation state
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
            components["conversation_history"],    # CRITICAL: First output - updated conversation state
            components["conversation_display"],
            components["current_answer"],
            components["thinking_output"],
            components["sources_output"],
            components["chunks_output"],
            components["thinking_output"],         # Visibility update for thinking
            components["question_input"]           # CRITICAL: Clear input after follow-up
        ]
    )

    # FIXED: New chat button - 6 outputs including conversation_history
    components["new_chat_btn"].click(
        fn=handlers.new_chat,
        outputs=[
            components["conversation_history"],    # CRITICAL: Reset conversation state
            components["conversation_display"],
            components["current_answer"],
            components["thinking_output"],
            components["sources_output"],
            components["question_input"]           # Clear input
        ]
    )

    return components


def get_initial_character_choices(char_api: CharacterAPIClient):
    """Get initial character choices for app load."""
    try:
        return RAGQueryHandlers(None, char_api).get_character_selector_choices()
    except Exception as e:
        logger.error(f"Failed to get initial character choices: {e}")
        return gr.update(choices=["None"], value="None")