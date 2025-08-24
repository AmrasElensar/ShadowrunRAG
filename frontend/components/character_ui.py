"""
Character Management UI Components for Shadowrun RAG System
Clean, organized UI builders with properly wired dropdowns and event handlers.
"""

import gradio as gr
import logging
from typing import Dict
from frontend.api_clients.character_client import CharacterAPIClient
from frontend.components.character_equipment_ui import CharacterEquipmentUI, CharacterEquipmentHandlers, wire_equipment_events

logger = logging.getLogger(__name__)


class CharacterUI:
    """Organized character management UI with proper event wiring."""

    def __init__(self, char_api: CharacterAPIClient):
        self.char_api = char_api
        self.selected_character_id = None
        self.character_data_cache = {}

    # ===== CHARACTER SELECTION & MANAGEMENT =====

    def build_character_selector_section(self):
        """Build the character selection and management section."""
        with gr.Column():
            gr.Markdown("### 🎭 Character Management")

            # Character selection dropdown
            character_selector = gr.Dropdown(
                label="Select Character",
                choices=[("No characters", None)],
                value=None,
                interactive=True
            )

            with gr.Row():
                refresh_btn = gr.Button("🔄 Refresh", size="sm")
                set_active_btn = gr.Button("⭐ Set Active", variant="primary", size="sm")
                delete_btn = gr.Button("🗑️ Delete", variant="stop", size="sm")

            character_status = gr.Markdown("Ready to manage characters")

            # Character creation section
            creation_components = self._build_character_creation_section()

            return {
                "character_selector": character_selector,
                "refresh_btn": refresh_btn,
                "set_active_btn": set_active_btn,
                "delete_btn": delete_btn,
                "character_status": character_status,
                **creation_components
            }

    def _build_character_creation_section(self):
        """Build character creation form."""
        with gr.Accordion("➕ Create New Character", open=False):
            new_char_name = gr.Textbox(label="Character Name", placeholder="Enter character name")

            with gr.Row():
                new_char_metatype = gr.Dropdown(
                    label="Metatype",
                    choices=["Human", "Elf", "Dwarf", "Ork", "Troll"],
                    value="Human"
                )
                new_char_archetype = gr.Textbox(
                    label="Archetype",
                    placeholder="e.g., Street Samurai, Decker"
                )

            create_btn = gr.Button("➕ Create Character", variant="primary")
            create_status = gr.Textbox(label="Creation Status", interactive=False, lines=2)

        return {
            "new_char_name": new_char_name,
            "new_char_metatype": new_char_metatype,
            "new_char_archetype": new_char_archetype,
            "create_btn": create_btn,
            "create_status": create_status
        }

    # ===== STATS & RESOURCES TAB =====

    def build_stats_resources_tab(self):
        """Build the stats and resources management tab."""
        with gr.Tab("📊 Stats & Resources"):
            with gr.Row():
                # Attributes section
                attributes_components = self._build_attributes_section()

                # Limits and initiative section
                limits_components = self._build_limits_section()

            # Resources section
            resources_components = self._build_resources_section()

            # Update buttons
            with gr.Row():
                update_stats_btn = gr.Button("💾 Save Stats", variant="primary")
                update_resources_btn = gr.Button("💾 Save Resources", variant="primary")

            stats_update_status = gr.Textbox(label="Update Status", interactive=False)

            return {
                **attributes_components,
                **limits_components,
                **resources_components,
                "update_stats_btn": update_stats_btn,
                "update_resources_btn": update_resources_btn,
                "stats_update_status": stats_update_status
            }

    def _build_attributes_section(self):
        """Build attributes input section."""
        with gr.Column():
            gr.Markdown("#### 🏋️ Attributes")

            with gr.Row():
                body_input = gr.Number(label="Body", value=1, minimum=1, maximum=12)
                agility_input = gr.Number(label="Agility", value=1, minimum=1, maximum=12)
                reaction_input = gr.Number(label="Reaction", value=1, minimum=1, maximum=12)

            with gr.Row():
                strength_input = gr.Number(label="Strength", value=1, minimum=1, maximum=12)
                charisma_input = gr.Number(label="Charisma", value=1, minimum=1, maximum=12)
                logic_input = gr.Number(label="Logic", value=1, minimum=1, maximum=12)

            with gr.Row():
                intuition_input = gr.Number(label="Intuition", value=1, minimum=1, maximum=12)
                willpower_input = gr.Number(label="Willpower", value=1, minimum=1, maximum=12)
                edge_input = gr.Number(label="Edge", value=1, minimum=1, maximum=7)

            essence_input = gr.Number(label="Essence", value=6.0, minimum=0, maximum=6, step=0.01)

        return {
            "body_input": body_input,
            "agility_input": agility_input,
            "reaction_input": reaction_input,
            "strength_input": strength_input,
            "charisma_input": charisma_input,
            "logic_input": logic_input,
            "intuition_input": intuition_input,
            "willpower_input": willpower_input,
            "edge_input": edge_input,
            "essence_input": essence_input
        }

    def _build_limits_section(self):
        """Build limits and initiative section."""
        with gr.Column():
            gr.Markdown("#### 🎯 Limits & Initiative")

            physical_limit_input = gr.Number(label="Physical Limit", value=1, minimum=1)
            mental_limit_input = gr.Number(label="Mental Limit", value=1, minimum=1)
            social_limit_input = gr.Number(label="Social Limit", value=1, minimum=1)
            initiative_input = gr.Number(label="Initiative", value=1, minimum=1)
            hot_sim_vr_input = gr.Number(label="Hot Sim VR", value=0, minimum=0)

        return {
            "physical_limit_input": physical_limit_input,
            "mental_limit_input": mental_limit_input,
            "social_limit_input": social_limit_input,
            "initiative_input": initiative_input,
            "hot_sim_vr_input": hot_sim_vr_input
        }

    def _build_resources_section(self):
        """Build resources section."""
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### 💰 Resources")

                with gr.Row():
                    nuyen_input = gr.Number(label="Nuyen", value=0, minimum=0)
                    street_cred_input = gr.Number(label="Street Cred", value=0, minimum=0)

                with gr.Row():
                    notoriety_input = gr.Number(label="Notoriety", value=0, minimum=0)
                    public_aware_input = gr.Number(label="Public Aware", value=0, minimum=0)

                with gr.Row():
                    total_karma_input = gr.Number(label="Total Karma", value=0, minimum=0)
                    available_karma_input = gr.Number(label="Available Karma", value=0, minimum=0)
                    edge_pool_input = gr.Number(label="Edge Pool", value=1, minimum=1)

        return {
            "nuyen_input": nuyen_input,
            "street_cred_input": street_cred_input,
            "notoriety_input": notoriety_input,
            "public_aware_input": public_aware_input,
            "total_karma_input": total_karma_input,
            "available_karma_input": available_karma_input,
            "edge_pool_input": edge_pool_input
        }

    # ===== SKILLS TAB (WITH FIXED DROPDOWNS) =====

    def build_skills_tab(self):
        """Build the skills management tab with properly wired dropdowns."""
        with gr.Tab("🎯 Skills"):
            with gr.Row():
                # Current skills display
                with gr.Column(scale=2):
                    gr.Markdown("#### 📋 Current Skills")

                    skills_table = gr.Dataframe(
                        headers=["Skill", "Rating", "Specialization", "Type", "Dice Pool"],
                        datatype=["str", "number", "str", "str", "number"],
                        value=[],
                        interactive=False,
                        label="Character Skills"
                    )

                    with gr.Row():
                        remove_skill_btn = gr.Button("🗑️ Remove Selected", variant="secondary")

                # Add skills interface (FIXED DROPDOWNS)
                skills_add_components = self._build_skills_add_section()

            skills_status = gr.Textbox(label="Skills Status", interactive=False, lines=3)

            return {
                "skills_table": skills_table,
                "remove_skill_btn": remove_skill_btn,
                "skills_status": skills_status,
                **skills_add_components
            }

    def _build_skills_add_section(self):
        """Build skills addition section with PROPERLY WIRED dropdowns."""
        with gr.Column(scale=1):
            gr.Markdown("#### ➕ Add Skills")

            skill_type_selector = gr.Radio(
                label="Skill Type",
                choices=["active", "knowledge", "language"],
                value="active"
            )

            # THIS IS THE KEY FIX - Dropdown starts empty and gets populated
            skill_dropdown = gr.Dropdown(
                label="Select Skill",
                choices=[("Loading skills...", None)],
                interactive=True,
                allow_custom_value=False
            )

            skill_rating_input = gr.Number(
                label="Rating",
                value=1,
                minimum=1,
                maximum=12
            )

            skill_specialization_input = gr.Textbox(
                label="Specialization",
                placeholder="Optional specialization"
            )

            add_skill_btn = gr.Button("➕ Add Skill", variant="primary")

        return {
            "skill_type_selector": skill_type_selector,
            "skill_dropdown": skill_dropdown,
            "skill_rating_input": skill_rating_input,
            "skill_specialization_input": skill_specialization_input,
            "add_skill_btn": add_skill_btn
        }

    # ===== QUALITIES TAB (WITH FIXED DROPDOWNS) =====

    def build_qualities_tab(self):
        """Build the qualities management tab with properly wired dropdowns."""
        with gr.Tab("⭐ Qualities"):
            with gr.Row():
                # Current qualities display
                with gr.Column(scale=2):
                    gr.Markdown("#### 📋 Current Qualities")

                    qualities_table = gr.Dataframe(
                        headers=["Quality", "Rating", "Type"],
                        datatype=["str", "str", "str"],
                        value=[],
                        interactive=False,
                        label="Character Qualities"
                    )

                    with gr.Row():
                        remove_quality_btn = gr.Button("🗑️ Remove Selected", variant="secondary")

                # Add qualities interface (FIXED DROPDOWNS)
                qualities_add_components = self._build_qualities_add_section()

            qualities_status = gr.Textbox(label="Qualities Status", interactive=False, lines=3)

            return {
                "qualities_table": qualities_table,
                "remove_quality_btn": remove_quality_btn,
                "qualities_status": qualities_status,
                **qualities_add_components
            }

    def _build_qualities_add_section(self):
        """Build qualities addition section with PROPERLY WIRED dropdowns."""
        with gr.Column(scale=1):
            gr.Markdown("#### ➕ Add Qualities")

            quality_type_selector = gr.Radio(
                label="Quality Type",
                choices=["positive", "negative"],
                value="positive"
            )

            # THIS IS THE KEY FIX - Dropdown starts empty and gets populated
            quality_dropdown = gr.Dropdown(
                label="Select Quality",
                choices=[("Loading qualities...", None)],
                interactive=True,
                allow_custom_value=False
            )

            quality_rating_input = gr.Number(
                label="Rating",
                value=0,
                minimum=0,
                maximum=6
            )

            add_quality_btn = gr.Button("➕ Add Quality", variant="primary")

        return {
            "quality_type_selector": quality_type_selector,
            "quality_dropdown": quality_dropdown,
            "quality_rating_input": quality_rating_input,
            "add_quality_btn": add_quality_btn
        }

    # ===== GEAR TAB (WITH FIXED DROPDOWNS) =====

    def build_gear_tab(self):
        """Build the gear management tab with properly wired dropdowns."""
        with gr.Tab("🎒 Gear"):
            with gr.Row():
                # Current gear display
                with gr.Column(scale=2):
                    gr.Markdown("#### 📋 Current Gear")

                    gear_table = gr.Dataframe(
                        headers=["Item", "Quantity", "Category", "Armor"],
                        datatype=["str", "number", "str", "number"],
                        value=[],
                        interactive=False,
                        label="Character Gear"
                    )

                    with gr.Row():
                        remove_gear_btn = gr.Button("🗑️ Remove Selected", variant="secondary")
                        total_armor_display = gr.Number(
                            label="Total Armor",
                            value=0,
                            interactive=False
                        )

                # Add gear interface (FIXED DROPDOWNS)
                gear_add_components = self._build_gear_add_section()

            gear_status = gr.Textbox(label="Gear Status", interactive=False, lines=3)

            return {
                "gear_table": gear_table,
                "remove_gear_btn": remove_gear_btn,
                "total_armor_display": total_armor_display,
                "gear_status": gear_status,
                **gear_add_components
            }

    def _build_gear_add_section(self):
        """Build gear addition section with PROPERLY WIRED dropdowns."""
        with gr.Column(scale=1):
            gr.Markdown("#### ➕ Add Gear")

            # Category dropdown gets populated on startup
            gear_category_selector = gr.Dropdown(
                label="Category",
                choices=[("Loading categories...", None)],
                interactive=True
            )

            # Gear dropdown starts empty, gets populated when category changes
            gear_dropdown = gr.Dropdown(
                label="Select Gear",
                choices=[],
                interactive=True,
                allow_custom_value=False
            )

            gear_quantity_input = gr.Number(
                label="Quantity",
                value=1,
                minimum=1
            )

            gear_rating_input = gr.Number(
                label="Rating",
                value=0,
                minimum=0,
                maximum=6
            )

            add_gear_btn = gr.Button("➕ Add Gear", variant="primary")

        return {
            "gear_category_selector": gear_category_selector,
            "gear_dropdown": gear_dropdown,
            "gear_quantity_input": gear_quantity_input,
            "gear_rating_input": gear_rating_input,
            "add_gear_btn": add_gear_btn
        }

    # ===== UTILITY TAB =====

    def build_utility_section(self):
        """Build utilities section."""
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🔧 Utilities")

                with gr.Row():
                    populate_ref_btn = gr.Button("📚 Populate Reference Data", variant="secondary")
                    export_btn = gr.Button("📤 Export for GM", variant="primary")
                    dice_pool_btn = gr.Button("🎲 Quick Dice Pool", variant="secondary")

                utility_status = gr.Textbox(label="Utility Status", interactive=False, lines=2)

        return {
            "populate_ref_btn": populate_ref_btn,
            "export_btn": export_btn,
            "dice_pool_btn": dice_pool_btn,
            "utility_status": utility_status
        }

    # ===== COMPLETE CHARACTER MANAGEMENT TAB BUILDER =====

    def build_complete_character_tab(self):
        """Build the complete character management tab with all sub-sections."""
        with gr.Tab("👤 Characters"):
            # Character selection and management header
            with gr.Row():
                # Character selector section (left side)
                with gr.Column(scale=1):
                    selector_components = self.build_character_selector_section()

            # Character details in tabbed interface (full width)
            with gr.Row():
                with gr.Column():
                    with gr.Tabs():
                        # Stats & Resources tab
                        stats_components = self.build_stats_resources_tab()

                        # Skills tab
                        skills_components = self.build_skills_tab()

                        # Qualities tab
                        qualities_components = self.build_qualities_tab()

                        with gr.Tab("🎒 Equipment"):
                            equipment_ui = CharacterEquipmentUI(self.char_api)
                            equipment_components = equipment_ui.build_equipment_tabs_section()

            # Bottom utilities section
            utility_components = self.build_utility_section()

            # Return all components for event wiring
            return {
                **selector_components,
                **stats_components,
                **skills_components,
                **qualities_components,
                **equipment_components,
                **utility_components
            }


# ===== EVENT HANDLERS (CLEAN SEPARATION) =====

class CharacterEventHandlers:
    """Clean event handlers for character management operations."""

    def __init__(self, char_api: CharacterAPIClient):
        self.char_api = char_api

    # ===== CHARACTER MANAGEMENT HANDLERS =====

    def refresh_character_list(self):
        """Refresh character dropdown choices."""
        try:
            choices = self.char_api.get_character_dropdown_choices()
            return gr.update(choices=choices), f"Found {len(choices)} characters"
        except Exception as e:
            logger.error(f"Failed to refresh character list: {e}")
            return gr.update(choices=[("Error loading", None)]), f"Error: {str(e)}"

    def create_character(self, name: str, metatype: str, archetype: str):
        """Create a new character."""
        if not name.strip():
            return "❌ Character name is required", gr.update(), ""

        result = self.char_api.create_character(name.strip(), metatype, archetype.strip())

        if "error" in result:
            return f"❌ Failed to create character: {result['error']}", gr.update(), ""

        # Refresh character list
        choices = self.char_api.get_character_dropdown_choices()
        char_dropdown = gr.update(choices=choices)

        return f"✅ Character '{name}' created successfully!", char_dropdown, ""

    def set_active_character(self, character_id):
        """Set the active character for queries."""
        if not character_id:
            return "No character selected"

        result = self.char_api.set_active_character(character_id)
        if "error" in result:
            return f"❌ Failed to set active character: {result['error']}"

        return f"✅ {result['message']}"

    def delete_character(self, character_id):
        """Delete the selected character."""
        if not character_id:
            return "No character selected", gr.update()

        result = self.char_api.delete_character(character_id)
        if "error" in result:
            return f"❌ Failed to delete character: {result['error']}", gr.update()

        # Refresh character list
        choices = self.char_api.get_character_dropdown_choices()
        char_dropdown = gr.update(choices=choices)

        return "✅ Character deleted successfully!", char_dropdown

    # ===== DROPDOWN POPULATION HANDLERS (THE KEY FIXES) =====

    def populate_skills_dropdown(self, skill_type: str):
        """Populate skills dropdown - THIS FIXES THE MAIN ISSUE."""
        try:
            choices = self.char_api.get_skills_dropdown_choices(skill_type)
            logger.info(f"Populated skills dropdown with {len(choices)} choices for {skill_type}")
            return gr.update(choices=choices, value=None)
        except Exception as e:
            logger.error(f"Failed to populate skills dropdown: {e}")
            return gr.update(choices=[("Error loading skills", None)], value=None)

    def populate_qualities_dropdown(self, quality_type: str):
        """Populate qualities dropdown - THIS FIXES THE MAIN ISSUE."""
        try:
            choices = self.char_api.get_qualities_dropdown_choices(quality_type)
            logger.info(f"Populated qualities dropdown with {len(choices)} choices for {quality_type}")
            return gr.update(choices=choices, value=None)
        except Exception as e:
            logger.error(f"Failed to populate qualities dropdown: {e}")
            return gr.update(choices=[("Error loading qualities", None)], value=None)

    def populate_gear_dropdown(self, category: str):
        """Populate gear dropdown - THIS FIXES THE MAIN ISSUE."""
        try:
            if not category:
                return gr.update(choices=[], value=None)

            choices = self.char_api.get_gear_dropdown_choices(category)
            logger.info(f"Populated gear dropdown with {len(choices)} choices for {category}")
            return gr.update(choices=choices, value=None)
        except Exception as e:
            logger.error(f"Failed to populate gear dropdown: {e}")
            return gr.update(choices=[("Error loading gear", None)], value=None)

    def populate_gear_categories(self):
        """Populate gear categories dropdown."""
        try:
            categories = self.char_api.get_gear_categories()
            choices = [("Select category...", None)] + [(cat, cat) for cat in categories]
            logger.info(f"Populated gear categories with {len(categories)} categories")
            return gr.update(choices=choices, value=None)
        except Exception as e:
            logger.error(f"Failed to populate gear categories: {e}")
            return gr.update(choices=[("Error loading categories", None)], value=None)

    # ===== CHARACTER DATA HANDLERS =====

    def update_stats(self, character_id, body, agility, reaction, strength, charisma,
                     logic, intuition, willpower, edge, essence, physical_limit,
                     mental_limit, social_limit, initiative, hot_sim_vr):
        """Update character stats."""
        if not character_id:
            return "No character selected"

        stats_data = {
            "body": body, "agility": agility, "reaction": reaction, "strength": strength,
            "charisma": charisma, "logic": logic, "intuition": intuition, "willpower": willpower,
            "edge": edge, "essence": essence, "physical_limit": physical_limit,
            "mental_limit": mental_limit, "social_limit": social_limit,
            "initiative": initiative, "hot_sim_vr": hot_sim_vr
        }

        result = self.char_api.update_character_stats(character_id, stats_data)
        if "error" in result:
            return f"❌ Failed to update stats: {result['error']}"

        return "✅ Character stats updated successfully!"

    def update_resources(self, character_id, nuyen, street_cred, notoriety, public_aware,
                         total_karma, available_karma, edge_pool):
        """Update character resources."""
        if not character_id:
            return "No character selected"

        resources_data = {
            "nuyen": nuyen, "street_cred": street_cred, "notoriety": notoriety,
            "public_aware": public_aware, "total_karma": total_karma,
            "available_karma": available_karma, "edge_pool": edge_pool
        }

        result = self.char_api.update_character_resources(character_id, resources_data)
        if "error" in result:
            return f"❌ Failed to update resources: {result['error']}"

        return "✅ Character resources updated successfully!"

    # ===== UTILITY HANDLERS =====

    def populate_reference_data(self):
        """Populate reference tables from rulebooks."""
        try:
            result = self.char_api.populate_reference_data()
            if "error" in result:
                return f"❌ Failed to populate reference data: {result['error']}"

            return "✅ Reference tables populated from rulebooks successfully!"
        except Exception as e:
            logger.error(f"Failed to populate reference data: {e}")
            return f"❌ Error: {str(e)}"

    def get_equipment_handlers(self):
        """Get equipment event handlers instance."""
        return CharacterEquipmentHandlers(self.char_api)


def wire_character_events(components: Dict, handlers: CharacterEventHandlers):
    """Wire up all character management events - ORGANIZED AND CLEAN."""

    # Character management events
    components["refresh_btn"].click(
        fn=handlers.refresh_character_list,
        outputs=[components["character_selector"], components["character_status"]]
    )

    components["create_btn"].click(
        fn=handlers.create_character,
        inputs=[
            components["new_char_name"],
            components["new_char_metatype"],
            components["new_char_archetype"]
        ],
        outputs=[
            components["create_status"],
            components["character_selector"],
            components["new_char_name"]
        ]
    )

    components["set_active_btn"].click(
        fn=handlers.set_active_character,
        inputs=[components["character_selector"]],
        outputs=[components["character_status"]]
    )

    components["delete_btn"].click(
        fn=handlers.delete_character,
        inputs=[components["character_selector"]],
        outputs=[components["character_status"], components["character_selector"]]
    )

    # Stats and resources updates
    components["update_stats_btn"].click(
        fn=handlers.update_stats,
        inputs=[
            components["character_selector"],
            components["body_input"], components["agility_input"], components["reaction_input"],
            components["strength_input"], components["charisma_input"], components["logic_input"],
            components["intuition_input"], components["willpower_input"], components["edge_input"],
            components["essence_input"], components["physical_limit_input"],
            components["mental_limit_input"], components["social_limit_input"],
            components["initiative_input"], components["hot_sim_vr_input"]
        ],
        outputs=[components["stats_update_status"]]
    )

    components["update_resources_btn"].click(
        fn=handlers.update_resources,
        inputs=[
            components["character_selector"],
            components["nuyen_input"], components["street_cred_input"],
            components["notoriety_input"], components["public_aware_input"],
            components["total_karma_input"], components["available_karma_input"],
            components["edge_pool_input"]
        ],
        outputs=[components["stats_update_status"]]
    )

    # THE KEY FIXES - Dropdown population events
    components["skill_type_selector"].change(
        fn=handlers.populate_skills_dropdown,
        inputs=[components["skill_type_selector"]],
        outputs=[components["skill_dropdown"]]
    )

    components["quality_type_selector"].change(
        fn=handlers.populate_qualities_dropdown,
        inputs=[components["quality_type_selector"]],
        outputs=[components["quality_dropdown"]]
    )

    components["gear_category_selector"].change(
        fn=handlers.populate_gear_dropdown,
        inputs=[components["gear_category_selector"]],
        outputs=[components["gear_dropdown"]]
    )

    # Utilities
    components["populate_ref_btn"].click(
        fn=handlers.populate_reference_data,
        outputs=[components["utility_status"]]
    )

    equipment_components = {k: v for k, v in components.items()
                            if any(prefix in k for prefix in ['gear_', 'weapon_', 'vehicle_',
                                                              'armor_', 'accessory_', 'program_',
                                                              'cyberdeck_', 'deck_'])}

    if equipment_components:
        equipment_handlers = handlers.get_equipment_handlers()
        wire_equipment_events(equipment_components, equipment_handlers)

    # Auto-populate dropdowns on app load
    def auto_populate_initial_data():
        """Auto-populate initial dropdown data."""
        try:
            # Populate gear categories on startup
            categories = handlers.char_api.get_gear_categories()
            gear_choices = [("Select category...", None)] + [(cat, cat) for cat in categories]

            # Populate default skill dropdown (active skills)
            skill_choices = handlers.char_api.get_skills_dropdown_choices("active")

            # Populate default quality dropdown (positive qualities)
            quality_choices = handlers.char_api.get_qualities_dropdown_choices("positive")

            logger.info(
                f"Auto-populated: {len(categories)} gear categories, {len(skill_choices)} skills, {len(quality_choices)} qualities")

            return (
                gr.update(choices=gear_choices),  # gear_category_selector
                gr.update(choices=skill_choices),  # skill_dropdown
                gr.update(choices=quality_choices)  # quality_dropdown
            )
        except Exception as e:
            logger.error(f"Failed to auto-populate dropdowns: {e}")
            return (
                gr.update(choices=[("Error loading", None)]),
                gr.update(choices=[("Error loading", None)]),
                gr.update(choices=[("Error loading", None)])
            )

    return auto_populate_initial_data  # Return function for app.load()