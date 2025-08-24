"""
Updated Character Management UI with separate equipment tabs
Clean organization for gear, weapons, vehicles, and accessories
"""

import gradio as gr
from typing import Dict, List, Tuple


class CharacterEquipmentUI:
    """Organized character equipment management with separate tabs."""

    def __init__(self, char_api):
        self.char_api = char_api

    def build_equipment_tabs_section(self):
        """Build the complete equipment management with organized tabs."""

        with gr.Tabs() as equipment_tabs:
            # ===== GEAR TAB =====
            with gr.Tab("🎒 Gear"):
                gear_components = self._build_gear_tab()

            # ===== WEAPONS TAB =====
            with gr.Tab("🔫 Weapons"):
                weapons_components = self._build_weapons_tab()

            # ===== VEHICLES TAB =====
            with gr.Tab("🚗 Vehicles"):
                vehicles_components = self._build_vehicles_tab()

            # ===== ARMOR TAB =====
            with gr.Tab("🛡️ Armor"):
                armor_components = self._build_armor_tab()

            # ===== ACCESSORIES TAB =====
            with gr.Tab("🔧 Accessories"):
                accessories_components = self._build_accessories_tab()

            # ===== PROGRAMS TAB =====
            with gr.Tab("💻 Programs"):
                programs_components = self._build_programs_tab()

        # Return all components for event wiring
        return {
            **gear_components,
            **weapons_components,
            **vehicles_components,
            **armor_components,
            **accessories_components,
            **programs_components
        }

    def _build_gear_tab(self):
        """Build general gear management tab (electronics, tools, magical items)."""
        with gr.Row():
            # Current gear display
            with gr.Column(scale=2):
                gr.Markdown("#### 📋 Current Gear")

                gear_table = gr.Dataframe(
                    headers=["Item", "Category", "Quantity", "Rating", "Cost"],
                    datatype=["str", "str", "number", "number", "number"],
                    value=[],
                    interactive=False,
                    label="Character Gear"
                )

                with gr.Row():
                    remove_gear_btn = gr.Button("🗑️ Remove Selected", variant="secondary")
                    total_gear_cost = gr.Number(label="Total Gear Value", value=0, interactive=False)

            # Add gear interface
            with gr.Column(scale=1):
                gr.Markdown("#### ➕ Add Gear")

                gear_category_selector = gr.Dropdown(
                    label="Category",
                    choices=[
                        ("Electronics", "electronics"),
                        ("Tools", "tools"),
                        ("Magical Items", "magical"),
                        ("Medical", "biotech"),
                        ("Survival", "survival"),
                        ("General", "general")
                    ],
                    interactive=True
                )

                gear_dropdown = gr.Dropdown(
                    label="Select Gear",
                    choices=[],
                    interactive=True
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

        gear_status = gr.Textbox(label="Gear Status", interactive=False, lines=2)

        return {
            "gear_table": gear_table,
            "remove_gear_btn": remove_gear_btn,
            "total_gear_cost": total_gear_cost,
            "gear_category_selector": gear_category_selector,
            "gear_dropdown": gear_dropdown,
            "gear_quantity_input": gear_quantity_input,
            "gear_rating_input": gear_rating_input,
            "add_gear_btn": add_gear_btn,
            "gear_status": gear_status
        }

    def _build_weapons_tab(self):
        """Build weapons management tab with proper weapon stats."""
        with gr.Row():
            # Current weapons display
            with gr.Column(scale=2):
                gr.Markdown("#### 🔫 Current Weapons")

                weapons_table = gr.Dataframe(
                    headers=["Weapon", "Type", "Accuracy", "Damage", "AP", "Mode", "Cost"],
                    datatype=["str", "str", "str", "str", "str", "str", "number"],
                    value=[],
                    interactive=False,
                    label="Character Weapons"
                )

                with gr.Row():
                    remove_weapon_btn = gr.Button("🗑️ Remove Selected", variant="secondary")
                    total_weapon_cost = gr.Number(label="Total Weapon Value", value=0, interactive=False)

            # Add weapons interface
            with gr.Column(scale=1):
                gr.Markdown("#### ➕ Add Weapons")

                weapon_type_selector = gr.Radio(
                    label="Weapon Type",
                    choices=[("Melee", "melee"), ("Ranged", "ranged")],
                    value="ranged"
                )

                weapon_category_selector = gr.Dropdown(
                    label="Category",
                    choices=[],  # Populated based on weapon type
                    interactive=True
                )

                weapon_dropdown = gr.Dropdown(
                    label="Select Weapon",
                    choices=[],
                    interactive=True
                )

                # Weapon modification options
                with gr.Accordion("🔧 Weapon Modifications", open=False):
                    weapon_accessories = gr.CheckboxGroup(
                        label="Accessories",
                        choices=[
                            "Smartgun System", "Laser Sight", "Sound Suppressor",
                            "Gas-Vent System", "Imaging Scope", "Foregrip"
                        ],
                        value=[]
                    )

                add_weapon_btn = gr.Button("➕ Add Weapon", variant="primary")

        weapons_status = gr.Textbox(label="Weapons Status", interactive=False, lines=2)

        return {
            "weapons_table": weapons_table,
            "remove_weapon_btn": remove_weapon_btn,
            "total_weapon_cost": total_weapon_cost,
            "weapon_type_selector": weapon_type_selector,
            "weapon_category_selector": weapon_category_selector,
            "weapon_dropdown": weapon_dropdown,
            "weapon_accessories": weapon_accessories,
            "add_weapon_btn": add_weapon_btn,
            "weapons_status": weapons_status
        }

    def _build_vehicles_tab(self):
        """Build vehicles management tab with vehicle stats."""
        with gr.Row():
            # Current vehicles display
            with gr.Column(scale=2):
                gr.Markdown("#### 🚗 Current Vehicles")

                vehicles_table = gr.Dataframe(
                    headers=["Vehicle", "Type", "Handling", "Speed", "Body", "Armor", "Cost"],
                    datatype=["str", "str", "number", "number", "number", "number", "number"],
                    value=[],
                    interactive=False,
                    label="Character Vehicles"
                )

                with gr.Row():
                    remove_vehicle_btn = gr.Button("🗑️ Remove Selected", variant="secondary")
                    total_vehicle_cost = gr.Number(label="Total Vehicle Value", value=0, interactive=False)

            # Add vehicles interface
            with gr.Column(scale=1):
                gr.Markdown("#### ➕ Add Vehicles")

                vehicle_type_selector = gr.Dropdown(
                    label="Vehicle Type",
                    choices=[
                        ("Cars", "car"),
                        ("Bikes", "bike"),
                        ("Trucks", "truck"),
                        ("Aircraft", "aircraft"),
                        ("Watercraft", "watercraft")
                    ],
                    interactive=True
                )

                vehicle_dropdown = gr.Dropdown(
                    label="Select Vehicle",
                    choices=[],
                    interactive=True
                )

                # Vehicle modification options
                with gr.Accordion("🔧 Vehicle Modifications", open=False):
                    vehicle_mods = gr.CheckboxGroup(
                        label="Modifications",
                        choices=[
                            "Rigger Interface", "Standard Weapon Mount",
                            "Heavy Weapon Mount", "Run-Flat Tires",
                            "Enhanced Security", "Off-Road Suspension"
                        ],
                        value=[]
                    )

                add_vehicle_btn = gr.Button("➕ Add Vehicle", variant="primary")

        vehicles_status = gr.Textbox(label="Vehicles Status", interactive=False, lines=2)

        return {
            "vehicles_table": vehicles_table,
            "remove_vehicle_btn": remove_vehicle_btn,
            "total_vehicle_cost": total_vehicle_cost,
            "vehicle_type_selector": vehicle_type_selector,
            "vehicle_dropdown": vehicle_dropdown,
            "vehicle_mods": vehicle_mods,
            "add_vehicle_btn": add_vehicle_btn,
            "vehicles_status": vehicles_status
        }

    def _build_armor_tab(self):
        """Build armor management tab with armor ratings."""
        with gr.Row():
            # Current armor display
            with gr.Column(scale=2):
                gr.Markdown("#### 🛡️ Current Armor")

                armor_table = gr.Dataframe(
                    headers=["Armor", "Type", "Armor Rating", "Modifications", "Cost"],
                    datatype=["str", "str", "number", "str", "number"],
                    value=[],
                    interactive=False,
                    label="Character Armor"
                )

                with gr.Row():
                    remove_armor_btn = gr.Button("🗑️ Remove Selected", variant="secondary")
                    total_armor_rating = gr.Number(label="Total Armor Rating", value=0, interactive=False)

            # Add armor interface
            with gr.Column(scale=1):
                gr.Markdown("#### ➕ Add Armor")

                armor_category_selector = gr.Dropdown(
                    label="Armor Type",
                    choices=[
                        ("Clothing", "clothing"),
                        ("Light Armor", "light"),
                        ("Heavy Armor", "heavy"),
                        ("Accessories", "accessories")
                    ],
                    interactive=True
                )

                armor_dropdown = gr.Dropdown(
                    label="Select Armor",
                    choices=[],
                    interactive=True
                )

                # Armor modifications
                with gr.Accordion("🔧 Armor Modifications", open=False):
                    armor_mods = gr.CheckboxGroup(
                        label="Modifications",
                        choices=[
                            "Chemical Protection", "Fire Resistance",
                            "Insulation", "Nonconductivity",
                            "Shock Frills", "Thermal Damping"
                        ],
                        value=[]
                    )

                    mod_rating = gr.Slider(
                        label="Modification Rating",
                        minimum=1,
                        maximum=6,
                        value=1,
                        step=1
                    )

                add_armor_btn = gr.Button("➕ Add Armor", variant="primary")

        armor_status = gr.Textbox(label="Armor Status", interactive=False, lines=2)

        return {
            "armor_table": armor_table,
            "remove_armor_btn": remove_armor_btn,
            "total_armor_rating": total_armor_rating,
            "armor_category_selector": armor_category_selector,
            "armor_dropdown": armor_dropdown,
            "armor_mods": armor_mods,
            "mod_rating": mod_rating,
            "add_armor_btn": add_armor_btn,
            "armor_status": armor_status
        }

    def _build_accessories_tab(self):
        """Build weapon accessories and ammunition tab."""
        with gr.Row():
            # Current accessories display
            with gr.Column(scale=2):
                gr.Markdown("#### 🔧 Current Accessories & Ammo")

                accessories_table = gr.Dataframe(
                    headers=["Item", "Type", "Mount", "Quantity", "Cost"],
                    datatype=["str", "str", "str", "number", "number"],
                    value=[],
                    interactive=False,
                    label="Accessories & Ammunition"
                )

                with gr.Row():
                    remove_accessory_btn = gr.Button("🗑️ Remove Selected", variant="secondary")

            # Add accessories interface
            with gr.Column(scale=1):
                gr.Markdown("#### ➕ Add Accessories")

                accessory_type_selector = gr.Radio(
                    label="Type",
                    choices=[("Weapon Accessories", "accessories"), ("Ammunition", "ammunition")],
                    value="accessories"
                )

                accessory_category_selector = gr.Dropdown(
                    label="Category",
                    choices=[],  # Populated based on type
                    interactive=True
                )

                accessory_dropdown = gr.Dropdown(
                    label="Select Item",
                    choices=[],
                    interactive=True
                )

                accessory_quantity_input = gr.Number(
                    label="Quantity",
                    value=1,
                    minimum=1
                )

                accessory_rating_input = gr.Number(
                    label="Rating (if applicable)",
                    value=0,
                    minimum=0,
                    maximum=6
                )

                add_accessory_btn = gr.Button("➕ Add Accessory", variant="primary")

        accessories_status = gr.Textbox(label="Accessories Status", interactive=False, lines=2)

        return {
            "accessories_table": accessories_table,
            "remove_accessory_btn": remove_accessory_btn,
            "accessory_type_selector": accessory_type_selector,
            "accessory_category_selector": accessory_category_selector,
            "accessory_dropdown": accessory_dropdown,
            "accessory_quantity_input": accessory_quantity_input,
            "accessory_rating_input": accessory_rating_input,
            "add_accessory_btn": add_accessory_btn,
            "accessories_status": accessories_status
        }

    def _build_programs_tab(self):
        """Build programs and cyberdeck management tab."""
        with gr.Row():
            # Current programs display
            with gr.Column(scale=2):
                gr.Markdown("#### 💻 Current Programs")

                programs_table = gr.Dataframe(
                    headers=["Program", "Type", "Rating", "Description"],
                    datatype=["str", "str", "number", "str"],
                    value=[],
                    interactive=False,
                    label="Character Programs"
                )

                with gr.Row():
                    remove_program_btn = gr.Button("🗑️ Remove Selected", variant="secondary")

                # Cyberdeck section
                with gr.Accordion("🖥️ Cyberdeck", open=True):
                    cyberdeck_selector = gr.Dropdown(
                        label="Select Cyberdeck",
                        choices=[],
                        interactive=True
                    )

                    with gr.Row():
                        deck_attack = gr.Number(label="Attack", value=0, interactive=False)
                        deck_sleaze = gr.Number(label="Sleaze", value=0, interactive=False)
                        deck_data_proc = gr.Number(label="Data Processing", value=0, interactive=False)
                        deck_firewall = gr.Number(label="Firewall", value=0, interactive=False)

                    set_cyberdeck_btn = gr.Button("⚡ Set Cyberdeck", variant="primary")

            # Add programs interface
            with gr.Column(scale=1):
                gr.Markdown("#### ➕ Add Programs")

                program_type_selector = gr.Dropdown(
                    label="Program Type",
                    choices=[
                        ("Common Programs", "common"),
                        ("Hacking Programs", "hacking"),
                        ("Autosofts", "autosoft")
                    ],
                    interactive=True
                )

                program_dropdown = gr.Dropdown(
                    label="Select Program",
                    choices=[],
                    interactive=True
                )

                program_rating_input = gr.Number(
                    label="Rating",
                    value=1,
                    minimum=1,
                    maximum=6
                )

                add_program_btn = gr.Button("➕ Add Program", variant="primary")

        programs_status = gr.Textbox(label="Programs Status", interactive=False, lines=2)

        return {
            "programs_table": programs_table,
            "remove_program_btn": remove_program_btn,
            "cyberdeck_selector": cyberdeck_selector,
            "deck_attack": deck_attack,
            "deck_sleaze": deck_sleaze,
            "deck_data_proc": deck_data_proc,
            "deck_firewall": deck_firewall,
            "set_cyberdeck_btn": set_cyberdeck_btn,
            "program_type_selector": program_type_selector,
            "program_dropdown": program_dropdown,
            "program_rating_input": program_rating_input,
            "add_program_btn": add_program_btn,
            "programs_status": programs_status
        }


# ===== EVENT HANDLERS FOR EQUIPMENT TABS =====

class CharacterEquipmentHandlers:
    """Event handlers for equipment management tabs."""

    def __init__(self, char_api):
        self.char_api = char_api

    # ===== DROPDOWN POPULATION HANDLERS =====

    def populate_gear_dropdown(self, category: str):
        """Populate gear dropdown based on category."""
        try:
            if not category:
                return gr.update(choices=[], value=None)

            # This would call your API to get gear by category
            choices = self.char_api.get_gear_dropdown_choices(category)
            return gr.update(choices=choices, value=None)
        except Exception as e:
            return gr.update(choices=[("Error loading gear", None)], value=None)

    def populate_weapon_category_dropdown(self, weapon_type: str):
        """Populate weapon category dropdown based on weapon type."""
        try:
            if weapon_type == "melee":
                choices = [("Blades", "blades"), ("Clubs", "clubs"), ("Exotic Melee", "exotic_melee")]
            elif weapon_type == "ranged":
                choices = [
                    ("Pistols", "pistols"), ("SMGs", "submachine_guns"), ("Assault Rifles", "assault_rifles"),
                    ("Sniper Rifles", "sniper_rifles"), ("Shotguns", "shotguns"), ("Archery", "archery"),
                    ("Machine Guns", "machine_guns"), ("Special", "special")
                ]
            else:
                choices = []

            return gr.update(choices=choices, value=None)
        except Exception as e:
            return gr.update(choices=[("Error", None)], value=None)

    def populate_weapon_dropdown(self, weapon_type: str, category: str):
        """Populate weapon dropdown based on type and category."""
        try:
            if not weapon_type or not category:
                return gr.update(choices=[], value=None)

            # This would call your API to get weapons by type and category
            choices = self.char_api.get_weapons_dropdown_choices(weapon_type, category)
            return gr.update(choices=choices, value=None)
        except Exception as e:
            return gr.update(choices=[("Error loading weapons", None)], value=None)

    def populate_vehicle_dropdown(self, vehicle_type: str):
        """Populate vehicle dropdown based on vehicle type."""
        try:
            if not vehicle_type:
                return gr.update(choices=[], value=None)

            choices = self.char_api.get_vehicles_dropdown_choices(vehicle_type)
            return gr.update(choices=choices, value=None)
        except Exception as e:
            return gr.update(choices=[("Error loading vehicles", None)], value=None)

    def populate_armor_dropdown(self, armor_category: str):
        """Populate armor dropdown based on category."""
        try:
            if not armor_category:
                return gr.update(choices=[], value=None)

            # Map categories to gear library subcategories
            category_mapping = {
                "clothing": "clothing",
                "light": "light_armor",
                "heavy": "heavy_armor",
                "accessories": "accessories"
            }

            subcategory = category_mapping.get(armor_category, armor_category)
            choices = self.char_api.get_armor_dropdown_choices(subcategory)
            return gr.update(choices=choices, value=None)
        except Exception as e:
            return gr.update(choices=[("Error loading armor", None)], value=None)

    def populate_accessory_category_dropdown(self, accessory_type: str):
        """Populate accessory category dropdown based on type."""
        try:
            if accessory_type == "accessories":
                choices = [
                    ("Optics", "optics"), ("Barrel", "barrel"), ("Stock", "stock"),
                    ("Electronics", "electronics"), ("Support", "support")
                ]
            elif accessory_type == "ammunition":
                choices = [
                    ("Standard", "standard"), ("Special", "special"),
                    ("Shotgun", "shotgun"), ("Grenades", "grenade")
                ]
            else:
                choices = []

            return gr.update(choices=choices, value=None)
        except Exception as e:
            return gr.update(choices=[("Error", None)], value=None)

    def populate_accessory_dropdown(self, accessory_type: str, category: str):
        """Populate accessory dropdown based on type and category."""
        try:
            if not accessory_type or not category:
                return gr.update(choices=[], value=None)

            if accessory_type == "accessories":
                choices = self.char_api.get_weapon_accessories_dropdown_choices(category)
            elif accessory_type == "ammunition":
                choices = self.char_api.get_ammunition_dropdown_choices(category)
            else:
                choices = []

            return gr.update(choices=choices, value=None)
        except Exception as e:
            return gr.update(choices=[("Error loading items", None)], value=None)

    def populate_program_dropdown(self, program_type: str):
        """Populate program dropdown based on program type."""
        try:
            if not program_type:
                return gr.update(choices=[], value=None)

            choices = self.char_api.get_programs_dropdown_choices(program_type)
            return gr.update(choices=choices, value=None)
        except Exception as e:
            return gr.update(choices=[("Error loading programs", None)], value=None)

    def populate_cyberdeck_dropdown(self):
        """Populate cyberdeck dropdown with available cyberdecks."""
        try:
            choices = self.char_api.get_cyberdecks_dropdown_choices()
            return gr.update(choices=choices, value=None)
        except Exception as e:
            return gr.update(choices=[("Error loading cyberdecks", None)], value=None)

    # ===== EQUIPMENT MANAGEMENT HANDLERS =====

    def add_gear_to_character(self, character_id, gear_name, category, quantity, rating):
        """Add gear item to character."""
        if not character_id or not gear_name:
            return "No character or gear selected", gr.update()

        gear_data = {
            "name": gear_name,
            "category": category,
            "quantity": quantity,
            "rating": rating
        }

        result = self.char_api.add_character_gear(character_id, gear_data)
        if "error" in result:
            return f"❌ Failed to add gear: {result['error']}", gr.update()

        # Refresh gear table
        updated_gear = self._get_character_gear_table(character_id)
        return "✅ Gear added successfully!", gr.update(value=updated_gear)

    def add_weapon_to_character(self, character_id, weapon_name, weapon_type, category, accessories):
        """Add weapon to character with optional accessories."""
        if not character_id or not weapon_name:
            return "No character or weapon selected", gr.update()

        weapon_data = {
            "name": weapon_name,
            "weapon_type": weapon_type,
            "category": category,
            "accessories": accessories  # List of selected accessories
        }

        result = self.char_api.add_character_weapon(character_id, weapon_data)
        if "error" in result:
            return f"❌ Failed to add weapon: {result['error']}", gr.update()

        # Refresh weapons table
        updated_weapons = self._get_character_weapons_table(character_id)
        return "✅ Weapon added successfully!", gr.update(value=updated_weapons)

    def add_vehicle_to_character(self, character_id, vehicle_name, vehicle_type, modifications):
        """Add vehicle to character with optional modifications."""
        if not character_id or not vehicle_name:
            return "No character or vehicle selected", gr.update()

        vehicle_data = {
            "name": vehicle_name,
            "vehicle_type": vehicle_type,
            "modifications": modifications
        }

        result = self.char_api.add_character_vehicle(character_id, vehicle_data)
        if "error" in result:
            return f"❌ Failed to add vehicle: {result['error']}", gr.update()

        # Refresh vehicles table
        updated_vehicles = self._get_character_vehicles_table(character_id)
        return "✅ Vehicle added successfully!", gr.update(value=updated_vehicles)

    def set_character_cyberdeck(self, character_id, cyberdeck_name):
        """Set character's cyberdeck and update stats display."""
        if not character_id or not cyberdeck_name:
            return "No character or cyberdeck selected", {}, {}, {}, {}

        # Get cyberdeck stats from library
        cyberdeck_data = self.char_api.get_cyberdeck_stats(cyberdeck_name)
        if "error" in cyberdeck_data:
            return f"❌ Failed to get cyberdeck stats: {cyberdeck_data['error']}", {}, {}, {}, {}

        # Set cyberdeck for character
        result = self.char_api.update_character_cyberdeck(character_id, cyberdeck_data)
        if "error" in result:
            return f"❌ Failed to set cyberdeck: {result['error']}", {}, {}, {}, {}

        # Return updated stat displays
        return (
            "✅ Cyberdeck set successfully!",
            gr.update(value=cyberdeck_data.get("attack", 0)),
            gr.update(value=cyberdeck_data.get("sleaze", 0)),
            gr.update(value=cyberdeck_data.get("data_processing", 0)),
            gr.update(value=cyberdeck_data.get("firewall", 0))
        )

    # ===== HELPER METHODS =====

    def _get_character_gear_table(self, character_id):
        """Get formatted gear table for character."""
        character_data = self.char_api.get_character(character_id)
        if "gear" in character_data:
            return [
                [item["name"], item["category"], item["quantity"], item["rating"], item["cost"]]
                for item in character_data["gear"]
            ]
        return []

    def _get_character_weapons_table(self, character_id):
        """Get formatted weapons table for character."""
        character_data = self.char_api.get_character(character_id)
        if "weapons" in character_data:
            return [
                [w["name"], w["weapon_type"], w["accuracy"], w["damage_code"],
                 w["armor_penetration"], w["mode_ammo"], w["cost"]]
                for w in character_data["weapons"]
            ]
        return []

    def _get_character_vehicles_table(self, character_id):
        """Get formatted vehicles table for character."""
        character_data = self.char_api.get_character(character_id)
        if "vehicles" in character_data:
            return [
                [v["name"], v["vehicle_type"], v["handling"], v["speed"],
                 v["body"], v["armor"], v["cost"]]
                for v in character_data["vehicles"]
            ]
        return []


def wire_equipment_events(components: Dict, handlers: CharacterEquipmentHandlers):
    """Wire up all equipment management events with proper dropdown cascading."""

    # ===== GEAR TAB EVENTS =====

    components["gear_category_selector"].change(
        fn=handlers.populate_gear_dropdown,
        inputs=[components["gear_category_selector"]],
        outputs=[components["gear_dropdown"]]
    )

    components["add_gear_btn"].click(
        fn=handlers.add_gear_to_character,
        inputs=[
            components["character_selector"],  # From main character selector
            components["gear_dropdown"],
            components["gear_category_selector"],
            components["gear_quantity_input"],
            components["gear_rating_input"]
        ],
        outputs=[components["gear_status"], components["gear_table"]]
    )

    # ===== WEAPONS TAB EVENTS =====

    components["weapon_type_selector"].change(
        fn=handlers.populate_weapon_category_dropdown,
        inputs=[components["weapon_type_selector"]],
        outputs=[components["weapon_category_selector"]]
    )

    components["weapon_category_selector"].change(
        fn=handlers.populate_weapon_dropdown,
        inputs=[components["weapon_type_selector"], components["weapon_category_selector"]],
        outputs=[components["weapon_dropdown"]]
    )

    components["add_weapon_btn"].click(
        fn=handlers.add_weapon_to_character,
        inputs=[
            components["character_selector"],
            components["weapon_dropdown"],
            components["weapon_type_selector"],
            components["weapon_category_selector"],
            components["weapon_accessories"]
        ],
        outputs=[components["weapons_status"], components["weapons_table"]]
    )

    # ===== VEHICLES TAB EVENTS =====

    components["vehicle_type_selector"].change(
        fn=handlers.populate_vehicle_dropdown,
        inputs=[components["vehicle_type_selector"]],
        outputs=[components["vehicle_dropdown"]]
    )

    components["add_vehicle_btn"].click(
        fn=handlers.add_vehicle_to_character,
        inputs=[
            components["character_selector"],
            components["vehicle_dropdown"],
            components["vehicle_type_selector"],
            components["vehicle_mods"]
        ],
        outputs=[components["vehicles_status"], components["vehicles_table"]]
    )

    # ===== ARMOR TAB EVENTS =====

    components["armor_category_selector"].change(
        fn=handlers.populate_armor_dropdown,
        inputs=[components["armor_category_selector"]],
        outputs=[components["armor_dropdown"]]
    )

    # ===== ACCESSORIES TAB EVENTS =====

    components["accessory_type_selector"].change(
        fn=handlers.populate_accessory_category_dropdown,
        inputs=[components["accessory_type_selector"]],
        outputs=[components["accessory_category_selector"]]
    )

    components["accessory_category_selector"].change(
        fn=handlers.populate_accessory_dropdown,
        inputs=[components["accessory_type_selector"], components["accessory_category_selector"]],
        outputs=[components["accessory_dropdown"]]
    )

    # ===== PROGRAMS TAB EVENTS =====

    components["program_type_selector"].change(
        fn=handlers.populate_program_dropdown,
        inputs=[components["program_type_selector"]],
        outputs=[components["program_dropdown"]]
    )

    components["set_cyberdeck_btn"].click(
        fn=handlers.set_character_cyberdeck,
        inputs=[components["character_selector"], components["cyberdeck_selector"]],
        outputs=[
            components["programs_status"],
            components["deck_attack"],
            components["deck_sleaze"],
            components["deck_data_proc"],
            components["deck_firewall"]
        ]
    )

    # ===== AUTO-POPULATE ON LOAD =====

    def auto_populate_equipment_dropdowns():
        """Auto-populate equipment dropdowns on app load."""
        try:
            # Populate cyberdeck dropdown
            cyberdeck_choices = handlers.char_api.get_cyberdecks_dropdown_choices()

            return gr.update(choices=cyberdeck_choices)
        except Exception as e:
            return gr.update(choices=[("Error loading", None)])

    return auto_populate_equipment_dropdowns