"""
Shared UI Utility Functions for Shadowrun RAG System
Common helpers, formatters, and utility functions used across UI components.
"""

import gradio as gr
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import time

logger = logging.getLogger(__name__)


# ===== FORMATTING HELPERS =====

def format_character_name_for_display(char_data: Dict) -> str:
    """Format character name for dropdown display."""
    if not char_data:
        return "Unknown Character"

    name = char_data.get('name', 'Unknown')
    metatype = char_data.get('metatype', 'Human')
    archetype = char_data.get('archetype', '')

    display_name = f"👤 {name} ({metatype})"
    if archetype:
        display_name += f" - {archetype}"

    return display_name


def format_skill_for_table(skill_data: Dict, character_stats: Dict = None) -> List:
    """Format skill data for table display with dice pool calculation."""
    name = skill_data.get('name', 'Unknown Skill')
    rating = skill_data.get('rating', 0)
    specialization = skill_data.get('specialization', '')
    skill_type = skill_data.get('skill_type', 'active')

    # Calculate dice pool if character stats available
    dice_pool = rating
    if character_stats and skill_data.get('attribute'):
        attr_name = skill_data['attribute'].lower()
        attr_value = character_stats.get(attr_name, 0)
        dice_pool = attr_value + rating

    return [name, rating, specialization, skill_type, dice_pool]


def format_quality_for_table(quality_data: Dict) -> List:
    """Format quality data for table display."""
    name = quality_data.get('name', 'Unknown Quality')
    rating = quality_data.get('rating', '')
    quality_type = quality_data.get('quality_type', 'positive')

    return [name, rating, quality_type]


def format_gear_for_table(gear_data: Dict) -> List:
    """Format gear data for table display."""
    name = gear_data.get('name', 'Unknown Item')
    quantity = gear_data.get('quantity', 1)
    category = gear_data.get('category', '')
    armor_value = gear_data.get('armor_value', 0)

    return [name, quantity, category, armor_value]


def format_weapon_for_table(weapon_data: Dict) -> List:
    """Format weapon data for table display."""
    name = weapon_data.get('name', 'Unknown Weapon')
    weapon_type = weapon_data.get('weapon_type', 'ranged')
    damage_code = weapon_data.get('damage_code', '')
    armor_penetration = weapon_data.get('armor_penetration', 0)

    return [name, weapon_type, damage_code, armor_penetration]


def format_vehicle_for_table(vehicle_data: Dict) -> List:
    """Format vehicle data for table display."""
    name = vehicle_data.get('name', 'Unknown Vehicle')
    vehicle_type = vehicle_data.get('vehicle_type', 'vehicle')
    handling = vehicle_data.get('handling', 0)
    speed = vehicle_data.get('speed', 0)

    return [name, vehicle_type, handling, speed]


def format_program_for_table(program_data: Dict) -> List:
    """Format program data for table display."""
    name = program_data.get('name', 'Unknown Program')
    rating = program_data.get('rating', 1)
    program_type = program_data.get('program_type', 'common')

    return [name, rating, program_type]


# ===== CHARACTER DATA HELPERS =====

def extract_character_data_for_forms(char_data: Dict) -> Tuple[Dict, Dict]:
    """Extract character data and format for form inputs."""
    if not char_data or "error" in char_data:
        return {}, {}

    # Extract stats with defaults
    stats = char_data.get('stats', {})
    stats_defaults = {
        'body': 1, 'agility': 1, 'reaction': 1, 'strength': 1,
        'charisma': 1, 'logic': 1, 'intuition': 1, 'willpower': 1,
        'edge': 1, 'essence': 6.0, 'physical_limit': 1,
        'mental_limit': 1, 'social_limit': 1, 'initiative': 1, 'hot_sim_vr': 0
    }

    # Merge with defaults
    formatted_stats = {key: stats.get(key, default) for key, default in stats_defaults.items()}

    # Extract resources with defaults
    resources = char_data.get('resources', {})
    resources_defaults = {
        'nuyen': 0, 'street_cred': 0, 'notoriety': 0, 'public_aware': 0,
        'total_karma': 0, 'available_karma': 0, 'edge_pool': 1
    }

    # Merge with defaults
    formatted_resources = {key: resources.get(key, default) for key, default in resources_defaults.items()}

    return formatted_stats, formatted_resources


def format_character_tables_data(char_data: Dict) -> Dict[str, List]:
    """Format all character data for table display."""
    if not char_data or "error" in char_data:
        return {
            "skills": [], "qualities": [], "gear": [],
            "weapons": [], "vehicles": [], "programs": []
        }

    # Get character stats for dice pool calculations
    stats = char_data.get('stats', {})

    # Format skills by type
    skills_display = []
    skills_by_type = char_data.get('skills', {})
    for skill_type, skills in skills_by_type.items():
        for skill in skills:
            skill_row = format_skill_for_table(skill, stats)
            skills_display.append(skill_row)

    # Format other data types
    qualities_display = [format_quality_for_table(q) for q in char_data.get('qualities', [])]
    gear_display = [format_gear_for_table(g) for g in char_data.get('gear', [])]
    weapons_display = [format_weapon_for_table(w) for w in char_data.get('weapons', [])]
    vehicles_display = [format_vehicle_for_table(v) for v in char_data.get('vehicles', [])]
    programs_display = [format_program_for_table(p) for p in char_data.get('programs', [])]

    return {
        "skills": skills_display,
        "qualities": qualities_display,
        "gear": gear_display,
        "weapons": weapons_display,
        "vehicles": vehicles_display,
        "programs": programs_display
    }


# ===== STATUS MESSAGE HELPERS =====

def create_success_message(message: str) -> str:
    """Create formatted success message."""
    return f"✅ {message}"


def create_error_message(message: str) -> str:
    """Create formatted error message."""
    return f"❌ {message}"


def create_warning_message(message: str) -> str:
    """Create formatted warning message."""
    return f"⚠️ {message}"


def create_info_message(message: str) -> str:
    """Create formatted info message."""
    return f"ℹ️ {message}"


# ===== PROGRESS TRACKING HELPERS =====

class ProgressTracker:
    """Simple progress tracking for UI operations."""

    def __init__(self):
        self.current_operation = None
        self.start_time = None
        self.steps_total = 0
        self.steps_completed = 0

    def start(self, operation_name: str, total_steps: int = 100):
        """Start tracking an operation."""
        self.current_operation = operation_name
        self.start_time = time.time()
        self.steps_total = total_steps
        self.steps_completed = 0
        logger.info(f"Started operation: {operation_name}")

    def update(self, steps_completed: int, details: str = ""):
        """Update progress."""
        self.steps_completed = steps_completed
        progress_pct = (steps_completed / self.steps_total * 100) if self.steps_total > 0 else 0

        elapsed = time.time() - self.start_time if self.start_time else 0
        status = f"{self.current_operation}: {progress_pct:.0f}% ({elapsed:.1f}s)"
        if details:
            status += f" - {details}"

        return status, progress_pct

    def complete(self, success_message: str = ""):
        """Mark operation as complete."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        final_message = success_message or f"{self.current_operation} completed"
        logger.info(f"Completed: {final_message} ({elapsed:.1f}s)")

        # Reset
        self.current_operation = None
        self.start_time = None

        return f"✅ {final_message} ({elapsed:.1f}s)", 100


# ===== DROPDOWN HELPERS =====

def create_dropdown_choices_with_empty(items: List[Dict],
                                       value_key: str,
                                       display_key: str,
                                       empty_text: str = "Select...") -> List[Tuple]:
    """Create dropdown choices with empty option."""
    if not items:
        return [(f"No items found", None)]

    choices = [(empty_text, None)]
    for item in items:
        display_value = item.get(display_key, 'Unknown')
        actual_value = item.get(value_key)
        choices.append((display_value, actual_value))

    return choices


def format_dropdown_choice_with_details(item: Dict,
                                        name_key: str = 'name',
                                        details_keys: List[str] = None) -> str:
    """Format dropdown choice with additional details."""
    name = item.get(name_key, 'Unknown')

    if not details_keys:
        return name

    details = []
    for key in details_keys:
        value = item.get(key)
        if value and value != 0:
            if key == 'karma_cost':
                details.append(f"{value} karma")
            elif key == 'base_cost':
                details.append(f"{value}¥")
            elif key == 'linked_attribute':
                details.append(f"({value})")
            else:
                details.append(str(value))

    if details:
        return f"{name} - {', '.join(details)}"
    return name


# ===== VALIDATION HELPERS =====

def validate_character_name(name: str) -> Tuple[bool, str]:
    """Validate character name input."""
    if not name or not name.strip():
        return False, "Character name is required"

    if len(name.strip()) < 2:
        return False, "Character name must be at least 2 characters"

    if len(name.strip()) > 50:
        return False, "Character name must be less than 50 characters"

    return True, ""


def validate_numeric_input(value: Union[int, float],
                           min_val: Union[int, float] = None,
                           max_val: Union[int, float] = None,
                           field_name: str = "Value") -> Tuple[bool, str]:
    """Validate numeric input with optional range checking."""
    try:
        numeric_value = float(value)

        if min_val is not None and numeric_value < min_val:
            return False, f"{field_name} must be at least {min_val}"

        if max_val is not None and numeric_value > max_val:
            return False, f"{field_name} must be no more than {max_val}"

        return True, ""

    except (ValueError, TypeError):
        return False, f"{field_name} must be a valid number"


# ===== FILE HANDLING HELPERS =====

def safe_file_write(file_path: Union[str, Path], content: str, encoding: str = 'utf-8') -> bool:
    """Safely write content to file with error handling."""
    try:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding=encoding)
        return True
    except Exception as e:
        logger.error(f"Failed to write file {file_path}: {e}")
        return False


def safe_file_read(file_path: Union[str, Path], encoding: str = 'utf-8') -> Optional[str]:
    """Safely read file content with error handling."""
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            return None
        return file_path.read_text(encoding=encoding)
    except Exception as e:
        logger.error(f"Failed to read file {file_path}: {e}")
        return None


# ===== GRADIO UPDATE HELPERS =====

def update_dropdown_choices(choices: List[Tuple],
                            selected_value: Any = None,
                            visible: bool = True) -> gr.update:
    """Helper to update dropdown choices."""
    update_args = {"choices": choices, "visible": visible}
    if selected_value is not None:
        update_args["value"] = selected_value
    return gr.update(**update_args)


def update_table_data(data: List[List],
                      headers: List[str] = None,
                      visible: bool = True) -> gr.update:
    """Helper to update table data."""
    update_args = {"value": data, "visible": visible}
    if headers:
        update_args["headers"] = headers
    return gr.update(**update_args)


def update_textbox_value(value: str,
                         interactive: bool = None,
                         visible: bool = True) -> gr.update:
    """Helper to update textbox value."""
    update_args = {"value": value, "visible": visible}
    if interactive is not None:
        update_args["interactive"] = interactive
    return gr.update(**update_args)


# ===== DICE POOL CALCULATION HELPERS =====

def calculate_dice_pool(attribute_value: int,
                        skill_rating: int,
                        specialization_applies: bool = False,
                        modifiers: List[int] = None) -> Tuple[int, str]:
    """Calculate dice pool with explanation."""
    base_pool = attribute_value + skill_rating

    explanation_parts = [f"Attribute {attribute_value}", f"Skill {skill_rating}"]

    if specialization_applies:
        base_pool += 2
        explanation_parts.append("Specialization +2")

    if modifiers:
        modifier_total = sum(modifiers)
        base_pool += modifier_total
        if modifier_total != 0:
            sign = "+" if modifier_total > 0 else ""
            explanation_parts.append(f"Modifiers {sign}{modifier_total}")

    final_pool = max(0, base_pool)  # Dice pools can't be negative
    explanation = " + ".join(explanation_parts) + f" = {final_pool} dice"

    return final_pool, explanation


def format_dice_result(dice_pool: int,
                       explanation: str,
                       character_name: str = None) -> str:
    """Format dice pool result for display."""
    if character_name:
        return f"🎲 **{character_name}'s Dice Pool:**\n\n{explanation}"
    else:
        return f"🎲 **Dice Pool Calculation:**\n\n{explanation}"


# ===== THEME AND STYLING HELPERS =====

def get_custom_css() -> str:
    """Return custom CSS for the application."""
    return """
    <style>
    /* Character Management Styles */
    .character-section {
        border-left: 3px solid #2196F3;
        padding-left: 1rem;
        margin: 0.5rem 0;
    }

    .character-active {
        border-left-color: #4CAF50;
        background: rgba(76, 175, 80, 0.1);
    }

    /* Status Message Styles */
    .status-success {
        color: #4CAF50;
        font-weight: 500;
    }

    .status-error {
        color: #f44336;
        font-weight: 500;
    }

    .status-warning {
        color: #FF9800;
        font-weight: 500;
    }

    .status-info {
        color: #2196F3;
        font-weight: 500;
    }

    /* Table Styles */
    .character-table {
        margin: 0.5rem 0;
    }

    .character-table .headers {
        background: rgba(33, 150, 243, 0.1);
        font-weight: 600;
    }

    /* Dropdown Styles */
    .character-dropdown {
        margin-bottom: 0.5rem;
    }

    .character-dropdown .selected {
        background: rgba(76, 175, 80, 0.2);
    }

    /* Button Styles */
    .character-btn-primary {
        background: #2196F3;
        color: white;
        font-weight: 500;
    }

    .character-btn-success {
        background: #4CAF50;
        color: white;
        font-weight: 500;
    }

    .character-btn-danger {
        background: #f44336;
        color: white;
        font-weight: 500;
    }

    /* Form Styles */
    .character-form-section {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        border: 1px solid rgba(0, 0, 0, 0.1);
    }

    .character-form-section h4 {
        margin-top: 0;
        color: #2196F3;
    }

    /* Progress Styles */
    .progress-container {
        margin: 1rem 0;
    }

    .progress-bar {
        background: #e0e0e0;
        border-radius: 4px;
        overflow: hidden;
    }

    .progress-fill {
        background: #2196F3;
        height: 100%;
        transition: width 0.3s ease;
    }

    /* Responsive adjustments */
    @media (max-width: 768px) {
        .character-section {
            padding-left: 0.5rem;
        }

        .character-form-section {
            padding: 0.5rem;
        }
    }
    </style>
    """


# ===== ERROR HANDLING HELPERS =====

class UIErrorHandler:
    """Centralized error handling for UI operations."""

    @staticmethod
    def handle_api_error(error_response: Dict, operation_name: str = "operation") -> str:
        """Handle API error response and return user-friendly message."""
        if isinstance(error_response, dict) and "error" in error_response:
            error_msg = error_response["error"]
            logger.error(f"{operation_name} failed: {error_msg}")
            return create_error_message(f"{operation_name} failed: {error_msg}")
        else:
            logger.error(f"{operation_name} failed: Unknown error")
            return create_error_message(f"{operation_name} failed: Unknown error")

    @staticmethod
    def handle_validation_error(validation_result: Tuple[bool, str],
                                field_name: str = "input") -> Optional[str]:
        """Handle validation error and return message if invalid."""
        is_valid, error_msg = validation_result
        if not is_valid:
            logger.warning(f"Validation failed for {field_name}: {error_msg}")
            return create_error_message(error_msg)
        return None

    @staticmethod
    def handle_exception(exception: Exception, operation_name: str = "operation") -> str:
        """Handle unexpected exceptions."""
        logger.exception(f"Exception in {operation_name}: {exception}")
        return create_error_message(f"{operation_name} failed: {str(exception)}")


# ===== LOGGING HELPERS =====

def setup_ui_logging(log_level: str = "INFO"):
    """Set up logging for UI components."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger.setLevel(numeric_level)


def log_user_action(action: str, details: Dict = None):
    """Log user actions for debugging."""
    log_msg = f"User action: {action}"
    if details:
        log_msg += f" - {details}"
    logger.info(log_msg)


# ===== EXPORT HELPERS =====

def format_character_for_export(char_data: Dict) -> str:
    """Format character data for text export."""
    if not char_data:
        return "No character data available"

    lines = [
        f"Character: {char_data.get('name', 'Unknown')}",
        f"Metatype: {char_data.get('metatype', 'Human')}",
        f"Archetype: {char_data.get('archetype', '')}\n"
    ]

    # Add stats
    stats = char_data.get('stats', {})
    if stats:
        lines.append("ATTRIBUTES:")
        for attr in ['body', 'agility', 'reaction', 'strength', 'willpower', 'logic', 'intuition', 'charisma']:
            value = stats.get(attr, 0)
            lines.append(f"  {attr.title()}: {value}")
        lines.append(f"  Edge: {stats.get('edge', 1)}")
        lines.append(f"  Essence: {stats.get('essence', 6.0)}\n")

    # Add skills summary
    skills = char_data.get('skills', {})
    if skills:
        active_skills = skills.get('active', [])
        if active_skills:
            lines.append("TOP SKILLS:")
            sorted_skills = sorted(active_skills, key=lambda x: x.get('rating', 0), reverse=True)
            for skill in sorted_skills[:10]:  # Top 10 skills
                name = skill.get('name', 'Unknown')
                rating = skill.get('rating', 0)
                lines.append(f"  {name}: {rating}")
            lines.append("")

    return "\n".join(lines)


# Module initialization
logger.info("UI Helpers module initialized")