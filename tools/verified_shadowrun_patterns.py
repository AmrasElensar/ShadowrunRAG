"""
VERIFIED Shadowrun content detection patterns - ONLY from your uploaded files.
This replaces the comprehensive patterns with verified content only.
"""

VERIFIED_SHADOWRUN_PATTERNS = {

    # GEAR/WEAPONS (verified from CoreRules-11-Street_Gear.md)
    "gear": {
        "manufacturers": ["ares", "colt", "ruger", "browning", "remington", "defiance",
                          "yamaha", "fichetti", "beretta", "taurus", "steyr", "enfield",
                          "cavalier arms", "ranger arms", "pjss", "parashield"],
        "weapon_types": ["pistol", "rifle", "shotgun", "smg", "assault rifle", "sniper rifle",
                         "machine gun", "crossbow", "heavy pistol", "light pistol", "hold-out pistol"],
        "specific_weapons": ["predator", "ares predator", "government 2066", "ultra-power",
                             "roomsweeper", "super warhawk", "crusader", "black scorpion", "ak-97",
                             "ares alpha", "desert strike", "crockett", "ranger sm-5"],
        "weapon_stats": ["accuracy", "damage", "ap", "mode", "rc", "ammo", "avail", "cost"],
        "availability_terms": ["availability", "restricted", "forbidden", "legal"]
    },

    # MAGIC (verified from CoreRules-8-Magic.md)
    "magic": {
        "specific_spells": ["fireball", "manabolt", "lightning bolt", "death touch", "manaball",
                            "flamethrower", "ball lightning", "shatter", "powerbolt", "powerball",
                            "levitate", "light", "magic fingers", "mana barrier", "physical barrier"],
        "spell_mechanics": ["force", "drain", "range", "duration", "type", "spellcasting + magic",
                            "drain resistance", "astral", "physical", "mana", "net hits"],
        "spell_types": ["direct", "indirect", "elemental", "environmental", "area", "touch",
                        "line of sight", "sustained"],
        "magic_skills": ["spellcasting", "counterspelling", "summoning", "binding", "banishing",
                         "alchemy", "artificing", "disenchanting", "arcana"],
        "traditions": ["hermetic", "shamanic"],  # Only these confirmed in files
        "awakened_types": ["mage", "shaman", "adept", "mystic adept", "aspected magician"],
        "astral_terms": ["astral space", "astral projection", "astral perception", "astral limit"],
        "drain_terms": ["drain value", "drain resistance", "stun damage", "physical damage"]
    },

    # MATRIX (verified from CoreRules-6-The_Matrix.md)
    "matrix": {
        "ic_programs": ["black ic", "white ic", "gray ic", "killer ic", "marker ic", "patrol ic",
                        "crash ic", "jammer ic", "probe ic", "blaster ic", "acid", "binder"],
        "matrix_attributes": ["attack", "sleaze", "data processing", "firewall"],
        "cyberdeck_brands": ["erika", "microdeck", "microtronica", "hermes", "novatech", "renraku",
                             "sony", "shiawase", "fairlight"],
        "matrix_actions": ["hack on the fly", "brute force", "data spike", "crash program",
                           "matrix perception", "hide", "trace user", "full matrix defense"],
        "matrix_damage": ["matrix damage", "biofeedback", "dumpshock", "link-lock", "bricking"],
        "matrix_programs": ["armor", "baby monitor", "biofeedback", "blackout", "decryption",
                            "exploit", "stealth", "toolbox", "virtual machine"],
        "matrix_personas": ["decker", "technomancer", "spider", "agent"],
        "matrix_tests": ["willpower + firewall", "intuition + firewall", "logic + firewall"]
    },

    # RIGGERS (verified from CoreRules-7-Riggers.md and CoreRules-11-Street_Gear.md)
    "riggers": {
        "drone_types": ["steel lynx", "doberman", "rotodrone", "fly-spy", "kanmushi", "crawler",
                        "horizon flying eye", "optic-x2", "duelist", "dalmatian"],
        "drone_brands": ["shiawase", "mct", "aztechnology", "lockheed", "ares", "gm-nissan",
                         "cyberspace designs"],
        "rigger_gear": ["rigger command console", "rcc", "control rig", "rigger interface"],
        "rigger_programs": ["maneuvering", "targeting", "stealth", "evasion", "clearsight",
                            "electronic warfare"],
        "vehicle_stats": ["handling", "speed", "acceleration", "body", "armor", "pilot", "sensor"],
        "rigger_actions": ["jumped in", "remote control", "autopilot", "noise reduction"]
    },

    # CHARACTER CREATION (verified from CoreRules-3-Creating A Shadowrunner.md)
    "character_creation": {
        "priority_system": ["priority a", "priority b", "priority c", "priority d", "priority e"],
        "metatypes": ["human", "elf", "dwarf", "ork", "troll"],  # Only these 5 confirmed
        "attributes": ["body", "agility", "reaction", "strength", "charisma", "intuition",
                       "logic", "willpower", "edge", "essence", "magic", "resonance"],
        "special_attributes": ["edge", "magic", "resonance"],
        "advancement": ["karma", "special attribute points", "attribute points"],
        "priority_categories": ["metatype", "attributes", "magic or resonance", "skills", "resources"]
    },

    # COMBAT (verified from CoreRules-5-Combat.md)
    "combat": {
        "combat_actions": ["attack", "defense", "full defense", "dodge", "parry", "block",
                           "interrupt action", "complex action", "simple action", "free action"],
        "defense_types": ["reaction + intuition", "full defense", "dodge", "parry", "block"],
        "initiative": ["initiative score", "initiative dice", "combat turn", "action phase"],
        "damage_types": ["physical damage", "stun damage", "matrix damage", "biofeedback"],
        "modifiers": ["wound modifier", "reach", "cover", "armor penetration"],
        "defense_modifiers": ["+3 moving vehicle", "-2 prone", "-1 previous attack", "+2 good cover"],
        "armor_mechanics": ["armor rating", "damage resistance", "ap modifier"]
    },

    # SOCIAL (verified from CoreRules-10-Helps and Hindrances.md)
    "social": {
        "contact_mechanics": ["connection rating", "loyalty rating", "favor rating"],
        "contact_services": ["legwork", "networking", "swag", "favors"],
        "social_tests": ["negotiation + charisma", "etiquette + charisma", "con + charisma"],
        "social_skills": ["negotiation", "etiquette", "con", "intimidation", "leadership"],
        "contact_ratings": ["connection 1-12", "loyalty 1-6", "favor rating 1-6"],
        "availability_tests": ["charisma + negotiation", "social limit", "availability rating"]
    },

    # SKILLS (verified references scattered across files)
    "skills": {
        "skill_mechanics": ["dice pool", "threshold", "extended test", "opposed test",
                            "teamwork test", "specialization", "skill group", "defaulting"],
        "magic_skills": ["spellcasting", "counterspelling", "summoning", "binding", "banishing",
                         "alchemy", "artificing", "disenchanting"],
        "resonance_skills": ["compiling", "registering"],  # Found in technomancer example
        "social_skills": ["negotiation", "etiquette", "con", "intimidation", "leadership"],
        "test_types": ["success test", "opposed test", "extended test", "teamwork test"]
    },

    # SETTING/GEAR (limited verified content)
    "setting": {
        "availability_system": ["restricted", "forbidden", "legal", "availability rating"],
        "delivery_times": ["6 hours", "1 day", "2 days", "1 week", "1 month"],
        "fencing": ["fence gear", "availability test", "delivery time"]
    }
}


# Enhanced detection functions using ONLY verified content
def detect_verified_magic_content(text_lower: str) -> dict:
    """Detect magic content using only verified spell names and mechanics."""
    magic_indicators = {
        "spells_found": [],
        "mechanics_found": [],
        "skills_found": [],
        "confidence_score": 0.0
    }

    patterns = VERIFIED_SHADOWRUN_PATTERNS["magic"]

    # Check for specific spells (high confidence)
    for spell in patterns["specific_spells"]:
        if spell in text_lower:
            magic_indicators["spells_found"].append(spell)
            magic_indicators["confidence_score"] += 0.3

    # Check for magic mechanics
    for mechanic in patterns["spell_mechanics"]:
        if mechanic in text_lower:
            magic_indicators["mechanics_found"].append(mechanic)
            magic_indicators["confidence_score"] += 0.2

    # Check for magic skills
    for skill in patterns["magic_skills"]:
        if skill in text_lower:
            magic_indicators["skills_found"].append(skill)
            magic_indicators["confidence_score"] += 0.25

    return magic_indicators


def detect_verified_rigger_content(text_lower: str) -> dict:
    """Detect rigger content using only verified drone names and mechanics."""
    rigger_indicators = {
        "drones_found": [],
        "gear_found": [],
        "programs_found": [],
        "confidence_score": 0.0
    }

    patterns = VERIFIED_SHADOWRUN_PATTERNS["riggers"]

    # Check for specific drones (high confidence)
    for drone in patterns["drone_types"]:
        if drone in text_lower:
            rigger_indicators["drones_found"].append(drone)
            rigger_indicators["confidence_score"] += 0.3

    # Check for rigger gear
    for gear in patterns["rigger_gear"]:
        if gear in text_lower:
            rigger_indicators["gear_found"].append(gear)
            rigger_indicators["confidence_score"] += 0.25

    # Check for rigger programs
    for program in patterns["rigger_programs"]:
        if program in text_lower:
            rigger_indicators["programs_found"].append(program)
            rigger_indicators["confidence_score"] += 0.2

    return rigger_indicators


def detect_verified_combat_content(text_lower: str) -> dict:
    """Detect combat content using verified mechanics from your files."""
    combat_indicators = {
        "actions_found": [],
        "initiative_found": [],
        "damage_found": [],
        "confidence_score": 0.0
    }

    patterns = VERIFIED_SHADOWRUN_PATTERNS["combat"]

    # Check for combat actions
    for action in patterns["combat_actions"]:
        if action in text_lower:
            combat_indicators["actions_found"].append(action)
            combat_indicators["confidence_score"] += 0.2

    # Check for initiative terms
    for term in patterns["initiative"]:
        if term in text_lower:
            combat_indicators["initiative_found"].append(term)
            combat_indicators["confidence_score"] += 0.25

    # Check for damage types
    for damage in patterns["damage_types"]:
        if damage in text_lower:
            combat_indicators["damage_found"].append(damage)
            combat_indicators["confidence_score"] += 0.2

    return combat_indicators


def detect_verified_social_content(text_lower: str) -> dict:
    """Detect social content using verified contact mechanics."""
    social_indicators = {
        "contacts_found": [],
        "tests_found": [],
        "services_found": [],
        "confidence_score": 0.0
    }

    patterns = VERIFIED_SHADOWRUN_PATTERNS["social"]

    # Check for contact mechanics
    for mechanic in patterns["contact_mechanics"]:
        if mechanic in text_lower:
            social_indicators["contacts_found"].append(mechanic)
            social_indicators["confidence_score"] += 0.3

    # Check for social tests
    for test in patterns["social_tests"]:
        if test in text_lower:
            social_indicators["tests_found"].append(test)
            social_indicators["confidence_score"] += 0.25

    # Check for contact services
    for service in patterns["contact_services"]:
        if service in text_lower:
            social_indicators["services_found"].append(service)
            social_indicators["confidence_score"] += 0.2

    return social_indicators


def detect_verified_character_creation_content(text_lower: str) -> dict:
    """Detect character creation content using verified priority system."""
    chargen_indicators = {
        "priorities_found": [],
        "metatypes_found": [],
        "attributes_found": [],
        "confidence_score": 0.0
    }

    patterns = VERIFIED_SHADOWRUN_PATTERNS["character_creation"]

    # Check for priority system
    for priority in patterns["priority_system"]:
        if priority in text_lower:
            chargen_indicators["priorities_found"].append(priority)
            chargen_indicators["confidence_score"] += 0.3

    # Check for metatypes
    for metatype in patterns["metatypes"]:
        if metatype in text_lower:
            chargen_indicators["metatypes_found"].append(metatype)
            chargen_indicators["confidence_score"] += 0.25

    # Check for attributes
    for attr in patterns["attributes"]:
        if attr in text_lower:
            chargen_indicators["attributes_found"].append(attr)
            chargen_indicators["confidence_score"] += 0.15

    return chargen_indicators


def detect_verified_skills_content(text_lower: str) -> dict:
    """Detect skills content using verified mechanics."""
    skills_indicators = {
        "mechanics_found": [],
        "tests_found": [],
        "skills_found": [],
        "confidence_score": 0.0
    }

    patterns = VERIFIED_SHADOWRUN_PATTERNS["skills"]

    # Check for skill mechanics (high confidence indicators)
    for mechanic in patterns["skill_mechanics"]:
        if mechanic in text_lower:
            skills_indicators["mechanics_found"].append(mechanic)
            skills_indicators["confidence_score"] += 0.25

    # Check for test types
    for test_type in patterns["test_types"]:
        if test_type in text_lower:
            skills_indicators["tests_found"].append(test_type)
            skills_indicators["confidence_score"] += 0.2

    # Check for specific skills
    all_skills = patterns["magic_skills"] + patterns["resonance_skills"] + patterns["social_skills"]
    for skill in all_skills:
        if skill in text_lower:
            skills_indicators["skills_found"].append(skill)
            skills_indicators["confidence_score"] += 0.15

    return skills_indicators


# Factory function for comprehensive detection
def create_verified_detector_set():
    """Create a set of all verified content detectors."""
    return {
        "Magic": detect_verified_magic_content,
        "Riggers": detect_verified_rigger_content,
        "Combat": detect_verified_combat_content,
        "Social": detect_verified_social_content,
        "Character_Creation": detect_verified_character_creation_content,
        "Skills": detect_verified_skills_content
    }