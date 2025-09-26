"""
Improved Shadowrun entity detection patterns based on actual MD file analysis.
Replaces verified_shadowrun_patterns.py with more comprehensive and flexible patterns.
"""

import re

# SIGNIFICANTLY EXPANDED patterns based on actual content analysis

IMPROVED_SHADOWRUN_PATTERNS = {

    # GREATLY EXPANDED MAGIC PATTERNS
    "magic": {
        # All spells found in CoreRules-8-Magic.md
        "specific_spells": [
            # Combat Spells
            "fireball", "manabolt", "lightning bolt", "death touch", "manaball",
            "flamethrower", "ball lightning", "shatter", "powerbolt", "powerball",
            "knockout", "stunbolt", "stunball",

            # Manipulation Spells
            "levitate", "magic fingers", "armor", "control actions", "mob control",
            "control thoughts", "mob mind", "fling", "ice sheet",

            # Environmental/Barrier Spells
            "light", "mana barrier", "physical barrier",

            # Detection Spells (based on context patterns)
            "clairvoyance", "detect magic", "detect life", "analyze truth", "mind probe",
            "combat sense", "danger sense", "enhanced senses",

            # Illusion Spells (common patterns)
            "invisibility", "improved invisibility", "silence", "confusion", "chaos",
            "entertainment", "mass confusion", "phantasm", "trid phantasm",

            # Health Spells (common patterns)
            "heal", "increase reflexes", "increase attribute", "decrease attribute",
            "antidote", "cure disease", "stabilize", "oxygenate"
        ],

        # Spell format indicators (much more flexible)
        "spell_format_patterns": [
            r'\*\*Type:\*\*\s*[MP]\s*;\s*\*\*Range:\*\*',
            r'\*\*Drain:\*\*\s*F\s*[-—–]\s*\d+',
            r'\*\*Duration:\*\*\s*[ISPA]',
            r'Type:\s*[MP]\s*Range:\s*\w+',
            r'Drain:\s*F[-—–]\d+'
        ],

        # Spell mechanics (expanded)
        "spell_mechanics": [
            "force", "drain", "range", "duration", "type", "spellcasting + magic",
            "drain resistance", "astral", "physical", "mana", "net hits",
            "spellcasting test", "magic rating", "threshold", "hits scored",
            "opposed test", "willpower + logic", "body + willpower",
            "sustained", "instantaneous", "permanent", "special",
            "line of sight", "touch", "area effect", "direct spell", "indirect spell"
        ],

        # Spell types and keywords (expanded)
        "spell_types": [
            "direct", "indirect", "elemental", "environmental", "area", "touch",
            "line of sight", "sustained", "detection", "health", "illusion",
            "manipulation", "combat", "physical", "mana", "mental"
        ],

        # Magic skills and terms
        "magic_skills": [
            "spellcasting", "counterspelling", "summoning", "binding", "banishing",
            "alchemy", "artificing", "disenchanting", "arcana", "assensing",
            "astral combat", "ritual spellcasting"
        ],

        # Context indicators for magic content
        "magic_context": [
            "magician", "mage", "shaman", "adept", "mystic adept", "awakened",
            "magic rating", "spell formula", "reagents", "foci", "focus",
            "tradition", "mentor spirit", "astral space", "astral projection",
            "magical lodge", "background count", "mana", "essence"
        ]
    },

    # GREATLY EXPANDED MATRIX/IC PATTERNS
    "matrix": {
        # All IC programs from CoreRules-6-The_Matrix.md
        "ic_programs": [
            "acid", "binder", "black ic", "blaster", "crash", "jammer",
            "killer", "marker", "patrol", "probe", "scramble",
            # Alternative names
            "grey ic", "gray ic", "white ic", "red ic", "blue ic"
        ],

        # IC attack pattern detection (more flexible)
        "ic_attack_patterns": [
            r'\*\*Attack:\*\*.*?v\.',
            r'Attack:\s*Host\s*Rating\s*x\s*2',
            r'Host\s*Rating\s*x\s*2\s*\[Attack\]',
            r'v\.\s*\w+\s*\+\s*\w+',
            r'causes.*?DV.*?damage',
            r'reduces?\s*your\s*\w+\s*by\s*\d+'
        ],

        # Matrix attributes and mechanics
        "matrix_attributes": [
            "attack", "sleaze", "data processing", "firewall", "device rating",
            "host rating", "matrix damage", "biofeedback", "dumpshock",
            "link-lock", "bricking", "marks", "overwatch score"
        ],

        # Matrix actions and terms
        "matrix_actions": [
            "hack on the fly", "brute force", "data spike", "crash program",
            "matrix perception", "hide", "trace user", "full matrix defense",
            "jack out", "reboot", "spoof command", "control device",
            "crack file", "edit file", "format device", "snoop", "trace icon"
        ],

        # Cyberdeck and gear
        "matrix_gear": [
            "cyberdeck", "commlink", "rcc", "rigger command console",
            "sim module", "trodes", "datajack", "sim rig",
            "erika", "microdeck", "microtronica", "hermes", "novatech",
            "renraku", "sony", "shiawase", "fairlight"
        ],

        # Matrix context indicators
        "matrix_context": [
            "decker", "hacker", "technomancer", "spider", "agent", "sprite",
            "host", "node", "grid", "matrix", "virtual reality", "augmented reality",
            "persona", "icon", "file", "program", "utility", "cybercombat"
        ]
    },

    # WEAPON PATTERNS (expanded but weapons already work well)
    "gear": {
        # Manufacturers (expanded list)
        "manufacturers": [
            "ares", "colt", "ruger", "browning", "remington", "defiance",
            "yamaha", "fichetti", "beretta", "taurus", "steyr", "enfield",
            "cavalier arms", "ranger arms", "pjss", "parashield", "ak-97",
            "mossberg", "winchester", "hk", "heckler koch", "fn", "sig sauer"
        ],

        # Weapon types (expanded)
        "weapon_types": [
            "pistol", "rifle", "shotgun", "smg", "assault rifle", "sniper rifle",
            "machine gun", "crossbow", "heavy pistol", "light pistol",
            "hold-out pistol", "machine pistol", "submachine gun", "carbine",
            "sporting rifle", "hunting rifle", "combat shotgun"
        ],

        # Known specific weapons (expanded from actual content)
        "specific_weapons": [
            "predator", "ares predator", "government 2066", "ultra-power",
            "roomsweeper", "super warhawk", "crusader", "black scorpion",
            "ak-97", "ares alpha", "desert strike", "crockett", "ranger sm-5",
            "fichetti security 600", "browning ultra-power", "colt government",
            "ruger super warhawk", "defiance t-250", "mossberg am-cmdt"
        ],

        # Weapon table detection patterns
        "weapon_table_patterns": [
            r'\|\s*(?:WEAPON|FIREARM|ACC|Accuracy|Damage|AP|Mode|RC|Ammo|Avail|Cost)\s*\|',
            r'\|\s*\w+\s*\|\s*\d+(?:\s*\(\d+\))?\s*\|\s*\d+[PS]',
            r'Acc\s*Damage\s*AP\s*Mode\s*RC\s*Ammo\s*Avail\s*Cost'
        ],

        # Weapon stats and terms
        "weapon_stats": [
            "accuracy", "damage", "ap", "armor penetration", "mode", "rc",
            "recoil compensation", "ammo", "ammunition", "avail", "availability",
            "cost", "single shot", "semi-auto", "burst fire", "full auto"
        ]
    }
}


def create_improved_pattern_detectors():
    """Create detection functions using improved patterns."""

    def detect_spell_content(text: str) -> dict:
        """Detect spell content with improved patterns."""
        text_lower = text.lower()

        # Check for spell format patterns (high confidence)
        format_score = 0
        for pattern in IMPROVED_SHADOWRUN_PATTERNS["magic"]["spell_format_patterns"]:
            if re.search(pattern, text, re.IGNORECASE):
                format_score += 2.0

        # Check for specific spells
        spell_score = 0
        found_spells = []
        for spell in IMPROVED_SHADOWRUN_PATTERNS["magic"]["specific_spells"]:
            if spell in text_lower:
                found_spells.append(spell)
                spell_score += 1.0

        # Check for magic mechanics
        mechanic_score = 0
        found_mechanics = []
        for mechanic in IMPROVED_SHADOWRUN_PATTERNS["magic"]["spell_mechanics"]:
            if mechanic in text_lower:
                found_mechanics.append(mechanic)
                mechanic_score += 0.5

        # Check for magic context
        context_score = 0
        for context in IMPROVED_SHADOWRUN_PATTERNS["magic"]["magic_context"]:
            if context in text_lower:
                context_score += 0.3

        total_score = format_score + spell_score + min(mechanic_score, 5.0) + min(context_score, 3.0)

        return {
            "is_spell_content": total_score >= 2.0,
            "confidence": min(total_score / 10.0, 1.0),
            "found_spells": found_spells[:5],  # Limit output
            "found_mechanics": found_mechanics[:5],
            "format_indicators": format_score > 0,
            "total_score": total_score
        }

    def detect_ic_content(text: str) -> dict:
        """Detect IC content with improved patterns."""
        text_upper = text.upper()
        text_lower = text.lower()

        # Check for IC attack patterns (very high confidence)
        attack_score = 0
        for pattern in IMPROVED_SHADOWRUN_PATTERNS["matrix"]["ic_attack_patterns"]:
            if re.search(pattern, text, re.IGNORECASE):
                attack_score += 3.0

        # Check for specific IC programs
        ic_score = 0
        found_programs = []
        for ic_program in IMPROVED_SHADOWRUN_PATTERNS["matrix"]["ic_programs"]:
            if ic_program.upper() in text_upper:
                found_programs.append(ic_program)
                ic_score += 1.5

        # Check for matrix attributes
        matrix_score = 0
        for attr in IMPROVED_SHADOWRUN_PATTERNS["matrix"]["matrix_attributes"]:
            if attr in text_lower:
                matrix_score += 0.5

        # Check for matrix context
        context_score = 0
        for context in IMPROVED_SHADOWRUN_PATTERNS["matrix"]["matrix_context"]:
            if context in text_lower:
                context_score += 0.3

        total_score = attack_score + ic_score + min(matrix_score, 4.0) + min(context_score, 2.0)

        return {
            "is_ic_content": total_score >= 2.0,
            "confidence": min(total_score / 10.0, 1.0),
            "found_programs": found_programs,
            "attack_patterns": attack_score > 0,
            "total_score": total_score
        }

    def detect_weapon_content(text: str) -> dict:
        """Detect weapon content (existing logic, minor improvements)."""
        text_upper = text.upper()
        text_lower = text.lower()

        # Check for weapon table patterns
        table_score = 0
        for pattern in IMPROVED_SHADOWRUN_PATTERNS["gear"]["weapon_table_patterns"]:
            if re.search(pattern, text, re.IGNORECASE):
                table_score += 3.0

        # Check for specific weapons
        weapon_score = 0
        found_weapons = []
        for weapon in IMPROVED_SHADOWRUN_PATTERNS["gear"]["specific_weapons"]:
            if weapon.lower() in text_lower:
                found_weapons.append(weapon)
                weapon_score += 1.0

        # Check for manufacturers
        mfg_score = 0
        for mfg in IMPROVED_SHADOWRUN_PATTERNS["gear"]["manufacturers"]:
            if mfg.upper() in text_upper:
                mfg_score += 0.5

        total_score = table_score + weapon_score + min(mfg_score, 3.0)

        return {
            "is_weapon_content": total_score >= 1.5,
            "confidence": min(total_score / 8.0, 1.0),
            "found_weapons": found_weapons[:5],
            "table_detected": table_score > 0,
            "total_score": total_score
        }

    return {
        "spell_detector": detect_spell_content,
        "ic_detector": detect_ic_content,
        "weapon_detector": detect_weapon_content
    }


# Update function for use in entity extraction
def create_improved_detector_set():
    """Create the improved detector set for entity extraction."""
    return create_improved_pattern_detectors()