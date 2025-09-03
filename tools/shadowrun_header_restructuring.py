#!/usr/bin/env python3
"""
Complete Shadowrun Header Restructuring Script with nested TOC support,
section context awareness, and false header removal for document coherence.
"""

import re
import json
from typing import Dict, List, Tuple, Optional, Set
from pathlib import Path


class NestedShadowrunHeaderFixer:
    def __init__(self, nested_toc_file: str = None, nested_toc_dict: Dict = None):
        """
        Initialize with either a nested TOC file or dictionary.
        """
        if nested_toc_file:
            with open(nested_toc_file, 'r', encoding='utf-8') as f:
                self.nested_toc = json.load(f)
        elif nested_toc_dict:
            self.nested_toc = nested_toc_dict
        else:
            raise ValueError("Must provide either nested_toc_file or nested_toc_dict")

        # Common variations and typos to normalize
        self.title_normalizations = {
            "SKILLS": "Skills",
            "SOCIAL SKILLS": "Social Skills",
            "THE MATRIX": "The Matrix",
            "MATRIX BASICS": "Matrix Basics",
            "WIRELESS WORLD": "Wireless World",
            "CRACKING THE MATRIX SPINE": "Cracking The Matrix Spine",
            "TECHNOMANCERS": "Technomancers",
            "(MIS)USING THE MATRIX": "(Mis)Using The Matrix",
            "PROGRAMS": "Programs",
            "HOSTS": "Hosts",
        }

        # Build lookup structures
        self.section_mapping = self._build_section_mapping()
        self.title_to_level = self._build_title_level_mapping()
        self.valid_titles = self._get_all_valid_titles()

    def _build_section_mapping(self) -> Dict[str, str]:
        """Build mapping of all titles to their top-level section."""
        mapping = {}

        def traverse(structure: Dict[str, any], parent_section: str = None):
            for title, data in structure.items():
                if isinstance(data, dict) and "level" in data:
                    if data["level"] == 1:
                        parent_section = title
                        mapping[title] = title
                    else:
                        mapping[title] = parent_section

                    if data.get("subsections"):
                        traverse(data["subsections"], parent_section)
                else:
                    traverse(data, parent_section)

        traverse(self.nested_toc.get("structure", {}))
        return mapping

    def _build_title_level_mapping(self) -> Dict[str, int]:
        """Build mapping of all titles to their levels."""
        mapping = {}

        def traverse(structure: Dict[str, any]):
            for title, data in structure.items():
                if isinstance(data, dict) and "level" in data:
                    mapping[title] = data["level"]
                    if data.get("subsections"):
                        traverse(data["subsections"])
                else:
                    traverse(data)

        traverse(self.nested_toc.get("structure", {}))
        return mapping

    def _get_all_valid_titles(self) -> Set[str]:
        """Get set of all valid titles from the nested TOC"""
        titles = set()

        def collect_titles(structure: Dict[str, any]):
            for title, data in structure.items():
                if isinstance(data, dict) and "level" in data:
                    titles.add(title)
                    titles.add(self.normalize_title(title))  # Add normalized version too
                    if data.get("subsections"):
                        collect_titles(data["subsections"])
                else:
                    collect_titles(data)

        collect_titles(self.nested_toc.get("structure", {}))
        return titles

    def determine_section_context(self, filename: str) -> Optional[str]:
        """
        Determine which main section this file represents based on filename.
        Returns the main section name to constrain title lookups.
        """
        filename_lower = filename.lower()
        print(filename_lower)

        # Define patterns for different sections
        section_mapping = {
            "11": "Street Gear",
            "street": "Street Gear",
            "gear": "Street Gear",
        }

        for pattern, section_name in section_mapping.items():
            if pattern in filename_lower:
                # Verify this section actually exists in our TOC
                if section_name in self.nested_toc.get("structure", {}):
                    return section_name

        return None

    def find_header_level_in_context(self, title: str, section_context: str = None) -> Optional[int]:
        """
        Find the correct header level for a title, optionally constrained by section context.
        """
        normalized = self.normalize_title(title)

        # If we have section context, search within that section's hierarchy first
        if section_context:
            # Get the actual section name from our mapping (handles variations)
            actual_section = None
            for toc_section in self.nested_toc.get("structure", {}).keys():
                if section_context.lower() in toc_section.lower():
                    actual_section = toc_section
                    break

            if actual_section:
                context_level = self._search_in_section(normalized, actual_section)
                if context_level is not None:
                    return context_level

        # Fallback to global search
        return self.title_to_level.get(normalized)

    def _search_in_section(self, title: str, section_name: str) -> Optional[int]:
        """Search for a title within a specific section's hierarchy."""
        section_data = self.nested_toc["structure"].get(section_name)
        if not section_data:
            return None

        def search_recursive(structure: Dict[str, any]) -> Optional[int]:
            for subtitle, data in structure.items():
                if isinstance(data, dict) and "level" in data:
                    # Normalize both titles for comparison
                    normalized_subtitle = self.normalize_title(subtitle)
                    normalized_target = self.normalize_title(title)

                    # Check if this is our target title
                    if normalized_subtitle == normalized_target:
                        return data["level"]

                    # Recursively search subsections
                    if data.get("subsections"):
                        result = search_recursive(data["subsections"])
                        if result:
                            return result
            return None

        # Check if it's the section title itself
        if self.normalize_title(section_name) == self.normalize_title(title):
            return section_data.get("level", 1)

        # Search in subsections
        if section_data.get("subsections"):
            return search_recursive(section_data["subsections"])

        return None

    def normalize_title(self, title: str) -> str:
        """Normalize title casing and common variations"""
        title = title.strip()

        # Handle common variations first
        if title in self.title_normalizations:
            return self.title_normalizations[title]

        # Handle special cases
        special_cases = {
            "THE MATRIX": "The Matrix",
            "PANS AND WANS": "PANs and WANs",
            "IC": "IC",
            "MATRIX BASICS": "Matrix Basics",
            "WIRELESS WORLD": "Wireless World",
            # Add more as needed
        }

        upper_title = title.upper()
        if upper_title in special_cases:
            return special_cases[upper_title]

        # Default: title case for everything else
        # But be careful with words that should stay capitalized
        words = title.split()
        normalized_words = []
        for word in words:
            if word.upper() in ['II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X']:
                normalized_words.append(word.upper())  # Roman numerals
            elif len(word) > 3 or word.upper() in ['AND', 'THE', 'FOR', 'WITH']:
                normalized_words.append(word.capitalize())
            else:
                normalized_words.append(word.lower())

        return ' '.join(normalized_words)

    def extract_existing_headers(self, content: str) -> List[Tuple[str, str, int]]:
        """Extract only markdown headers (lines starting with #)"""
        headers = []
        lines = content.split('\n')

        for i, line in enumerate(lines):
            line = line.strip()

            # Only look for actual markdown headers
            match = re.match(r'^(#{1,6})\s*(.+?)$', line)
            if match:
                title = match.group(2).strip()
                headers.append((line, title, i))

        return headers

    def fix_headers(self, content: str, filename: str = "") -> str:
        """Fix all headers in the content based on nested TOC structure"""
        section_context = self.determine_section_context(filename)

        if section_context:
            print(f"Processing file with section context: {section_context}")
        else:
            print(f"Processing file without specific section context")

        lines = content.split('\n')
        headers = self.extract_existing_headers(content)
        processed_lines = set()

        for original_line, title, line_num in headers:
            if line_num in processed_lines:
                continue

            normalized_title = self.normalize_title(title)
            print(f"Checking header: '{title}' -> '{normalized_title}' at line {line_num + 1}")

            # Check if this title exists in our TOC
            if normalized_title in self.valid_titles:
                # This is a valid header - fix its level
                correct_level = self.find_header_level_in_context(title, section_context)

                if correct_level:
                    new_header = '#' * correct_level + ' ' + normalized_title

                    if lines[line_num].strip() != new_header:
                        context_info = f" (in {section_context})" if section_context else ""
                        print(f"Fixing header at line {line_num + 1}{context_info}:")
                        print(f"  OLD: {lines[line_num]}")
                        print(f"  NEW: {new_header}")
                        lines[line_num] = new_header
                else:
                    print(f"Warning: Valid title '{title}' found but could not determine level at line {line_num + 1}")
            else:
                # This is a false header - remove the markdown formatting
                print(f"Removing false header: '{title}' at line {line_num + 1}")
                print(f"  OLD: {lines[line_num]}")
                print(f"  NEW: {title}")
                lines[line_num] = title  # Just use the text without #

            processed_lines.add(line_num)

        return '\n'.join(lines)

    def process_file(self, input_file: str, output_file: str = None):
        """Process a single markdown file"""
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                content = f.read()

            print(f"\nProcessing: {input_file}")
            fixed_content = self.fix_headers(content, Path(input_file).name)

            output_path = output_file if output_file else input_file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(fixed_content)

            print(f"✓ Headers fixed and saved to: {output_path}")

        except Exception as e:
            print(f"✗ Error processing {input_file}: {e}")

    def debug_section_context(self, filename: str):
        """Debug section context determination"""
        context = self.determine_section_context(filename)
        print(f"Filename: {filename}")
        print(f"Determined context: {context}")

        if context:
            print(f"Available titles in {context}:")
            section_data = self.nested_toc["structure"].get(context, {})
            if section_data.get("subsections"):
                self._print_titles_in_section(section_data["subsections"])

        return context

    def _print_titles_in_section(self, structure: Dict[str, any], indent: int = 0):
        """Recursively print titles in a section"""
        for title, data in structure.items():
            if isinstance(data, dict) and "level" in data:
                print("  " * indent + f"{title} (Level {data['level']})")
                if data.get("subsections"):
                    self._print_titles_in_section(data["subsections"], indent + 1)


def main():
    """Main execution function"""
    # You'll need to provide the nested TOC file
    nested_toc_file = "data/processed_markdown_cleaned/toc.json"  # Generated by nested_toc_extractor.py

    try:
        fixer = NestedShadowrunHeaderFixer(nested_toc_file=nested_toc_file)

        # Example usage
        test_files = [
            #"data/processed_markdown_cleaned/Shadowrun5thEdition-CoreRules-1-Life in the Sixth World.md",
            #"data/processed_markdown_cleaned/Shadowrun5thEdition-CoreRules-2-Shadowrun Concepts.md",
            #"data/processed_markdown_cleaned/Shadowrun5thEdition-CoreRules-3-Creating A Shadowrunner.md",
            #"data/processed_markdown_cleaned/Shadowrun5thEdition-CoreRules-4-Skills.md",
            #"data/processed_markdown_cleaned/Shadowrun5thEdition-CoreRules-5-Combat.md",
            #"data/processed_markdown_cleaned/Shadowrun5thEdition-CoreRules-6-The Matrix.md",
            #"data/processed_markdown_cleaned/Shadowrun5thEdition-CoreRules-7-Riggers.md",
            #"data/processed_markdown_cleaned/Shadowrun5thEdition-CoreRules-8-Magic.md",
            #"data/processed_markdown_cleaned/Shadowrun5thEdition-CoreRules-9-Gamemaster advice.md",
            #"data/processed_markdown_cleaned/Shadowrun5thEdition-CoreRules-10-Helps and Hindrances.md",
            "data/processed_markdown_cleaned/Shadowrun5thEdition-CoreRules-11-Street_Gear.md"
        ]

        for file_path in test_files:
            try:
                fixer.debug_section_context(file_path)
                fixer.process_file(file_path)
            except FileNotFoundError:
                print(f"File not found: {file_path}")
            except Exception as e:
                print(f"Error with {file_path}: {e}")

    except FileNotFoundError:
        print(f"Nested TOC file not found: {nested_toc_file}")
        print("Generate it first using: python nested_toc_extractor.py your_pdf.pdf")


if __name__ == "__main__":
    main()