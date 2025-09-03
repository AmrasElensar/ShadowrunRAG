#!/usr/bin/env python3
"""
Enhanced PDF TOC Extractor that outputs nested dictionary structure
to avoid title conflicts in multi-section documents
"""

import fitz
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional


def extract_nested_toc_from_pdf(pdf_path: str, output_file: str = None) -> Dict[str, Any]:
    """
    Extract table of contents as a nested dictionary structure.

    Returns:
    {
        "document_title": "Full Document Name",
        "structure": {
            "Section Name": {
                "level": 1,
                "page": 10,
                "subsections": {
                    "Subsection Name": {
                        "level": 2,
                        "page": 12,
                        "subsections": {...}
                    }
                }
            }
        }
    }
    """

    doc = fitz.open(pdf_path)
    toc_data = doc.get_toc()
    doc.close()

    if not toc_data:
        print("No TOC found in PDF")
        return {"document_title": Path(pdf_path).stem, "structure": {}}

    # Build nested structure
    nested_toc = {
        "document_title": Path(pdf_path).stem,
        "structure": {}
    }

    # Stack to keep track of current context at each level
    level_stack = [nested_toc["structure"]]  # Start with root

    for level, title, page in toc_data:
        title = title.strip()

        # Adjust stack to current level
        while len(level_stack) > level:
            level_stack.pop()

        # Current container is the last item in stack
        current_container = level_stack[-1]

        # Create entry
        entry = {
            "level": level,
            "page": page,
            "subsections": {}
        }

        # Add to current container
        current_container[title] = entry

        # Push this entry's subsections onto stack for next items
        level_stack.append(entry["subsections"])

    # Save to file if requested
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(nested_toc, f, indent=2, ensure_ascii=False)
        print(f"Nested TOC saved to: {output_file}")

    return nested_toc


def print_nested_structure(structure: Dict[str, Any], indent: int = 0):
    """Pretty print the nested structure"""
    for title, data in structure.items():
        if isinstance(data, dict) and "level" in data:
            prefix = "  " * indent + "#" * data["level"]
            print(f"{prefix} {title} (p.{data['page']})")
            if data.get("subsections"):
                print_nested_structure(data["subsections"], indent + 1)
        elif isinstance(data, dict):
            # Handle nested dictionary (like subsections)
            print_nested_structure(data, indent)
        else:
            # Skip non-dict items (like document_title string)
            continue


def get_section_mapping(nested_toc: Dict[str, Any]) -> Dict[str, str]:
    """
    Generate a mapping of section titles to their top-level parent.
    This helps identify which main section a subtitle belongs to.

    Returns: {"subtitle": "main_section_name", ...}
    """
    mapping = {}

    def traverse(structure: Dict[str, Any], parent_section: str = None):
        for title, data in structure.items():
            if isinstance(data, dict) and "level" in data:
                # If this is a level 1 item, it's a main section
                if data["level"] == 1:
                    parent_section = title
                    mapping[title] = title  # Section maps to itself
                else:
                    # Subsection maps to its main section
                    mapping[title] = parent_section

                # Recurse into subsections
                if data.get("subsections"):
                    traverse(data["subsections"], parent_section)
            else:
                # Handle root level
                traverse(data, parent_section)

    traverse(nested_toc.get("structure", {}))
    return mapping


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python nested_toc_extractor.py <pdf_file> [output_json]")
        sys.exit(1)

    pdf_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else f"{Path(pdf_file).stem}_nested_toc.json"

    print(f"Extracting nested TOC from: {pdf_file}")
    nested_toc = extract_nested_toc_from_pdf(pdf_file, output_file)

    print(f"\nDocument: {nested_toc['document_title']}")
    print("Structure:")
    print_nested_structure(nested_toc["structure"])  # Pass only the structure part

    print(f"\nSection mapping:")
    mapping = get_section_mapping(nested_toc)
    for subtitle, main_section in mapping.items():
        if subtitle != main_section:  # Only show subsections
            print(f"  '{subtitle}' -> '{main_section}'")