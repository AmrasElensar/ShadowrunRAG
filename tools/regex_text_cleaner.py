#!/usr/bin/env python3
"""
Regex-based text cleaner for extracted PDF content.
Reliable alternative to LLM preprocessing.
"""

import re
import logging
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)


class RegexTextCleaner:
    """Clean extracted text using reliable regex patterns."""

    def __init__(self):
        # OCR artifact replacements
        self.ocr_replacements = {
            'Ã¢â‚¬"': '—',  # Em dash
            'Ã¢â‚¬â„¢': "'",  # Right single quote
            'Ã¢â‚¬Å"': '"',  # Left double quote
            'Ã¢â‚¬': '"',  # Right double quote
            'Ã¢â‚¬Â¢': '•',  # Bullet point
            'Ã¢â€ ': '→',  # Right arrow
            'Ã¢â‚¬Ëœ': "'",  # Left single quote
            'Ã¢â‚¬Â¦': '...',  # Ellipsis
            'Ã¢Ë†': '-',  # Minus sign
            'Ã‚ ': ' ',  # Non-breaking space
        }

    def clean_file(self, input_file: Path, output_file: Path = None) -> Path:
        """Clean a single markdown file."""
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        if output_file is None:
            output_file = input_file.parent / f"{input_file.stem}_cleaned{input_file.suffix}"

        logger.info(f"Cleaning {input_file.name} -> {output_file.name}")

        # Read, clean, write
        content = input_file.read_text(encoding='utf-8')
        cleaned_content = self.clean_text(content)
        output_file.write_text(cleaned_content, encoding='utf-8')

        # Report changes
        reduction = len(content) - len(cleaned_content)
        logger.info(f"Cleaned: {reduction} chars removed, {len(cleaned_content)} remaining")

        return output_file

    def clean_directory(self, input_dir: Path, output_dir: Path = None) -> Dict[str, Path]:
        """Clean all markdown files in directory."""
        input_dir = Path(input_dir)
        if output_dir is None:
            output_dir = input_dir.parent / f"{input_dir.name}_cleaned"

        output_dir.mkdir(exist_ok=True)

        results = {}
        for md_file in input_dir.glob("*.md"):
            output_file = output_dir / md_file.name
            cleaned_file = self.clean_file(md_file, output_file)
            results[str(md_file)] = cleaned_file

        logger.info(f"Cleaned {len(results)} files")
        return results

    def clean_text(self, text: str) -> str:
        """Clean text using regex patterns - simplified approach."""

        # Protect << patterns before HTML cleaning
        text = re.sub(r'<<\s*([^>]+?)\s+(\d+)', r'TEMP_DOUBLE_OPEN \1 \2', text)

        # Your existing HTML tag removal
        text = re.sub(r'<[^>]+>', '', text)

        # Restore protected patterns
        text = re.sub(r'TEMP_DOUBLE_OPEN\s+([^>]+?)\s+(\d+)', r'<< \1 \2', text)

        # 2. Remove page references (consolidated patterns)
        page_patterns = [
            r'\(see\s+\*\*[^*]+\*\*,\s*p\.\s*\d+\)',  # (see **Something**, p. 160)
            r'\([pP]\.\s*\d+\)',  # (p. 123)
            r'\*\*[pP]\.\s*\d+\*\*',  # **p. 123**
            r'[pP]age\s+\d+',  # Page 123
        ]

        for pattern in page_patterns:
            text = re.sub(pattern, '', text)

        # 3. Remove image references
        text = re.sub(r'!\[\]\([^)]+\.(jpeg|jpg|png|gif|bmp)\)', '', text)

        # 4. Remove navigation breadcrumbs (be more specific)
        text = re.sub(r'^\s*\d+\s+[\w\s]+\s*>>\s*$', '', text, flags=re.MULTILINE)

        # Remove cross-reference patterns with escaped asterisks
        text = re.sub(r'\*\s*\\\*\s*See\s+[^,]+,\s*p\.\s*\d+\*', '', text)

        # Remove bold references with page numbers
        text = re.sub(r'\*\*[^*]+\*\*,\s*p\.\s*\d+\.?', '', text)

        # Remove standalone > characters on their own lines
        text = re.sub(r'^\s*>\s*$', '', text, flags=re.MULTILINE)

        # Remove simple page references in parentheses
        text = re.sub(r'\(see\s+p\.\s*\d+\)\.?', '', text)

        # 5. Fix OCR artifacts
        for old, new in self.ocr_replacements.items():
            text = text.replace(old, new)

        # 6. Clean whitespace WITHOUT trying to preserve table alignment
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Remove excess blank lines
        text = re.sub(r'[ \t]+', ' ', text)  # Normalize spaces
        text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)  # Trailing spaces

        return text.strip()


# Integration functions
def create_regex_cleaner() -> RegexTextCleaner:
    """Factory function."""
    return RegexTextCleaner()


def clean_extracted_files(input_dir: str, output_dir: str = None):
    """Clean extracted markdown files."""
    cleaner = create_regex_cleaner()
    input_path = Path(input_dir)
    output_path = Path(output_dir) if output_dir else None

    if input_path.is_file():
        return cleaner.clean_file(input_path, output_path)
    else:
        return cleaner.clean_directory(input_path, output_path)


# CLI interface
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Clean extracted PDF text with regex")
    parser.add_argument("input", help="Input file or directory")
    parser.add_argument("-o", "--output", help="Output file or directory")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    results = clean_extracted_files(args.input, args.output)

    if isinstance(results, Path):
        print(f"Cleaned: {results}")
    else:
        print(f"Cleaned {len(results)} files")


if __name__ == "__main__":
    main()