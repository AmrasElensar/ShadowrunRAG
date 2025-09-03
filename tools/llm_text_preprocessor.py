"""
LLM-based text preprocessor for cleaning and restructuring extracted PDF content.
Integrates with existing PDFProcessor pipeline.
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Optional, Dict, Any
import ollama
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for LLM preprocessing."""
    model_name: str = "qwen2.5-coder"
    chunk_size: int = 2000  # Process in larger chunks for context
    timeout: int = 60
    max_retries: int = 3
    debug_mode: bool = False


class LLMTextPreprocessor:
    """Clean and restructure extracted text using local LLM."""

    def __init__(self, config: PreprocessingConfig = None, toc_data: list = None):
        self.config = config or PreprocessingConfig()
        self.model_name = self.config.model_name
        self.toc_data = toc_data or []  # TOC structure from PDF extraction

        # Initialize connection
        self._ensure_model_available()

        # Enhanced prompt with TOC context
        self.preprocessing_prompt = """You are an expert markdown document cleaner.

    REMOVE:
    1. All HTML tags: <span>, <div>, </span>, etc.
    2. Page references: "(p. 123)", "see p. 45", "**p. 123**"
    3. Remove not needed empty lines in text blocks
    CRITICAL: 
    - DO NOT ADD TEXT, REWRITE EXISTING TEXT OR REMOVE ANY OTHER TEXT THEN WHAT IS SPECIFIED ABOVE.

    Original text:
    {text}

    Return ONLY the cleaned text. No explanations, no metadata, just the corrected content."""

        self.section_detection_prompt = """Analyze this Shadowrun content and identify what section it belongs to and what type of content it is.

Content:
{text}

Respond with JSON only:
{{
    "section": "Combat|Magic|Matrix|Skills|Gear|Character_Creation|General",
    "content_type": "rules|examples|tables|narrative|headers",
    "topic": "brief description of main topic"
}}"""

    def _ensure_model_available(self):
        """Check if model is available and suggest alternatives."""
        try:
            models_response = ollama.list()
            available_models = []

            for model in models_response.get('models', []):
                model_name = (
                        model.get('name') or
                        model.get('model') or
                        model.get('id') or
                        str(model)
                )
                if model_name:
                    available_models.append(model_name)

            if self.model_name not in available_models:
                logger.warning(f"Model {self.model_name} not found")

                # Suggest alternatives
                alternatives = [m for m in available_models if 'qwen' in m.lower() or 'llama' in m.lower()]
                if alternatives:
                    self.model_name = alternatives[0]
                    logger.info(f"Using alternative model: {self.model_name}")
                else:
                    logger.error(f"No suitable models found. Available: {available_models}")
                    raise ValueError("No compatible LLM models available")

        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            raise

    def preprocess_file(self, input_file: Path, output_file: Path = None) -> Path:
        """Preprocess a single markdown file."""
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        if output_file is None:
            output_file = input_file.parent / f"{input_file.stem}_cleaned{input_file.suffix}"

        logger.info(f"Preprocessing {input_file.name} -> {output_file.name}")

        # Read input
        content = input_file.read_text(encoding='utf-8')

        # Process in chunks
        cleaned_content = self._process_content(content, str(input_file))

        # Write output
        output_file.write_text(cleaned_content, encoding='utf-8')

        logger.info(f"Cleaned content saved to {output_file}")
        return output_file

    def preprocess_directory(self, input_dir: Path, output_dir: Path = None) -> Dict[str, Path]:
        """Preprocess all markdown files in directory."""
        input_dir = Path(input_dir)
        if output_dir is None:
            output_dir = input_dir.parent / f"{input_dir.name}_cleaned"

        output_dir.mkdir(exist_ok=True)

        markdown_files = list(input_dir.glob("*.md"))
        if not markdown_files:
            logger.warning(f"No markdown files found in {input_dir}")
            return {}

        results = {}

        for md_file in markdown_files:
            try:
                output_file = output_dir / md_file.name
                cleaned_file = self.preprocess_file(md_file, output_file)
                results[str(md_file)] = cleaned_file

            except Exception as e:
                logger.error(f"Failed to preprocess {md_file}: {e}")

        logger.info(f"Preprocessed {len(results)} files successfully")
        return results

    def _process_content(self, content: str, source: str) -> str:
        """Process content using TOC-aware splitting."""

        # Quick cleanup first
        content = self._quick_cleanup(content)

        # Check content size and decide on processing approach
        estimated_tokens = len(content.split()) * 1.3  # Rough token estimation

        if estimated_tokens < 30000:  # Process as single chunk
            logger.info(f"Processing as single chunk (~{int(estimated_tokens)} tokens)")
            return self._clean_section_with_toc(content, self.toc_data)

        else:  # Split using TOC boundaries
            logger.info(f"Content too large (~{int(estimated_tokens)} tokens), splitting by TOC")
            return self._process_with_toc_splitting(content)

    def _process_with_toc_splitting(self, content: str) -> str:
        """Split content using TOC boundaries and process each section."""

        if not self.toc_data:
            logger.warning("No TOC data available, falling back to header-based splitting")
            return self._fallback_splitting(content)

        # Find TOC sections in the content
        toc_sections = self._find_toc_sections_in_content(content)

        if not toc_sections:
            logger.warning("Could not match TOC to content, falling back to header splitting")
            return self._fallback_splitting(content)

        cleaned_sections = []

        for i, section in enumerate(toc_sections):
            section_title = section['title']
            section_content = section['content']
            section_toc = section.get('subsections', [])

            logger.info(
                f"Processing TOC section {i + 1}/{len(toc_sections)}: '{section_title}' ({len(section_content)} chars)")

            # Clean this section with its specific TOC context
            cleaned_section = self._clean_section_with_toc(section_content, section_toc, section_title)
            cleaned_sections.append(cleaned_section)

            # Brief pause between sections
            time.sleep(1.0)

        return '\n\n'.join(cleaned_sections)

    def _find_toc_sections_in_content(self, content: str) -> list:
        """Find TOC section boundaries in the actual content."""

        sections = []
        content_lines = content.split('\n')

        # Filter to level 2 sections only
        level_2_toc = [toc for toc in self.toc_data if toc.get('level') == 2]

        # Create mapping of TOC titles to their positions in content
        toc_positions = []
        for toc_entry in level_2_toc:
            title = toc_entry.get('title', '').strip()
            if not title:
                continue

            # Look for this title in the content (fuzzy matching)
            position = self._find_title_in_content(title, content_lines)
            if position is not None:
                toc_positions.append({
                    'title': title,
                    'position': position,
                    'level': toc_entry.get('level', 1),
                    'toc_entry': toc_entry
                })

        # Sort by position
        toc_positions.sort(key=lambda x: x['position'])

        # Extract sections between TOC boundaries
        for i, toc_pos in enumerate(toc_positions):
            start_pos = toc_pos['position']

            # Find end position (next major section or end of content)
            if i < len(toc_positions) - 1:
                # Look for next section of same or higher level
                current_level = toc_pos['level']
                end_pos = None

                for next_toc in toc_positions[i + 1:]:
                    if next_toc['level'] <= current_level:
                        end_pos = next_toc['position']
                        break

                if end_pos is None:
                    end_pos = len(content_lines)
            else:
                end_pos = len(content_lines)

            # Extract section content
            section_lines = content_lines[start_pos:end_pos]
            section_content = '\n'.join(section_lines).strip()

            if len(section_content) > 100:  # Only include substantial sections

                # Find subsections within this section
                subsections = []
                for toc_entry in self.toc_data:
                    if (toc_entry.get('level', 1) > toc_pos['level'] and
                            self._title_appears_in_text(toc_entry.get('title', ''), section_content)):
                        subsections.append(toc_entry)

                sections.append({
                    'title': toc_pos['title'],
                    'content': section_content,
                    'level': toc_pos['level'],
                    'subsections': subsections
                })

        return sections

    def _find_title_in_content(self, title: str, content_lines: list) -> Optional[int]:
        """Find where a TOC title appears in the content lines."""

        # Clean title for comparison
        clean_title = re.sub(r'[^\w\s]', '', title.lower()).strip()

        for i, line in enumerate(content_lines):
            clean_line = re.sub(r'[^\w\s]', '', line.lower()).strip()

            # Direct match
            if clean_title in clean_line:
                return i

            # Fuzzy match for partial titles
            if len(clean_title) > 5:
                title_words = clean_title.split()
                if len(title_words) > 1:
                    # Check if most words from title appear in line
                    matching_words = sum(1 for word in title_words if word in clean_line)
                    if matching_words >= len(title_words) * 0.7:  # 70% of words match
                        return i

        return None

    def _title_appears_in_text(self, title: str, text: str) -> bool:
        """Check if a title appears anywhere in the text."""
        clean_title = re.sub(r'[^\w\s]', '', title.lower()).strip()
        clean_text = re.sub(r'[^\w\s]', '', text.lower()).strip()
        return clean_title in clean_text

    def _clean_section_with_toc(self, text: str, toc_context: list = None, section_title: str = "") -> str:
        """Clean a section with TOC structure guidance."""

        estimated_tokens = len(text.split()) * 1.3
        logger.info(f"Processing section: '{section_title}' (~{int(estimated_tokens)} tokens)")
        logger.info(f"TOC context items: {len(toc_context) if toc_context else 0}")

        # Build TOC context for the prompt
        toc_context_str = ""
        if toc_context:
            toc_context_str = f"""
            DOCUMENT STRUCTURE CONTEXT:
            This content is part of the "{section_title}" section. The expected subsections based on the table of contents are:
            """
            for toc_entry in toc_context[:10]:  # Limit to avoid prompt bloat
                level_indent = "  " * (toc_entry.get('level', 1) - 1)
                toc_context_str += f"{level_indent}- {toc_entry.get('title', 'Untitled')}\n"

            toc_context_str += "\nUse this structure to organize headers appropriately.\n"

        for attempt in range(self.config.max_retries):
            try:
                prompt = self.preprocessing_prompt.format(
                    text=text,
                    toc_context=toc_context_str
                )

                response = ollama.generate(
                    model=self.model_name,
                    prompt=prompt,
                    options={
                        'num_ctx': 32768,  # Reduced but still generous context
                    },
                    stream=False
                )

                cleaned = response['response'].strip()

                # Check if too short first (might need retry)
                if len(cleaned) < len(text) * 0.3:
                    logger.warning(f"LLM output too short, attempt {attempt + 1}")
                    continue

                # Check if too long (definite failure, return original)
                if len(cleaned) > len(text) * 1.05:
                    logger.error(f"Model added content: {len(cleaned)} vs {len(text)}")
                    logger.error(f"Excess content preview: {cleaned[len(text):len(text) + 200]}")
                    return text

                # Log success metrics only for valid output
                logger.info(f"Model returned {len(cleaned)} chars vs input {len(text)} chars")
                logger.info(f"First 200 chars of output: {repr(cleaned[:200])}")

                return cleaned

            except Exception as e:
                logger.warning(f"LLM cleaning attempt {attempt + 1} failed: {e}")
                if attempt == self.config.max_retries - 1:
                    logger.error(f"All attempts failed, returning original")
                return text

        time.sleep(2 ** attempt)

        return text

    def _fallback_splitting(self, content: str) -> str:
        """Fallback to simple header-based splitting when TOC is not available."""

        # Split by major headers only (single # at start of line)
        sections = re.split(r'\n(?=# [^#])', content)

        cleaned_sections = []
        for i, section in enumerate(sections):
            if len(section.strip()) < 100:
                cleaned_sections.append(section)
                continue

            logger.info(f"Processing fallback section {i + 1}/{len(sections)} ({len(section)} chars)")
            cleaned_section = self._clean_section_with_toc(section)
            cleaned_sections.append(cleaned_section)
            time.sleep(0.5)

        return '\n'.join(cleaned_sections)

    def _quick_cleanup(self, text: str) -> str:
        """Quick regex-based cleanup before LLM processing."""

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        # Fix common OCR artifacts
        # Fix common OCR artifacts
        replacements = {
            'â€"': '—',  # Em dash
            'â€™': "'",  # Right single quote
            'â€œ': '"',  # Left double quote
            'â€': '"',  # Right double quote
            'â€¢': '•',  # Bullet point
            'â†': '→',
            'â€˜': "'",  # Left single quote
            'â€¦': '...',  # Ellipsis
            'âˆ': ' - ',  # Minus sign
            'Â ': ' ',  # Non-breaking space that got mangled
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        # Clean up excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)

        return text

    def _split_by_sections(self, text: str) -> list:
        """Split text by headers while preserving context."""

        # Find section boundaries (# headers)
        lines = text.split('\n')
        sections = []
        current_section = []

        for line in lines:
            if line.strip().startswith('#') and not line.strip().startswith('##'):
                # Save previous section
                if current_section:
                    sections.append('\n'.join(current_section))
                    current_section = []

            current_section.append(line)

        # Add final section
        if current_section:
            sections.append('\n'.join(current_section))

        return sections

    def _detect_section_type(self, text: str) -> Dict[str, str]:
        """Use LLM to detect section type for better processing."""

        # Use first 500 chars for detection
        sample = text[:500]

        try:
            prompt = self.section_detection_prompt.format(text=sample)

            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={'temperature': 0.7, 'num_predict': 100, 'top_p': 0.8, 'top_k': 20, 'min_p': 0},
                stream=False
            )

            result = json.loads(response['response'].strip())
            return result

        except Exception as e:
            logger.warning(f"Section detection failed: {e}")
            return {
                "section": "General",
                "content_type": "rules",
                "topic": "unknown"
            }

# Integration function for PDFProcessor
def create_preprocessor(model_name: str = "qwen2.5-coder", debug_mode: bool = False, toc_data: list = None) -> LLMTextPreprocessor:
    """Factory function to create preprocessor with optional TOC data."""
    config = PreprocessingConfig(
        model_name=model_name,
        debug_mode=debug_mode
    )
    return LLMTextPreprocessor(config, toc_data)


# CLI interface
def main():
    """Command line interface."""
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess extracted PDF text with LLM")
    parser.add_argument("input", help="Input file or directory")
    parser.add_argument("-o", "--output", help="Output file or directory")
    parser.add_argument("-m", "--model", default="qwen2.5:14b-instruct", help="LLM model to use")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create preprocessor
    preprocessor = create_preprocessor(args.model, args.debug)

    input_path = Path(args.input)

    if input_path.is_file():
        # Single file
        output_path = Path(args.output) if args.output else None
        result = preprocessor.preprocess_file(input_path, output_path)
        print(f"Cleaned file: {result}")

    elif input_path.is_dir():
        # Directory
        output_dir = Path(args.output) if args.output else None
        results = preprocessor.preprocess_directory(input_path, output_dir)
        print(f"Processed {len(results)} files:")
        for original, cleaned in results.items():
            print(f"  {original} -> {cleaned}")

    else:
        print(f"Error: {input_path} is not a valid file or directory")


if __name__ == "__main__":
    main()