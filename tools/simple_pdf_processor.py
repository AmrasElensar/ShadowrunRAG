"""Simplified PDF processor using only TOC-guided extraction."""

import json
import logging
import time
import fitz  # PyMuPDF
from pathlib import Path
from typing import Dict, Optional, Callable, List, Tuple
from tools.regex_text_cleaner import create_regex_cleaner
from tools.pdf_toc_extractor import extract_nested_toc_from_pdf
from tools.shadowrun_header_restructuring import NestedShadowrunHeaderFixer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimplePDFProcessor:
    """Simplified PDF processor using only TOC-guided extraction."""

    def __init__(self, progress_callback: Optional[Callable[[str, float, str], None]] = None):
        self.progress_callback = progress_callback or self._default_progress_callback
        
        # Initialize text cleaner
        self.regex_cleaner = create_regex_cleaner()
        self.header_fixer = NestedShadowrunHeaderFixer()

    def _default_progress_callback(self, stage: str, progress: float, details: str):
        """Default progress callback that just logs."""
        logger.info(f"Progress: {stage} ({progress}%) - {details}")

    def _update_progress(self, stage: str, progress: float, details: str):
        """Update progress with callback."""
        try:
            self.progress_callback(stage, progress, details)
        except Exception as e:
            logger.warning(f"Progress callback failed: {e}")

    def process_pdf(self, pdf_path: str, force_reparse: bool = False) -> List[str]:
        """Process PDF using TOC-guided extraction only."""
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        self._update_progress("init", 5, "Starting TOC-guided extraction...")

        try:
            # Extract using TOC-guided method
            markdown_files = self._extract_with_toc(pdf_path, force_reparse)
            
            if not markdown_files:
                raise Exception("TOC-guided extraction failed to produce any files")

            self._update_progress("complete", 100, f"Successfully created {len(markdown_files)} files")
            return markdown_files

        except Exception as e:
            logger.error(f"PDF processing failed: {e}")
            self._update_progress("error", -1, f"Processing failed: {str(e)}")
            raise

    def _extract_with_toc(self, pdf_path: Path, force_reparse: bool) -> List[str]:
        """Extract using TOC-guided method."""
        self._update_progress("toc_init", 15, "Analyzing PDF structure...")
        
        try:
            # Open PDF and extract TOC
            doc = fitz.open(str(pdf_path))
            self._update_progress("toc_extraction", 25, "Extracting table of contents...")
            
            # Get TOC using the existing extractor
            toc_data = extract_nested_toc_from_pdf(str(pdf_path))
            
            if not toc_data:
                raise Exception("No TOC found in PDF")

            self._update_progress("toc_processing", 40, "Processing TOC sections...")

            # Process each TOC section
            markdown_files = []
            total_sections = len(toc_data)
            
            for i, toc_entry in enumerate(toc_data):
                section_progress = 40 + (i / total_sections) * 30  # 40-70% range
                
                title = toc_entry.get('title', f'Section_{i+1}')
                start_page = toc_entry.get('page', 0)
                
                # Determine end page (next section start or document end)
                if i + 1 < len(toc_data):
                    end_page = toc_data[i + 1].get('page', doc.page_count) - 1
                else:
                    end_page = doc.page_count - 1

                self._update_progress("toc_processing", section_progress, f"Processing: {title}")

                # Extract text from pages
                section_text = self._extract_pages_text(doc, start_page, end_page)
                
                if section_text.strip():
                    # Clean the text
                    cleaned_text = self._clean_text(section_text, title)
                    
                    # Create markdown file
                    markdown_file = self._save_section_markdown(pdf_path, title, cleaned_text, i+1)
                    markdown_files.append(markdown_file)

            doc.close()
            
            self._update_progress("toc_complete", 75, f"TOC extraction complete - {len(markdown_files)} sections")
            return markdown_files

        except Exception as e:
            logger.error(f"TOC extraction failed: {e}")
            raise

    def _extract_pages_text(self, doc: fitz.Document, start_page: int, end_page: int) -> str:
        """Extract text from a range of pages."""
        text_parts = []
        
        for page_num in range(start_page, min(end_page + 1, doc.page_count)):
            try:
                page = doc[page_num]
                page_text = page.get_text("text")
                if page_text.strip():
                    text_parts.append(page_text)
            except Exception as e:
                logger.warning(f"Failed to extract text from page {page_num}: {e}")
                continue
        
        return "\n\n".join(text_parts)

    def _clean_text(self, text: str, section_title: str) -> str:
        """Clean extracted text using regex cleaner and header fixer."""
        self._update_progress("cleaning", 78, f"Cleaning text for: {section_title}")
        
        try:
            # Apply regex cleaning
            cleaned = self.regex_cleaner.clean_text(text)
            
            # Apply header restructuring
            fixed = self.header_fixer.fix_headers(cleaned)
            
            return fixed
        except Exception as e:
            logger.warning(f"Text cleaning failed for {section_title}: {e}")
            return text  # Return original text if cleaning fails

    def _save_section_markdown(self, pdf_path: Path, section_title: str, content: str, section_num: int) -> str:
        """Save section as markdown file."""
        self._update_progress("saving", 85, f"Saving: {section_title}")
        
        # Create output directory
        output_dir = Path("data/processed_markdown")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create safe filename
        safe_title = self._sanitize_filename(section_title)
        base_name = pdf_path.stem
        filename = f"{base_name}-{section_num:02d}-{safe_title}.md"
        
        output_path = output_dir / filename
        
        try:
            # Create markdown content with metadata
            markdown_content = self._create_markdown_with_metadata(
                section_title, content, pdf_path.name, section_num
            )
            
            # Write file
            output_path.write_text(markdown_content, encoding='utf-8')
            
            logger.info(f"Saved section: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to save section {section_title}: {e}")
            raise

    def _sanitize_filename(self, title: str) -> str:
        """Create a safe filename from section title."""
        import re
        # Remove or replace problematic characters
        safe = re.sub(r'[<>:"/\\|?*]', '', title)
        safe = re.sub(r'\s+', '_', safe)
        safe = safe.strip('._')
        
        # Limit length
        if len(safe) > 50:
            safe = safe[:50]
        
        return safe or "untitled"

    def _create_markdown_with_metadata(self, title: str, content: str, source_file: str, section_num: int) -> str:
        """Create markdown content with YAML front matter."""
        
        # Determine primary section based on title
        primary_section = self._classify_section(title)
        
        # Create YAML front matter
        metadata = {
            'title': title,
            'source': source_file,
            'section_number': section_num,
            'primary_section': primary_section,
            'extraction_method': 'toc_guided',
            'edition': 'sr5',  # Fixed to SR5
            'document_type': 'rulebook'  # Fixed to rulebook
        }
        
        yaml_content = "---\n"
        for key, value in metadata.items():
            if isinstance(value, str):
                yaml_content += f'{key}: "{value}"\n'
            else:
                yaml_content += f'{key}: {value}\n'
        yaml_content += "---\n\n"
        
        # Combine metadata and content
        return yaml_content + f"# {title}\n\n" + content

    def _classify_section(self, title: str) -> str:
        """Classify section based on title keywords."""
        title_lower = title.lower()
        
        # Define section keywords
        section_mapping = {
            'Matrix': ['matrix', 'hacking', 'cyberdeck', 'icon', 'host', 'data', 'program', 'decker', 'technomancer'],
            'Combat': ['combat', 'weapon', 'armor', 'damage', 'initiative', 'attack', 'defense', 'firearm', 'melee'],
            'Magic': ['magic', 'spell', 'conjuring', 'enchanting', 'astral', 'mana', 'spirit', 'adept', 'mage', 'shaman'],
            'Riggers': ['rigger', 'drone', 'vehicle', 'pilot', 'sensor', 'autosofts', 'jumped'],
            'Social': ['social', 'contact', 'etiquette', 'leadership', 'negotiation', 'street cred'],
            'Skills': ['skill', 'attribute', 'test', 'specialization', 'knowledge'],
            'Character Creation': ['character', 'creation', 'priority', 'metatype', 'attribute', 'quality', 'background']
        }
        
        # Check each section
        for section, keywords in section_mapping.items():
            if any(keyword in title_lower for keyword in keywords):
                return section
        
        # Default section
        return 'General'

# Backward compatibility - alias the old class name
PDFProcessor = SimplePDFProcessor
