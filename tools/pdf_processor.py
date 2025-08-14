"""PDF to Markdown converter using unstructured.io for superior semantic parsing."""
import json
import os
from pathlib import Path
from typing import Dict, List
import logging

# unstructured imports
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import (
    Title,
    NarrativeText,
    ListItem,
    Table,
    Image,
    Footer,
    Header,
)
from unstructured.cleaners.core import clean_extra_whitespace, clean

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFProcessor:
    """
    Advanced PDF processor using unstructured.io for Shadowrun rulebooks.
    Preserves semantic structure, tables, and hierarchy for optimal RAG performance.
    """

    def __init__(
        self,
        output_dir: str = "data/processed_markdown",
        chunk_size: int = 1024,
        use_ocr: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        self.use_ocr = use_ocr  # Enable OCR for scanned pages

    def process_pdf(self, pdf_path: str, force_reparse: bool = False) -> Dict[str, str]:
        """
        Convert PDF to structured Markdown using unstructured.io.
        Returns dict of {filename: content} for each semantic section.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        pdf_name = pdf_path.stem
        output_subdir = self.output_dir / pdf_name
        output_subdir.mkdir(exist_ok=True)

        # Skip if already processed (unless forced)
        json_meta = output_subdir / "_parsed_metadata.json"
        if json_meta.exists() and not force_reparse:
            logger.info(f"Skipping {pdf_name} (already processed). Use force_reparse=True to reprocess.")
            # Optionally load existing files
            return {
                str(f): f.read_text(encoding='utf-8')
                for f in output_subdir.glob("*.md")
                if f.name != "_parsed_metadata.json"
            }

        logger.info(f"Processing {pdf_path} with unstructured...")

        try:
            # Step 1: Extract elements with layout-aware parsing
            elements = partition_pdf(
                filename=str(pdf_path),
                strategy="hi_res",                    # High-resolution layout analysis
                infer_table_structure=True,          # Detect tables
                ocr_languages="eng",                 # Use Tesseract for image text
                skip_infer_table_types=[],           # Analyze all pages
                pdf_processing_mode="continuous",    # Best for long docs
                extract_images=False,                # Optional: set True to extract images
            )

            # Step 2: Clean and normalize text
            cleaned_elements = []
            for elem in elements:
                if isinstance(elem, (NarrativeText, Title, ListItem)):
                    text = clean_extra_whitespace(str(elem))
                    text = clean(text, bullets=False)  # Preserve list markers
                    elem.text = text
                cleaned_elements.append(elem)

            # Step 3: Group under titles (semantic chunking)
            chunks = chunk_by_title(
                cleaned_elements,
                max_characters=self.chunk_size * 4,   # ~500–800 tokens
                overlap=self.chunk_size // 2,         # 50% overlap for continuity
            )

            # Step 4: Convert to Markdown and save
            saved_files = {}
            for i, chunk in enumerate(chunks):
                # Determine title
                if hasattr(chunk, "metadata") and chunk.metadata.parent_id:
                    # Try to get parent title
                    title = str(chunk).split("\n")[0] if str(chunk).strip() else f"Section_{i+1}"
                else:
                    title = str(chunk).strip().split("\n")[0] if str(chunk).strip() else f"Section_{i+1}"
                    title = title[:100]  # Truncate long titles

                # Clean filename
                safe_title = "".join(c for c in title if c.isalnum() or c in " _-").rstrip()[:60]
                safe_title = safe_title.strip().replace(" ", "_") or f"section_{i+1}"

                # Format as Markdown
                md_content = self._format_as_markdown(chunk, index=i)

                # Save
                file_path = output_subdir / f"{safe_title}.md"
                file_path.write_text(md_content, encoding='utf-8')
                saved_files[str(file_path)] = md_content

            # Save metadata to mark as processed
            json_meta.write_text(
                json.dumps({
                    "source": str(pdf_path),
                    "processed_at": os.getenv("DATE", __import__("datetime").datetime.utcnow().isoformat()),
                    "chunk_count": len(chunks),
                    "use_ocr": self.use_ocr,
                    "strategy": "hi_res"
                }, indent=2)
            )

            logger.info(f"Processed {pdf_name}: {len(saved_files)} sections saved")
            return saved_files

        except Exception as e:
            logger.error(f"Failed to process {pdf_path}: {e}", exc_info=True)
            raise

    def _format_as_markdown(self, chunk, index: int = 0) -> str:
        """
        Format unstructured chunk as clean Markdown.
        Preserves headings, lists, and tables.
        """
        lines = []

        # Traverse elements in chunk
        for elem in chunk:
            text = str(elem).strip()
            if not text:
                continue

            if isinstance(elem, Title):
                level = self._estimate_heading_level(elem)
                lines.append(f"{'#' * level} {text}")
            elif isinstance(elem, ListItem):
                lines.append(f"- {text}")
            elif isinstance(elem, Table):
                lines.append("\n" + text + "\n")  # Tables already formatted by unstructured
            elif isinstance(elem, Image):
                # Optional: include image ref
                # lines.append(f"![Image](image_{index}.jpg)")
                pass
            elif isinstance(elem, (Header, Footer)):
                continue  # Skip
            else:
                # NarrativeText or other
                lines.append(text)

        return "\n\n".join(lines)

    def _estimate_heading_level(self, title_elem) -> int:
        """
        Estimate heading level based on font size, style, or position.
        unstructured sometimes includes this in metadata.
        """
        try:
            meta = title_elem.metadata
            level = getattr(meta, "heading_level", None)
            if level and isinstance(level, int) and 1 <= level <= 3:
                return level
        except:
            pass

        # Fallback: use text pattern
        text = str(title_elem)
        if text.isupper() and len(text) < 50:
            return 1
        if text.istitle() and len(text.split()) <= 6:
            return 2
        return 2  # Default to H2