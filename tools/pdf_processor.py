"""PDF to Markdown converter with staged debugging approach."""
import json
import os
from pathlib import Path
from typing import Dict, List
import logging
import traceback

# unstructured imports
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import (
    Title, NarrativeText, ListItem, Table, Image, Footer, Header, Text
)
from unstructured.cleaners.core import clean_extra_whitespace, clean

logging.basicConfig(level=logging.DEBUG)  # More verbose logging
logger = logging.getLogger(__name__)

class PDFProcessor:
    def __init__(
        self,
        output_dir: str = "data/processed_markdown",
        chunk_size: int = 1024,
        use_ocr: bool = True,
        debug_mode: bool = True
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        self.use_ocr = use_ocr
        self.debug_mode = debug_mode

        # Create debug directory
        if debug_mode:
            self.debug_dir = self.output_dir / "_debug"
            self.debug_dir.mkdir(exist_ok=True)

    def process_pdf(self, pdf_path: str, force_reparse: bool = False) -> Dict[str, str]:
        """Process PDF with staged debugging."""
        logger.info("🔥 TESTING: PDF processor code is LIVE and updated!")
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        pdf_name = pdf_path.stem
        output_subdir = self.output_dir / pdf_name
        output_subdir.mkdir(exist_ok=True)

        # Skip if already processed
        json_meta = output_subdir / "_parsed_metadata.json"
        if json_meta.exists() and not force_reparse:
            logger.info(f"Skipping {pdf_name} (already processed)")
            return self._load_existing_files(output_subdir)

        logger.info(f"Processing {pdf_path}")

        try:
            # Stage 1: Try hi_res first
            elements = self._extract_elements_staged(pdf_path)

            if not elements:
                logger.error("No elements extracted!")
                return {}

            # Stage 2: Debug element analysis
            self._debug_elements(elements, pdf_name)

            # Stage 3: Clean and filter elements
            cleaned_elements = self._clean_elements(elements)

            if not cleaned_elements:
                logger.warning("No valid elements after cleaning!")
                return self._emergency_fallback(pdf_path, output_subdir, pdf_name)

            # Stage 4: Chunk with fallbacks
            chunks = self._create_chunks_staged(cleaned_elements)

            if not chunks:
                logger.warning("No chunks created!")
                return self._emergency_fallback(pdf_path, output_subdir, pdf_name)

            # Stage 5: Save markdown files
            saved_files = self._save_chunks(chunks, output_subdir, pdf_name)

            # Save metadata
            self._save_metadata(json_meta, pdf_path, len(elements), len(cleaned_elements), len(chunks))

            logger.info(f"Successfully processed {pdf_name}: {len(saved_files)} files created")
            return saved_files

        except Exception as e:
            logger.error(f"Processing failed: {e}")
            logger.error(traceback.format_exc())
            # Try emergency fallback
            return self._emergency_fallback(pdf_path, output_subdir, pdf_name)

    def _extract_elements_staged(self, pdf_path: Path):
        """Try different extraction strategies in order of quality."""
        strategies = [
            ("hi_res", {
                "strategy": "hi_res",
                "infer_table_structure": True,
                "ocr_languages": "eng" if self.use_ocr else None,
                "extract_images": False,
            }),
            ("auto", {
                "strategy": "auto",
                "infer_table_structure": True,
                "ocr_languages": "eng" if self.use_ocr else None,
            }),
            ("fast", {
                "strategy": "fast",
                "ocr_languages": "eng" if self.use_ocr else None,
            })
        ]

        for strategy_name, kwargs in strategies:
            try:
                logger.info(f"Trying {strategy_name} strategy...")
                elements = partition_pdf(filename=str(pdf_path), **kwargs)
                logger.info(f"{strategy_name} strategy succeeded: {len(elements)} elements")
                return elements

            except Exception as e:
                logger.warning(f"{strategy_name} strategy failed: {e}")
                if strategy_name == "hi_res":
                    logger.info("hi_res failed - this might be a dependency issue, not hardware")
                continue

        logger.error("All extraction strategies failed!")
        return []

    def _debug_elements(self, elements, pdf_name):
        """Analyze and debug elements."""
        if not self.debug_mode:
            return

        # Count element types
        element_types = {}
        text_samples = {}

        for i, elem in enumerate(elements):
            elem_type = type(elem).__name__
            element_types[elem_type] = element_types.get(elem_type, 0) + 1

            # Sample text for each type
            if elem_type not in text_samples:
                text_samples[elem_type] = str(elem)[:200] + "..." if len(str(elem)) > 200 else str(elem)

        logger.info(f"Element types: {element_types}")

        # Save debug info
        debug_file = self.debug_dir / f"{pdf_name}_elements_debug.json"
        debug_info = {
            "total_elements": len(elements),
            "element_types": element_types,
            "text_samples": text_samples,
            "first_10_elements": [
                {
                    "type": type(elem).__name__,
                    "text": str(elem)[:300] + "..." if len(str(elem)) > 300 else str(elem)
                }
                for elem in elements[:10]
            ]
        }

        debug_file.write_text(json.dumps(debug_info, indent=2, ensure_ascii=False))
        logger.info(f"Debug info saved to {debug_file}")

    def _clean_elements(self, elements):
        """Clean and filter elements with inclusive approach."""
        cleaned = []

        for elem in elements:
            elem_text = str(elem).strip()
            if not elem_text or len(elem_text) < 3:  # Skip very short elements
                continue

            # Include most text-bearing elements
            if isinstance(elem, (NarrativeText, Title, ListItem, Text, Table)):
                text = clean_extra_whitespace(elem_text)
                text = clean(text, bullets=False)
                if text.strip():
                    elem.text = text
                    cleaned.append(elem)
            elif hasattr(elem, 'text') and elem.text and elem.text.strip():
                # Catch other elements with text
                text = clean_extra_whitespace(elem.text)
                text = clean(text, bullets=False)
                if text.strip():
                    elem.text = text
                    cleaned.append(elem)

        logger.info(f"Cleaned elements: {len(cleaned)} from {len(elements)}")
        return cleaned

    def _create_chunks_staged(self, elements):
        """Try chunking with multiple fallbacks."""
        # Try 1: chunk_by_title
        try:
            logger.info("Attempting chunk_by_title...")
            chunks = chunk_by_title(
                elements,
                max_characters=self.chunk_size * 4,
                overlap=self.chunk_size // 2,
            )
            if chunks:
                logger.info(f"chunk_by_title succeeded: {len(chunks)} chunks")
                return chunks
        except Exception as e:
            logger.warning(f"chunk_by_title failed: {e}")

        # Try 2: Manual size-based chunking
        logger.info("Using manual chunking fallback...")
        return self._manual_chunk(elements)

    def _manual_chunk(self, elements):
        """Size-based chunking fallback."""
        chunks = []
        current_chunk = []
        current_size = 0
        max_size = self.chunk_size * 4

        for elem in elements:
            elem_text = str(elem)
            elem_size = len(elem_text)

            # Start new chunk if current would be too large
            if current_size + elem_size > max_size and current_chunk:
                chunk_obj = type('Chunk', (), {
                    'elements': current_chunk.copy(),
                    'text': '\n\n'.join(str(e) for e in current_chunk)
                })
                chunks.append(chunk_obj)
                current_chunk = []
                current_size = 0

            current_chunk.append(elem)
            current_size += elem_size

        # Add final chunk
        if current_chunk:
            chunk_obj = type('Chunk', (), {
                'elements': current_chunk.copy(),
                'text': '\n\n'.join(str(e) for e in current_chunk)
            })
            chunks.append(chunk_obj)

        logger.info(f"Manual chunking created {len(chunks)} chunks")
        return chunks

    def _save_chunks(self, chunks, output_dir, pdf_name):
        """Save chunks as markdown files."""
        saved_files = {}

        for i, chunk in enumerate(chunks):
            try:
                # Get chunk text
                if hasattr(chunk, 'text'):
                    chunk_text = chunk.text
                else:
                    chunk_text = str(chunk)

                if not chunk_text.strip():
                    continue

                # Generate title
                lines = chunk_text.split('\n')
                first_line = next((line.strip() for line in lines if line.strip()), f"Section_{i+1}")

                # Clean filename
                safe_title = "".join(c for c in first_line if c.isalnum() or c in " _-").strip()[:50]
                safe_title = safe_title.replace(" ", "_") or f"section_{i+1}"

                # Format content
                md_content = self._format_chunk_content(chunk, chunk_text, i)

                # Save with unique filename
                file_path = output_dir / f"{safe_title}.md"
                counter = 1
                while file_path.exists():
                    file_path = output_dir / f"{safe_title}_{counter}.md"
                    counter += 1

                file_path.write_text(md_content, encoding='utf-8')
                saved_files[str(file_path)] = md_content

            except Exception as e:
                logger.warning(f"Failed to save chunk {i}: {e}")
                continue

        return saved_files

    def _format_chunk_content(self, chunk, chunk_text, index):
        """Format chunk as markdown."""
        if hasattr(chunk, 'elements'):
            # Process individual elements
            lines = []
            for elem in chunk.elements:
                text = str(elem).strip()
                if not text:
                    continue

                if isinstance(elem, Title):
                    level = self._estimate_heading_level(elem)
                    lines.append(f"{'#' * level} {text}")
                elif isinstance(elem, ListItem):
                    lines.append(f"- {text}")
                elif isinstance(elem, Table):
                    lines.append(f"\n{text}\n")
                else:
                    lines.append(text)

            return '\n\n'.join(lines) if lines else chunk_text
        else:
            return chunk_text

    def _estimate_heading_level(self, title_elem):
        """Estimate heading level."""
        try:
            if hasattr(title_elem, 'metadata') and hasattr(title_elem.metadata, 'heading_level'):
                level = title_elem.metadata.heading_level
                if 1 <= level <= 6:
                    return level
        except:
            pass

        text = str(title_elem)
        if text.isupper() and len(text) < 50:
            return 1
        return 2

    def _emergency_fallback(self, pdf_path, output_dir, pdf_name):
        """Last resort: extract raw text."""
        try:
            import fitz  # PyMuPDF fallback
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()

            if text.strip():
                file_path = output_dir / f"{pdf_name}_fallback.md"
                md_content = f"# {pdf_name}\n\n{text}"
                file_path.write_text(md_content, encoding='utf-8')
                logger.info(f"Emergency fallback created: {file_path}")
                return {str(file_path): md_content}
        except Exception as e:
            logger.error(f"Emergency fallback failed: {e}")

        return {}

    def _load_existing_files(self, output_dir):
        """Load existing markdown files."""
        return {
            str(f): f.read_text(encoding='utf-8')
            for f in output_dir.glob("*.md")
        }

    def _save_metadata(self, json_meta, pdf_path, total_elements, cleaned_elements, chunks):
        """Save processing metadata."""
        json_meta.write_text(json.dumps({
            "source": str(pdf_path),
            "processed_at": __import__("datetime").datetime.utcnow().isoformat(),
            "total_elements": total_elements,
            "cleaned_elements": cleaned_elements,
            "chunks_created": chunks,
            "processor_version": "debug_v1"
        }, indent=2))