"""PDF to Markdown converter with REAL progress tracking via unstructured logging hooks."""
import json
import os
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Callable
import traceback

# unstructured imports
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import (
    Title, NarrativeText, ListItem, Table, Image, Footer, Header, Text
)
from unstructured.cleaners.core import clean_extra_whitespace, clean

# Hook into unstructured logging
from unstructured.logger import logger as unstructured_logger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnstructuredProgressHandler(logging.Handler):
    """Custom logging handler to capture real progress from unstructured library."""

    def __init__(self, progress_callback: Callable, job_id: str, filename: str):
        super().__init__()
        self.progress_callback = progress_callback
        self.job_id = job_id
        self.filename = filename
        self.start_time = time.time()
        self.current_progress = 5
        self._in_callback = False  # Add this to prevent recursion

        # Stage mapping based on unstructured log patterns
        self.stage_patterns = {
            # Early stages
            "Reading PDF": ("reading", 10, "Reading PDF content..."),
            "Loading the Table agent": ("table_detection", 15, "Loading table detection model..."),
            "hi_res strategy": ("extraction", 25, "Using high-resolution extraction..."),
            "fast strategy": ("extraction", 25, "Using fast text extraction..."),
            "auto strategy": ("extraction", 25, "Auto-selecting extraction strategy..."),

            # Progress indicators
            "hi_res strategy succeeded": ("extraction_complete", 50, "Element extraction complete"),
            "fast strategy succeeded": ("extraction_complete", 50, "Text extraction complete"),
            "extraction complete": ("extraction_complete", 50, "Document parsing complete"),

            # Element processing
            "Element types": ("analyzing", 60, "Analyzing document structure..."),
            "Cleaned elements": ("cleaning", 70, "Cleaning extracted content..."),

            # Chunking stages
            "Attempting chunk_by_title": ("chunking", 80, "Creating semantic chunks..."),
            "chunk_by_title succeeded": ("chunking_complete", 85, "Chunking complete"),
            "Manual chunking": ("chunking", 80, "Using fallback chunking..."),
            "Manual chunking created": ("chunking_complete", 85, "Chunking complete"),

            # Final stages
            "Debug info saved": ("saving", 90, "Saving processed content..."),
            "Successfully processed": ("complete", 100, "Processing complete!"),
        }

    def emit(self, record):
        """Process log records and extract progress information."""
        try:
            message = record.getMessage()

            # Check for known progress patterns
            for pattern, (stage, progress, description) in self.stage_patterns.items():
                if pattern in message:
                    self.current_progress = max(self.current_progress, progress)

                    # Extract additional details from log message
                    details = description
                    if "succeeded" in message and "elements" in message:
                        # Extract element count: "hi_res strategy succeeded: 150 elements"
                        try:
                            count = message.split("succeeded:")[1].split("elements")[0].strip()
                            details = f"{description} ({count} elements found)"
                        except:
                            pass
                    elif "Cleaned elements" in message:
                        # Extract cleaning info: "Cleaned elements: 120 from 150"
                        try:
                            parts = message.split(":")
                            if len(parts) > 1:
                                details = f"Cleaned and filtered elements: {parts[1].strip()}"
                        except:
                            pass
                    elif "chunking created" in message:
                        # Extract chunk count
                        try:
                            count = message.split("created")[1].split("chunks")[0].strip()
                            details = f"Created {count} semantic chunks"
                        except:
                            pass

                    # Send progress update (synchronously) - with recursion protection
                    self._send_progress_update(stage, self.current_progress, details)
                    break

            # Special handling for error patterns
            if any(error_word in message.lower() for error_word in ["error", "failed", "exception"]):
                if "processing failed" not in message.lower():  # Avoid our own error messages
                    self._send_progress_update("error", -1, f"Processing error: {message}")

        except Exception as e:
            # Don't let logging errors break the main process
            logger.warning(f"Progress handler error: {e}")

    def _send_progress_update(self, stage: str, progress: float, details: str):
        """Send progress update via callback - SYNCHRONOUS ONLY."""
        if not self.progress_callback:
            return

        # Prevent recursion
        if self._in_callback:
            logger.debug(f"Preventing recursive callback for {stage}")
            return

        self._in_callback = True
        try:
            # Call the callback synchronously with job_id
            self.progress_callback(self.job_id, stage, progress, details)
        except Exception as e:
            logger.warning(f"Progress callback failed: {e}")
        finally:
            self._in_callback = False

class PDFProcessor:
    def __init__(
        self,
        output_dir: str = "data/processed_markdown",
        chunk_size: int = 1024,
        use_ocr: bool = True,
        debug_mode: bool = True,
        progress_callback: Optional[Callable] = None
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        self.use_ocr = use_ocr
        self.debug_mode = debug_mode
        self.progress_callback = progress_callback  # Now expects synchronous callback

        # Create debug directory
        if debug_mode:
            self.debug_dir = self.output_dir / "_debug"
            self.debug_dir.mkdir(exist_ok=True)

    def process_pdf(self, pdf_path: str, force_reparse: bool = False, job_id: str = None) -> Dict[str, str]:
        """Process PDF with REAL progress tracking via unstructured logging."""
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

        logger.info(f"Processing {pdf_path} with REAL progress tracking")

        # Generate job_id if not provided
        if not job_id:
            job_id = f"{pdf_name}_{int(time.time())}"

        progress_handler = None

        try:
            # Install our custom logging handler
            if self.progress_callback:
                progress_handler = UnstructuredProgressHandler(
                    self.progress_callback, job_id, pdf_name
                )
                # Add to both unstructured logger and root logger to catch everything
                unstructured_logger.addHandler(progress_handler)
                logging.getLogger().addHandler(progress_handler)

                # Initial progress
                self._send_progress(job_id, 5, "starting", "Setting up PDF processing...")

            # Stage 1: Extract elements using unstructured (this is where the real work happens)
            logger.info("Reading PDF for file: %s ...", pdf_path)
            elements = self._extract_elements_with_fallbacks(pdf_path)

            if not elements:
                logger.error("No elements extracted!")
                return {}

            logger.info("hi_res strategy succeeded: %d elements", len(elements))

            # Stage 2: Debug and analyze
            if self.debug_mode:
                self._debug_elements(elements, pdf_name)

            # Stage 3: Clean elements
            cleaned_elements = self._clean_elements(elements)
            if not cleaned_elements:
                logger.warning("No valid elements after cleaning!")
                return self._emergency_fallback(pdf_path, output_subdir, pdf_name)

            logger.info("Cleaned elements: %d from %d", len(cleaned_elements), len(elements))

            # Stage 4: Create chunks
            logger.info("Attempting chunk_by_title...")
            chunks = self._create_chunks_with_fallbacks(cleaned_elements)

            if not chunks:
                logger.warning("No chunks created!")
                return self._emergency_fallback(pdf_path, output_subdir, pdf_name)

            logger.info("chunk_by_title succeeded: %d chunks", len(chunks))

            # Stage 5: Save files
            self._send_progress(job_id, 90, "saving", "Saving markdown files...")
            saved_files = self._save_chunks(chunks, output_subdir, pdf_name)

            # Save metadata
            self._save_metadata(json_meta, pdf_path, len(elements), len(cleaned_elements), len(chunks))

            # Final success
            logger.info("Successfully processed %s: %d files created", pdf_name, len(saved_files))
            self._send_progress(job_id, 100, "complete", f"Processing complete! Created {len(saved_files)} files.")

            return saved_files

        except Exception as e:
            logger.error("Processing failed: %s", str(e))
            logger.error(traceback.format_exc())
            self._send_progress(job_id, -1, "error", f"Processing failed: {str(e)}")
            # Try emergency fallback
            return self._emergency_fallback(pdf_path, output_subdir, pdf_name)

        finally:
            # Clean up logging handler
            if progress_handler:
                unstructured_logger.removeHandler(progress_handler)
                logging.getLogger().removeHandler(progress_handler)

    def _send_progress(self, job_id: str, progress: float, stage: str, details: str):
        """Send progress update if callback is available - SYNCHRONOUS."""
        if not self.progress_callback:
            return

        try:
            # Call synchronously with job_id
            self.progress_callback(job_id, stage, progress, details)
        except Exception as e:
            logger.warning(f"Failed to send progress update: {e}")

    def _extract_elements_with_fallbacks(self, pdf_path: Path):
        """Extract elements using multiple strategies with fallbacks."""
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
                logger.info("Trying %s strategy...", strategy_name)
                elements = partition_pdf(filename=str(pdf_path), **kwargs)
                logger.info("%s strategy succeeded: %d elements", strategy_name, len(elements))
                return elements

            except Exception as e:
                logger.warning("%s strategy failed: %s", strategy_name, str(e))
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

        for elem in elements:
            elem_type = type(elem).__name__
            element_types[elem_type] = element_types.get(elem_type, 0) + 1

            if elem_type not in text_samples:
                text = str(elem)
                text_samples[elem_type] = text[:200] + "..." if len(text) > 200 else text

        logger.info("Element types: %s", element_types)

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
        logger.info("Debug info saved to %s", debug_file)

    def _clean_elements(self, elements):
        """Clean and filter elements."""
        cleaned = []

        for elem in elements:
            elem_text = str(elem).strip()
            if not elem_text or len(elem_text) < 3:
                continue

            # Include most text-bearing elements
            if isinstance(elem, (NarrativeText, Title, ListItem, Text, Table)):
                text = clean_extra_whitespace(elem_text)
                text = clean(text, bullets=False)
                if text.strip():
                    elem.text = text
                    cleaned.append(elem)
            elif hasattr(elem, 'text') and elem.text and elem.text.strip():
                text = clean_extra_whitespace(elem.text)
                text = clean(text, bullets=False)
                if text.strip():
                    elem.text = text
                    cleaned.append(elem)

        logger.info("Cleaned elements: %d from %d", len(cleaned), len(elements))
        return cleaned

    def _create_chunks_with_fallbacks(self, elements):
        """Create chunks with multiple fallback strategies."""
        # Try 1: chunk_by_title
        try:
            logger.info("Attempting chunk_by_title...")
            chunks = chunk_by_title(
                elements,
                max_characters=self.chunk_size * 4,
                overlap=self.chunk_size // 2,
            )
            if chunks:
                logger.info("chunk_by_title succeeded: %d chunks", len(chunks))
                return chunks
        except Exception as e:
            logger.warning("chunk_by_title failed: %s", str(e))

        # Try 2: Manual chunking fallback
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

        logger.info("Manual chunking created %d chunks", len(chunks))
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
                logger.warning("Failed to save chunk %d: %s", i, str(e))
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
                logger.info("Emergency fallback created: %s", file_path)
                return {str(file_path): md_content}
        except Exception as e:
            logger.error("Emergency fallback failed: %s", str(e))

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
            "processor_version": "sync_progress_v2"
        }, indent=2))