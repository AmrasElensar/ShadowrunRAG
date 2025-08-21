"""PDF to Markdown converter with enhanced metadata extraction and document type awareness."""
import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Callable
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

class SimpleProgressHandler(logging.Handler):
    """Simplified logging handler to capture progress from unstructured library."""

    def __init__(self, progress_callback: Callable[[str, float, str], None], filename: str):
        super().__init__()
        self.progress_callback = progress_callback
        self.filename = filename
        self.start_time = time.time()
        self.current_progress = 5

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

            # Skip our own progress messages to avoid recursion
            if ("Progress [" in message or
                "Progress callback failed" in message or
                "Progress handler error" in message):
                return

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

                    # Send progress update (simple synchronous call)
                    self._send_progress_update(stage, self.current_progress, details)
                    break

            # Special handling for error patterns (but avoid our own messages)
            if (any(error_word in message.lower() for error_word in ["error", "failed", "exception"]) and
                "processing failed" not in message.lower() and
                "progress" not in message.lower()):
                self._send_progress_update("error", -1, f"Processing error: {message}")

        except Exception as e:
            # Don't let logging errors break the main process - and don't log this to avoid recursion
            pass

    def _send_progress_update(self, stage: str, progress: float, details: str):
        """Send progress update via callback (simplified synchronous version)."""
        if not self.progress_callback:
            return

        try:
            # Simple synchronous callback - no threading complexity
            self.progress_callback(stage, progress, details)
        except Exception as e:
            # Don't log this to avoid recursion - just silently fail
            pass

class PDFProcessor:
    def __init__(
        self,
        output_dir: str = "data/processed_markdown",
        chunk_size: int = 1024,  # Updated to match indexer
        use_ocr: bool = True,
        debug_mode: bool = True,
        progress_callback: Optional[Callable[[str, float, str], None]] = None,
        document_type: str = "rulebook"  # New parameter for document type
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        self.use_ocr = use_ocr
        self.debug_mode = debug_mode
        self.progress_callback = progress_callback
        self.document_type = document_type  # Store document type for processing

        # Create debug directory
        if debug_mode:
            self.debug_dir = self.output_dir / "_debug"
            self.debug_dir.mkdir(exist_ok=True)

    def _detect_shadowrun_content(self, text: str, filename: str) -> Dict[str, str]:
        """Enhanced Shadowrun-specific content detection."""
        content_lower = text[:3000].lower()  # Analyze first 3000 chars
        filename_lower = filename.lower()

        metadata = {
            "edition": "unknown",
            "document_type": self.document_type,  # Use provided type as default
            "primary_focus": "general",
            "detected_sections": []
        }

        # Edition detection with more patterns
        edition_patterns = {
            "SR6": ["shadowrun 6", "6th edition", "sr6", "sixth edition", "catalyst game labs 2019", "catalyst 2019"],
            "SR5": ["shadowrun 5", "5th edition", "sr5", "fifth edition", "catalyst game labs 2013", "catalyst 2013"],
            "SR4": ["shadowrun 4", "4th edition", "sr4", "fourth edition", "catalyst game labs 2005", "wizkids"],
            "SR3": ["shadowrun 3", "3rd edition", "sr3", "third edition", "fasa corporation"],
            "SR2": ["shadowrun 2", "2nd edition", "sr2", "second edition"],
            "SR1": ["shadowrun 1", "1st edition", "sr1", "first edition", "fasa 1989"]
        }

        for edition, patterns in edition_patterns.items():
            if any(pattern in content_lower or pattern in filename_lower for pattern in patterns):
                metadata["edition"] = edition
                break

        # Document type refinement based on content analysis
        type_indicators = {
            "character_sheet": [
                "character sheet", "player character", "npc", "attributes:", "skills:",
                "metatype:", "priority system", "karma", "nuyen", "contacts:",
                "cyberware:", "bioware:", "spells known", "adept powers"
            ],
            "rulebook": [
                "table of contents", "chapter", "game master", "dice pool",
                "threshold", "glitch", "critical glitch", "extended test",
                "opposed test", "teamwork test", "rule", "modifier"
            ],
            "universe_info": [
                "sixth world", "awakening", "crash of", "matrix 2.0",
                "corporate court", "dragon", "immortal elf", "history",
                "timeline", "shadowrunner", "mr. johnson", "fixers"
            ],
            "adventure": [
                "adventure", "scenario", "gamemaster", "handout",
                "scene", "encounter", "npc stats", "plot hook",
                "mission", "shadowrun", "team briefing"
            ]
        }

        # Score each type
        type_scores = {}
        for doc_type, indicators in type_indicators.items():
            score = sum(1 for indicator in indicators if indicator in content_lower)
            type_scores[doc_type] = score

        # Override document type if strong indicators found
        if type_scores:
            best_type = max(type_scores, key=type_scores.get)
            if type_scores[best_type] > 2:  # Threshold for confidence
                metadata["document_type"] = best_type

        # Section detection
        section_patterns = {
            "Combat": ["combat", "initiative", "damage", "armor", "weapon", "attack", "defense"],
            "Magic": ["magic", "spell", "astral", "summoning", "enchanting", "adept", "mage"],
            "Matrix": ["matrix", "hacking", "cyberdeck", "programs", "ic", "host", "decker"],
            "Riggers": ["rigger", "drone", "vehicle", "pilot", "autosofts", "jumped in"],
            "Character Creation": ["character creation", "priority", "attributes", "skills", "metatype"],
            "Gear": ["gear", "equipment", "cyberware", "bioware", "weapons", "armor"],
            "Gamemaster": ["gamemaster", "gm", "adventure", "npc", "campaign"],
            "Setting": ["seattle", "corps", "corporations", "sixth world", "awakening"]
        }

        detected_sections = []
        for section, patterns in section_patterns.items():
            if any(pattern in content_lower for pattern in patterns):
                detected_sections.append(section)

        metadata["detected_sections"] = detected_sections

        # Determine primary focus
        if detected_sections:
            # Use most mentioned section as primary focus
            section_counts = {}
            for section, patterns in section_patterns.items():
                count = sum(content_lower.count(pattern) for pattern in patterns)
                section_counts[section] = count

            if section_counts:
                metadata["primary_focus"] = max(section_counts, key=section_counts.get)

        return metadata

    def process_pdf(self, pdf_path: str, force_reparse: bool = False) -> Dict[str, str]:
        """Process PDF with enhanced metadata extraction and document type awareness."""
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

        logger.info(f"Processing {pdf_path} as {self.document_type} with enhanced metadata extraction")

        # Set up progress tracking via logging
        progress_handler = None

        try:
            # Install our simplified logging handler
            if self.progress_callback:
                progress_handler = SimpleProgressHandler(self.progress_callback, pdf_name)

                # Add to both unstructured logger and root logger to catch everything
                unstructured_logger.addHandler(progress_handler)
                logging.getLogger().addHandler(progress_handler)

                # Initial progress
                self.progress_callback("starting", 5, f"Setting up {self.document_type} processing...")

            # Stage 1: Extract elements using unstructured
            logger.info("Reading PDF for file: %s ...", pdf_path)
            elements = self._extract_elements_with_fallbacks(pdf_path)

            if not elements:
                logger.error("No elements extracted!")
                return {}

            logger.info("hi_res strategy succeeded: %d elements", len(elements))

            # Stage 1.5: Enhanced content analysis
            if self.progress_callback:
                self.progress_callback("analyzing", 55, "Analyzing Shadowrun content...")

            # Get full text for content analysis
            full_text = "\n".join(str(elem) for elem in elements[:50])  # First 50 elements for analysis
            content_metadata = self._detect_shadowrun_content(full_text, pdf_name)

            # Stage 2: Debug and analyze
            if self.debug_mode:
                self._debug_elements(elements, pdf_name, content_metadata)

            # Stage 3: Clean elements
            cleaned_elements = self._clean_elements(elements)
            if not cleaned_elements:
                logger.warning("No valid elements after cleaning!")
                return self._emergency_fallback(pdf_path, output_subdir, pdf_name)

            logger.info("Cleaned elements: %d from %d", len(cleaned_elements), len(elements))

            # Stage 4: Create chunks with document type awareness
            logger.info("Attempting chunk_by_title...")
            chunks = self._create_chunks_with_fallbacks(cleaned_elements)

            if not chunks:
                logger.warning("No chunks created!")
                return self._emergency_fallback(pdf_path, output_subdir, pdf_name)

            logger.info("chunk_by_title succeeded: %d chunks", len(chunks))

            # Stage 5: Save files with enhanced metadata
            if self.progress_callback:
                self.progress_callback("saving", 90, "Saving markdown files with metadata...")
            saved_files = self._save_chunks(chunks, output_subdir, pdf_name, content_metadata)

            # Save enhanced metadata
            self._save_metadata(json_meta, pdf_path, len(elements), len(cleaned_elements), len(chunks), content_metadata)

            # Final success
            logger.info("Successfully processed %s: %d files created", pdf_name, len(saved_files))
            if self.progress_callback:
                self.progress_callback("complete", 100, f"Processing complete! Created {len(saved_files)} files as {self.document_type}.")

            return saved_files

        except Exception as e:
            logger.error("Processing failed: %s", str(e))
            logger.error(traceback.format_exc())
            if self.progress_callback:
                self.progress_callback("error", -1, f"Processing failed: {str(e)}")
            # Try emergency fallback
            return self._emergency_fallback(pdf_path, output_subdir, pdf_name)

        finally:
            # Clean up logging handler
            if progress_handler:
                unstructured_logger.removeHandler(progress_handler)
                logging.getLogger().removeHandler(progress_handler)

    def _extract_elements_with_fallbacks(self, pdf_path: Path):
        """Extract elements using multiple strategies with document type optimization."""
        # Adjust strategy based on document type
        if self.document_type == "character_sheet":
            # Character sheets need table detection but less complex analysis
            strategies = [
                ("fast", {
                    "strategy": "fast",
                    "ocr_languages": "eng" if self.use_ocr else None,
                    "infer_table_structure": True,
                    "include_page_breaks": True,  # ← Add this to help with layout
                }),
                ("hi_res", {  # Fallback
                    "strategy": "hi_res",
                    "infer_table_structure": False,
                    "extract_images": False,
                    "ocr_languages": "eng" if self.use_ocr else None,
                })
            ]
        else:
            # Rulebooks and universe info need full analysis
            strategies = [
                ("auto", {
                    "strategy": "auto",
                    "infer_table_structure": True,
                    "ocr_languages": "eng" if self.use_ocr else None,
                    "include_page_breaks": True,
                }),
                ("fast", {
                    "strategy": "fast",
                    "ocr_languages": "eng" if self.use_ocr else None,
                    "infer_table_structure": True,
                    "include_page_breaks": True,
                }),
                ("hi_res", {
                    "strategy": "hi_res",
                    "infer_table_structure": False,
                    "extract_images": False,
                    "ocr_languages": "eng" if self.use_ocr else None,
                })
            ]

        for strategy_name, kwargs in strategies:
            try:
                logger.info("Trying %s strategy for %s...", strategy_name, self.document_type)
                elements = partition_pdf(filename=str(pdf_path), **kwargs)
                logger.info("%s strategy succeeded: %d elements", strategy_name, len(elements))
                return elements

            except Exception as e:
                logger.warning("%s strategy failed: %s", strategy_name, str(e))
                continue

        logger.error("All extraction strategies failed!")
        return []

    def _debug_elements(self, elements, pdf_name, content_metadata):
        """Enhanced debug analysis with content metadata."""
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

        # Save debug info with enhanced metadata
        debug_file = self.debug_dir / f"{pdf_name}_elements_debug.json"
        debug_info = {
            "total_elements": len(elements),
            "element_types": element_types,
            "text_samples": text_samples,
            "content_metadata": content_metadata,
            "document_type": self.document_type,
            "first_10_elements": [
                {
                    "type": type(elem).__name__,
                    "text": str(elem)[:300] + "..." if len(str(elem)) > 300 else str(elem)
                }
                for elem in elements[:10]
            ]
        }

        debug_file.write_text(json.dumps(debug_info, indent=2, ensure_ascii=False))
        logger.info("Enhanced debug info saved to %s", debug_file)

    def _clean_elements(self, elements):
        """Clean and filter elements with document type awareness."""
        cleaned = []

        for elem in elements:
            elem_text = str(elem).strip()
            if not elem_text or len(elem_text) < 3:
                continue

            # More aggressive filtering for character sheets
            if self.document_type == "character_sheet":
                # Skip very short fragments that are likely form fields
                if len(elem_text) < 10 and not any(char.isdigit() for char in elem_text):
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
        """Create chunks with document type optimization."""
        # Adjust chunking strategy based on document type
        if self.document_type == "character_sheet":
            # Character sheets benefit from smaller, more precise chunks
            chunk_size = self.chunk_size // 2
        else:
            chunk_size = self.chunk_size

        # Try 1: chunk_by_title
        try:
            logger.info("Attempting chunk_by_title with size %d...", chunk_size)
            chunks = chunk_by_title(
                elements,
                max_characters=chunk_size * 4,
                overlap=chunk_size // 4,
            )
            if chunks:
                logger.info("chunk_by_title succeeded: %d chunks", len(chunks))
                return chunks
        except Exception as e:
            logger.warning("chunk_by_title failed: %s", str(e))

        # Try 2: Manual chunking fallback
        logger.info("Using manual chunking fallback...")
        return self._manual_chunk(elements, chunk_size)

    def _manual_chunk(self, elements, chunk_size):
        """Size-based chunking fallback with document type awareness."""
        chunks = []
        current_chunk = []
        current_size = 0
        max_size = chunk_size * 4

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

    def _save_chunks(self, chunks, output_dir, pdf_name, content_metadata):
        """Save chunks as markdown files with enhanced metadata headers."""
        saved_files = {}

        # Create metadata header for all files
        metadata_header = self._create_metadata_header(pdf_name, content_metadata)

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

                # Format content with metadata
                md_content = self._format_chunk_content(chunk, chunk_text, i, metadata_header)

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

    def _create_metadata_header(self, pdf_name: str, content_metadata: Dict) -> str:
        """Create YAML front matter for markdown files."""
        return f"""---
title: "{pdf_name}"
document_type: "{content_metadata.get('document_type', 'unknown')}"
edition: "{content_metadata.get('edition', 'unknown')}"
primary_focus: "{content_metadata.get('primary_focus', 'general')}"
detected_sections: {content_metadata.get('detected_sections', [])}
processed_date: "{time.strftime('%Y-%m-%d %H:%M:%S')}"
---

"""

    def _format_chunk_content(self, chunk, chunk_text, index, metadata_header):
        """Format chunk as markdown with metadata header."""
        content = ""

        # Add metadata header only to first chunk or if requested
        if index == 0:
            content += metadata_header

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

            content += '\n\n'.join(lines) if lines else chunk_text
        else:
            content += chunk_text

        return content

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
        """Last resort: extract raw text with document type awareness."""
        try:
            import fitz  # PyMuPDF fallback
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()

            if text.strip():
                # Analyze content even in fallback
                content_metadata = self._detect_shadowrun_content(text, pdf_name)
                metadata_header = self._create_metadata_header(pdf_name, content_metadata)

                file_path = output_dir / f"{pdf_name}_fallback.md"
                md_content = f"{metadata_header}# {pdf_name}\n\n{text}"
                file_path.write_text(md_content, encoding='utf-8')
                logger.info("Emergency fallback created with metadata: %s", file_path)
                if self.progress_callback:
                    self.progress_callback("complete", 100, f"Completed using text extraction fallback for {self.document_type}")
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

    def _save_metadata(self, json_meta, pdf_path, total_elements, cleaned_elements, chunks, content_metadata):
        """Save enhanced processing metadata."""
        json_meta.write_text(json.dumps({
            "source": str(pdf_path),
            "processed_at": __import__("datetime").datetime.utcnow().isoformat(),
            "total_elements": total_elements,
            "cleaned_elements": cleaned_elements,
            "chunks_created": chunks,
            "processor_version": "enhanced_v2",
            "document_type": self.document_type,
            "content_metadata": content_metadata
        }, indent=2))