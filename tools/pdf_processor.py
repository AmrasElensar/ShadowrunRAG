"""Enhanced PDF to Markdown converter with Marker GPU acceleration and PyMuPDF+TOC fallback.

This replaces the original pdf_processor.py with a 4-tier extraction hierarchy:
1. Marker GPU acceleration (primary - handles multi-column layouts)
2. PyMuPDF + TOC fallback (leverages PDF bookmarks for structure)
3. Unstructured emergency fallback (fast strategy only)
4. Basic text extraction (absolute last resort)

Usage:
    from tools.pdf_processor import PDFProcessor  # Backward compatible
    processor = PDFProcessor(document_type="rulebook", use_gpu=True)
    result = processor.process_pdf("path/to/shadowrun.pdf")
"""

import json
import logging
import time
import fitz  # PyMuPDF
from pathlib import Path
from typing import Dict, Optional, Callable, List, Tuple
import traceback
import re

# Marker imports (GPU-accelerated extraction)
try:
    from marker.convert import convert_single_pdf
    from marker.models import load_all_models
    MARKER_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("✅ Marker PDF processing available (GPU-accelerated)")
except ImportError:
    MARKER_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("⚠️ Marker not available - install with: pip install marker-pdf")

# unstructured imports (emergency fallback)
try:
    from unstructured.partition.pdf import partition_pdf
    from unstructured.documents.elements import Title, NarrativeText, ListItem, Table, Text
    from unstructured.cleaners.core import clean_extra_whitespace, clean
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("⚠️ Unstructured not available - limited fallback options")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleProgressHandler(logging.Handler):
    """Capture progress from various PDF processing libraries."""

    def __init__(self, progress_callback: Callable[[str, float, str], None], filename: str):
        super().__init__()
        self.progress_callback = progress_callback
        self.filename = filename
        self.current_progress = 5

        # Enhanced stage patterns for all extraction methods
        self.stage_patterns = {
            # Marker GPU stages
            "Loading models": ("marker_init", 10, "Loading Marker GPU models..."),
            "Converting PDF": ("marker_processing", 30, "Marker GPU acceleration active..."),
            "Extracting text": ("marker_extraction", 60, "GPU extracting text and layout..."),
            "Processing complete": ("marker_complete", 75, "Marker extraction complete"),

            # PyMuPDF + TOC stages
            "Reading PDF structure": ("toc_init", 15, "Analyzing PDF structure..."),
            "Extracting TOC": ("toc_extraction", 25, "Extracting table of contents..."),
            "Processing sections": ("toc_processing", 50, "Processing TOC sections..."),
            "TOC processing complete": ("toc_complete", 70, "TOC extraction complete"),

            # Unstructured emergency fallback
            "Reading PDF": ("unstructured_reading", 20, "Emergency fallback: reading PDF..."),
            "fast strategy": ("unstructured_fast", 40, "Emergency: fast text extraction..."),

            # Common completion stages
            "Chunking": ("chunking", 80, "Creating content chunks..."),
            "Saving": ("saving", 90, "Saving markdown files..."),
            "Successfully processed": ("complete", 100, "Processing complete!"),
        }

    def emit(self, record):
        """Extract progress from log messages."""
        try:
            message = record.getMessage()

            # Skip recursive progress messages
            if any(skip in message for skip in ["Progress [", "Progress callback", "Progress handler"]):
                return

            # Match progress patterns
            for pattern, (stage, progress, description) in self.stage_patterns.items():
                if pattern in message:
                    self.current_progress = max(self.current_progress, progress)
                    self._send_progress_update(stage, self.current_progress, description)
                    break

            # Handle errors
            if any(error in message.lower() for error in ["error", "failed", "exception"]) and "progress" not in message.lower():
                self._send_progress_update("error", -1, f"Error: {message}")

        except Exception:
            pass  # Silent fail to avoid recursion

    def _send_progress_update(self, stage: str, progress: float, details: str):
        """Send progress update safely."""
        try:
            if self.progress_callback:
                self.progress_callback(stage, progress, details)
        except Exception:
            pass

class EnhancedPDFProcessor:
    """Enhanced PDF processor with 4-tier extraction hierarchy."""

    def __init__(
            self,
            output_dir: str = "data/processed_markdown",
            chunk_size: int = 1024,
            use_ocr: bool = True,
            debug_mode: bool = True,
            progress_callback: Optional[Callable[[str, float, str], None]] = None,
            document_type: str = "rulebook",
            use_gpu: bool = True
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        self.use_ocr = use_ocr
        self.debug_mode = debug_mode
        self.progress_callback = progress_callback
        self.document_type = document_type
        self.use_gpu = use_gpu

        # GPU model cache for Marker (expensive to load)
        self._marker_models = None
        self._models_loading = False

        # Create debug directory
        if debug_mode:
            self.debug_dir = self.output_dir / "_debug"
            self.debug_dir.mkdir(exist_ok=True)

        logger.info(f"Enhanced PDF Processor initialized:")
        logger.info(f"  - Document type: {document_type}")
        logger.info(f"  - GPU acceleration: {use_gpu}")
        logger.info(f"  - Marker available: {MARKER_AVAILABLE}")
        logger.info(f"  - Unstructured available: {UNSTRUCTURED_AVAILABLE}")

    def _load_marker_models(self):
        """Load Marker models once and cache them (GPU-intensive operation)."""
        if not MARKER_AVAILABLE or not self.use_gpu:
            return None

        if self._marker_models is not None:
            return self._marker_models

        if self._models_loading:
            # Another thread is loading, wait briefly
            for _ in range(30):  # Wait up to 30 seconds
                time.sleep(1)
                if self._marker_models is not None:
                    return self._marker_models
            return None

        try:
            self._models_loading = True

            if self.progress_callback:
                self.progress_callback("marker_init", 10, "Loading Marker GPU models (this may take a moment)...")

            logger.info("🚀 Loading Marker models for GPU acceleration...")
            start_time = time.time()

            # Load all required models for GPU processing
            self._marker_models = load_all_models()

            load_time = time.time() - start_time
            logger.info(f"✅ Marker models loaded successfully ({load_time:.1f}s)")

            if self.progress_callback:
                self.progress_callback("marker_ready", 15, f"Marker GPU models ready ({load_time:.1f}s)")

        except Exception as e:
            logger.error(f"❌ Failed to load Marker models: {e}")
            self._marker_models = None

        finally:
            self._models_loading = False

        return self._marker_models

    def _extract_with_marker(self, pdf_path: Path) -> Optional[str]:
        """Primary extraction method: Marker GPU acceleration for multi-column layouts."""
        if not MARKER_AVAILABLE or not self.use_gpu:
            logger.info("Marker not available or GPU disabled, skipping...")
            return None

        try:
            if self.progress_callback:
                self.progress_callback("marker_processing", 20, "Starting Marker GPU extraction...")

            # Load models (cached after first use)
            models = self._load_marker_models()
            if not models:
                logger.warning("Could not load Marker models")
                return None

            logger.info(f"🎯 Processing {pdf_path.name} with Marker GPU acceleration...")

            # Configure Marker for optimal extraction
            start_time = time.time()

            # Convert PDF to markdown using Marker's GPU acceleration
            full_text, images, out_meta = convert_single_pdf(
                str(pdf_path),
                models,
                max_pages=None,  # Process all pages
                langs=["en"],  # English content
                batch_multiplier=2,  # GPU optimization for RTX 4090
                start_page=None,  # Process from beginning
                ocr_all_pages=self.use_ocr  # OCR fallback if needed
            )

            extraction_time = time.time() - start_time

            if self.progress_callback:
                self.progress_callback("marker_complete", 70, f"Marker extraction complete ({extraction_time:.1f}s)")

            # Validate extraction quality
            if not full_text or len(full_text.strip()) < 100:
                logger.warning("Marker extraction produced insufficient content")
                return None

            logger.info(f"✅ Marker success: {len(full_text)} chars in {extraction_time:.1f}s")

            # Save debug info if enabled
            if self.debug_mode:
                debug_file = self.debug_dir / f"{pdf_path.stem}_marker_debug.json"
                debug_info = {
                    "extraction_method": "marker_gpu",
                    "processing_time": extraction_time,
                    "content_length": len(full_text),
                    "metadata": out_meta,
                    "images_found": len(images) if images else 0,
                    "success": True
                }
                debug_file.write_text(json.dumps(debug_info, indent=2))

            return full_text

        except Exception as e:
            logger.error(f"❌ Marker extraction failed: {e}")
            if self.progress_callback:
                self.progress_callback("marker_failed", -1, f"Marker failed: {str(e)}")

            # Save debug info for failed extraction
            if self.debug_mode:
                debug_file = self.debug_dir / f"{pdf_path.stem}_marker_error.json"
                debug_info = {
                    "extraction_method": "marker_gpu",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "success": False
                }
                debug_file.write_text(json.dumps(debug_info, indent=2))

            return None

    def _clear_gpu_cache(self):
        """Clear GPU memory cache after processing."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("🧹 GPU cache cleared")
        except ImportError:
            pass  # PyTorch not available

    def _extract_toc_structure(self, pdf_document) -> List[Dict]:
        """Extract table of contents structure from PDF for semantic chunking."""
        try:
            toc = pdf_document.get_toc()
            if not toc:
                logger.info("No TOC found in PDF")
                return []

            # Convert PyMuPDF TOC to structured format
            structured_toc = []
            for level, title, page_num in toc:
                # Clean and validate title
                clean_title = title.strip()
                if not clean_title or len(clean_title) < 2:
                    continue

                structured_toc.append({
                    "level": level,
                    "title": clean_title,
                    "page": max(0, page_num - 1),  # Convert to 0-based, ensure non-negative
                    "text": ""  # Will be filled with extracted text
                })

            logger.info(f"📚 Extracted TOC with {len(structured_toc)} sections")
            return structured_toc

        except Exception as e:
            logger.warning(f"TOC extraction failed: {e}")
            return []

    def _extract_with_pymupdf_toc(self, pdf_path: Path) -> Optional[str]:
        """Secondary method: PyMuPDF with TOC-aware semantic chunking."""
        try:
            if self.progress_callback:
                self.progress_callback("toc_init", 25, "Analyzing PDF structure with PyMuPDF...")

            logger.info(f"📖 Processing {pdf_path.name} with PyMuPDF + TOC extraction...")

            pdf_document = fitz.open(pdf_path)

            # Extract TOC structure for semantic organization
            if self.progress_callback:
                self.progress_callback("toc_extraction", 35, "Extracting table of contents...")

            toc_structure = self._extract_toc_structure(pdf_document)

            if toc_structure:
                # TOC-based semantic extraction
                if self.progress_callback:
                    self.progress_callback("toc_processing", 50, f"Processing {len(toc_structure)} TOC sections...")

                result = self._extract_by_toc_sections(pdf_document, toc_structure)
            else:
                # Fallback to intelligent page-by-page extraction
                if self.progress_callback:
                    self.progress_callback("page_processing", 45, "No TOC found, using intelligent page extraction...")

                result = self._extract_page_by_page_smart(pdf_document)

            pdf_document.close()

            if result and len(result.strip()) > 100:
                if self.progress_callback:
                    self.progress_callback("toc_complete", 70, "PyMuPDF extraction complete")

                logger.info(f"✅ PyMuPDF + TOC success: {len(result)} characters")
                return result
            else:
                logger.warning("PyMuPDF + TOC produced insufficient content")
                return None

        except Exception as e:
            logger.error(f"❌ PyMuPDF + TOC extraction failed: {e}")
            if 'pdf_document' in locals():
                pdf_document.close()
            return None

    def _extract_by_toc_sections(self, pdf_document, toc_structure: List[Dict]) -> str:
        """Extract text organized by TOC sections for perfect semantic chunking."""
        sections = []
        total_sections = len(toc_structure)

        for i, section in enumerate(toc_structure):
            try:
                # Determine page range for this section
                start_page = section["page"]
                end_page = (toc_structure[i + 1]["page"] if i + 1 < total_sections
                            else pdf_document.page_count)

                # Ensure valid page range
                start_page = max(0, min(start_page, pdf_document.page_count - 1))
                end_page = max(start_page + 1, min(end_page, pdf_document.page_count))

                # Extract text for this section
                section_text = ""
                for page_num in range(start_page, end_page):
                    try:
                        page = pdf_document[page_num]
                        page_text = page.get_text()

                        # Clean up common PDF artifacts
                        page_text = self._clean_pdf_text(page_text)

                        if page_text.strip():
                            section_text += page_text + "\n"
                    except Exception as page_error:
                        logger.warning(f"Failed to extract page {page_num}: {page_error}")
                        continue

                # Format as markdown section with proper heading level
                if section_text.strip():
                    level_marker = "#" * min(section["level"], 6)
                    formatted_section = f"\n{level_marker} {section['title']}\n\n{section_text.strip()}\n"
                    sections.append(formatted_section)

                    logger.debug(f"Extracted section '{section['title']}': {len(section_text)} chars")

            except Exception as e:
                logger.warning(f"Failed to extract section '{section.get('title', 'Unknown')}': {e}")
                continue

        full_text = "\n".join(sections)
        logger.info(f"✅ TOC-based extraction: {len(sections)} sections, {len(full_text)} characters")
        return full_text

    def _extract_page_by_page_smart(self, pdf_document) -> str:
        """Intelligent page-by-page extraction with basic structure detection."""
        pages = []

        for page_num in range(pdf_document.page_count):
            try:
                page = pdf_document[page_num]
                page_text = page.get_text()

                # Clean up text
                page_text = self._clean_pdf_text(page_text)

                if page_text.strip():
                    # Try to detect if this looks like a chapter/section start
                    lines = page_text.split('\n')
                    first_lines = [line.strip() for line in lines[:5] if line.strip()]

                    # Look for chapter/section indicators
                    is_chapter_start = False
                    if first_lines:
                        first_line = first_lines[0]
                        chapter_indicators = ['chapter', 'section', 'part ', 'appendix', 'introduction', 'conclusion']
                        if any(indicator in first_line.lower() for indicator in chapter_indicators):
                            is_chapter_start = True

                    # Format with appropriate heading
                    if is_chapter_start and first_lines:
                        formatted_page = f"\n## {first_lines[0]}\n\n{page_text.strip()}\n"
                    else:
                        formatted_page = f"\n### Page {page_num + 1}\n\n{page_text.strip()}\n"

                    pages.append(formatted_page)

            except Exception as e:
                logger.warning(f"Failed to extract page {page_num}: {e}")
                continue

        full_text = "\n".join(pages)
        logger.info(f"✅ Smart page extraction: {len(pages)} pages, {len(full_text)} characters")
        return full_text

    def _clean_pdf_text(self, text: str) -> str:
        """Clean common PDF extraction artifacts."""
        if not text:
            return ""

        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Multiple newlines to double
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces to single
        text = re.sub(r'^\s+|\s+$', '', text, flags=re.MULTILINE)  # Trim lines

        # Remove common PDF artifacts
        text = re.sub(r'\f', '\n', text)  # Form feeds to newlines
        text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)  # Rejoin hyphenated words

        # Remove isolated page numbers and headers/footers (simple heuristic)
        lines = text.split('\n')
        cleaned_lines = []

        for line in lines:
            line = line.strip()
            # Skip lines that are just numbers (likely page numbers)
            if line.isdigit() and len(line) <= 3:
                continue
            # Skip very short lines that might be headers/footers
            if len(line) < 3:
                continue
            cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    def _extract_with_unstructured_fallback(self, pdf_path: Path) -> Optional[str]:
        """Tertiary method: Emergency unstructured fallback (fast strategy only)."""
        if not UNSTRUCTURED_AVAILABLE:
            logger.warning("Unstructured not available for emergency fallback")
            return None

        try:
            if self.progress_callback:
                self.progress_callback("unstructured_fallback", 30, "Emergency: using unstructured fallback...")

            logger.info(f"🚨 Emergency fallback: Using unstructured for {pdf_path.name}")

            # Use only fast strategy for emergency (avoid problematic hi-res/auto)
            elements = partition_pdf(
                filename=str(pdf_path),
                strategy="fast",  # Fast only - avoid column mixing
                ocr_languages="eng" if self.use_ocr else None,
                include_page_breaks=True,
                infer_table_structure=False  # Disable to avoid layout issues
            )

            if not elements:
                logger.warning("Unstructured returned no elements")
                return None

            # Convert elements to clean markdown
            text_parts = []
            for elem in elements:
                elem_text = str(elem).strip()
                if not elem_text or len(elem_text) < 3:
                    continue

                # Basic markdown formatting
                if isinstance(elem, Title):
                    # Estimate heading level
                    if elem_text.isupper() and len(elem_text) < 50:
                        text_parts.append(f"\n# {elem_text}\n")
                    else:
                        text_parts.append(f"\n## {elem_text}\n")
                elif isinstance(elem, ListItem):
                    text_parts.append(f"- {elem_text}")
                elif isinstance(elem, Table):
                    text_parts.append(f"\n{elem_text}\n")  # Keep tables as-is
                else:
                    # Clean the text
                    clean_text = clean_extra_whitespace(elem_text)
                    text_parts.append(clean_text)

            full_text = "\n\n".join(text_parts)

            if len(full_text.strip()) > 100:
                logger.info(f"✅ Unstructured fallback: {len(elements)} elements, {len(full_text)} characters")
                return full_text
            else:
                logger.warning("Unstructured fallback produced insufficient content")
                return None

        except Exception as e:
            logger.error(f"❌ Unstructured fallback failed: {e}")
            return None

    def _extract_emergency_text(self, pdf_path: Path) -> str:
        """Last resort: Basic PyMuPDF text extraction (always succeeds)."""
        try:
            if self.progress_callback:
                self.progress_callback("emergency_extraction", 35, "Last resort: basic text extraction...")

            logger.warning(f"🆘 Emergency extraction for {pdf_path.name}")

            pdf_document = fitz.open(pdf_path)
            text_parts = []

            for page_num in range(pdf_document.page_count):
                try:
                    page = pdf_document[page_num]
                    page_text = page.get_text()

                    # Basic cleaning
                    page_text = self._clean_pdf_text(page_text)

                    if page_text.strip():
                        text_parts.append(f"\n## Page {page_num + 1}\n\n{page_text.strip()}")

                except Exception as page_error:
                    logger.warning(f"Failed to extract page {page_num}: {page_error}")
                    text_parts.append(f"\n## Page {page_num + 1}\n\n[Page extraction failed]")

            pdf_document.close()

            full_text = "\n\n".join(text_parts) if text_parts else f"# {pdf_path.stem}\n\n[No text could be extracted]"

            logger.info(f"✅ Emergency extraction: {len(text_parts)} pages, {len(full_text)} characters")
            return full_text

        except Exception as e:
            logger.error(f"❌ Emergency extraction failed: {e}")
            # Absolute last resort
            return f"# {pdf_path.stem}\n\nExtraction failed completely: {str(e)}"

    def _extract_pdf_content(self, pdf_path: Path) -> str:
        """Main extraction orchestrator: 4-tier hierarchy with fallbacks."""

        extraction_methods = [
            ("Marker GPU", self._extract_with_marker, "🎯"),
            ("PyMuPDF + TOC", self._extract_with_pymupdf_toc, "📚"),
            ("Unstructured Emergency", self._extract_with_unstructured_fallback, "🚨"),
            ("Basic Text", self._extract_emergency_text, "🆘")
        ]

        extraction_start = time.time()

        for method_name, method_func, icon in extraction_methods:
            method_start = time.time()

            try:
                logger.info(f"{icon} Attempting {method_name} extraction...")

                # Emergency method always returns something
                if method_name == "Basic Text":
                    result = method_func(pdf_path)
                    logger.info(f"✅ {method_name} completed (fallback)")
                    return result

                # Try extraction method
                result = method_func(pdf_path)
                method_time = time.time() - method_start

                # Validate result quality
                if result and len(result.strip()) > 100:
                    logger.info(f"✅ {method_name} successful in {method_time:.1f}s")

                    # Clear GPU cache after Marker
                    if method_name == "Marker GPU":
                        self._clear_gpu_cache()

                    return result
                else:
                    logger.warning(f"❌ {method_name} insufficient content ({len(result) if result else 0} chars)")

            except Exception as e:
                method_time = time.time() - method_start
                logger.error(f"❌ {method_name} failed after {method_time:.1f}s: {e}")
                continue

        # Should never reach here due to emergency fallback, but just in case
        total_time = time.time() - extraction_start
        logger.error(f"💥 All extraction methods failed after {total_time:.1f}s")
        return f"# {pdf_path.stem}\n\nAll extraction methods failed"

    def _detect_shadowrun_content(self, text: str, filename: str) -> Dict[str, str]:
        """Enhanced Shadowrun-specific content detection and classification."""
        content_lower = text[:3000].lower()  # Analyze first 3K characters
        filename_lower = filename.lower()

        metadata = {
            "edition": "unknown",
            "document_type": self.document_type,
            "primary_focus": "general",
            "detected_sections": [],
            "extraction_quality": "unknown"
        }

        # Edition detection with comprehensive patterns
        edition_patterns = {
            "SR6": ["shadowrun 6", "6th edition", "sr6", "sixth edition", "catalyst game labs 2019", "catalyst 2019"],
            "SR5": ["shadowrun 5", "5th edition", "sr5", "fifth edition", "catalyst game labs 2013", "catalyst 2013",
                    "fifth world"],
            "SR4": ["shadowrun 4", "4th edition", "sr4", "fourth edition", "catalyst game labs 2005", "wizkids",
                    "fanpro"],
            "SR3": ["shadowrun 3", "3rd edition", "sr3", "third edition", "fasa corporation", "fasa corp"],
            "SR2": ["shadowrun 2", "2nd edition", "sr2", "second edition", "fasa 1992"],
            "SR1": ["shadowrun 1", "1st edition", "sr1", "first edition", "fasa 1989", "original shadowrun"]
        }

        for edition, patterns in edition_patterns.items():
            if any(pattern in content_lower or pattern in filename_lower for pattern in patterns):
                metadata["edition"] = edition
                break

        # Document type refinement based on content analysis
        type_indicators = {
            "character_sheet": [
                "character sheet", "player character", "npc", "attributes:", "skills:", "metatype:",
                "priority system", "karma", "nuyen", "contacts:", "cyberware:", "bioware:",
                "spells known", "adept powers", "essence", "magic rating", "street cred"
            ],
            "rulebook": [
                "table of contents", "chapter", "game master", "dice pool", "threshold",
                "glitch", "critical glitch", "extended test", "opposed test", "teamwork test",
                "rule", "modifier", "game mechanics", "core rules"
            ],
            "universe_info": [
                "sixth world", "awakening", "crash of", "matrix 2.0", "corporate court",
                "dragon", "immortal elf", "history", "timeline", "shadowrunner",
                "mr. johnson", "fixers", "seattle", "megacorp", "zaibatsu"
            ],
            "adventure": [
                "adventure", "scenario", "gamemaster", "handout", "scene", "encounter",
                "npc stats", "plot hook", "mission", "shadowrun", "team briefing",
                "background", "getting started", "scenes"
            ]
        }

        # Score each type and override if strong indicators found
        type_scores = {}
        for doc_type, indicators in type_indicators.items():
            score = sum(1 for indicator in indicators if indicator in content_lower)
            type_scores[doc_type] = score

        if type_scores:
            best_type = max(type_scores, key=type_scores.get)
            if type_scores[best_type] > 3:  # Confidence threshold
                metadata["document_type"] = best_type

        # Section detection for better filtering
        section_patterns = {
            "Combat": ["combat", "initiative", "damage", "armor", "weapon", "attack", "defense", "wound"],
            "Magic": ["magic", "spell", "astral", "summoning", "enchanting", "adept", "mage", "spirit"],
            "Matrix": ["matrix", "hacking", "cyberdeck", "programs", "ic", "host", "decker", "technomancer"],
            "Riggers": ["rigger", "drone", "vehicle", "pilot", "autosofts", "jumped in", "control rig"],
            "Character Creation": ["character creation", "priority", "attributes", "skills", "metatype",
                                   "build points"],
            "Gear": ["gear", "equipment", "cyberware", "bioware", "weapons", "armor", "electronics"],
            "Gamemaster": ["gamemaster", "gm", "adventure", "npc", "campaign", "plot", "scenario"],
            "Setting": ["seattle", "sixth world", "corporations", "shadowrun", "awakening", "crash"],
            "Social": ["social", "etiquette", "negotiation", "leadership", "contacts", "reputation"]
        }

        detected_sections = []
        section_counts = {}

        for section, patterns in section_patterns.items():
            count = sum(content_lower.count(pattern) for pattern in patterns)
            section_counts[section] = count
            if count > 0:
                detected_sections.append(section)

        metadata["detected_sections"] = detected_sections

        # Determine primary focus based on most mentioned section
        if section_counts:
            metadata["primary_focus"] = max(section_counts, key=section_counts.get)

        # Assess extraction quality
        if len(text) > 5000 and any(keyword in content_lower for keyword in ["shadowrun", "dice", "test", "skill"]):
            metadata["extraction_quality"] = "high"
        elif len(text) > 1000:
            metadata["extraction_quality"] = "medium"
        else:
            metadata["extraction_quality"] = "low"

        return metadata

    def _create_simple_chunks(self, text: str, chunk_size: int = None) -> List[str]:
        """Create semantic chunks optimized for Shadowrun content."""
        if chunk_size is None:
            chunk_size = self.chunk_size

        # Target characters (rough estimate: 1 token ≈ 4 characters)
        target_chars = chunk_size * 4

        # Split by markdown headers first to preserve structure
        header_pattern = r'\n(#{1,6}\s+.+?)(?=\n)'
        sections = re.split(header_pattern, text)

        chunks = []
        current_chunk = ""

        for i, section in enumerate(sections):
            section = section.strip()
            if not section:
                continue

            # If this section would make chunk too large, finalize current chunk
            if len(current_chunk) + len(section) > target_chars and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = section
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + section
                else:
                    current_chunk = section

        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # If no chunks created or chunks too large, fallback to paragraph-based
        if not chunks or any(len(chunk) > target_chars * 2 for chunk in chunks):
            chunks = self._create_paragraph_chunks(text, target_chars)

        logger.info(f"Created {len(chunks)} chunks from {len(text)} characters")
        return chunks

    def _create_paragraph_chunks(self, text: str, target_chars: int) -> List[str]:
        """Fallback chunking by paragraphs when header-based fails."""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) > target_chars and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph

        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks if chunks else [text]  # Emergency: return full text as single chunk

    def process_pdf(self, pdf_path: str, force_reparse: bool = False) -> Dict[str, str]:
        """Main processing method: Enhanced PDF extraction with 4-tier hierarchy."""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        pdf_name = pdf_path.stem
        output_subdir = self.output_dir / pdf_name
        output_subdir.mkdir(exist_ok=True)

        # Skip if already processed (unless forced)
        json_meta = output_subdir / "_enhanced_metadata.json"
        if json_meta.exists() and not force_reparse:
            logger.info(f"⏭️ Skipping {pdf_name} (already processed with enhanced pipeline)")
            return self._load_existing_files(output_subdir)

        logger.info(f"🚀 Processing {pdf_path} as {self.document_type} with enhanced 4-tier extraction")

        # Set up progress tracking
        progress_handler = None
        processing_start = time.time()

        try:
            if self.progress_callback:
                progress_handler = SimpleProgressHandler(self.progress_callback, pdf_name)
                logging.getLogger().addHandler(progress_handler)
                self.progress_callback("starting", 5, f"Starting enhanced {self.document_type} processing...")

            # STAGE 1: Enhanced PDF extraction (4-tier hierarchy)
            if self.progress_callback:
                self.progress_callback("extracting", 10, "Running enhanced extraction pipeline...")

            full_text = self._extract_pdf_content(pdf_path)

            if not full_text or len(full_text.strip()) < 50:
                logger.error("❌ Insufficient content extracted from all methods!")
                return {}

            extraction_time = time.time() - processing_start
            logger.info(f"✅ Extraction completed in {extraction_time:.1f}s")

            # STAGE 2: Enhanced content analysis
            if self.progress_callback:
                self.progress_callback("analyzing", 60, "Analyzing Shadowrun content with enhanced detection...")

            content_metadata = self._detect_shadowrun_content(full_text, pdf_name)

            # Log detected metadata
            logger.info(
                f"📊 Detected: {content_metadata['document_type']} | {content_metadata['edition']} | {content_metadata['primary_focus']}")

            # STAGE 3: Semantic chunking
            if self.progress_callback:
                self.progress_callback("chunking", 75, "Creating semantic chunks optimized for RAG...")

            chunks = self._create_simple_chunks(full_text)

            if not chunks:
                logger.warning("⚠️ No chunks created, using full text")
                chunks = [full_text]

            # STAGE 4: Save enhanced markdown files
            if self.progress_callback:
                self.progress_callback("saving", 90, "Saving enhanced markdown files...")

            saved_files = self._save_enhanced_chunks(chunks, output_subdir, pdf_name, content_metadata)

            # STAGE 5: Save comprehensive metadata
            self._save_enhanced_metadata(json_meta, pdf_path, len(chunks), content_metadata, extraction_time)

            # Final success
            total_time = time.time() - processing_start
            logger.info(f"🎉 Successfully processed {pdf_name} in {total_time:.1f}s: {len(saved_files)} files created")

            if self.progress_callback:
                self.progress_callback("complete", 100,
                                       f"Enhanced processing complete! Created {len(saved_files)} files in {total_time:.1f}s")

            return saved_files

        except Exception as e:
            total_time = time.time() - processing_start
            logger.error(f"💥 Processing failed after {total_time:.1f}s: {e}")
            logger.error(traceback.format_exc())

            if self.progress_callback:
                self.progress_callback("error", -1, f"Enhanced processing failed: {str(e)}")
            return {}

        finally:
            # Always clean up
            if progress_handler:
                logging.getLogger().removeHandler(progress_handler)
            # Clear GPU cache if we used it
            self._clear_gpu_cache()

    def _save_enhanced_chunks(self, chunks: List[str], output_dir: Path, pdf_name: str, content_metadata: Dict) -> Dict[
        str, str]:
        """Save chunks as enhanced markdown files with rich metadata."""
        saved_files = {}
        metadata_header = self._create_enhanced_metadata_header(pdf_name, content_metadata)

        for i, chunk_text in enumerate(chunks):
            if not chunk_text.strip():
                continue

            try:
                # Generate intelligent title from chunk content
                title = self._extract_chunk_title(chunk_text, i)

                # Clean filename (safe for filesystem)
                safe_title = "".join(c for c in title if c.isalnum() or c in " _-").strip()[:50]
                safe_title = safe_title.replace(" ", "_") or f"chunk_{i + 1:03d}"

                # Format with enhanced metadata
                md_content = f"{metadata_header}\n\n{chunk_text}"

                # Save with unique filename
                file_path = output_dir / f"{safe_title}.md"
                counter = 1
                while file_path.exists():
                    file_path = output_dir / f"{safe_title}_{counter:02d}.md"
                    counter += 1

                file_path.write_text(md_content, encoding='utf-8')
                saved_files[str(file_path)] = md_content

                logger.debug(f"💾 Saved chunk {i + 1}: {file_path.name}")

            except Exception as e:
                logger.warning(f"Failed to save chunk {i + 1}: {e}")
                continue

        return saved_files

    def _extract_chunk_title(self, chunk_text: str, chunk_index: int) -> str:
        """Extract meaningful title from chunk content."""
        lines = [line.strip() for line in chunk_text.split('\n') if line.strip()]

        # Look for markdown headers first
        for line in lines[:5]:
            if line.startswith('#'):
                return re.sub(r'^#+\s*', '', line).strip()

        # Look for title-like content (short, capitalized lines)
        for line in lines[:3]:
            if len(line) < 100 and len(line) > 5:
                # Check if it looks like a title (mostly title case, no lowercase articles)
                words = line.split()
                if len(words) <= 8 and any(word[0].isupper() for word in words):
                    return line

        # Fallback: use first meaningful sentence
        for line in lines[:3]:
            if len(line) > 10 and len(line) < 150:
                return line

        # Final fallback
        return f"Section_{chunk_index + 1:03d}"

    def _create_enhanced_metadata_header(self, pdf_name: str, content_metadata: Dict) -> str:
        """Create comprehensive YAML front matter for markdown files."""
        return f"""---
title: "{pdf_name}"
document_type: "{content_metadata.get('document_type', 'unknown')}"
edition: "{content_metadata.get('edition', 'unknown')}"
primary_focus: "{content_metadata.get('primary_focus', 'general')}"
detected_sections: {content_metadata.get('detected_sections', [])}
extraction_quality: "{content_metadata.get('extraction_quality', 'unknown')}"
processor_version: "enhanced_4tier_v1"
processed_date: "{time.strftime('%Y-%m-%d %H:%M:%S')}"
extraction_hierarchy: ["marker_gpu", "pymupdf_toc", "unstructured_emergency", "basic_text"]
---"""

    def _load_existing_files(self, output_dir: Path) -> Dict[str, str]:
        """Load existing markdown files."""
        return {
            str(f): f.read_text(encoding='utf-8')
            for f in output_dir.glob("*.md")
            if not f.name.startswith('_')  # Skip metadata files
        }

    def _save_enhanced_metadata(self, json_meta: Path, pdf_path: Path, chunks_created: int,
                                content_metadata: Dict, extraction_time: float):
        """Save comprehensive processing metadata."""
        metadata = {
            "source": str(pdf_path),
            "processed_at": time.strftime('%Y-%m-%d %H:%M:%S'),
            "processor_version": "enhanced_4tier_v1",
            "extraction_time_seconds": round(extraction_time, 2),
            "chunks_created": chunks_created,
            "document_type": self.document_type,
            "content_metadata": content_metadata,
            "extraction_hierarchy": ["marker_gpu", "pymupdf_toc", "unstructured_emergency", "basic_text"],
            "processing_config": {
                "chunk_size": self.chunk_size,
                "use_gpu": self.use_gpu,
                "use_ocr": self.use_ocr,
                "debug_mode": self.debug_mode
            },
            "capabilities": {
                "marker_available": MARKER_AVAILABLE,
                "unstructured_available": UNSTRUCTURED_AVAILABLE,
                "gpu_acceleration": self.use_gpu and MARKER_AVAILABLE
            }
        }

        json_meta.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
        logger.debug(f"💾 Saved enhanced metadata: {json_meta}")


# ===========================================
# BACKWARD COMPATIBILITY & CONVENIENCE ALIASES
# ===========================================

# Main class alias for backward compatibility
PDFProcessor = EnhancedPDFProcessor


# Convenience function for quick processing
def process_pdf_enhanced(pdf_path: str, document_type: str = "rulebook",
                         use_gpu: bool = True, progress_callback=None) -> Dict[str, str]:
    """Convenience function for enhanced PDF processing."""
    processor = EnhancedPDFProcessor(
        document_type=document_type,
        use_gpu=use_gpu,
        progress_callback=progress_callback
    )
    return processor.process_pdf(pdf_path, force_reparse=True)


# ===========================================
# MODULE EXPORTS
# ===========================================

__all__ = [
    'EnhancedPDFProcessor',
    'PDFProcessor',  # Backward compatibility
    'process_pdf_enhanced',
    'MARKER_AVAILABLE',
    'UNSTRUCTURED_AVAILABLE'
]

if __name__ == "__main__":
    # Quick test/demo
    import sys

    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        print(f"🎯 Testing enhanced PDF processor on: {pdf_path}")


        def progress_printer(stage, progress, details):
            print(f"Progress [{stage}]: {progress}% - {details}")


        result = process_pdf_enhanced(pdf_path, progress_callback=progress_printer)
        print(f"✅ Processed successfully: {len(result)} files created")
    else:
        print("Usage: python pdf_processor.py <path_to_pdf>")
        print("🎲 Enhanced Shadowrun RAG PDF Processor v1.0")
        print("Features: Marker GPU, PyMuPDF+TOC, Unstructured fallback, Emergency extraction")