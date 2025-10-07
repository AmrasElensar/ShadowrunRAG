"""Simplified PDF processor - TOC-guided extraction only."""

# Import the simplified processor
from tools.simple_pdf_processor import SimplePDFProcessor

class PDFProcessor:
    """Simplified PDF processor wrapper for backward compatibility."""
    
    def __init__(self, document_type: str = "rulebook", extraction_method: str = "toc_guided", 
                 vision_model: str = None, progress_callback=None):
        """Initialize with simplified processor."""
        # Only support toc_guided method now
        if extraction_method != "toc_guided":
            import logging
            logging.warning(f"Extraction method '{extraction_method}' not supported. Using 'toc_guided'.")
        
        self.processor = SimplePDFProcessor(progress_callback=progress_callback)
    
    def process_pdf(self, pdf_path: str, force_reparse: bool = False):
        """Process PDF using TOC-guided extraction."""
        return self.processor.process_pdf(pdf_path, force_reparse)
