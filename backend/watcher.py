"""File system watcher for automatic indexing."""

import time
from pathlib import Path
from typing import Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import logging
from .indexer import IncrementalIndexer
from tools.pdf_processor import PDFProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFWatcher(FileSystemEventHandler):
    """Watch for PDF changes and trigger processing."""
    
    def __init__(
        self,
        watch_dir: str = "data/raw_pdfs",
        indexer: Optional[IncrementalIndexer] = None,
        processor: Optional[PDFProcessor] = None
    ):
        self.watch_dir = Path(watch_dir)
        self.watch_dir.mkdir(parents=True, exist_ok=True)
        
        self.indexer = indexer or IncrementalIndexer()
        self.processor = processor or PDFProcessor()
        
        # Track processing to avoid duplicates
        self.processing = set()
    
    def on_created(self, event):
        """Handle new file creation."""
        if not event.is_directory and event.src_path.endswith('.pdf'):
            self.process_pdf(event.src_path)
    
    def on_modified(self, event):
        """Handle file modification."""
        if not event.is_directory and event.src_path.endswith('.pdf'):
            self.process_pdf(event.src_path)
    
    def on_deleted(self, event):
        """Handle file deletion."""
        if not event.is_directory and event.src_path.endswith('.pdf'):
            self.remove_pdf(event.src_path)
    
    def process_pdf(self, pdf_path: str):
        """Process and index a PDF."""
        pdf_path = Path(pdf_path)
        
        if str(pdf_path) in self.processing:
            return
        
        self.processing.add(str(pdf_path))
        
        try:
            logger.info(f"Processing PDF: {pdf_path.name}")
            
            # Convert to markdown
            saved_files = self.processor.process_pdf(pdf_path)
            
            # Index the markdown files
            self.indexer.index_directory("data/processed_markdown")
            
            logger.info(f"Successfully processed and indexed: {pdf_path.name}")
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
        finally:
            self.processing.discard(str(pdf_path))
    
    def remove_pdf(self, pdf_path: str):
        """Remove PDF data from index."""
        pdf_path = Path(pdf_path)
        pdf_name = pdf_path.stem
        
        # Find and remove markdown files
        markdown_dir = Path("data/processed_markdown") / pdf_name
        if markdown_dir.exists():
            for md_file in markdown_dir.glob("*.md"):
                self.indexer.remove_document(str(md_file))
            
            # Remove directory
            import shutil
            shutil.rmtree(markdown_dir)
            
            logger.info(f"Removed indexed data for: {pdf_name}")

def start_watcher(watch_dir: str = "data/raw_pdfs"):
    """Start the file system watcher."""
    event_handler = PDFWatcher(watch_dir)
    observer = Observer()
    observer.schedule(event_handler, watch_dir, recursive=False)
    observer.start()
    
    logger.info(f"Watching directory: {watch_dir}")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()