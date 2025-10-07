"""Simplified FastAPI backend for Shadowrun RAG - Core functionality only."""

import ollama
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, Dict, List
from pathlib import Path
import logging
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor

from backend.indexer import EnhancedIncrementalIndexer
from backend.simple_retriever import SimpleRetriever
from backend.simple_models import (
    HealthCheckResponse, UploadResponse, JobStatusResponse, 
    QueryResponse, IndexResponse, QueryRequest, IndexRequest,
    DocumentStatsResponse, SystemStatusResponse
)
from tools.pdf_processor import PDFProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Shadowrun RAG API - Simplified")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple progress tracking for uploads only
class SimpleProgressTracker:
    """Simple progress tracker for file processing."""

    def __init__(self):
        self.active_jobs: Dict[str, Dict] = {}
        self.lock = threading.Lock()

    def update_progress(self, job_id: str, stage: str, progress: float, details: str = ""):
        progress_data = {
            "job_id": job_id,
            "stage": stage,
            "progress": max(0, min(100, progress)),
            "details": details,
            "timestamp": time.time()
        }

        with self.lock:
            self.active_jobs[job_id] = progress_data

        logger.info(f"Progress [{job_id}]: {stage} ({progress}%) - {details}")

    def get_job_status(self, job_id: str) -> Optional[Dict]:
        with self.lock:
            return self.active_jobs.get(job_id)

    def cleanup_job(self, job_id: str, delay: int = 300):
        def cleanup():
            time.sleep(delay)
            with self.lock:
                if job_id in self.active_jobs:
                    del self.active_jobs[job_id]
                    logger.info(f"Cleaned up job: {job_id}")

        cleanup_thread = threading.Thread(target=cleanup, daemon=True)
        cleanup_thread.start()

# Global instances
progress_tracker = SimpleProgressTracker()
executor = ThreadPoolExecutor(max_workers=2)
indexer = EnhancedIncrementalIndexer()
retriever = SimpleRetriever()

logger.info("Simplified Shadowrun RAG API initialized")

@app.get("/", response_model=HealthCheckResponse)
def root():
    """Health check."""
    return HealthCheckResponse(
        status="online",
        service="Simplified Shadowrun RAG API",
        active_jobs=len(progress_tracker.active_jobs)
    )

def process_pdf_with_progress(pdf_path: str, job_id: str):
    """Process PDF using only TOC-based marker extraction."""
    
    def progress_callback(stage: str, progress: float, details: str):
        progress_tracker.update_progress(job_id, stage, progress, details)

    try:
        # Only use TOC-based marker extraction
        processor = PDFProcessor(
            document_type="rulebook",  # Fixed to rulebook
            extraction_method="toc_guided",  # Only TOC-guided method
            progress_callback=progress_callback
        )

        result = processor.process_pdf(pdf_path, force_reparse=True)

        progress_tracker.update_progress(
            job_id, "complete", 100,
            f"Processing complete! Created {len(result)} files."
        )

        # Schedule cleanup
        progress_tracker.cleanup_job(job_id)
        return result

    except Exception as e:
        progress_tracker.update_progress(
            job_id, "error", -1,
            f"Processing failed: {str(e)}"
        )
        logger.error(f"Processing failed: {e}")
        raise

@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process a PDF using TOC-guided extraction."""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(400, "Only PDF files are allowed")

    job_id = f"pdf_{file.filename}_{int(time.time() * 1000)}"

    try:
        # Save file
        save_path = Path("data/raw_pdfs") / file.filename
        save_path.parent.mkdir(parents=True, exist_ok=True)

        content = await file.read()
        save_path.write_bytes(content)

        # Save metadata
        metadata = {
            "extraction_method": "toc_guided",
            "uploaded_at": time.time()
        }
        metadata_file = save_path.with_suffix('.meta.json')
        metadata_file.write_text(json.dumps(metadata))

        # Start processing in background
        def start_processing():
            try:
                process_pdf_with_progress(str(save_path), job_id)
            except Exception as e:
                logger.error(f"Background processing failed: {e}")

        thread = threading.Thread(target=start_processing, daemon=True)
        thread.start()

        return UploadResponse(
            job_id=job_id,
            filename=file.filename,
            status="processing",
            message="PDF uploaded. Processing with TOC-guided extraction.",
            poll_url=f"/job/{job_id}"
        )

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(500, f"Upload failed: {str(e)}")

@app.get("/job/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get current status of a processing job."""
    status = progress_tracker.get_job_status(job_id)
    if status:
        return JobStatusResponse(**status)
    else:
        return JobStatusResponse(
            job_id=job_id,
            stage="not_found",
            progress=0,
            details="Job not found or completed",
            timestamp=time.time()
        )

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Simple query with basic section filtering."""
    try:
        # Build simple filter - only section filtering
        where_filter = {}
        if request.section and request.section != "All":
            where_filter["primary_section"] = request.section

        final_filter = where_filter if where_filter else None

        results = retriever.query(
            question=request.question,
            n_results=request.n_results,
            where_filter=final_filter,
            model=request.model
        )
        return QueryResponse(**results)

    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        raise HTTPException(500, str(e))

@app.post("/query_stream")
async def query_stream(request: QueryRequest):
    """Simple streaming query."""
    try:
        # Build simple filter
        where_filter = {}
        if request.section and request.section != "All":
            where_filter["primary_section"] = request.section

        final_filter = where_filter if where_filter else None

        def generate():
            try:
                # Get search results first
                search_results = retriever.search(
                    question=request.question,
                    n_results=request.n_results,
                    where_filter=final_filter
                )

                if not search_results['documents']:
                    yield "No relevant information found in the indexed documents."
                    return

                # Stream the generation
                for token in retriever.query_stream(
                    question=request.question,
                    n_results=request.n_results,
                    where_filter=final_filter,
                    model=request.model
                ):
                    yield token

                # Send metadata after streaming
                metadata_packet = {
                    "sources": list({meta.get('source', 'Unknown') for meta in search_results['metadatas']}),
                    "chunks": search_results['documents'],
                    "distances": search_results['distances'],
                    "metadatas": search_results['metadatas'],
                    "applied_filters": final_filter,
                    "done": True
                }

                metadata_json = json.dumps(metadata_packet, ensure_ascii=False)
                yield f"\n\n__METADATA_START__\n{metadata_json}\n__METADATA_END__\n"

            except Exception as e:
                logger.error(f"Generation error: {e}")
                yield f"\n\nError: {str(e)}"

        return StreamingResponse(generate(), media_type="text/plain")

    except Exception as e:
        logger.error(f"Stream error: {e}")
        raise HTTPException(500, str(e))

@app.post("/index", response_model=IndexResponse)
async def index_documents(request: IndexRequest):
    """Manually trigger indexing."""
    try:
        indexer.index_directory(
            request.directory,
            request.force_reindex
        )
        return IndexResponse(
            status="success",
            message="Indexing complete"
        )
    except Exception as e:
        logger.error(f"Indexing error: {e}")
        raise HTTPException(500, str(e))

@app.get("/stats", response_model=DocumentStatsResponse)
async def get_document_stats():
    """Get basic statistics about indexed documents."""
    try:
        results = retriever.collection.get()

        stats = {
            "total_chunks": len(results.get('ids', [])),
            "sections": {},
            "sources": set()
        }

        for metadata in results.get('metadatas', []):
            if not metadata:
                continue

            # Count sections
            section = metadata.get('primary_section', 'unknown')
            stats["sections"][section] = stats["sections"].get(section, 0) + 1

            # Track unique sources
            if 'source' in metadata:
                stats["sources"].add(Path(metadata['source']).stem)

        stats["unique_documents"] = len(stats["sources"])
        stats["sources"] = list(stats["sources"])  # Convert set to list

        return DocumentStatsResponse(**stats)
    except Exception as e:
        logger.error(f"Error getting document stats: {e}")
        raise HTTPException(500, f"Error getting stats: {str(e)}")

@app.get("/models")
async def list_models():
    """List available Ollama models."""
    try:
        models_response = ollama.list()
        models = []
        for model in models_response.get('models', []):
            model_name = (
                model.get('name') or
                model.get('model') or
                model.get('id') or
                str(model)
            )
            if model_name:
                models.append(model_name)
        return {"models": models}
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        return {"models": [], "error": str(e)}

@app.get("/status", response_model=SystemStatusResponse)
async def status():
    """Return system status."""
    try:
        results = retriever.collection.get()
        doc_count = len(results.get('ids', []))
        sources = set()
        for meta in results.get('metadatas', []):
            if meta and 'source' in meta:
                sources.add(Path(meta['source']).stem)

        # Get model information
        models_available = []
        try:
            models_response = ollama.list()
            for model in models_response.get('models', []):
                model_name = (
                    model.get('name') or
                    model.get('model') or
                    model.get('id') or
                    str(model)
                )
                if model_name:
                    models_available.append(model_name)
        except Exception:
            models_available = []

        return SystemStatusResponse(
            status="online",
            indexed_documents=len(sources),
            indexed_chunks=doc_count,
            active_jobs=len(progress_tracker.active_jobs),
            models_available=models_available
        )
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return SystemStatusResponse(status="degraded", error=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
