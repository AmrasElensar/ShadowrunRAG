"""FastAPI backend server with simplified polling-only progress tracking."""
import ollama
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import shutil
from pathlib import Path
import logging
import asyncio
import json
import time
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor
from .indexer import IncrementalIndexer
from .retriever import Retriever
from tools.pdf_processor import PDFProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Shadowrun RAG API")

# CORS for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# SIMPLIFIED progress tracking (polling-only)
class PollingProgressTracker:
    """Simple progress tracker for polling-based updates."""

    def __init__(self):
        self.active_jobs: Dict[str, Dict] = {}
        self.lock = threading.Lock()
        logger.info("🔧 PollingProgressTracker initialized")

    def update_progress(self, job_id: str, stage: str, progress: float, details: str = ""):
        """Update progress (thread-safe)."""
        progress_data = {
            "job_id": job_id,
            "stage": stage,
            "progress": max(0, min(100, progress)),  # Clamp 0-100
            "details": details,
            "timestamp": time.time()
        }

        with self.lock:
            self.active_jobs[job_id] = progress_data

        # Log for debugging
        logger.info(f"Progress [{job_id}]: {stage} ({progress}%) - {details}")

    def get_job_status(self, job_id: str) -> Optional[Dict]:
        """Get current job status."""
        with self.lock:
            return self.active_jobs.get(job_id)

    def cleanup_job(self, job_id: str, delay: int = 300):  # 5 minutes cleanup
        """Remove completed job after delay."""
        def cleanup():
            time.sleep(delay)
            with self.lock:
                if job_id in self.active_jobs:
                    del self.active_jobs[job_id]
                    logger.info(f"🧹 Cleaned up job: {job_id}")

        # Run cleanup in background thread
        cleanup_thread = threading.Thread(target=cleanup, daemon=True)
        cleanup_thread.start()

    def get_all_jobs(self) -> Dict[str, Dict]:
        """Get all active jobs (for debugging)."""
        with self.lock:
            return self.active_jobs.copy()

# Global tracker instance
progress_tracker = PollingProgressTracker()

# Thread pool for CPU-intensive tasks
executor = ThreadPoolExecutor(max_workers=2)

# Initialize components
indexer = IncrementalIndexer()
retriever = Retriever()

# Log initialization
logger.info("🚀 FastAPI app initialized with components:")
logger.info(f"   - PollingProgressTracker: ✅")
logger.info(f"   - IncrementalIndexer: ✅")
logger.info(f"   - Retriever: ✅")

class QueryRequest(BaseModel):
    question: str
    n_results: int = 5
    query_type: str = "general"
    filter_source: Optional[str] = None
    filter_section: Optional[str] = None
    filter_subsection: Optional[str] = None
    character_role: Optional[str] = None
    character_stats: Optional[str] = None
    edition: Optional[str] = None
    model: Optional[str] = None

class IndexRequest(BaseModel):
    directory: str = "data/processed_markdown"
    force_reindex: bool = False

@app.get("/")
def root():
    """Health check."""
    return {
        "status": "online",
        "service": "Shadowrun RAG API",
        "active_jobs": len(progress_tracker.active_jobs),
        "tracking_method": "polling"
    }

def process_pdf_with_progress(pdf_path: str, job_id: str):
    """Process PDF with progress tracking (synchronous version)."""
    try:
        # Create processor with progress callback
        def progress_callback(stage, progress, details):
            progress_tracker.update_progress(job_id, stage, progress, details)

        progress_tracker.update_progress(job_id, "starting", 5, "Initializing PDF processing...")

        processor = PDFProcessor(progress_callback=progress_callback)

        # Process the PDF
        result = processor.process_pdf(pdf_path, force_reparse=True)

        # Index the results
        progress_tracker.update_progress(job_id, "indexing", 95, "Adding to search index...")
        indexer.index_directory("data/processed_markdown")

        progress_tracker.update_progress(job_id, "complete", 100, f"Processing complete! Created {len(result)} files.")

        # Schedule cleanup
        progress_tracker.cleanup_job(job_id)

        return result

    except Exception as e:
        progress_tracker.update_progress(job_id, "error", -1, f"Processing failed: {str(e)}")
        logger.error(f"Processing failed: {e}")
        logger.error(traceback.format_exc())
        raise

@app.post("/upload")
async def upload_pdf_with_progress(file: UploadFile = File(...)):
    """Upload and process a PDF with progress tracking."""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(400, "Only PDF files are allowed")

    # Generate unique job ID
    job_id = f"{file.filename}_{int(time.time() * 1000)}"

    try:
        # Save file quickly
        save_path = Path("data/raw_pdfs") / file.filename
        save_path.parent.mkdir(parents=True, exist_ok=True)

        content = await file.read()
        save_path.write_bytes(content)

        # Start processing in background thread
        def start_processing():
            try:
                process_pdf_with_progress(str(save_path), job_id)
            except Exception as e:
                logger.error(f"Background processing failed: {e}")

        thread = threading.Thread(target=start_processing, daemon=True)
        thread.start()

        return {
            "job_id": job_id,
            "filename": file.filename,
            "status": "processing",
            "message": "PDF uploaded. Processing started with progress tracking.",
            "poll_url": f"/job/{job_id}"
        }

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(500, f"Upload failed: {str(e)}")

@app.get("/job/{job_id}")
async def get_job_status(job_id: str):
    """Get current status of a processing job."""
    status = progress_tracker.get_job_status(job_id)
    if status:
        return status
    else:
        return {
            "job_id": job_id,
            "status": "not_found",
            "message": "Job not found or completed",
            "timestamp": time.time()
        }

@app.get("/jobs")
async def list_all_jobs():
    """List all active jobs (for debugging)."""
    return {
        "active_jobs": progress_tracker.get_all_jobs(),
        "count": len(progress_tracker.active_jobs)
    }

@app.post("/query")
async def query(request: QueryRequest):
    """Query the RAG system with full context."""
    try:
        # Build metadata filter
        where_filter = {}

        if request.filter_source and request.filter_source.strip():
            where_filter["source"] = {"$contains": request.filter_source}
        if request.filter_section and request.filter_section.strip():
            where_filter["Section"] = request.filter_section
        if request.filter_subsection and request.filter_subsection.strip():
            where_filter["Subsection"] = request.filter_subsection

        # Role-based fallback
        role_to_section = {
            "decker": "Matrix", "hacker": "Matrix", "mage": "Magic", "adept": "Magic",
            "street_samurai": "Combat", "rigger": "Riggers", "technomancer": "Technomancy"
        }
        if (request.character_role and
                request.character_role in role_to_section and
                "Section" not in where_filter):
            where_filter["Section"] = role_to_section[request.character_role]

        final_filter = where_filter if where_filter else None

        results = retriever.query(
            question=request.question,
            n_results=request.n_results,
            query_type=request.query_type,
            where_filter=final_filter,
            character_role=request.character_role,
            character_stats=request.character_stats,
            edition=request.edition
        )
        return results

    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        raise HTTPException(500, str(e))

@app.post("/query_stream")
async def query_stream(request: QueryRequest):
    """Stream the answer and include full metadata at the end."""
    try:
        # Build filter
        where_filter = {}

        if request.filter_source and request.filter_source.strip():
            where_filter["source"] = {"$contains": request.filter_source}
        if request.filter_section and request.filter_section.strip():
            where_filter["Section"] = request.filter_section
        if request.filter_subsection and request.filter_subsection.strip():
            where_filter["Subsection"] = request.filter_subsection

        role_to_section = {
            "decker": "Matrix", "hacker": "Matrix", "mage": "Magic", "adept": "Magic",
            "street_samurai": "Combat", "rigger": "Riggers", "technomancer": "Technomancy"
        }
        if (request.character_role and
                request.character_role in role_to_section and
                "Section" not in where_filter):
            where_filter["Section"] = role_to_section[request.character_role]

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
                        query_type=request.query_type,
                        where_filter=final_filter,
                        character_role=request.character_role,
                        character_stats=request.character_stats,
                        edition=request.edition,
                        model=request.model
                ):
                    yield token

                # Send metadata after streaming
                metadata_packet = {
                    "sources": list({meta.get('source', 'Unknown') for meta in search_results['metadatas']}),
                    "chunks": search_results['documents'],
                    "distances": search_results['distances'],
                    "metadatas": search_results['metadatas'],
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

@app.post("/index")
async def index_documents(request: IndexRequest):
    """Manually trigger indexing."""
    try:
        indexer.index_directory(
            request.directory,
            request.force_reindex
        )
        return {"status": "success", "message": "Indexing complete"}
    except Exception as e:
        logger.error(f"Indexing error: {e}")
        raise HTTPException(500, str(e))

@app.get("/documents")
async def list_documents():
    """List all indexed documents."""
    try:
        results = retriever.collection.get()
        sources = set()

        for metadata in results.get('metadatas', []):
            if metadata and 'source' in metadata:
                sources.add(metadata['source'])

        return {"documents": sorted(list(sources))}
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        return {"documents": []}

@app.get("/models")
async def list_models():
    """List available Ollama models."""
    try:
        import ollama
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
        return {"models": ["llama3", "mistral", "codellama"], "error": str(e)}

@app.get("/status")
async def status():
    """Return system status and indexed document count."""
    try:
        results = retriever.collection.get()
        doc_count = len(results.get('ids', []))
        sources = set()
        for meta in results.get('metadatas', []):
            if meta and 'source' in meta:
                sources.add(Path(meta['source']).parent.name)

        return {
            "status": "online",
            "indexed_documents": len(sources),
            "indexed_chunks": doc_count,
            "active_jobs": len(progress_tracker.active_jobs),
            "tracking_method": "polling",
            "models_available": [m['name'] for m in ollama.list().get('models', [])]
        }
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return {"status": "degraded", "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)