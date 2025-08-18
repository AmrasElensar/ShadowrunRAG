"""FastAPI backend server with FIXED WebSocket progress tracking using synchronous callbacks."""
import ollama
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Set
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

# GLOBAL progress tracking with automatic cleanup
class GlobalProgressTracker:
    """Global progress tracker with automatic cleanup and synchronous updates."""

    def __init__(self):
        self.active_jobs: Dict[str, Dict] = {}
        self.websocket_connections: Set[WebSocket] = set()
        self.lock = threading.Lock()  # Use threading lock for sync operations
        self.cleanup_tasks: Dict[str, asyncio.Task] = {}
        logger.info("🔧 GlobalProgressTracker initialized")

    def update_progress_sync(self, job_id: str, stage: str, progress: float, details: str = ""):
        """Synchronous progress update - called from PDF processor thread."""

        # Prevent recursion - check if we're already processing this exact update
        progress_key = f"{job_id}:{stage}:{progress}"
        if hasattr(self, '_in_progress_update') and self._in_progress_update == progress_key:
            logger.warning(f"Preventing recursive progress update for {progress_key}")
            return

        self._in_progress_update = progress_key

        try:
            progress_data = {
                "job_id": job_id,
                "stage": stage,
                "progress": max(0, min(100, progress)),  # Clamp 0-100
                "details": details,
                "timestamp": time.time()
            }

            with self.lock:
                # Store globally (survives disconnections)
                self.active_jobs[job_id] = progress_data

            # Schedule async broadcast in the main event loop
            try:
                # Get the main event loop safely
                loop = asyncio.get_event_loop()
                if loop and loop.is_running():
                    # Schedule coroutine to run in the main loop
                    asyncio.run_coroutine_threadsafe(
                        self.broadcast_to_all(progress_data),
                        loop
                    )
                else:
                    logger.warning("No running event loop found for broadcast")
            except RuntimeError:
                # Handle case where there's no event loop in this thread
                logger.warning("No event loop available in current thread for broadcast")
            except Exception as e:
                logger.warning(f"Could not schedule broadcast: {e}")

            # Log for debugging
            logger.info(f"Progress [{job_id}]: {stage} ({progress}%) - {details}")

            # Schedule cleanup if complete or error
            if stage in ["complete", "error"] and job_id not in self.cleanup_tasks:
                self._schedule_cleanup(job_id)

        finally:
            # Clear recursion guard
            if hasattr(self, '_in_progress_update') and self._in_progress_update == progress_key:
                delattr(self, '_in_progress_update')

    async def broadcast_to_all(self, message: dict):
        """Broadcast message to all connected WebSocket clients."""
        if not self.websocket_connections:
            logger.debug("No WebSocket connections to broadcast to")
            return

        disconnected = set()
        message_json = json.dumps(message)

        logger.info(f"Broadcasting to {len(self.websocket_connections)} WebSocket clients")

        for connection in list(self.websocket_connections):
            try:
                await connection.send_text(message_json)
            except Exception as e:
                logger.warning(f"Failed to send to WebSocket: {e}")
                disconnected.add(connection)

        # Clean up disconnected clients
        self.websocket_connections.difference_update(disconnected)
        if disconnected:
            logger.info(f"Cleaned up {len(disconnected)} disconnected WebSocket clients")

    async def add_connection(self, websocket: WebSocket):
        """Add new WebSocket connection and send current progress."""
        await websocket.accept()
        self.websocket_connections.add(websocket)
        logger.info(f"✅ WebSocket connected. Total connections: {len(self.websocket_connections)}")

        # Send current active jobs to new connection
        with self.lock:
            for job_data in self.active_jobs.values():
                try:
                    await websocket.send_text(json.dumps(job_data))
                    logger.debug(f"Sent existing job data to new WebSocket: {job_data['job_id']}")
                except:
                    pass  # Connection might have closed immediately

    def remove_connection(self, websocket: WebSocket):
        """Remove WebSocket connection."""
        self.websocket_connections.discard(websocket)
        logger.info(f"❌ WebSocket disconnected. Remaining connections: {len(self.websocket_connections)}")

    def get_job_status(self, job_id: str) -> Optional[Dict]:
        """Get current job status (for polling fallback)."""
        with self.lock:
            return self.active_jobs.get(job_id)

    def _schedule_cleanup(self, job_id: str):
        """Schedule job cleanup after completion."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                task = asyncio.run_coroutine_threadsafe(
                    self._cleanup_job_async(job_id, delay=30),
                    loop
                )
                self.cleanup_tasks[job_id] = task
        except Exception as e:
            logger.warning(f"Could not schedule cleanup for {job_id}: {e}")

    async def _cleanup_job_async(self, job_id: str, delay: int = 30):
        """Remove completed job after delay."""
        await asyncio.sleep(delay)
        with self.lock:
            if job_id in self.active_jobs:
                status = self.active_jobs[job_id].get('stage')
                if status in ['complete', 'error']:
                    del self.active_jobs[job_id]
                    logger.info(f"🧹 Cleaned up completed job: {job_id}")

        # Remove from cleanup tasks
        if job_id in self.cleanup_tasks:
            del self.cleanup_tasks[job_id]

    def clear_stale_jobs(self, max_age_seconds: int = 3600):
        """Clear jobs older than max_age_seconds."""
        current_time = time.time()
        with self.lock:
            stale_jobs = []
            for job_id, job_data in self.active_jobs.items():
                age = current_time - job_data.get('timestamp', current_time)
                if age > max_age_seconds:
                    stale_jobs.append(job_id)

            for job_id in stale_jobs:
                del self.active_jobs[job_id]
                logger.info(f"🧹 Cleared stale job: {job_id}")

            return len(stale_jobs)

# Global tracker instance - Initialize immediately
progress_tracker = GlobalProgressTracker()

# Thread pool for CPU-intensive tasks
executor = ThreadPoolExecutor(max_workers=2)

# Initialize components
indexer = IncrementalIndexer()
retriever = Retriever()

# Periodic cleanup task
async def periodic_cleanup():
    """Background task to clean up stale jobs periodically."""
    while True:
        await asyncio.sleep(600)  # Every 10 minutes
        try:
            cleared = progress_tracker.clear_stale_jobs(max_age_seconds=3600)
            if cleared > 0:
                logger.info(f"Periodic cleanup: cleared {cleared} stale jobs")
        except Exception as e:
            logger.error(f"Periodic cleanup error: {e}")

# Start cleanup task on startup
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(periodic_cleanup())
    logger.info("🚀 Started periodic cleanup task")

# Log initialization
logger.info("🚀 FastAPI app initialized with components:")
logger.info(f"   - GlobalProgressTracker: ✅")
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

def process_pdf_with_sync_progress(pdf_path: str, job_id: str):
    """Process PDF with synchronous progress callback."""
    try:
        # Create processor with synchronous progress callback
        def sync_progress_callback(job_id: str, stage: str, progress: float, details: str):
            # Add recursion protection here too
            try:
                progress_tracker.update_progress_sync(job_id, stage, progress, details)
            except Exception as e:
                logger.warning(f"Progress update failed: {e}")

        processor = PDFProcessor(progress_callback=sync_progress_callback)

        # Initial progress
        progress_tracker.update_progress_sync(job_id, "starting", 5, "Initializing PDF processing...")

        # Process the PDF
        result = processor.process_pdf(pdf_path, force_reparse=True, job_id=job_id)

        # Index the results
        progress_tracker.update_progress_sync(job_id, "indexing", 95, "Indexing processed content...")
        indexer.index_directory("data/processed_markdown")

        # Complete
        progress_tracker.update_progress_sync(job_id, "complete", 100, f"Processing complete! Created {len(result)} files.")

        return result

    except Exception as e:
        logger.error(f"PDF processing failed: {e}")
        logger.error(traceback.format_exc())
        try:
            progress_tracker.update_progress_sync(job_id, "error", -1, f"Processing failed: {str(e)}")
        except:
            pass  # Ignore errors in error handling
        raise

@app.websocket("/ws/progress")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time progress updates."""
    logger.info("🔌 New WebSocket connection attempt")

    try:
        await progress_tracker.add_connection(websocket)

        # Keep connection alive and handle any incoming messages
        while True:
            try:
                # Wait for any message (ping/pong, etc.)
                await websocket.receive_text()
            except WebSocketDisconnect:
                logger.info("WebSocket client disconnected normally")
                break
            except Exception as e:
                logger.warning(f"WebSocket receive error: {e}")
                break

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected during handshake")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        progress_tracker.remove_connection(websocket)

@app.get("/")
def root():
    """Health check."""
    return {
        "status": "online",
        "service": "Shadowrun RAG API",
        "websocket_connections": len(progress_tracker.websocket_connections),
        "active_jobs": len(progress_tracker.active_jobs)
    }

@app.post("/upload")
async def upload_pdf_with_progress(file: UploadFile = File(...)):
    """Upload and process a PDF with real-time progress tracking."""
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

        # Process in background using thread pool
        loop = asyncio.get_event_loop()
        loop.run_in_executor(
            executor,
            process_pdf_with_sync_progress,
            str(save_path),
            job_id
        )

        return {
            "job_id": job_id,
            "filename": file.filename,
            "status": "processing",
            "message": "PDF uploaded. Processing started with real-time progress tracking."
        }

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(500, f"Upload failed: {str(e)}")

@app.get("/job/{job_id}")
async def get_job_status(job_id: str):
    """Get current status of a processing job (polling fallback)."""
    status = progress_tracker.get_job_status(job_id)
    if status:
        return status
    else:
        return {"status": "not_found", "message": "Job not found or completed"}

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

        # Include active job info
        active_job_summary = []
        for job_id, job_data in progress_tracker.active_jobs.items():
            active_job_summary.append({
                "job_id": job_id[-8:],  # Last 8 chars for privacy
                "stage": job_data.get("stage"),
                "progress": job_data.get("progress")
            })

        # Safer model list retrieval
        try:
            models_list = ollama.list().get('models', [])
            model_names = []
            for m in models_list:
                if isinstance(m, dict):
                    # Try different possible keys for model name
                    model_name = m.get('name') or m.get('model') or m.get('id') or str(m)
                    if model_name:
                        model_names.append(model_name)
                else:
                    model_names.append(str(m))
            models_available = model_names
        except Exception as e:
            logger.error(f"Error getting model list: {e}")
            models_available = ["llama3"]  # fallback

        return {
            "status": "online",
            "indexed_documents": len(sources),
            "indexed_chunks": doc_count,
            "active_jobs": len(progress_tracker.active_jobs),
            "active_job_details": active_job_summary,
            "websocket_connections": len(progress_tracker.websocket_connections),
            "models_available": models_available
        }
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return {"status": "degraded", "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)