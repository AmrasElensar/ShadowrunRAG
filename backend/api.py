"""FastAPI backend server with WebSocket progress tracking - FIXED VERSION."""
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
# WATCHER DISABLED - was causing auto-processing
# from .watcher import PDFWatcher
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

# Global progress tracking
active_jobs: Dict[str, Dict] = {}
websocket_connections: Set[WebSocket] = set()

# Thread pool for CPU-intensive tasks
executor = ThreadPoolExecutor(max_workers=2)

# Global task tracking
cleanup_tasks = set()

class ProgressLogger:
    """Custom logger that broadcasts progress via WebSocket."""

    def __init__(self, job_id: str, filename: str):
        self.job_id = job_id
        self.filename = filename
        self.start_time = time.time()

    async def log_progress(self, stage: str, progress: float, details: str = ""):
        """Log progress and broadcast to WebSocket clients."""
        message = {
            "job_id": self.job_id,
            "filename": self.filename,
            "stage": stage,
            "progress": progress,
            "details": details,
            "elapsed_time": time.time() - self.start_time,
            "timestamp": time.time()
        }

        # Store in active jobs
        active_jobs[self.job_id] = message

        # Broadcast to all WebSocket connections
        disconnected = set()
        for connection in websocket_connections:
            try:
                await connection.send_text(json.dumps(message))
            except:
                disconnected.add(connection)

        # Clean up disconnected clients
        websocket_connections.difference_update(disconnected)

        # Also log normally
        logger.info(f"Progress [{self.job_id}]: {stage} ({progress}%) - {details}")

async def cleanup_job(job_id: str):
    """Remove completed job from active tracking after delay."""
    await asyncio.sleep(30)  # Keep visible for 30 seconds
    if job_id in active_jobs:
        del active_jobs[job_id]
        logger.info(f"Cleaned up job: {job_id}")

def schedule_cleanup(job_id: str):
    """Schedule cleanup task and track it properly."""
    task = asyncio.create_task(cleanup_job(job_id))
    cleanup_tasks.add(task)
    task.add_done_callback(cleanup_tasks.discard)  # Remove from set when done

# Initialize components
indexer = IncrementalIndexer()
retriever = Retriever()
processor = PDFProcessor()
# WATCHER DISABLED - was causing the auto-processing issue
# watcher = PDFWatcher(indexer=indexer, processor=processor)

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

@app.websocket("/ws/progress")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time progress updates."""
    await websocket.accept()
    websocket_connections.add(websocket)
    logger.info(f"WebSocket connected. Total connections: {len(websocket_connections)}")

    try:
        # Send current active jobs on connect
        for job_id, progress in active_jobs.items():
            await websocket.send_text(json.dumps(progress))

        # Keep connection alive
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        websocket_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Remaining connections: {len(websocket_connections)}")

@app.get("/")
def root():
    """Health check."""
    return {"status": "online", "service": "Shadowrun RAG API"}

async def process_pdf_with_progress_async(pdf_path: str, job_id: str):
    """Process PDF with progress tracking in a separate thread."""
    progress_logger = ProgressLogger(job_id, Path(pdf_path).name)

    def cpu_intensive_processing():
        """Run the actual PDF processing in a thread."""
        try:
            # Create a new processor instance for this thread
            thread_processor = PDFProcessor()

            # Process the PDF (this is the slow part)
            result = thread_processor.process_pdf(pdf_path, force_reparse=True)

            # Trigger indexing after processing
            thread_indexer = IncrementalIndexer()
            thread_indexer.index_directory("data/processed_markdown")

            return result

        except Exception as e:
            logger.error(f"PDF processing failed: {e}")
            raise e

    try:
        # Only send real progress updates, not fake ones
        await progress_logger.log_progress("starting", 5, f"Starting PDF processing...")

        # Run the CPU-intensive work in the thread pool
        # All the real progress will come from the PDFProcessor logs
        result = await asyncio.get_event_loop().run_in_executor(
            executor, cpu_intensive_processing
        )

        # Only update when we're actually done
        await progress_logger.log_progress("complete", 100, f"Processing complete! Created {len(result)} files.")

        # Clean up from active jobs after a delay - properly scheduled
        schedule_cleanup(job_id)

        return result

    except Exception as e:
        await progress_logger.log_progress("error", -1, f"Processing failed: {str(e)}")
        logger.error(f"Processing failed: {e}")
        logger.error(traceback.format_exc())
        raise

@app.post("/upload")
async def upload_pdf_with_progress(
    file: UploadFile = File(...)
):
    """Upload and process a PDF with WebSocket progress tracking."""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(400, "Only PDF files are allowed")

    # Generate unique job ID
    job_id = f"{file.filename}_{int(time.time() * 1000)}"

    try:
        # Save file quickly and return immediately
        save_path = Path("data/raw_pdfs") / file.filename
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Read and save file
        content = await file.read()
        save_path.write_bytes(content)

        # Start processing in completely separate thread to avoid blocking
        def start_processing():
            # Create new event loop for this thread
            asyncio.set_event_loop(asyncio.new_event_loop())
            loop = asyncio.get_event_loop()
            try:
                loop.run_until_complete(process_pdf_with_progress_async(str(save_path), job_id))
            finally:
                loop.close()

        # Start in daemon thread
        thread = threading.Thread(target=start_processing, daemon=True)
        thread.start()

        # Return immediately - don't wait for processing
        return {
            "job_id": job_id,
            "filename": file.filename,
            "status": "processing",
            "message": "PDF uploaded successfully. Processing started in background."
        }

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(500, f"Upload failed: {str(e)}")

@app.get("/job/{job_id}")
async def get_job_status(job_id: str):
    """Get current status of a processing job."""
    if job_id in active_jobs:
        return active_jobs[job_id]
    else:
        return {"status": "not_found", "message": "Job not found or completed"}

@app.post("/query")
async def query(request: QueryRequest):
    """Query the RAG system with full context."""
    try:
        # Build metadata filter - FIXED: Only add non-empty conditions
        where_filter = {}

        if request.filter_source and request.filter_source.strip():
            where_filter["source"] = {"$contains": request.filter_source}
        if request.filter_section and request.filter_section.strip():
            where_filter["Section"] = request.filter_section
        if request.filter_subsection and request.filter_subsection.strip():
            where_filter["Subsection"] = request.filter_subsection

        # Role-based fallback (only if no manual section)
        role_to_section = {
            "decker": "Matrix", "hacker": "Matrix", "mage": "Magic", "adept": "Magic",
            "street_samurai": "Combat", "rigger": "Riggers", "technomancer": "Technomancy"
        }
        if (request.character_role and
                request.character_role in role_to_section and
                "Section" not in where_filter):
            where_filter["Section"] = role_to_section[request.character_role]

        # Pass None if empty dict
        final_filter = where_filter if where_filter else None

        # Query retriever
        results = retriever.query(
            question=request.question,
            n_results=request.n_results,
            query_type=request.query_type,
            where_filter=final_filter,  # Fixed: Use final_filter
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
        # Build filter - FIXED: Only add non-empty conditions
        where_filter = {}

        if request.filter_source and request.filter_source.strip():
            where_filter["source"] = {"$contains": request.filter_source}
        if request.filter_section and request.filter_section.strip():
            where_filter["Section"] = request.filter_section
        if request.filter_subsection and request.filter_subsection.strip():
            where_filter["Subsection"] = request.filter_subsection

        # Role-based section filter (only if no manual section set)
        role_to_section = {
            "decker": "Matrix", "hacker": "Matrix", "mage": "Magic", "adept": "Magic",
            "street_samurai": "Combat", "rigger": "Riggers", "technomancer": "Technomancy"
        }
        if (request.character_role and
                request.character_role in role_to_section and
                "Section" not in where_filter):  # Don't override manual section
            where_filter["Section"] = role_to_section[request.character_role]

        # Pass None if empty dict (ChromaDB requirement)
        final_filter = where_filter if where_filter else None

        def generate():
            try:
                # Get search results first (non-streaming)
                search_results = retriever.search(
                    question=request.question,
                    n_results=request.n_results,
                    where_filter=final_filter
                )

                if not search_results['documents']:
                    yield "No relevant information found in the indexed documents."
                    return

                # Stream the actual generation (REAL streaming)
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

                # Send metadata after streaming completes
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
        # Get unique sources from ChromaDB
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
            "active_jobs": len(active_jobs),
            "websocket_connections": len(websocket_connections),
            "models_available": [m['name'] for m in ollama.list().get('models', [])]
        }
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return {"status": "degraded", "error": str(e)}

@app.get("/prompts")
async def list_prompts():
    """List available prompt templates (placeholder)."""
    return {
        "prompts": [
            "general_rule",
            "combat_query",
            "magic_query",
            "matrix_query",
            "character_action"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)