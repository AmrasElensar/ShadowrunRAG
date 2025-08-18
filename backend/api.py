"""FastAPI backend server with FIXED WebSocket progress tracking using global state."""
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

# GLOBAL progress tracking (NOT session-dependent)
class GlobalProgressTracker:
    """Global progress tracker that survives WebSocket disconnections."""

    def __init__(self):
        self.active_jobs: Dict[str, Dict] = {}
        self.websocket_connections: Set[WebSocket] = set()
        self.lock = asyncio.Lock()
        logger.info("🔧 GlobalProgressTracker initialized")

    async def update_progress(self, job_id: str, stage: str, progress: float, details: str = ""):
        """Update progress and broadcast to ALL connected clients."""
        progress_data = {
            "job_id": job_id,
            "stage": stage,
            "progress": max(0, min(100, progress)),  # Clamp 0-100
            "details": details,
            "timestamp": time.time()
        }

        async with self.lock:
            # Store globally (survives disconnections)
            self.active_jobs[job_id] = progress_data

        # Broadcast to ALL connected WebSocket clients
        await self.broadcast_to_all(progress_data)

        # Also log for debugging
        logger.info(f"Progress [{job_id}]: {stage} ({progress}%) - {details}")

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
        async with self.lock:
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
        return self.active_jobs.get(job_id)

    async def cleanup_job(self, job_id: str, delay: int = 30):
        """Remove completed job after delay."""
        await asyncio.sleep(delay)
        async with self.lock:
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]
                logger.info(f"🧹 Cleaned up job: {job_id}")

# Global tracker instance - Initialize immediately
progress_tracker = GlobalProgressTracker()

# Thread pool for CPU-intensive tasks
executor = ThreadPoolExecutor(max_workers=2)

# Initialize components
indexer = IncrementalIndexer()
retriever = Retriever()

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

async def process_pdf_with_real_progress(pdf_path: str, job_id: str):
    """Process PDF with REAL progress tracking from unstructured logs."""

    def cpu_intensive_processing():
        """Run PDF processing in thread with progress tracking."""
        try:
            # Create processor with progress callback
            processor = PDFProcessor(progress_callback=lambda stage, progress, details:
                asyncio.create_task(progress_tracker.update_progress(job_id, stage, progress, details))
            )

            # Process the PDF (this hooks into unstructured logging)
            result = processor.process_pdf(pdf_path, force_reparse=True)

            # Index the results
            indexer.index_directory("data/processed_markdown")

            return result

        except Exception as e:
            logger.error(f"PDF processing failed: {e}")
            raise e

    try:
        await progress_tracker.update_progress(job_id, "starting", 5, "Initializing PDF processing...")

        # Run CPU-intensive work in thread pool
        result = await asyncio.get_event_loop().run_in_executor(
            executor, cpu_intensive_processing
        )

        await progress_tracker.update_progress(job_id, "complete", 100, f"Processing complete! Created {len(result)} files.")

        # Schedule cleanup
        asyncio.create_task(progress_tracker.cleanup_job(job_id))

        return result

    except Exception as e:
        await progress_tracker.update_progress(job_id, "error", -1, f"Processing failed: {str(e)}")
        logger.error(f"Processing failed: {e}")
        logger.error(traceback.format_exc())
        raise

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

        # Start processing in background thread
        def start_processing():
            asyncio.set_event_loop(asyncio.new_event_loop())
            loop = asyncio.get_event_loop()
            try:
                loop.run_until_complete(process_pdf_with_real_progress(str(save_path), job_id))
            finally:
                loop.close()

        thread = threading.Thread(target=start_processing, daemon=True)
        thread.start()

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

        return {
            "status": "online",
            "indexed_documents": len(sources),
            "indexed_chunks": doc_count,
            "active_jobs": len(progress_tracker.active_jobs),
            "websocket_connections": len(progress_tracker.websocket_connections),
            "models_available": [m['name'] for m in ollama.list().get('models', [])]
        }
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return {"status": "degraded", "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)