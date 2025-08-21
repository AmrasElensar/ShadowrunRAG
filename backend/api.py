"""FastAPI backend server with enhanced document type support and improved filtering."""
import ollama
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, Dict
from pathlib import Path
import logging
import json
import time
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor
from .indexer import IncrementalIndexer
from .retriever import Retriever
from .models import (
    HealthCheckResponse, UploadResponse, JobStatusResponse, JobsListResponse, JobInfo,
    QueryResponse, IndexResponse, DocumentsResponse, ModelsResponse, SystemStatusResponse
)
from tools.pdf_processor import PDFProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Shadowrun RAG API")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enhanced progress tracking
class PollingProgressTracker:
    """Enhanced progress tracker for polling-based updates with document type awareness."""

    def __init__(self):
        self.active_jobs: Dict[str, Dict] = {}
        self.lock = threading.Lock()
        logger.info("🔧 Enhanced PollingProgressTracker initialized")

    def update_progress(self, job_id: str, stage: str, progress: float, details: str = "", document_type: str = "unknown"):
        """Update progress with document type information (thread-safe)."""
        progress_data = {
            "job_id": job_id,
            "stage": stage,
            "progress": max(0, min(100, progress)),
            "details": details,
            "document_type": document_type,
            "timestamp": time.time()
        }

        with self.lock:
            self.active_jobs[job_id] = progress_data

        # Log for debugging
        logger.info(f"Progress [{job_id}] ({document_type}): {stage} ({progress}%) - {details}")

    def get_job_status(self, job_id: str) -> Optional[Dict]:
        """Get current job status."""
        with self.lock:
            return self.active_jobs.get(job_id)

    def cleanup_job(self, job_id: str, delay: int = 300):
        """Remove completed job after delay."""
        def cleanup():
            time.sleep(delay)
            with self.lock:
                if job_id in self.active_jobs:
                    del self.active_jobs[job_id]
                    logger.info(f"🧹 Cleaned up job: {job_id}")

        cleanup_thread = threading.Thread(target=cleanup, daemon=True)
        cleanup_thread.start()

    def get_all_jobs(self) -> Dict[str, Dict]:
        """Get all active jobs."""
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
logger.info("🚀 Enhanced FastAPI app initialized with components:")
logger.info(f"   - Enhanced PollingProgressTracker: ✅")
logger.info(f"   - Enhanced IncrementalIndexer: ✅")
logger.info(f"   - Enhanced Retriever: ✅")

class QueryRequest(BaseModel):
    question: str
    n_results: int = 5
    query_type: str = "general"
    filter_source: Optional[str] = None
    filter_section: Optional[str] = None
    filter_subsection: Optional[str] = None
    filter_document_type: Optional[str] = None  # New filter
    filter_edition: Optional[str] = None        # New filter
    character_role: Optional[str] = None
    character_stats: Optional[str] = None
    edition: Optional[str] = "SR5"              # Default to SR5
    model: Optional[str] = None

class IndexRequest(BaseModel):
    directory: str = "data/processed_markdown"
    force_reindex: bool = False

@app.get("/", response_model=HealthCheckResponse)
def root():
    """Health check."""
    return HealthCheckResponse(
        status="online",
        service="Enhanced Shadowrun RAG API",
        active_jobs=len(progress_tracker.active_jobs),
        tracking_method="polling_enhanced"
    )

def process_pdf_with_progress(pdf_path: str, job_id: str, document_type: str = "rulebook"):
    """Process PDF with enhanced progress tracking and document type awareness."""
    try:
        # Create processor with progress callback and document type
        def progress_callback(stage, progress, details):
            progress_tracker.update_progress(job_id, stage, progress, details, document_type)

        progress_tracker.update_progress(
            job_id, "starting", 5,
            f"Initializing {document_type} processing...",
            document_type
        )

        processor = PDFProcessor(
            progress_callback=progress_callback,
            document_type=document_type
        )

        # Process the PDF
        result = processor.process_pdf(pdf_path, force_reparse=True)

        # Index the results
        progress_tracker.update_progress(
            job_id, "indexing", 95,
            "Adding to search index with enhanced metadata...",
            document_type
        )
        indexer.index_directory("data/processed_markdown")

        progress_tracker.update_progress(
            job_id, "complete", 100,
            f"Processing complete! Created {len(result)} files as {document_type}.",
            document_type
        )

        # Schedule cleanup
        progress_tracker.cleanup_job(job_id)

        return result

    except Exception as e:
        progress_tracker.update_progress(
            job_id, "error", -1,
            f"Processing failed: {str(e)}",
            document_type
        )
        logger.error(f"Processing failed: {e}")
        logger.error(traceback.format_exc())
        raise

@app.post("/upload", response_model=UploadResponse)
async def upload_pdf_with_progress(
    file: UploadFile = File(...),
    document_type: str = Form(default="rulebook", description="Document type: rulebook, character_sheet, universe_info, adventure")
):
    """Upload and process a PDF with document type specification."""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(400, "Only PDF files are allowed")

    # Validate document type
    valid_types = ["rulebook", "character_sheet", "universe_info", "adventure"]
    if document_type not in valid_types:
        raise HTTPException(400, f"Invalid document type. Must be one of: {valid_types}")

    # Generate unique job ID
    job_id = f"{document_type}_{file.filename}_{int(time.time() * 1000)}"

    try:
        # Save file quickly
        save_path = Path("data/raw_pdfs") / file.filename
        save_path.parent.mkdir(parents=True, exist_ok=True)

        content = await file.read()
        save_path.write_bytes(content)

        # Start processing in background thread
        def start_processing():
            try:
                process_pdf_with_progress(str(save_path), job_id, document_type)
            except Exception as e:
                logger.error(f"Background processing failed: {e}")

        thread = threading.Thread(target=start_processing, daemon=True)
        thread.start()

        return UploadResponse(
            job_id=job_id,
            filename=file.filename,
            status="processing",
            message=f"PDF uploaded as {document_type}. Processing started with enhanced tracking.",
            poll_url=f"/job/{job_id}"
        )

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(500, f"Upload failed: {str(e)}")

@app.get("/job/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get current status of a processing job with enhanced information."""
    status = progress_tracker.get_job_status(job_id)
    if status:
        return JobStatusResponse(**status)
    else:
        return JobStatusResponse(
            job_id=job_id,
            status="not_found",
            message="Job not found or completed",
            timestamp=time.time()
        )

@app.get("/jobs", response_model=JobsListResponse)
async def list_all_jobs():
    """List all active jobs with document type information."""
    active_jobs_raw = progress_tracker.get_all_jobs()

    active_jobs_formatted = {}
    for job_id, job_data in active_jobs_raw.items():
        active_jobs_formatted[job_id] = JobInfo(
            job_id=job_data.get("job_id", job_id),
            stage=job_data.get("stage"),
            progress=job_data.get("progress"),
            details=job_data.get("details"),
            timestamp=job_data.get("timestamp")
        )

    return JobsListResponse(
        active_jobs=active_jobs_formatted,
        count=len(active_jobs_formatted)
    )

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Enhanced query with improved filtering logic."""
    try:
        # Build enhanced metadata filter
        where_filter = {}

        # Document type filter
        if request.filter_document_type and request.filter_document_type.strip():
            where_filter["document_type"] = request.filter_document_type

        # Edition filter (prioritize request edition over global preference)
        edition_to_use = request.filter_edition or request.edition
        if edition_to_use and edition_to_use != "None":
            where_filter["edition"] = edition_to_use

        # Source filter
        if request.filter_source and request.filter_source.strip():
            where_filter["source"] = {"$contains": request.filter_source}

        # Section filtering with character role precedence
        if request.character_role and request.character_role != "None":
            # Character role takes precedence over manual section filter
            role_to_section = {
                "decker": "Matrix",
                "hacker": "Matrix",
                "mage": "Magic",
                "adept": "Magic",
                "street_samurai": "Combat",
                "rigger": "Riggers",
                "technomancer": "Matrix",
                "face": "Social"  # Added Face role
            }

            role_section = role_to_section.get(request.character_role.lower())
            if role_section:
                where_filter["main_section"] = role_section
                logger.info(f"Character role '{request.character_role}' mapped to section '{role_section}'")
        elif request.filter_section and request.filter_section.strip() and request.filter_section != "All":
            # Manual section filter only if no character role
            where_filter["main_section"] = request.filter_section

        # Subsection filter
        if request.filter_subsection and request.filter_subsection.strip():
            where_filter["subsection"] = request.filter_subsection

        # Log filter for debugging
        logger.info(f"Applied filters: {where_filter}")

        final_filter = where_filter if where_filter else None

        results = retriever.query(
            question=request.question,
            n_results=request.n_results,
            query_type=request.query_type,
            where_filter=final_filter,
            character_role=request.character_role,
            character_stats=request.character_stats,
            edition=edition_to_use
        )
        return QueryResponse(**results)

    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        raise HTTPException(500, str(e))

@app.post("/query_stream", responses={200: {"content": {"text/plain": {"schema": {"type": "string"}}}}})
async def query_stream(request: QueryRequest):
    """Enhanced streaming with <think> tag detection and improved filtering."""
    try:
        # Build enhanced filter (same logic as query endpoint)
        where_filter = {}

        if request.filter_document_type and request.filter_document_type.strip():
            where_filter["document_type"] = request.filter_document_type

        edition_to_use = request.filter_edition or request.edition
        if edition_to_use and edition_to_use != "None":
            where_filter["edition"] = edition_to_use

        if request.filter_source and request.filter_source.strip():
            where_filter["source"] = {"$contains": request.filter_source}

        # Character role precedence logic
        if request.character_role and request.character_role != "None":
            role_to_section = {
                "decker": "Matrix", "hacker": "Matrix", "mage": "Magic", "adept": "Magic",
                "street_samurai": "Combat", "rigger": "Riggers", "technomancer": "Matrix", "face": "Social"
            }
            role_section = role_to_section.get(request.character_role.lower())
            if role_section:
                where_filter["main_section"] = role_section
        elif request.filter_section and request.filter_section.strip() and request.filter_section != "All":
            where_filter["main_section"] = request.filter_section

        if request.filter_subsection and request.filter_subsection.strip():
            where_filter["subsection"] = request.filter_subsection

        logger.info(f"Stream query filters: {where_filter}")
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

                # Track thinking content separately
                thinking_content = ""
                main_content = ""
                in_thinking = False

                # Stream the generation with <think> tag detection
                for token in retriever.query_stream(
                        question=request.question,
                        n_results=request.n_results,
                        query_type=request.query_type,
                        where_filter=final_filter,
                        character_role=request.character_role,
                        character_stats=request.character_stats,
                        edition=edition_to_use,
                        model=request.model
                ):
                    # Detect thinking tags
                    if "<think>" in token:
                        in_thinking = True
                        # Split token if it contains both tag and content
                        parts = token.split("<think>")
                        if parts[0]:  # Content before <think>
                            main_content += parts[0]
                            yield parts[0]
                        if len(parts) > 1:  # Content after <think>
                            thinking_content += parts[1]
                        # Send thinking marker
                        yield "\n__THINKING_START__\n"
                        if len(parts) > 1 and parts[1]:
                            yield parts[1]
                        continue

                    if "</think>" in token and in_thinking:
                        in_thinking = False
                        # Split token if it contains both content and closing tag
                        parts = token.split("</think>")
                        if parts[0]:  # Content before </think>
                            thinking_content += parts[0]
                            yield parts[0]
                        # Send thinking end marker
                        yield "\n__THINKING_END__\n"
                        if len(parts) > 1:  # Content after </think>
                            main_content += parts[1]
                            yield parts[1]
                        continue

                    # Regular content
                    if in_thinking:
                        thinking_content += token
                    else:
                        main_content += token

                    yield token

                # Send metadata after streaming
                metadata_packet = {
                    "sources": list({meta.get('source', 'Unknown') for meta in search_results['metadatas']}),
                    "chunks": search_results['documents'],
                    "distances": search_results['distances'],
                    "metadatas": search_results['metadatas'],
                    "thinking_content": thinking_content if thinking_content else None,
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
    """Manually trigger enhanced indexing."""
    try:
        indexer.index_directory(
            request.directory,
            request.force_reindex
        )
        return IndexResponse(
            status="success",
            message="Enhanced indexing complete with metadata extraction"
        )
    except Exception as e:
        logger.error(f"Indexing error: {e}")
        raise HTTPException(500, str(e))

@app.get("/documents", response_model=DocumentsResponse)
async def list_documents():
    """List all indexed documents with enhanced metadata."""
    try:
        results = retriever.collection.get()
        sources = set()

        for metadata in results.get('metadatas', []):
            if metadata and 'source' in metadata:
                sources.add(metadata['source'])

        return DocumentsResponse(documents=sorted(list(sources)))
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        return DocumentsResponse(documents=[])

@app.get("/document_stats")
async def get_document_stats():
    """Get statistics about indexed documents by type and edition."""
    try:
        results = retriever.collection.get()

        stats = {
            "total_chunks": len(results.get('ids', [])),
            "document_types": {},
            "editions": {},
            "sections": {},
            "sources": set()
        }

        for metadata in results.get('metadatas', []):
            if not metadata:
                continue

            # Count document types
            doc_type = metadata.get('document_type', 'unknown')
            stats["document_types"][doc_type] = stats["document_types"].get(doc_type, 0) + 1

            # Count editions
            edition = metadata.get('edition', 'unknown')
            stats["editions"][edition] = stats["editions"].get(edition, 0) + 1

            # Count sections
            section = metadata.get('main_section', 'unknown')
            stats["sections"][section] = stats["sections"].get(section, 0) + 1

            # Track unique sources
            if 'source' in metadata:
                stats["sources"].add(Path(metadata['source']).parent.name)

        stats["unique_documents"] = len(stats["sources"])
        stats["sources"] = list(stats["sources"])  # Convert set to list for JSON

        return stats
    except Exception as e:
        logger.error(f"Error getting document stats: {e}")
        return {"error": str(e)}

@app.get("/document/{file_path:path}")
async def get_document_content(file_path: str):
    """Get content of a specific document file."""
    try:
        full_path = Path("data/processed_markdown") / file_path

        if not str(full_path.resolve()).startswith(str(Path("data/processed_markdown").resolve())):
            raise HTTPException(400, "Invalid file path")

        if not full_path.exists():
            raise HTTPException(404, "File not found")

        content = full_path.read_text(encoding='utf-8')
        return {"content": content, "file_path": file_path}

    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        raise HTTPException(500, f"Error reading file: {str(e)}")

@app.post("/search_documents")
async def search_documents(request: dict):
    """Enhanced document search with metadata filtering."""
    try:
        query = request.get("query", "").lower()
        group_filter = request.get("group")
        doc_type_filter = request.get("document_type")  # New filter
        edition_filter = request.get("edition")         # New filter
        page = request.get("page", 1)
        page_size = request.get("page_size", 20)

        processed_dir = Path("data/processed_markdown")
        results = []

        for file_path in processed_dir.rglob("*.md"):
            group_name = file_path.parent.name

            # Apply filters
            if group_filter and group_name != group_filter:
                continue

            # Check document metadata if filters specified
            if doc_type_filter or edition_filter:
                try:
                    content = file_path.read_text(encoding='utf-8')
                    # Simple YAML front matter parsing
                    if content.startswith('---'):
                        yaml_end = content.find('---', 3)
                        if yaml_end > 0:
                            yaml_content = content[3:yaml_end]

                            if doc_type_filter:
                                if f'document_type: "{doc_type_filter}"' not in yaml_content:
                                    continue

                            if edition_filter:
                                if f'edition: "{edition_filter}"' not in yaml_content:
                                    continue
                except:
                    # Skip files we can't read
                    continue

            # Search filename and content
            filename_match = query in file_path.name.lower()
            content_match = False

            if not filename_match and query:
                try:
                    content = file_path.read_text(encoding='utf-8')
                    content_match = query in content.lower()
                except:
                    content_match = False

            if not query or filename_match or content_match:
                relative_path = str(file_path.relative_to(processed_dir))
                results.append({
                    "file_path": relative_path,
                    "filename": file_path.name,
                    "group": group_name,
                    "match_type": "filename" if filename_match else "content" if content_match else "all"
                })

        # Pagination
        total = len(results)
        start = (page - 1) * page_size
        end = start + page_size
        paginated_results = results[start:end]

        return {
            "results": paginated_results,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size
        }

    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(500, str(e))

@app.get("/models", response_model=ModelsResponse)
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

        return ModelsResponse(models=models)

    except Exception as e:
        logger.error(f"Error listing models: {e}")
        return ModelsResponse(models=[], error=str(e))

@app.get("/status", response_model=SystemStatusResponse)
async def status():
    """Return enhanced system status with metadata breakdown."""
    try:
        results = retriever.collection.get()
        doc_count = len(results.get('ids', []))
        sources = set()
        for meta in results.get('metadatas', []):
            if meta and 'source' in meta:
                sources.add(Path(meta['source']).parent.name)

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
        except Exception as model_error:
            logger.warning(f"Failed to get models: {model_error}")
            models_available = []

        return SystemStatusResponse(
            status="online",
            indexed_documents=len(sources),
            indexed_chunks=doc_count,
            active_jobs=len(progress_tracker.active_jobs),
            tracking_method="polling_enhanced",
            models_available=models_available
        )
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return SystemStatusResponse(status="degraded", error=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)