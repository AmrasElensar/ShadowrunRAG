"""FastAPI backend server with enhanced document type support and improved filtering."""
import ollama
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
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
    QueryResponse, IndexResponse, DocumentsResponse, ModelsResponse, SystemStatusResponse, CharacterCreateRequest,
    CharacterStatsUpdate, CharacterResourcesUpdate, SkillAddRequest, QualityAddRequest, GearAddRequest,
    WeaponAddRequest, VehicleAddRequest, CyberdeckUpdate, ProgramAddRequest
)
from tools.pdf_processor import PDFProcessor
from backend.character_manager import get_character_manager
from backend.extractors import populate_reference_tables

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
                search_results = retriever.search_with_linked_chunks(
                    question=request.question,
                    n_results=request.n_results,
                    where_filter=final_filter,
                    character_role=request.character_role,
                    fetch_linked=True  # or make this configurable
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


# ===== CHARACTER MANAGEMENT ENDPOINTS =====

@app.get("/characters")
async def list_characters():
    """Get list of all characters."""
    try:
        manager = get_character_manager()
        characters = manager.get_character_list()
        return {"characters": characters}
    except Exception as e:
        logger.error(f"Error listing characters: {e}")
        raise HTTPException(500, f"Failed to list characters: {str(e)}")


@app.post("/characters")
async def create_character(request: CharacterCreateRequest):
    """Create a new character."""
    try:
        manager = get_character_manager()
        character_id = manager.create_character(request.name, request.metatype, request.archetype)
        return {
            "character_id": character_id,
            "message": f"Character '{request.name}' created successfully"
        }
    except Exception as e:
        logger.error(f"Error creating character: {e}")
        raise HTTPException(500, f"Failed to create character: {str(e)}")


@app.get("/characters/{character_id}")
async def get_character(character_id: int):
    """Get complete character data."""
    try:
        manager = get_character_manager()
        character_data = manager.get_character_full_data(character_id)
        if not character_data:
            raise HTTPException(404, "Character not found")
        return {"character": character_data}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting character: {e}")
        raise HTTPException(500, f"Failed to get character: {str(e)}")


@app.delete("/characters/{character_id}")
async def delete_character(character_id: int):
    """Delete a character."""
    try:
        manager = get_character_manager()
        success = manager.delete_character(character_id)
        if not success:
            raise HTTPException(404, "Character not found")
        return {"message": "Character deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting character: {e}")
        raise HTTPException(500, f"Failed to delete character: {str(e)}")


@app.get("/characters/active")
async def get_active_character():
    """Get the currently active character."""
    try:
        manager = get_character_manager()
        active_id = manager.db.get_active_character_id()
        if not active_id:
            return {"active_character": None}

        character_data = manager.get_character_full_data(active_id)
        return {"active_character": character_data}
    except Exception as e:
        logger.error(f"Error getting active character: {e}")
        raise HTTPException(500, f"Failed to get active character: {str(e)}")


@app.post("/characters/{character_id}/activate")
async def set_active_character(character_id: int):
    """Set the active character."""
    try:
        manager = get_character_manager()
        # Verify character exists
        character_data = manager.get_character_full_data(character_id)
        if not character_data:
            raise HTTPException(404, "Character not found")

        manager.db.set_active_character_id(character_id)
        return {"message": f"Active character set to '{character_data['name']}'"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting active character: {e}")
        raise HTTPException(500, f"Failed to set active character: {str(e)}")


# ===== CHARACTER DATA UPDATE ENDPOINTS =====

@app.put("/characters/{character_id}/stats")
async def update_character_stats(character_id: int, stats: CharacterStatsUpdate):
    """Update character statistics."""
    try:
        manager = get_character_manager()
        success = manager.update_character_stats(character_id, stats.dict())
        if not success:
            raise HTTPException(404, "Character not found or update failed")
        return {"message": "Character stats updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating character stats: {e}")
        raise HTTPException(500, f"Failed to update stats: {str(e)}")


@app.put("/characters/{character_id}/resources")
async def update_character_resources(character_id: int, resources: CharacterResourcesUpdate):
    """Update character resources."""
    try:
        manager = get_character_manager()
        success = manager.update_character_resources(character_id, resources.dict())
        if not success:
            raise HTTPException(404, "Character not found or update failed")
        return {"message": "Character resources updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating character resources: {e}")
        raise HTTPException(500, f"Failed to update resources: {str(e)}")


# ===== SKILLS MANAGEMENT =====

@app.post("/characters/{character_id}/skills")
async def add_character_skill(character_id: int, skill: SkillAddRequest):
    """Add a skill to a character."""
    try:
        manager = get_character_manager()
        success = manager.add_character_skill(character_id, skill.dict())
        if not success:
            raise HTTPException(400, "Failed to add skill")
        return {"message": f"Skill '{skill.name}' added successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding character skill: {e}")
        raise HTTPException(500, f"Failed to add skill: {str(e)}")


@app.put("/characters/{character_id}/skills/{skill_name}")
async def update_character_skill(character_id: int, skill_name: str, skill: SkillAddRequest):
    """Update an existing character skill."""
    try:
        manager = get_character_manager()
        success = manager.update_character_skill(character_id, skill_name, skill.dict())
        if not success:
            raise HTTPException(404, "Skill not found")
        return {"message": f"Skill '{skill_name}' updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating character skill: {e}")
        raise HTTPException(500, f"Failed to update skill: {str(e)}")


@app.delete("/characters/{character_id}/skills/{skill_name}")
async def remove_character_skill(character_id: int, skill_name: str):
    """Remove a skill from a character."""
    try:
        manager = get_character_manager()
        success = manager.remove_character_skill(character_id, skill_name)
        if not success:
            raise HTTPException(404, "Skill not found")
        return {"message": f"Skill '{skill_name}' removed successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing character skill: {e}")
        raise HTTPException(500, f"Failed to remove skill: {str(e)}")


# ===== QUALITIES MANAGEMENT =====

@app.post("/characters/{character_id}/qualities")
async def add_character_quality(character_id: int, quality: QualityAddRequest):
    """Add a quality to a character."""
    try:
        manager = get_character_manager()
        success = manager.add_character_quality(character_id, quality.dict())
        if not success:
            raise HTTPException(400, "Failed to add quality")
        return {"message": f"Quality '{quality.name}' added successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding character quality: {e}")
        raise HTTPException(500, f"Failed to add quality: {str(e)}")


@app.delete("/characters/{character_id}/qualities/{quality_name}")
async def remove_character_quality(character_id: int, quality_name: str):
    """Remove a quality from a character."""
    try:
        manager = get_character_manager()
        success = manager.remove_character_quality(character_id, quality_name)
        if not success:
            raise HTTPException(404, "Quality not found")
        return {"message": f"Quality '{quality_name}' removed successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing character quality: {e}")
        raise HTTPException(500, f"Failed to remove quality: {str(e)}")


# ===== GEAR MANAGEMENT =====

@app.post("/characters/{character_id}/gear")
async def add_character_gear(character_id: int, gear: GearAddRequest):
    """Add gear to a character."""
    try:
        manager = get_character_manager()
        success = manager.add_character_gear(character_id, gear.dict())
        if not success:
            raise HTTPException(400, "Failed to add gear")
        return {"message": f"Gear '{gear.name}' added successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding character gear: {e}")
        raise HTTPException(500, f"Failed to add gear: {str(e)}")


@app.delete("/characters/{character_id}/gear/{gear_id}")
async def remove_character_gear(character_id: int, gear_id: int):
    """Remove gear from a character."""
    try:
        manager = get_character_manager()
        success = manager.remove_character_gear(character_id, gear_id)
        if not success:
            raise HTTPException(404, "Gear not found")
        return {"message": "Gear removed successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing character gear: {e}")
        raise HTTPException(500, f"Failed to remove gear: {str(e)}")


# ===== WEAPONS MANAGEMENT =====

@app.post("/characters/{character_id}/weapons")
async def add_character_weapon(character_id: int, weapon: WeaponAddRequest):
    """Add weapon to a character."""
    try:
        manager = get_character_manager()
        success = manager.add_character_weapon(character_id, weapon.dict())
        if not success:
            raise HTTPException(400, "Failed to add weapon")
        return {"message": f"Weapon '{weapon.name}' added successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding character weapon: {e}")
        raise HTTPException(500, f"Failed to add weapon: {str(e)}")


@app.delete("/characters/{character_id}/weapons/{weapon_id}")
async def remove_character_weapon(character_id: int, weapon_id: int):
    """Remove weapon from a character."""
    try:
        manager = get_character_manager()
        success = manager.remove_character_weapon(character_id, weapon_id)
        if not success:
            raise HTTPException(404, "Weapon not found")
        return {"message": "Weapon removed successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing character weapon: {e}")
        raise HTTPException(500, f"Failed to remove weapon: {str(e)}")


# ===== VEHICLES MANAGEMENT =====

@app.post("/characters/{character_id}/vehicles")
async def add_character_vehicle(character_id: int, vehicle: VehicleAddRequest):
    """Add vehicle/drone to a character."""
    try:
        manager = get_character_manager()
        vehicle_data = vehicle.dict()
        success = manager.add_character_vehicle(character_id, vehicle_data)
        if not success:
            raise HTTPException(400, "Failed to add vehicle")
        return {"message": f"Vehicle '{vehicle.name}' added successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding character vehicle: {e}")
        raise HTTPException(500, f"Failed to add vehicle: {str(e)}")


@app.delete("/characters/{character_id}/vehicles/{vehicle_id}")
async def remove_character_vehicle(character_id: int, vehicle_id: int):
    """Remove vehicle from a character."""
    try:
        manager = get_character_manager()
        success = manager.remove_character_vehicle(character_id, vehicle_id)
        if not success:
            raise HTTPException(404, "Vehicle not found")
        return {"message": "Vehicle removed successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing character vehicle: {e}")
        raise HTTPException(500, f"Failed to remove vehicle: {str(e)}")


# ===== CYBERDECK & PROGRAMS =====

@app.put("/characters/{character_id}/cyberdeck")
async def update_character_cyberdeck(character_id: int, cyberdeck: CyberdeckUpdate):
    """Update character's cyberdeck."""
    try:
        manager = get_character_manager()
        success = manager.update_character_cyberdeck(character_id, cyberdeck.dict())
        if not success:
            raise HTTPException(400, "Failed to update cyberdeck")
        return {"message": "Cyberdeck updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating character cyberdeck: {e}")
        raise HTTPException(500, f"Failed to update cyberdeck: {str(e)}")


@app.post("/characters/{character_id}/programs")
async def add_character_program(character_id: int, program: ProgramAddRequest):
    """Add program to character's cyberdeck."""
    try:
        manager = get_character_manager()
        success = manager.add_character_program(character_id, program.dict())
        if not success:
            raise HTTPException(400, "Failed to add program")
        return {"message": f"Program '{program.name}' added successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding character program: {e}")
        raise HTTPException(500, f"Failed to add program: {str(e)}")


@app.delete("/characters/{character_id}/programs/{program_id}")
async def remove_character_program(character_id: int, program_id: int):
    """Remove program from character."""
    try:
        manager = get_character_manager()
        success = manager.remove_character_program(character_id, program_id)
        if not success:
            raise HTTPException(404, "Program not found")
        return {"message": "Program removed successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing character program: {e}")
        raise HTTPException(500, f"Failed to remove program: {str(e)}")


# ===== REFERENCE DATA ENDPOINTS =====

@app.get("/reference/skills")
async def get_skills_library(skill_type: Optional[str] = None):
    """Get available skills from rulebooks for dropdowns."""
    try:
        manager = get_character_manager()
        skills = manager.get_skills_library(skill_type)
        return {"skills": skills}
    except Exception as e:
        logger.error(f"Error getting skills library: {e}")
        raise HTTPException(500, f"Failed to get skills: {str(e)}")


@app.get("/reference/qualities")
async def get_qualities_library(quality_type: Optional[str] = None):
    """Get available qualities from rulebooks for dropdowns."""
    try:
        manager = get_character_manager()
        qualities = manager.get_qualities_library(quality_type)
        return {"qualities": qualities}
    except Exception as e:
        logger.error(f"Error getting qualities library: {e}")
        raise HTTPException(500, f"Failed to get qualities: {str(e)}")


@app.get("/reference/gear")
async def get_gear_library(category: Optional[str] = None):
    """Get available gear from rulebooks for dropdowns."""
    try:
        manager = get_character_manager()
        gear = manager.get_gear_library(category)
        return {"gear": gear}
    except Exception as e:
        logger.error(f"Error getting gear library: {e}")
        raise HTTPException(500, f"Failed to get gear: {str(e)}")


@app.get("/reference/gear/categories")
async def get_gear_categories():
    """Get list of available gear categories."""
    try:
        manager = get_character_manager()
        categories = manager.get_gear_categories()
        return {"categories": categories}
    except Exception as e:
        logger.error(f"Error getting gear categories: {e}")
        raise HTTPException(500, f"Failed to get categories: {str(e)}")


@app.post("/reference/populate")
async def populate_reference_data():
    """Populate reference tables from processed rulebooks."""
    try:
        populate_reference_tables()
        return {"message": "Reference tables populated successfully from rulebooks"}
    except Exception as e:
        logger.error(f"Error populating reference data: {e}")
        raise HTTPException(500, f"Failed to populate reference data: {str(e)}")


# ===== EXPORT ENDPOINTS =====

@app.get("/characters/{character_id}/export/json")
async def export_character_json(character_id: int):
    """Export character as JSON for sharing with GM."""
    try:
        manager = get_character_manager()
        json_data = manager.export_character_json(character_id)
        if not json_data:
            raise HTTPException(404, "Character not found")

        # Return as downloadable file
        from fastapi.responses import Response
        return Response(
            content=json_data,
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename=character_{character_id}.json"}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting character JSON: {e}")
        raise HTTPException(500, f"Failed to export character: {str(e)}")


@app.get("/characters/{character_id}/export/csv")
async def export_character_csv(character_id: int):
    """Export character as CSV for sharing with GM."""
    try:
        manager = get_character_manager()
        csv_data = manager.export_character_csv(character_id)
        if not csv_data:
            raise HTTPException(404, "Character not found")

        # Return as downloadable file
        from fastapi.responses import Response
        return Response(
            content=csv_data,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=character_{character_id}.csv"}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting character CSV: {e}")
        raise HTTPException(500, f"Failed to export character: {str(e)}")


# ===== DICE POOL & QUERY HELPERS =====

@app.get("/characters/{character_id}/dice_pool/{skill_name}")
async def get_skill_dice_pool(character_id: int, skill_name: str):
    """Get dice pool for a specific skill."""
    try:
        manager = get_character_manager()
        dice_pool, explanation = manager.get_character_skill_dice_pool(character_id, skill_name)
        return {
            "character_id": character_id,
            "skill_name": skill_name,
            "dice_pool": dice_pool,
            "explanation": explanation
        }
    except Exception as e:
        logger.error(f"Error getting dice pool: {e}")
        raise HTTPException(500, f"Failed to get dice pool: {str(e)}")


@app.get("/characters/{character_id}/context")
async def get_character_query_context(character_id: int):
    """Get character context string for RAG queries."""
    try:
        manager = get_character_manager()
        context = manager.generate_character_context_for_query(character_id)
        return {
            "character_id": character_id,
            "context": context
        }
    except Exception as e:
        logger.error(f"Error getting character context: {e}")
        raise HTTPException(500, f"Failed to get character context: {str(e)}")


@app.get("/debug/context-real")
async def debug_context_real(
        query: str,
        character_role: str = None,
        edition: str = "SR5"
):
    """Debug endpoint showing EXACT context sent to LLM without calling LLM"""

    # Use your exact search path
    where_filter = {"edition": edition} if edition else None
    enhanced_filter = retriever.build_enhanced_filter(character_role, where_filter)

    search_results = retriever.search_with_linked_chunks(
        query,
        n_results=5,
        where_filter=enhanced_filter,
        character_role=character_role,
        fetch_linked=False
    )

    if not search_results['documents']:
        return {"error": "No documents found"}

    # Build the exact context that would be sent to the LLM
    context = retriever.build_enhanced_context_with_sequence(search_results)

    return {
        "raw_context_sent_to_llm": context,
        "chunks_count": len(search_results['documents']),
        "linked_chunks_fetched": search_results.get('linked_chunks_fetched', 0),
        "first_chunk_preview": search_results['documents'][0][:200] if search_results['documents'] else None,
        "metadatas": search_results['metadatas']
    }


@app.get("/debug/context-text")
async def debug_context_as_text(
        query: str,
        character_role: str = None,
        edition: str = "SR5"
):
    # Use your exact search path
    where_filter = {"edition": edition} if edition else None
    enhanced_filter = retriever.build_enhanced_filter(character_role, where_filter)

    search_results = retriever.search_with_linked_chunks(
        query,
        n_results=5,
        where_filter=enhanced_filter,
        character_role=character_role,
        fetch_linked=False
    )

    if not search_results['documents']:
        return {"error": "No documents found"}

    # Build the exact context that would be sent to the LLM
    context = retriever.build_enhanced_context_with_sequence(search_results)

    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(
        content=context,  # This will preserve actual newlines
        headers={"Content-Type": "text/plain; charset=utf-8"}
    )


@app.get("/debug/collection-info")
async def debug_collection():
    try:
        count = retriever.collection.count()

        if count > 0:
            sample = retriever.collection.peek(limit=1)

            if sample and 'embeddings' in sample and len(sample['embeddings']) > 0:
                embedding_dim = len(sample['embeddings'][0])
                return {
                    "collection_name": retriever.collection.name,
                    "embedding_dimension": embedding_dim,
                    "document_count": count,
                    "sample_metadata": sample['metadatas'][0] if sample['metadatas'] else None
                }
            else:
                return {"error": "No embeddings found", "sample_keys": list(sample.keys()) if sample else None}
        else:
            return {"error": "Empty collection"}
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}


@app.get("/debug/metadata-values")
async def debug_metadata_values():
    """See what metadata values actually exist in ChromaDB"""
    try:
        # Get a sample of documents to see metadata structure
        sample = retriever.collection.peek(limit=20)

        main_sections = set()
        editions = set()

        for metadata in sample['metadatas']:
            if metadata:
                if 'main_section' in metadata:
                    main_sections.add(metadata['main_section'])
                if 'edition' in metadata:
                    editions.add(metadata['edition'])

        return {
            "main_sections": list(main_sections),
            "editions": list(editions),
            "sample_metadata": sample['metadatas'][:3]  # First 3 for inspection
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)