"""FastAPI backend server."""
import ollama
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import shutil
from pathlib import Path
import logging
from .indexer import IncrementalIndexer
from .retriever import Retriever
from .watcher import PDFWatcher
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

# Initialize components
indexer = IncrementalIndexer()
retriever = Retriever()
processor = PDFProcessor()
watcher = PDFWatcher(indexer=indexer, processor=processor)

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

class IndexRequest(BaseModel):
    directory: str = "data/processed_markdown"
    force_reindex: bool = False

@app.get("/")
def root():
    """Health check."""
    return {"status": "online", "service": "Shadowrun RAG API"}

@app.post("/upload")
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Upload and process a PDF."""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(400, "Only PDF files are allowed")
    
    # Save to raw_pdfs directory
    save_path = Path("data/raw_pdfs") / file.filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with save_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Process in background
    background_tasks.add_task(watcher.process_pdf, str(save_path))
    
    return {
        "filename": file.filename,
        "status": "processing",
        "message": "PDF uploaded and queued for processing"
    }

@app.post("/query")
async def query(request: QueryRequest):
    """Query the RAG system with full context."""
    try:
        # Build metadata filter
        where_filter = {}
        if request.filter_source:
            where_filter["source"] = {"$contains": request.filter_source}
        if request.filter_section:
            where_filter["Section"] = request.filter_section
        if request.filter_subsection:
            where_filter["Subsection"] = request.filter_subsection

        # Role-based fallback
        role_to_section = {
            "decker": "Matrix",
            "hacker": "Matrix",
            "mage": "Magic",
            "adept": "Magic",
            "street_samurai": "Combat",
            "rigger": "Riggers",
            "technomancer": "Technomancy"
        }
        if request.character_role and request.character_role in role_to_section:
            where_filter["Section"] = role_to_section[request.character_role]

        # Query retriever
        results = retriever.query(
            question=request.question,
            n_results=request.n_results,
            query_type=request.query_type,
            where_filter=where_filter,
            character_role=request.character_role,
            character_stats=request.character_stats,
            edition=request.edition
        )
        return results

    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
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
        models = ollama.list()
        return {
            "models": [model['name'] for model in models.get('models', [])]
        }
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        return {"models": ["llama3", "mistral", "codellama"]}

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