"""Simplified Pydantic models for the basic Shadowrun RAG API."""

from pydantic import BaseModel
from typing import Optional, List, Dict
import time

# Request Models
class QueryRequest(BaseModel):
    question: str
    n_results: int = 5
    section: Optional[str] = None  # Only basic section filtering: Matrix, Combat, Magic, etc.
    model: Optional[str] = None

class IndexRequest(BaseModel):
    directory: str = "data/processed_markdown"
    force_reindex: bool = False

# Response Models
class HealthCheckResponse(BaseModel):
    status: str
    service: str
    active_jobs: int

class UploadResponse(BaseModel):
    job_id: str
    filename: str
    status: str
    message: str
    poll_url: str

class JobStatusResponse(BaseModel):
    job_id: str
    stage: str
    progress: float
    details: str
    timestamp: float

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    chunks: List[str]
    distances: List[float]
    metadatas: List[Dict]

class IndexResponse(BaseModel):
    status: str
    message: str

class DocumentStatsResponse(BaseModel):
    total_chunks: int
    unique_documents: int
    sections: Dict[str, int]
    sources: List[str]

class SystemStatusResponse(BaseModel):
    status: str
    indexed_documents: int = 0
    indexed_chunks: int = 0
    active_jobs: int = 0
    models_available: List[str] = []
    error: Optional[str] = None
