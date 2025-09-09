"""Pydantic response models for FastAPI endpoints."""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class CharacterCreateRequest(BaseModel):
    name: str
    metatype: str = "Human"
    archetype: str = ""

class CharacterStatsUpdate(BaseModel):
    body: int = 1
    agility: int = 1
    reaction: int = 1
    strength: int = 1
    charisma: int = 1
    logic: int = 1
    intuition: int = 1
    willpower: int = 1
    edge: int = 1
    essence: float = 6.0
    physical_limit: int = 1
    mental_limit: int = 1
    social_limit: int = 1
    initiative: int = 1
    hot_sim_vr: int = 0

class CharacterResourcesUpdate(BaseModel):
    nuyen: int = 0
    street_cred: int = 0
    notoriety: int = 0
    public_aware: int = 0
    total_karma: int = 0
    available_karma: int = 0
    edge_pool: int = 1

class SkillAddRequest(BaseModel):
    name: str
    rating: int = 1
    specialization: str = ""
    skill_type: str = "active"
    skill_group: str = ""
    attribute: str = ""

class QualityAddRequest(BaseModel):
    name: str
    rating: int = 0
    karma_cost: int = 0
    description: str = ""
    quality_type: str = "positive"

class GearAddRequest(BaseModel):
    name: str
    category: str = ""
    subcategory: str = ""
    quantity: int = 1
    rating: int = 0
    armor_value: int = 0
    cost: int = 0
    availability: str = ""
    description: str = ""
    custom_properties: Dict[str, Any] = {}

class WeaponAddRequest(BaseModel):
    name: str
    weapon_type: str = "ranged"
    mode_ammo: str = ""
    accuracy: int = 0
    damage_code: str = ""
    armor_penetration: int = 0
    recoil_compensation: int = 0
    cost: int = 0
    availability: str = ""
    description: str = ""

class VehicleAddRequest(BaseModel):
    name: str
    vehicle_type: str = "vehicle"
    handling: int = 0
    speed: int = 0
    acceleration: int = 0
    body: int = 0
    armor: int = 0
    pilot: int = 0
    sensor: int = 0
    seats: int = 0
    cost: int = 0
    availability: str = ""
    description: str = ""

class CyberdeckUpdate(BaseModel):
    name: str = ""
    device_rating: int = 1
    attack: int = 0
    sleaze: int = 0
    firewall: int = 0
    data_processing: int = 0
    matrix_damage: int = 0
    cost: int = 0
    availability: str = ""
    description: str = ""

class ProgramAddRequest(BaseModel):
    name: str
    rating: int = 1
    program_type: str = "common"
    description: str = ""

# ===== RAG REQUEST MODELS =====

class QueryRequest(BaseModel):
    """Request model for RAG queries."""
    question: str
    n_results: int = 5
    query_type: str = "general"
    filter_source: Optional[str] = None
    filter_section: Optional[str] = None
    filter_subsection: Optional[str] = None
    filter_document_type: Optional[str] = None
    filter_edition: Optional[str] = None
    character_role: Optional[str] = None
    character_stats: Optional[str] = None
    edition: Optional[str] = "SR5"
    model: Optional[str] = None
    conversation_context: Optional[str] = None

class IndexRequest(BaseModel):
    """Request model for indexing operations."""
    directory: str = "data/processed_markdown"
    force_reindex: bool = False

# ===== RESPONSE MODELS =====

class HealthCheckResponse(BaseModel):
    """Response for root health check endpoint."""
    status: str = Field(..., description="Service status")
    service: str = Field(..., description="Service name")
    active_jobs: Optional[int] = Field(None, description="Number of active processing jobs")
    tracking_method: Optional[str] = Field(None, description="Progress tracking method used")

class UploadResponse(BaseModel):
    """Response for PDF upload endpoint."""
    job_id: str = Field(..., description="Unique job identifier")
    filename: str = Field(..., description="Name of uploaded file")
    status: str = Field(..., description="Upload status")
    message: str = Field(..., description="Human-readable status message")
    poll_url: Optional[str] = Field(None, description="URL to poll for job status")

class JobStatusResponse(BaseModel):
    """Response for job status polling."""
    job_id: str = Field(..., description="Job identifier")
    stage: Optional[str] = Field(None, description="Current processing stage")
    progress: Optional[float] = Field(None, description="Progress percentage (0-100)")
    details: Optional[str] = Field(None, description="Detailed status message")
    timestamp: Optional[float] = Field(None, description="Last update timestamp")
    status: Optional[str] = Field(None, description="Job status (for not found jobs)")
    message: Optional[str] = Field(None, description="Status message (for not found jobs)")

class JobInfo(BaseModel):
    """Information about a single job."""
    job_id: str = Field(..., description="Job identifier")
    stage: Optional[str] = Field(None, description="Current processing stage")
    progress: Optional[float] = Field(None, description="Progress percentage")
    details: Optional[str] = Field(None, description="Status details")
    timestamp: Optional[float] = Field(None, description="Last update timestamp")

class JobsListResponse(BaseModel):
    """Response for listing all active jobs."""
    active_jobs: Dict[str, JobInfo] = Field(..., description="Dictionary of active jobs")
    count: int = Field(..., description="Number of active jobs")

class QueryResponse(BaseModel):
    """Response for RAG query endpoint."""
    answer: str = Field(..., description="Generated answer")
    sources: List[str] = Field(..., description="List of source documents")
    chunks: List[str] = Field(..., description="Retrieved text chunks")
    distances: List[float] = Field(..., description="Similarity distances for chunks")
    metadatas: List[Dict[str, Any]] = Field(..., description="Metadata for each chunk")

class IndexResponse(BaseModel):
    """Response for indexing operations."""
    status: str = Field(..., description="Operation status")
    message: str = Field(..., description="Result message")

class DocumentsResponse(BaseModel):
    """Response for document listing."""
    documents: List[str] = Field(..., description="List of indexed document paths")

class ModelsResponse(BaseModel):
    """Response for available models."""
    models: List[str] = Field(..., description="List of available model names")
    error: Optional[str] = Field(None, description="Error message if model fetching failed")

class SystemStatusResponse(BaseModel):
    """Response for system status endpoint."""
    status: str = Field(..., description="System status")
    indexed_documents: Optional[int] = Field(None, description="Number of indexed documents")
    indexed_chunks: Optional[int] = Field(None, description="Number of indexed text chunks")
    active_jobs: Optional[int] = Field(None, description="Number of active processing jobs")
    tracking_method: Optional[str] = Field(None, description="Progress tracking method")
    models_available: Optional[List[str]] = Field(None, description="Available model names")
    error: Optional[str] = Field(None, description="Error message if status check failed")