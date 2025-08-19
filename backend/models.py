"""Pydantic response models for FastAPI endpoints."""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime


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


class JobsListResponse(BaseModel):
    """Response for listing all active jobs."""
    active_jobs: Dict[str, Dict[str, Any]] = Field(..., description="Dictionary of active jobs")
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