"""Pydantic models for request/response validation."""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Job status enumeration."""
    UPLOADING = "uploading"
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EditParams(BaseModel):
    """FlowEdit parameters with sensible defaults."""
    
    T_steps: int = Field(default=28, ge=1, le=100, description="Number of timesteps")
    n_avg: int = Field(default=1, ge=1, le=10, description="Number of averaging iterations")
    src_guidance_scale: float = Field(default=1.5, gt=0, le=20, description="Source guidance scale")
    tar_guidance_scale: float = Field(default=5.5, gt=0, le=20, description="Target guidance scale")
    n_min: int = Field(default=0, ge=0, le=50, description="Minimum step for editing")
    n_max: int = Field(default=24, ge=0, le=50, description="Maximum step for editing")


class CreateJobResponse(BaseModel):
    """Response when creating a new job."""
    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Initial job status")
    message: str = Field(default="Job created successfully", description="Status message")


class SubmitEditResponse(BaseModel):
    """Response when submitting an edit request."""
    job_id: str = Field(..., description="Job identifier")
    status: JobStatus = Field(..., description="Job status after submission")
    message: str = Field(default="Edit submitted successfully", description="Status message")


class JobInfo(BaseModel):
    """Detailed job information."""
    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Current job status")
    created_at: datetime = Field(..., description="Job creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    
    # Edit parameters
    source_prompt: Optional[str] = Field(default=None, description="Source prompt")
    target_prompt: Optional[str] = Field(default=None, description="Target prompt")
    edit_params: Optional[EditParams] = Field(default=None, description="Edit parameters")
    
    # Result information
    result_path: Optional[str] = Field(default=None, description="Path to edited image")
    
    # Error information
    error_message: Optional[str] = Field(default=None, description="Error message if job failed")
    error_traceback: Optional[str] = Field(default=None, description="Error traceback if job failed")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(default="healthy", description="Service status")
    model_loaded: bool = Field(default=False, description="Whether FlowEdit model is loaded")
    active_jobs: int = Field(default=0, ge=0, description="Number of active jobs")
    total_jobs: int = Field(default=0, ge=0, description="Total number of jobs")
