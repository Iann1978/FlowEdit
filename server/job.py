"""Job class for representing a single FlowEdit job."""

import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from .models import JobInfo, JobStatus, EditParams


class Job:
    """Represents a single FlowEdit job with its state and operations."""
    
    def __init__(self, job_info: JobInfo, job_dir: Path):
        """Initialize a job.
        
        Args:
            job_info: JobInfo Pydantic model containing job state
            job_dir: Path to the job's directory
        """
        self._info = job_info
        self._job_dir = job_dir
    
    def to_info(self) -> JobInfo:
        """Convert Job to JobInfo for API responses.
        
        Returns:
            JobInfo instance (same reference, not a copy)
        """
        return self._info
    
    def update_status(
        self,
        status: JobStatus,
        error_message: Optional[str] = None,
        error_traceback: Optional[str] = None,
    ) -> None:
        """Update job status.
        
        Args:
            status: New status
            error_message: Error message if failed (optional)
            error_traceback: Error traceback if failed (optional)
        """
        self._info.status = status
        self._info.updated_at = datetime.now()
        
        if error_message is not None:
            self._info.error_message = error_message
        if error_traceback is not None:
            self._info.error_traceback = error_traceback
    
    def set_edit_info(
        self,
        source_prompt: str,
        target_prompt: str,
        edit_params: EditParams,
    ) -> None:
        """Set edit information (prompts and parameters).
        
        Args:
            source_prompt: Source prompt
            target_prompt: Target prompt
            edit_params: Edit parameters
        """
        self._info.source_prompt = source_prompt
        self._info.target_prompt = target_prompt
        self._info.edit_params = edit_params
        self._info.updated_at = datetime.now()
    
    def set_result_path(self, result_path: Path) -> None:
        """Set result image path.
        
        Args:
            result_path: Path to edited image
        """
        self._info.result_path = str(result_path)
        self._info.updated_at = datetime.now()
    
    def upload_image(self, file_path: Path) -> bool:
        """Upload input image to job.
        
        Args:
            file_path: Path to image file to save
            
        Returns:
            True if successful, False if job directory doesn't exist
        """
        if not self._job_dir.exists():
            return False
        
        # Copy file to job directory
        target_path = self._job_dir / "input" / "image.png"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, target_path)
        
        self._info.updated_at = datetime.now()
        return True
    
    def cancel(self) -> bool:
        """Cancel a job (if it's pending or running).
        
        Returns:
            True if job was cancelled, False otherwise
        """
        if self._info.status in [JobStatus.PENDING, JobStatus.RUNNING]:
            self.update_status(JobStatus.CANCELLED)
            return True
        return False
    
    def get_dir(self) -> Path:
        """Get the job directory path.
        
        Returns:
            Path to job directory
        """
        return self._job_dir
    
    def get_input_image_path(self) -> Path:
        """Get path to input image.
        
        Returns:
            Path to input image file
        """
        return self._job_dir / "input" / "image.png"
    
    def get_output_image_path(self) -> Path:
        """Get path to output image.
        
        Returns:
            Path to output image file
        """
        return self._job_dir / "output" / "edited_image.png"
