"""FastAPI application for FlowEdit image editing service."""

import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from .config import (
    get_device,
    get_host,
    get_jobs_dir,
    get_max_concurrent_jobs,
    get_model_path,
    get_port,
)
from .flowedit_service import flowedit_service
from .job_manager import JobManager
from .models import (
    CreateJobResponse,
    EditParams,
    HealthResponse,
    JobInfo,
    JobStatus,
    SubmitEditResponse,
)
from .utils import load_image_from_upload, preprocess_image, save_image, validate_image

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create logs directory
_log_dir = Path(__file__).parent / "logs"
_log_dir.mkdir(parents=True, exist_ok=True)

# Configure file handler for API logs
_log_file = _log_dir / "api.log"
_file_handler = logging.FileHandler(_log_file, mode='a', encoding='utf-8')
_file_handler.setLevel(logging.DEBUG)

# Configure log format
_log_format = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
_file_handler.setFormatter(_log_format)

# Add file handler to logger
logger.addHandler(_file_handler)

# Configuration from environment variables
JOBS_DIR = get_jobs_dir()
MAX_CONCURRENT_JOBS = get_max_concurrent_jobs()

# Initialize FastAPI app
app = FastAPI(
    title="FlowEdit Service",
    description="REST API for text-guided image editing using FLUX model",
    version="1.0.0",
)

# Initialize job manager
job_manager = JobManager(jobs_dir=JOBS_DIR)

# Track active editing tasks
_active_tasks: dict[str, asyncio.Task] = {}


@app.on_event("startup")
async def startup_event():
    """Load FlowEdit model on startup."""
    logger.info("Starting FlowEdit server...")
    try:
        model_path = get_model_path()
        device = get_device()
        flowedit_service.load_model(model_path=model_path, device=device)
        logger.info("FlowEdit model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load FlowEdit model: {e}")
        raise


@app.post("/jobs", response_model=CreateJobResponse, status_code=201)
async def create_job() -> CreateJobResponse:
    """Create a new edit job.
    
    Returns:
        CreateJobResponse with job_id and UPLOADING status
    """
    logger.info("POST /jobs - Creating new job")
    job_id = job_manager.create_job()
    logger.info(f"POST /jobs - Created job: {job_id}")
    
    return CreateJobResponse(
        job_id=job_id,
        status=JobStatus.UPLOADING,
        message="Job created successfully",
    )


@app.post("/jobs/{job_id}/edit", response_model=SubmitEditResponse)
async def submit_edit(
    job_id: str,
    image: UploadFile = File(..., description="Input image file (JPEG or PNG)"),
    source_prompt: str = Form(..., description="Source prompt describing the input image"),
    target_prompt: str = Form(..., description="Target prompt describing desired edit"),
    edit_params_json: Optional[str] = Form(None, description="Optional JSON string with edit parameters"),
) -> SubmitEditResponse:
    """Submit an edit request.
    
    Args:
        job_id: Job identifier
        image: Input image file
        source_prompt: Source prompt
        target_prompt: Target prompt
        edit_params_json: Optional JSON string with T_steps, n_avg, src_guidance_scale, tar_guidance_scale, n_min, n_max
        
    Returns:
        SubmitEditResponse with job_id and PENDING status
    """
    logger.info(f"POST /jobs/{job_id}/edit - Edit submission requested")
    
    # Get job
    job = job_manager._get_job_object(job_id)
    if job is None:
        logger.error(f"POST /jobs/{job_id}/edit - Job not found: {job_id}")
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    if job._info.status != JobStatus.UPLOADING:
        logger.warning(f"POST /jobs/{job_id}/edit - Job not in UPLOADING state (current: {job._info.status})")
        raise HTTPException(
            status_code=400,
            detail=f"Job {job_id} is not in UPLOADING state (current: {job._info.status})",
        )
    
    # Parse edit parameters
    edit_params = EditParams()  # Use defaults
    if edit_params_json:
        try:
            params_dict = json.loads(edit_params_json)
            edit_params = EditParams(**params_dict)
        except Exception as e:
            logger.error(f"POST /jobs/{job_id}/edit - Invalid edit params JSON: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid edit params JSON: {str(e)}",
            )
    
    # Load and validate image
    try:
        pil_image = load_image_from_upload(image)
        validate_image(pil_image)
        pil_image = preprocess_image(pil_image)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"POST /jobs/{job_id}/edit - Image processing error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Image processing error: {str(e)}",
        )
    
    # Save image temporarily
    temp_image_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            pil_image.save(tmp_file.name, "PNG")
            temp_image_path = Path(tmp_file.name)
        
        # Upload image to job
        if not job.upload_image(temp_image_path):
            logger.error(f"POST /jobs/{job_id}/edit - Failed to upload image")
            raise HTTPException(status_code=500, detail="Failed to upload image")
        
        # Set edit information
        job.set_edit_info(source_prompt, target_prompt, edit_params)
        
        # Update status to PENDING
        job.update_status(JobStatus.PENDING)
        
        # Check concurrent job limit
        active_count = job_manager.get_active_jobs_count()
        if active_count > MAX_CONCURRENT_JOBS:
            logger.warning(f"POST /jobs/{job_id}/edit - Maximum concurrent jobs ({MAX_CONCURRENT_JOBS}) reached")
            raise HTTPException(
                status_code=503,
                detail=f"Maximum concurrent jobs ({MAX_CONCURRENT_JOBS}) reached. Please wait for a job to complete.",
            )
        
        # Start editing task
        task = asyncio.create_task(
            run_edit_job(
                job_id=job_id,
                job_manager=job_manager,
                input_image_path=job.get_input_image_path(),
                source_prompt=source_prompt,
                target_prompt=target_prompt,
                edit_params=edit_params,
                output_image_path=job.get_output_image_path(),
            )
        )
        _active_tasks[job_id] = task
        
        # Clean up task when done
        def cleanup_task(job_id: str):
            if job_id in _active_tasks:
                del _active_tasks[job_id]
        
        task.add_done_callback(lambda _: cleanup_task(job_id))
        
        logger.info(f"POST /jobs/{job_id}/edit - Edit submitted successfully")
        return SubmitEditResponse(
            job_id=job_id,
            status=JobStatus.PENDING,
            message="Edit submitted successfully",
        )
        
    finally:
        # Clean up temp file
        if temp_image_path and temp_image_path.exists():
            temp_image_path.unlink()


async def run_edit_job(
    job_id: str,
    job_manager: JobManager,
    input_image_path: Path,
    source_prompt: str,
    target_prompt: str,
    edit_params: EditParams,
    output_image_path: Path,
):
    """Run edit job asynchronously.
    
    Args:
        job_id: Job identifier
        job_manager: Job manager instance
        input_image_path: Path to input image
        source_prompt: Source prompt
        target_prompt: Target prompt
        edit_params: Edit parameters
        output_image_path: Path to save output image
    """
    job = job_manager._get_job_object(job_id)
    if job is None:
        logger.error(f"run_edit_job - Job not found: {job_id}")
        return
    
    try:
        # Update status to RUNNING
        job.update_status(JobStatus.RUNNING)
        logger.info(f"run_edit_job - Starting edit for job {job_id}")
        
        # Process edit (this is CPU/GPU intensive, run in thread pool)
        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(
            None,
            flowedit_service.process_edit,
            input_image_path,
            source_prompt,
            target_prompt,
            output_image_path,
            edit_params.T_steps,
            edit_params.n_avg,
            edit_params.src_guidance_scale,
            edit_params.tar_guidance_scale,
            edit_params.n_min,
            edit_params.n_max,
        )
        
        if success:
            # Set result path and update status to COMPLETED
            job.set_result_path(output_image_path)
            job.update_status(JobStatus.COMPLETED)
            logger.info(f"run_edit_job - Edit completed successfully for job {job_id}")
        else:
            # Update status to FAILED
            job.update_status(
                JobStatus.FAILED,
                error_message="Edit processing failed",
            )
            logger.error(f"run_edit_job - Edit failed for job {job_id}")
            
    except Exception as e:
        # Update status to FAILED with error message
        error_message = str(e)
        error_traceback = None
        try:
            import traceback
            error_traceback = traceback.format_exc()
        except:
            pass
        
        job.update_status(
            JobStatus.FAILED,
            error_message=error_message,
            error_traceback=error_traceback,
        )
        logger.error(f"run_edit_job - Exception during edit for job {job_id}: {e}", exc_info=True)


@app.get("/jobs/{job_id}", response_model=JobInfo)
async def get_job_status(job_id: str) -> JobInfo:
    """Get job status and information.
    
    Args:
        job_id: Job identifier
        
    Returns:
        JobInfo with current status, progress, and error information
    """
    logger.info(f"GET /jobs/{job_id} - Job status requested")
    job = job_manager.get_job(job_id)
    if job is None:
        logger.error(f"GET /jobs/{job_id} - Job not found: {job_id}")
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    logger.info(f"GET /jobs/{job_id} - Returning status: {job.status}")
    return job


@app.get("/jobs/{job_id}/result")
async def get_job_result(job_id: str) -> FileResponse:
    """Download edited image result.
    
    Args:
        job_id: Job identifier
        
    Returns:
        PNG file download
    """
    logger.info(f"GET /jobs/{job_id}/result - Result requested")
    job = job_manager.get_job(job_id)
    if job is None:
        logger.error(f"GET /jobs/{job_id}/result - Job not found: {job_id}")
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    if job.status != JobStatus.COMPLETED:
        logger.warning(f"GET /jobs/{job_id}/result - Job not completed (status: {job.status})")
        raise HTTPException(
            status_code=400,
            detail=f"Job {job_id} is not completed (status: {job.status})",
        )
    
    if job.result_path is None:
        logger.error(f"GET /jobs/{job_id}/result - Result path not found")
        raise HTTPException(status_code=404, detail="Result not found")
    
    result_path = Path(job.result_path)
    if not result_path.exists():
        logger.error(f"GET /jobs/{job_id}/result - Result file does not exist: {result_path}")
        raise HTTPException(status_code=404, detail="Result file does not exist")
    
    logger.info(f"GET /jobs/{job_id}/result - Returning result file")
    return FileResponse(
        result_path,
        media_type="image/png",
        filename=f"{job_id}_edited.png",
    )


@app.delete("/jobs/{job_id}")
async def cancel_job(job_id: str) -> JSONResponse:
    """Cancel or delete a job.
    
    Handles both active editing jobs (PENDING/RUNNING) and incomplete uploads (UPLOADING).
    
    Args:
        job_id: Job identifier
        
    Returns:
        JSON response with cancellation status
    """
    logger.info(f"DELETE /jobs/{job_id} - Cancellation requested")
    job = job_manager.get_job(job_id)
    if job is None:
        logger.error(f"DELETE /jobs/{job_id} - Job not found: {job_id}")
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    # Handle different job states
    if job.status == JobStatus.UPLOADING:
        # Cleanup upload job (full cleanup: delete directory and remove from memory)
        logger.info(f"DELETE /jobs/{job_id} - Cleaning up upload job")
        success = job_manager.cleanup_job(job_id)
        if success:
            logger.info(f"DELETE /jobs/{job_id} - Upload job cancelled successfully")
            return JSONResponse(
                content={"message": f"Job {job_id} cancelled successfully"},
                status_code=200,
            )
        else:
            logger.error(f"DELETE /jobs/{job_id} - Failed to cancel upload job")
            raise HTTPException(status_code=500, detail="Failed to cancel upload job")
    
    elif job.status in [JobStatus.PENDING, JobStatus.RUNNING]:
        # Cancel editing job
        # Cancel the async task if it exists
        if job_id in _active_tasks:
            task = _active_tasks[job_id]
            task.cancel()
            del _active_tasks[job_id]
            logger.info(f"DELETE /jobs/{job_id} - Cancelled active task")
        
        # Update job status via cleanup_job (keeps directory/memory for logs/results)
        success = job_manager.cleanup_job(job_id)
        
        if success:
            logger.info(f"DELETE /jobs/{job_id} - Editing job cancelled successfully")
            return JSONResponse(
                content={"message": f"Job {job_id} cancelled successfully"},
                status_code=200,
            )
        else:
            logger.error(f"DELETE /jobs/{job_id} - Failed to cancel job")
            raise HTTPException(status_code=500, detail="Failed to cancel job")
    else:
        # Cannot cancel completed/failed/cancelled jobs
        logger.warning(f"DELETE /jobs/{job_id} - Job cannot be cancelled (status: {job.status})")
        raise HTTPException(
            status_code=400,
            detail=f"Job {job_id} cannot be cancelled (status: {job.status})",
        )


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint.
    
    Returns:
        HealthResponse with service status, model loaded status, and job counts
    """
    active_jobs = job_manager.get_active_jobs_count()
    total_jobs = len(job_manager.list_jobs())
    
    logger.info(f"GET /health - Health check: model_loaded={flowedit_service.is_loaded}, {active_jobs} active jobs, {total_jobs} total jobs")
    
    return HealthResponse(
        status="healthy",
        model_loaded=flowedit_service.is_loaded,
        active_jobs=active_jobs,
        total_jobs=total_jobs,
    )


if __name__ == "__main__":
    import uvicorn
    
    port = get_port()
    host = get_host()
    
    uvicorn.run(app, host=host, port=port)
