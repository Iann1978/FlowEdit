"""Configuration management for FlowEdit server."""

import os
from pathlib import Path


def get_jobs_dir() -> Path:
    """Get jobs directory from environment or default.
    
    Returns:
        Path to jobs directory
    """
    jobs_dir = os.getenv("FLOWEDIT_JOBS_DIR", None)
    if jobs_dir is None:
        # Default to server/flowedit/server/jobs/ relative to this file
        script_dir = Path(__file__).parent
        jobs_dir = str(script_dir / "jobs")
    return Path(jobs_dir)


def get_model_path() -> str:
    """Get FLUX model path from environment or default.
    
    Returns:
        Model path string (default: "black-forest-labs/FLUX.1-dev")
    """
    return os.getenv("FLOWEDIT_MODEL_PATH", "black-forest-labs/FLUX.1-dev")


def get_device() -> str:
    """Get device from environment or auto-detect.
    
    Returns:
        Device string ("cuda" or "cpu")
    """
    device = os.getenv("FLOWEDIT_DEVICE", None)
    if device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return device


def get_max_concurrent_jobs() -> int:
    """Get maximum concurrent jobs from environment or default.
    
    Returns:
        Maximum concurrent jobs (default: 1)
    """
    return int(os.getenv("FLOWEDIT_MAX_CONCURRENT_JOBS", "1"))


def get_port() -> int:
    """Get server port from environment or default.
    
    Returns:
        Server port (default: 8001)
    """
    return int(os.getenv("PORT", "8001"))


def get_host() -> str:
    """Get server host from environment or default.
    
    Returns:
        Server host (default: "0.0.0.0")
    """
    return os.getenv("HOST", "0.0.0.0")
