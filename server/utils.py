"""Image processing utilities for FlowEdit server."""

from pathlib import Path
from typing import Tuple

from fastapi import UploadFile, HTTPException
from PIL import Image
import io


def load_image_from_upload(upload_file: UploadFile) -> Image.Image:
    """Load PIL Image from FastAPI UploadFile.
    
    Args:
        upload_file: FastAPI UploadFile object
        
    Returns:
        PIL Image object
        
    Raises:
        HTTPException: If image cannot be loaded or format is invalid
    """
    try:
        # Read image data
        image_data = upload_file.file.read()
        
        # Reset file pointer for potential reuse
        upload_file.file.seek(0)
        
        # Open image with PIL
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary (handles RGBA, P, etc.)
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        return image
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to load image: {str(e)}"
        )


def validate_image(image: Image.Image, max_size: int = 4096) -> Tuple[int, int]:
    """Validate image dimensions and constraints.
    
    Args:
        image: PIL Image object
        max_size: Maximum dimension in pixels (default: 4096)
        
    Returns:
        Tuple of (width, height)
        
    Raises:
        HTTPException: If image is invalid
    """
    width, height = image.size
    
    if width <= 0 or height <= 0:
        raise HTTPException(
            status_code=400,
            detail="Image dimensions must be positive"
        )
    
    if width > max_size or height > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"Image dimensions exceed maximum size of {max_size}x{max_size}"
        )
    
    return width, height


def preprocess_image(image: Image.Image) -> Image.Image:
    """Preprocess image for FlowEdit (crop to dimensions divisible by 16).
    
    This matches the preprocessing in run_script.py to avoid resizing issues.
    
    Args:
        image: PIL Image object
        
    Returns:
        Preprocessed PIL Image
    """
    width, height = image.size
    
    # Crop to dimensions divisible by 16
    new_width = width - (width % 16)
    new_height = height - (height % 16)
    
    if new_width != width or new_height != height:
        image = image.crop((0, 0, new_width, new_height))
    
    return image


def save_image(image: Image.Image, output_path: Path) -> None:
    """Save PIL Image to file.
    
    Args:
        image: PIL Image object
        output_path: Path to save image
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, "PNG")
