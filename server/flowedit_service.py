"""FlowEdit Model Service Singleton
Handles FLUX model initialization and image editing inference.
"""

import os
import logging
import random
import numpy as np
import torch
from pathlib import Path
from PIL import Image

# Import FlowEdit utilities from parent directory
import sys
flowedit_dir = Path(__file__).parent.parent
sys.path.insert(0, str(flowedit_dir))
from FlowEdit_utils import FlowEditFLUX

logger = logging.getLogger(__name__)


class FlowEditService:
    """Singleton service for FlowEdit model inference."""
    
    _instance = None
    _pipe = None
    _scheduler = None
    _device = None
    _is_loaded = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FlowEditService, cls).__new__(cls)
        return cls._instance
    
    def load_model(
        self,
        model_path: str = None,
        device: str = None,
    ):
        """
        Load FLUX model. This should be called once at startup.
        
        Args:
            model_path: Path to FLUX model. If None, uses default.
            device: Device to use ('cuda' or 'cpu'). If None, auto-detects.
        """
        if self._is_loaded:
            logger.warning("Model already loaded. Skipping reload.")
            return
        
        from diffusers import FluxPipeline
        
        # Set defaults if not provided
        if model_path is None:
            model_path = "black-forest-labs/FLUX.1-dev"
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Loading FLUX model: {model_path}")
        logger.info(f"Using device: {device}")
        
        try:
            # Load FLUX pipeline
            self._pipe = FluxPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
            self._scheduler = self._pipe.scheduler
            self._pipe = self._pipe.to(device)
            self._device = torch.device(device)
            
            self._is_loaded = True
            logger.info("FLUX model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load FLUX model: {e}")
            raise
    
    def process_edit(
        self,
        input_image_path: Path,
        source_prompt: str,
        target_prompt: str,
        output_image_path: Path,
        T_steps: int = 28,
        n_avg: int = 1,
        src_guidance_scale: float = 1.5,
        tar_guidance_scale: float = 5.5,
        n_min: int = 0,
        n_max: int = 24,
        seed: int = None,
    ) -> bool:
        """
        Process image edit using FlowEdit.
        
        Args:
            input_image_path: Path to input image
            source_prompt: Source prompt describing the input image
            target_prompt: Target prompt describing desired edit
            output_image_path: Path to save edited image
            T_steps: Number of timesteps (default: 28)
            n_avg: Number of averaging iterations (default: 1)
            src_guidance_scale: Source guidance scale (default: 1.5)
            tar_guidance_scale: Target guidance scale (default: 5.5)
            n_min: Minimum step for editing (default: 0)
            n_max: Maximum step for editing (default: 24)
            seed: Random seed (default: None, uses random seed)
            
        Returns:
            True if successful, False otherwise
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Set random seed if provided
            if seed is not None:
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            
            # Load and preprocess image
            image = Image.open(input_image_path)
            # Crop image to have both dimensions divisible by 16 - avoids issues with resizing
            image = image.crop((0, 0, image.width - image.width % 16, image.height - image.height % 16))
            
            # Preprocess image for pipeline
            image_src = self._pipe.image_processor.preprocess(image)
            # Cast image to half precision
            image_src = image_src.to(self._device).half()
            
            # Encode image to latents
            with torch.autocast("cuda"), torch.inference_mode():
                x0_src_denorm = self._pipe.vae.encode(image_src).latent_dist.mode()
            x0_src = (x0_src_denorm - self._pipe.vae.config.shift_factor) * self._pipe.vae.config.scaling_factor
            # Send to device
            x0_src = x0_src.to(self._device)
            
            # Process edit using FlowEditFLUX
            negative_prompt = ""  # FLUX doesn't use negative prompts
            x0_tar = FlowEditFLUX(
                self._pipe,
                self._scheduler,
                x0_src,
                source_prompt,
                target_prompt,
                negative_prompt,
                T_steps,
                n_avg,
                src_guidance_scale,
                tar_guidance_scale,
                n_min,
                n_max,
            )
            
            # Decode latents back to image
            x0_tar_denorm = (x0_tar / self._pipe.vae.config.scaling_factor) + self._pipe.vae.config.shift_factor
            with torch.autocast("cuda"), torch.inference_mode():
                image_tar = self._pipe.vae.decode(x0_tar_denorm, return_dict=False)[0]
            image_tar = self._pipe.image_processor.postprocess(image_tar)
            
            # Save edited image
            output_image_path.parent.mkdir(parents=True, exist_ok=True)
            image_tar[0].save(output_image_path, "PNG")
            
            logger.info(f"Successfully processed edit and saved to {output_image_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing edit: {e}", exc_info=True)
            return False
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded


# Global service instance
flowedit_service = FlowEditService()
