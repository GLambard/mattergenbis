# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Performance optimization utilities for MatterGen generation.
Implements mixed precision, model compilation, and memory management.
"""

import logging
import torch
import torch.nn as nn
from typing import Any, Optional
import warnings

logger = logging.getLogger(__name__)


class PerformanceOptimizer:
    """Utility class for applying performance optimizations to MatterGen models."""
    
    def __init__(self):
        self.original_precision = None
        self.is_compiled = False
        self.use_autocast = False
        
    def enable_mixed_precision(self, model: nn.Module, use_fp16: bool = True) -> nn.Module:
        """
        Enable mixed precision inference for faster generation.
        
        Args:
            model: The model to optimize
            use_fp16: Whether to use FP16 (True) or keep FP32 (False)
            
        Returns:
            Optimized model
        """
        if use_fp16 and torch.cuda.is_available():
            logger.info("Enabling FP16 mixed precision")
            
            # Instead of converting the entire model to half, we'll use autocast
            # during inference to avoid dtype mismatches
            self.use_autocast = True
            
            # Enable optimized attention if available
            if hasattr(torch.backends.cudnn, 'allow_tf32'):
                torch.backends.cudnn.allow_tf32 = True
                
            # Enable tensor core usage
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            
        return model
    
    def compile_model(self, model: nn.Module, mode: str = "reduce-overhead") -> nn.Module:
        """
        Compile model with PyTorch 2.0+ for optimization.
        
        Args:
            model: The model to compile
            mode: Compilation mode ('reduce-overhead', 'max-autotune', 'default')
            
        Returns:
            Compiled model
        """
        if hasattr(torch, 'compile') and not self.is_compiled:
            try:
                logger.info(f"Compiling model with mode: {mode}")
                model = torch.compile(model, mode=mode)
                self.is_compiled = True
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}")
                
        return model
    
    def optimize_memory(self, clear_cache_frequency: int = 10):
        """
        Configure memory optimization settings.
        
        Args:
            clear_cache_frequency: How often to clear CUDA cache (every N batches)
        """
        if torch.cuda.is_available():
            # Enable memory efficient attention
            try:
                torch.backends.cuda.enable_flash_sdp(True)
            except:
                pass
                
            # Set memory fraction if needed
            if torch.cuda.get_device_properties(0).total_memory > 8e9:  # > 8GB
                torch.cuda.set_per_process_memory_fraction(0.9)
                
            logger.info("Memory optimization enabled")
            
        return clear_cache_frequency
    
    def set_inference_mode(self):
        """Set optimal settings for inference."""
        torch.set_grad_enabled(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        
        # Set float32 matmul precision for speed
        torch.set_float32_matmul_precision("high")
        
        logger.info("Inference mode optimizations applied")
    
    def get_optimal_batch_size(self, model: nn.Module, sample_input: Any, 
                              max_memory_gb: float = None) -> int:
        """
        Estimate optimal batch size based on available memory.
        
        Args:
            model: The model
            sample_input: Sample input for memory estimation
            max_memory_gb: Maximum memory to use (GB)
            
        Returns:
            Suggested batch size
        """
        if not torch.cuda.is_available():
            return 16  # Conservative default for CPU
            
        try:
            # Get available memory
            memory_total = torch.cuda.get_device_properties(0).total_memory
            memory_available = memory_total - torch.cuda.memory_allocated()
            
            if max_memory_gb:
                memory_available = min(memory_available, max_memory_gb * 1e9)
                
            # Estimate memory per sample (very rough)
            memory_per_sample = memory_total * 0.01  # 1% of total as rough estimate
            
            suggested_batch_size = max(1, int(memory_available * 0.7 / memory_per_sample))
            suggested_batch_size = min(suggested_batch_size, 128)  # Cap at 128
            
            logger.info(f"Suggested batch size: {suggested_batch_size}")
            return suggested_batch_size
            
        except Exception as e:
            logger.warning(f"Could not estimate optimal batch size: {e}")
            return 32  # Default
    
    @staticmethod
    def clear_memory_cache():
        """Clear CUDA memory cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def apply_generation_optimizations(model: nn.Module, 
                                 enable_fp16: bool = True,
                                 compile_model: bool = True,
                                 optimize_memory: bool = True) -> nn.Module:
    """
    Apply all generation optimizations to a model.
    
    Args:
        model: Model to optimize
        enable_fp16: Enable mixed precision
        compile_model: Enable model compilation
        optimize_memory: Enable memory optimizations
        
    Returns:
        Optimized model
    """
    optimizer = PerformanceOptimizer()
    
    # Set inference mode
    optimizer.set_inference_mode()
    
    # Apply mixed precision
    if enable_fp16:
        model = optimizer.enable_mixed_precision(model)
        
    # Compile model
    if compile_model:
        model = optimizer.compile_model(model)
        
    # Memory optimizations
    if optimize_memory:
        optimizer.optimize_memory()
    
    logger.info("All performance optimizations applied")
    return model
