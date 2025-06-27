# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Multi-GPU optimization utilities for MatterGen Phase 3.
Provides DataParallel and DistributedDataParallel support for faster generation.
"""

import os
import logging
import warnings
from typing import Optional, Union, Dict, Any
from functools import wraps

# Try to import torch components with fallbacks
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import torch.distributed as dist
    from torch.nn.parallel import DataParallel, DistributedDataParallel
    DISTRIBUTED_AVAILABLE = True
except ImportError:
    DISTRIBUTED_AVAILABLE = False

logger = logging.getLogger(__name__)


class MultiGPUConfig:
    """Configuration for multi-GPU operations."""
    
    def __init__(
        self,
        strategy: str = "auto",  # "auto", "dp", "ddp", "single"
        device_ids: Optional[list] = None,
        master_port: str = "12355",
        backend: str = "nccl",
        find_unused_parameters: bool = False,
        bucket_cap_mb: int = 25,
        max_gpus: Optional[int] = None,
    ):
        self.strategy = strategy
        self.device_ids = device_ids
        self.master_port = master_port
        self.backend = backend
        self.find_unused_parameters = find_unused_parameters
        self.bucket_cap_mb = bucket_cap_mb
        self.max_gpus = max_gpus
        
        # Auto-detect available GPUs
        self.num_gpus = torch.cuda.device_count()
        if self.max_gpus:
            self.num_gpus = min(self.num_gpus, self.max_gpus)
            
        # Set device IDs if not provided
        if self.device_ids is None and self.num_gpus > 0:
            self.device_ids = list(range(self.num_gpus))
        elif self.device_ids:
            self.num_gpus = len(self.device_ids)
            
        # Auto-select strategy
        if self.strategy == "auto":
            if self.num_gpus <= 1:
                self.strategy = "single"
            elif self.num_gpus <= 4:
                self.strategy = "dp"  # DataParallel for smaller setups
            else:
                self.strategy = "ddp"  # DistributedDataParallel for larger setups


def setup_distributed(rank: int, world_size: int, master_port: str = "12355", backend: str = "nccl"):
    """Initialize distributed training environment."""
    if not TORCH_AVAILABLE or not DISTRIBUTED_AVAILABLE:
        raise RuntimeError("Distributed training not available")
        
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = master_port
    
    # Initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Clean up distributed training environment."""
    if DISTRIBUTED_AVAILABLE and dist.is_initialized():
        dist.destroy_process_group()


def is_distributed_available() -> bool:
    """Check if distributed training is available."""
    return DISTRIBUTED_AVAILABLE and TORCH_AVAILABLE and torch.cuda.is_available()


def get_optimal_batch_size(base_batch_size: int, num_gpus: int, strategy: str) -> int:
    """Calculate optimal batch size for multi-GPU setup."""
    if strategy == "single" or num_gpus <= 1:
        return base_batch_size
    
    # For DataParallel and DistributedDataParallel, we can scale batch size
    # Conservative scaling to avoid OOM
    if num_gpus == 2:
        return base_batch_size * 2
    elif num_gpus <= 4:
        return base_batch_size * 3  # Conservative scaling
    else:
        return base_batch_size * 4  # Cap at 4x for stability


class MultiGPUWrapper:
    """Wrapper for multi-GPU model operations."""
    
    def __init__(self, config: MultiGPUConfig):
        self.config = config
        self._is_distributed_setup = False
        
    def wrap_model(self, model: nn.Module) -> nn.Module:
        """Wrap model for multi-GPU execution."""
        if self.config.strategy == "single" or self.config.num_gpus <= 1:
            logger.info("Using single GPU/CPU execution")
            return model
            
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to single device")
            return model
            
        if self.config.strategy == "dp":
            return self._setup_data_parallel(model)
        elif self.config.strategy == "ddp":
            return self._setup_distributed_data_parallel(model)
        else:
            logger.warning(f"Unknown strategy {self.config.strategy}, using single device")
            return model
            
    def _setup_data_parallel(self, model: nn.Module) -> nn.Module:
        """Setup DataParallel wrapper."""
        if not TORCH_AVAILABLE:
            return model
            
        try:
            logger.info(f"Setting up DataParallel on {self.config.num_gpus} GPUs: {self.config.device_ids}")
            
            # Move model to first device
            model = model.to(f"cuda:{self.config.device_ids[0]}")
            
            # Wrap with DataParallel
            wrapped_model = DataParallel(model, device_ids=self.config.device_ids)
            
            # Preserve original model reference for attribute access
            if not hasattr(wrapped_model, '_original_module'):
                wrapped_model._original_module = model
            
            logger.info("DataParallel setup complete")
            return wrapped_model
            
        except Exception as e:
            logger.error(f"Failed to setup DataParallel: {e}")
            logger.info("Falling back to single GPU")
            return model.to("cuda:0" if torch.cuda.is_available() else "cpu")
            
    def _setup_distributed_data_parallel(self, model: nn.Module) -> nn.Module:
        """Setup DistributedDataParallel wrapper."""
        if not TORCH_AVAILABLE or not DISTRIBUTED_AVAILABLE:
            logger.warning("Distributed training not available, falling back to DataParallel")
            return self._setup_data_parallel(model)
            
        try:
            if not is_distributed_available():
                logger.warning("Distributed training not available, falling back to DataParallel")
                return self._setup_data_parallel(model)
                
            # For single-process multi-GPU (common in inference)
            if not dist.is_initialized():
                logger.info("Initializing single-process distributed setup")
                setup_distributed(0, 1, self.config.master_port, self.config.backend)
                self._is_distributed_setup = True
            
            logger.info(f"Setting up DistributedDataParallel on {self.config.num_gpus} GPUs")
            
            # Move model to current device
            device = torch.cuda.current_device()
            model = model.to(device)
            
            # Wrap with DistributedDataParallel
            wrapped_model = DistributedDataParallel(
                model,
                device_ids=[device],
                find_unused_parameters=self.config.find_unused_parameters,
                bucket_cap_mb=self.config.bucket_cap_mb,
            )
            
            # Preserve original model reference for attribute access
            if not hasattr(wrapped_model, '_original_module'):
                wrapped_model._original_module = model
            
            logger.info("DistributedDataParallel setup complete")
            return wrapped_model
            
        except Exception as e:
            logger.error(f"Failed to setup DistributedDataParallel: {e}")
            logger.info("Falling back to DataParallel")
            return self._setup_data_parallel(model)
            
    def cleanup(self):
        """Clean up multi-GPU resources."""
        if self._is_distributed_setup:
            cleanup_distributed()
            self._is_distributed_setup = False


def multi_gpu_safe(func):
    """Decorator to handle multi-GPU operations safely."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.error("CUDA OOM in multi-GPU operation. Try reducing batch size.")
                torch.cuda.empty_cache()
                raise
            elif "NCCL" in str(e):
                logger.error("NCCL error in distributed operation. Falling back to single GPU.")
                # Could implement fallback logic here
                raise
            else:
                raise
    return wrapper


class MultiGPUMemoryManager:
    """Memory management for multi-GPU setups."""
    
    def __init__(self, cleanup_frequency: int = 5):
        self.cleanup_frequency = cleanup_frequency
        self.step_count = 0
        
    def step(self):
        """Call this after each batch/generation step."""
        self.step_count += 1
        if self.step_count % self.cleanup_frequency == 0:
            self.cleanup_memory()
            
    def cleanup_memory(self):
        """Clean up GPU memory across all devices."""
        if torch.cuda.is_available():
            for device_id in range(torch.cuda.device_count()):
                with torch.cuda.device(device_id):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()


def optimize_multi_gpu_generation(
    model: nn.Module,
    batch_size: int,
    multi_gpu_config: Optional[MultiGPUConfig] = None,
    enable_memory_cleanup: bool = True,
) -> tuple[nn.Module, int, MultiGPUWrapper]:
    """
    Optimize model and batch size for multi-GPU generation.
    
    Args:
        model: The model to optimize
        batch_size: Base batch size
        multi_gpu_config: Multi-GPU configuration
        enable_memory_cleanup: Whether to enable automatic memory cleanup
        
    Returns:
        Tuple of (optimized_model, optimized_batch_size, gpu_wrapper)
    """
    if multi_gpu_config is None:
        multi_gpu_config = MultiGPUConfig()
        
    logger.info(f"Optimizing for multi-GPU with strategy: {multi_gpu_config.strategy}")
    logger.info(f"Available GPUs: {multi_gpu_config.num_gpus}")
    
    # Create GPU wrapper
    gpu_wrapper = MultiGPUWrapper(multi_gpu_config)
    
    # Wrap model for multi-GPU
    optimized_model = gpu_wrapper.wrap_model(model)
    
    # Optimize batch size
    optimized_batch_size = get_optimal_batch_size(
        batch_size, multi_gpu_config.num_gpus, multi_gpu_config.strategy
    )
    
    logger.info(f"Optimized batch size: {batch_size} -> {optimized_batch_size}")
    
    return optimized_model, optimized_batch_size, gpu_wrapper


def get_multi_gpu_info() -> Dict[str, Any]:
    """Get information about multi-GPU setup."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "num_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
        "distributed_available": is_distributed_available(),
        "distributed_initialized": dist.is_initialized() if dist.is_available() else False,
    }
    
    if torch.cuda.is_available():
        info["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        info["gpu_memory"] = [
            {
                "total": torch.cuda.get_device_properties(i).total_memory,
                "allocated": torch.cuda.memory_allocated(i),
                "cached": torch.cuda.memory_reserved(i),
            }
            for i in range(torch.cuda.device_count())
        ]
    
    return info


def print_multi_gpu_info():
    """Print multi-GPU setup information."""
    info = get_multi_gpu_info()
    
    print("\n" + "="*50)
    print("MULTI-GPU SETUP INFORMATION")
    print("="*50)
    print(f"CUDA Available: {info['cuda_available']}")
    print(f"Number of GPUs: {info['num_gpus']}")
    
    if info['cuda_available'] and info['num_gpus'] > 0:
        print(f"Current Device: {info['current_device']}")
        print(f"Distributed Available: {info['distributed_available']}")
        print(f"Distributed Initialized: {info['distributed_initialized']}")
        
        print("\nGPU Details:")
        for i, (name, memory) in enumerate(zip(info['gpu_names'], info['gpu_memory'])):
            print(f"  GPU {i}: {name}")
            print(f"    Total Memory: {memory['total'] / 1e9:.1f} GB")
            print(f"    Allocated: {memory['allocated'] / 1e9:.1f} GB")
            print(f"    Cached: {memory['cached'] / 1e9:.1f} GB")
    
    print("="*50)
