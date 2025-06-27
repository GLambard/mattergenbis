# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Performance optimization utilities for MatterGen generation.
Implements mixed precision, model compilation, memory management, and Phase 3 multi-GPU support.
"""

import logging
import gc
import psutil
import threading
import time
import torch
import torch.nn as nn
from typing import Any, Optional, Dict, List
import warnings
from dataclasses import dataclass

# Phase 3 imports - with better error handling
try:
    from .multi_gpu_optimizer import MultiGPUConfig, MultiGPUWrapper, optimize_multi_gpu_generation, print_multi_gpu_info
    MULTI_GPU_AVAILABLE = True
except ImportError as e:
    MULTI_GPU_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.debug(f"Multi-GPU optimizer not available: {e}")

try:
    from .graph_cache import setup_graph_caching, clear_all_caches, print_cache_stats
    GRAPH_CACHE_AVAILABLE = True
except ImportError as e:
    GRAPH_CACHE_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.debug(f"Graph cache not available: {e}")

PHASE3_AVAILABLE = MULTI_GPU_AVAILABLE and GRAPH_CACHE_AVAILABLE

if not PHASE3_AVAILABLE:
    logger = logging.getLogger(__name__)
    logger.warning("Phase 3 modules not fully available, some optimizations will be disabled")

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for all optimization features."""
    # Phase 2 optimizations
    enable_mixed_precision: bool = True
    enable_model_compilation: bool = True
    enable_memory_optimization: bool = True
    
    # Phase 3 optimizations
    enable_multi_gpu: bool = True
    enable_graph_caching: bool = True
    enable_gradient_checkpointing: bool = False
    enable_memory_monitoring: bool = True
    
    # Multi-GPU settings
    multi_gpu_strategy: str = "auto"  # "auto", "dp", "ddp", "single"
    max_gpus: Optional[int] = None
    
    # Memory settings
    memory_cleanup_frequency: int = 10
    max_memory_usage_gb: Optional[float] = None
    enable_swap_attention: bool = True
    
    # Caching settings
    graph_cache_size: int = 1000
    enable_disk_cache: bool = True
    
    # Hardware-specific
    optimize_for_hardware: bool = True
    enable_tensor_cores: bool = True


class MemoryMonitor:
    """Monitor memory usage and trigger cleanup when needed."""
    
    def __init__(self, max_memory_gb: Optional[float] = None, cleanup_threshold: float = 0.85):
        self.max_memory_gb = max_memory_gb
        self.cleanup_threshold = cleanup_threshold
        self.monitoring = False
        self.monitor_thread = None
        self.cleanup_callbacks = []
        
    def add_cleanup_callback(self, callback):
        """Add a callback to be called during memory cleanup."""
        self.cleanup_callbacks.append(callback)
        
    def start_monitoring(self, check_interval: float = 5.0):
        """Start memory monitoring in background thread."""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop, 
            args=(check_interval,), 
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("Memory monitoring started")
        
    def stop_monitoring(self):
        """Stop memory monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        logger.info("Memory monitoring stopped")
        
    def _monitor_loop(self, check_interval: float):
        """Background monitoring loop."""
        while self.monitoring:
            try:
                self._check_memory()
                time.sleep(check_interval)
            except Exception as e:
                logger.error(f"Error in memory monitoring: {e}")
                
    def _check_memory(self):
        """Check memory usage and trigger cleanup if needed."""
        # Check CUDA memory
        if torch.cuda.is_available():
            for device_id in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(device_id)
                reserved = torch.cuda.memory_reserved(device_id)
                total = torch.cuda.get_device_properties(device_id).total_memory
                
                usage_ratio = reserved / total
                if usage_ratio > self.cleanup_threshold:
                    logger.warning(f"High GPU memory usage on device {device_id}: {usage_ratio:.2%}")
                    self._trigger_cleanup()
                    
        # Check system memory
        memory = psutil.virtual_memory()
        if memory.percent > self.cleanup_threshold * 100:
            logger.warning(f"High system memory usage: {memory.percent:.1f}%")
            self._trigger_cleanup()
            
    def _trigger_cleanup(self):
        """Trigger memory cleanup."""
        logger.info("Triggering memory cleanup")
        
        # Run cleanup callbacks
        for callback in self.cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in cleanup callback: {e}")
                
        # Standard cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        gc.collect()


class PerformanceOptimizer:
    """Enhanced utility class for applying performance optimizations to MatterGen models."""
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        self.original_precision = None
        self.is_compiled = False
        self.use_autocast = False
        self.memory_monitor = None
        self.multi_gpu_wrapper = None
        
        # Initialize Phase 3 components
        if PHASE3_AVAILABLE and self.config.enable_memory_monitoring:
            self.memory_monitor = MemoryMonitor(self.config.max_memory_usage_gb)
            
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
                
            # Phase 3: Enhanced tensor core optimization
            if self.config.enable_tensor_cores:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.benchmark = True
                
                # Enable Flash Attention if available
                try:
                    torch.backends.cuda.enable_flash_sdp(True)
                    logger.info("Flash Attention enabled")
                except AttributeError:
                    pass
            
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
                
                # Phase 3: Hardware-specific compilation options
                compile_options = {}
                if self.config.optimize_for_hardware:
                    # Use more aggressive optimization for multi-GPU
                    if torch.cuda.device_count() > 1:
                        compile_options.update({
                            "dynamic": True,
                            "fullgraph": False,  # Allow breaks for multi-GPU
                        })
                
                model = torch.compile(model, mode=mode, **compile_options)
                self.is_compiled = True
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}")
                
        return model
    
    def setup_multi_gpu(self, model: nn.Module, batch_size: int) -> tuple[nn.Module, int]:
        """
        Setup multi-GPU support for the model.
        
        Args:
            model: Model to setup for multi-GPU
            batch_size: Base batch size
            
        Returns:
            Tuple of (optimized_model, optimized_batch_size)
        """
        if not MULTI_GPU_AVAILABLE or not self.config.enable_multi_gpu:
            logger.info("Multi-GPU optimization not available or disabled")
            return model, batch_size
            
        logger.info("Setting up multi-GPU support")
        
        try:
            # Create multi-GPU configuration
            multi_gpu_config = MultiGPUConfig(
                strategy=self.config.multi_gpu_strategy,
                max_gpus=self.config.max_gpus,
            )
            
            # Apply multi-GPU optimization
            optimized_model, optimized_batch_size, self.multi_gpu_wrapper = optimize_multi_gpu_generation(
                model, batch_size, multi_gpu_config
            )
            
            return optimized_model, optimized_batch_size
        except Exception as e:
            logger.warning(f"Multi-GPU setup failed: {e}, falling back to single GPU")
            return model, batch_size
    
    def setup_graph_caching(self):
        """Setup graph operation caching."""
        if not GRAPH_CACHE_AVAILABLE or not self.config.enable_graph_caching:
            logger.info("Graph caching not available or disabled")
            return
            
        logger.info("Setting up graph operation caching")
        try:
            setup_graph_caching(
                enable_caching=True,
                cache_size=self.config.graph_cache_size,
                enable_disk_cache=self.config.enable_disk_cache,
            )
        except Exception as e:
            logger.warning(f"Graph caching setup failed: {e}")
    
    def enable_gradient_checkpointing(self, model: nn.Module) -> nn.Module:
        """Enable gradient checkpointing for memory efficiency."""
        if not self.config.enable_gradient_checkpointing:
            return model
            
        try:
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled")
            elif hasattr(model, 'set_gradient_checkpointing'):
                model.set_gradient_checkpointing(True)
                logger.info("Gradient checkpointing enabled")
        except Exception as e:
            logger.warning(f"Could not enable gradient checkpointing: {e}")
            
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
                
            # Phase 3: Enhanced memory management
            if self.config.enable_swap_attention:
                try:
                    # Enable memory efficient attention patterns
                    torch.backends.cuda.enable_mem_efficient_sdp(True)
                    logger.info("Memory efficient attention enabled")
                except AttributeError:
                    pass
                    
            # Set memory fraction if needed
            if torch.cuda.get_device_properties(0).total_memory > 8e9:  # > 8GB
                torch.cuda.set_per_process_memory_fraction(0.9)
                
            # Start memory monitoring
            if self.memory_monitor and self.config.enable_memory_monitoring:
                self.memory_monitor.add_cleanup_callback(self.clear_memory_cache)
                self.memory_monitor.start_monitoring()
                
            logger.info("Memory optimization enabled")
            
        return clear_cache_frequency

    def set_inference_mode(self):
        """Set optimal settings for inference."""
        torch.set_grad_enabled(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        
        # Set float32 matmul precision for speed
        torch.set_float32_matmul_precision("high")
        
        # Phase 3: Additional inference optimizations
        if self.config.optimize_for_hardware:
            # Enable optimized kernels
            try:
                torch.jit.set_fusion_strategy([('STATIC', 20), ('DYNAMIC', 20)])
            except:
                pass
        
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
            
            # Phase 3: Adjust for multi-GPU
            if PHASE3_AVAILABLE and self.config.enable_multi_gpu and torch.cuda.device_count() > 1:
                num_gpus = min(torch.cuda.device_count(), self.config.max_gpus or torch.cuda.device_count())
                suggested_batch_size = max(suggested_batch_size, num_gpus * 2)  # Ensure minimum per GPU
            
            logger.info(f"Suggested batch size: {suggested_batch_size}")
            return suggested_batch_size
            
        except Exception as e:
            logger.warning(f"Could not estimate optimal batch size: {e}")
            return 32  # Default
    
    @staticmethod
    def clear_memory_cache():
        """Clear CUDA memory cache and other caches."""
        if torch.cuda.is_available():
            # Clear CUDA cache on all devices
            for device_id in range(torch.cuda.device_count()):
                with torch.cuda.device(device_id):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
        
        # Clear graph caches if available
        if GRAPH_CACHE_AVAILABLE:
            try:
                clear_all_caches()
            except Exception as e:
                logger.warning(f"Could not clear graph caches: {e}")
            
        # Python garbage collection
        gc.collect()
    
    def cleanup(self):
        """Clean up optimizer resources."""
        if self.memory_monitor:
            self.memory_monitor.stop_monitoring()
            
        if self.multi_gpu_wrapper:
            self.multi_gpu_wrapper.cleanup()
            
        self.clear_memory_cache()
        
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        stats = {
            "mixed_precision": self.use_autocast,
            "compiled": self.is_compiled,
            "cuda_available": torch.cuda.is_available(),
            "num_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "phase3_available": PHASE3_AVAILABLE,
        }
        
        if torch.cuda.is_available():
            stats["gpu_memory"] = [
                {
                    "device": i,
                    "allocated_gb": torch.cuda.memory_allocated(i) / 1e9,
                    "reserved_gb": torch.cuda.memory_reserved(i) / 1e9,
                    "total_gb": torch.cuda.get_device_properties(i).total_memory / 1e9,
                }
                for i in range(torch.cuda.device_count())
            ]
        
        return stats


def apply_all_optimizations(model: nn.Module, 
                          batch_size: int,
                          config: Optional[OptimizationConfig] = None) -> tuple[nn.Module, int, PerformanceOptimizer]:
    """
    Apply all available optimizations to a model.
    
    Args:
        model: Model to optimize
        batch_size: Base batch size
        config: Optimization configuration
        
    Returns:
        Tuple of (optimized_model, optimized_batch_size, optimizer_instance)
    """
    if config is None:
        config = OptimizationConfig()
        
    optimizer = PerformanceOptimizer(config)
    
    # Phase 2 optimizations
    optimizer.set_inference_mode()
    
    if config.enable_mixed_precision:
        model = optimizer.enable_mixed_precision(model)
        
    if config.enable_model_compilation:
        model = optimizer.compile_model(model)
        
    if config.enable_memory_optimization:
        optimizer.optimize_memory()
    
    # Phase 3 optimizations
    if MULTI_GPU_AVAILABLE and config.enable_multi_gpu:
        model, batch_size = optimizer.setup_multi_gpu(model, batch_size)
        
    if GRAPH_CACHE_AVAILABLE and config.enable_graph_caching:
        optimizer.setup_graph_caching()
        
    if config.enable_gradient_checkpointing:
        model = optimizer.enable_gradient_checkpointing(model)
    
    logger.info(f"All optimizations applied. Final batch size: {batch_size}")
    return model, batch_size, optimizer


# Legacy function for backward compatibility
def apply_generation_optimizations(model: nn.Module, 
                                 enable_fp16: bool = True,
                                 compile_model: bool = True,
                                 optimize_memory: bool = True) -> nn.Module:
    """
    Apply all generation optimizations to a model (Phase 2 compatibility).
    
    Args:
        model: Model to optimize
        enable_fp16: Enable mixed precision
        compile_model: Enable model compilation
        optimize_memory: Enable memory optimizations
        
    Returns:
        Optimized model
    """
    config = OptimizationConfig(
        enable_mixed_precision=enable_fp16,
        enable_model_compilation=compile_model,
        enable_memory_optimization=optimize_memory,
    )
    
    optimized_model, _, _ = apply_all_optimizations(model, 32, config)
    return optimized_model


def print_optimization_info():
    """Print information about available optimizations."""
    print("\n" + "="*60)
    print("MATTERGEN OPTIMIZATION FEATURES")
    print("="*60)
    print(f"Phase 2 (Basic): Available")
    print(f"  - Mixed Precision: {'✓' if torch.cuda.is_available() else '✗'}")
    print(f"  - Model Compilation: {'✓' if hasattr(torch, 'compile') else '✗'}")
    print(f"  - Memory Optimization: ✓")
    
    print(f"\nPhase 3 (Advanced): {'Available' if PHASE3_AVAILABLE else 'Partially Available'}")
    print(f"  - Multi-GPU Support: {'✓' if MULTI_GPU_AVAILABLE and torch.cuda.device_count() > 1 else ('Available but Single GPU' if MULTI_GPU_AVAILABLE else '✗')}")
    print(f"  - Graph Caching: {'✓' if GRAPH_CACHE_AVAILABLE else '✗'}")
    print(f"  - Memory Monitoring: {'✓' if PHASE3_AVAILABLE else '✗'}")
    print(f"  - Hardware Optimization: ✓")
    
    # Print hardware info
    if torch.cuda.is_available():
        if MULTI_GPU_AVAILABLE:
            try:
                print_multi_gpu_info()
            except Exception as e:
                print(f"\nMulti-GPU info not available: {e}")
        else:
            print(f"\nHardware Info:")
            print(f"  CUDA Available: True")
            print(f"  Number of GPUs: {torch.cuda.device_count()}")
    
    print("="*60)
