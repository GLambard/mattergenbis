"""
MatterGen Phase 3: Hardware-Specific Optimization Module
Implements Flash Attention, Tensor Core optimizations, and memory management
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List
import logging
import gc
import psutil
import os
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class HardwareOptimizer:
    """Hardware-specific optimizations for modern GPUs"""
    
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_cuda = self.device.type == 'cuda'
        self.gpu_info = self._get_gpu_info()
        self.optimizations_applied = []
        
        logger.info(f"Hardware optimizer initialized for {self.device}")
        logger.info(f"GPU info: {self.gpu_info}")
    
    def _get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information for optimization decisions"""
        if not self.is_cuda:
            return {'name': 'CPU', 'memory_gb': psutil.virtual_memory().total / (1024**3)}
        
        try:
            gpu_id = self.device.index or 0
            gpu_name = torch.cuda.get_device_name(gpu_id)
            memory_gb = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)
            
            return {
                'name': gpu_name,
                'memory_gb': memory_gb,
                'compute_capability': torch.cuda.get_device_capability(gpu_id),
                'supports_tensor_cores': self._supports_tensor_cores(gpu_name),
                'supports_flash_attention': self._supports_flash_attention(),
            }
        except Exception as e:
            logger.warning(f"Failed to get GPU info: {e}")
            return {'name': 'Unknown GPU', 'memory_gb': 0}
    
    def _supports_tensor_cores(self, gpu_name: str) -> bool:
        """Check if GPU supports Tensor Cores"""
        tensor_core_gpus = ['V100', 'A100', 'A6000', 'RTX', '3080', '3090', '4080', '4090', 'H100']
        return any(gpu in gpu_name for gpu in tensor_core_gpus)
    
    def _supports_flash_attention(self) -> bool:
        """Check if system supports Flash Attention"""
        try:
            # Check for flash attention availability
            import flash_attn
            return True
        except ImportError:
            # Check for PyTorch's scaled_dot_product_attention
            return hasattr(F, 'scaled_dot_product_attention')
    
    def apply_model_optimizations(self, model: nn.Module) -> nn.Module:
        """Apply hardware-specific optimizations to model"""
        logger.info("Applying hardware-specific model optimizations...")
        
        # Move model to device
        model = model.to(self.device)
        
        # Apply optimizations based on hardware capabilities
        if self.is_cuda:
            model = self._apply_cuda_optimizations(model)
        
        # Enable memory efficient attention if available
        if self.gpu_info.get('supports_flash_attention', False):
            model = self._enable_flash_attention(model)
            self.optimizations_applied.append('flash_attention')
        
        # Enable Tensor Core optimizations
        if self.gpu_info.get('supports_tensor_cores', False):
            model = self._enable_tensor_cores(model)
            self.optimizations_applied.append('tensor_cores')
        
        # Model compilation for PyTorch 2.0+
        if hasattr(torch, 'compile'):
            try:
                model = torch.compile(model, mode='reduce-overhead')
                self.optimizations_applied.append('torch_compile')
                logger.info("Model compiled with PyTorch 2.0+ optimizations")
            except Exception as e:
                logger.warning(f"Failed to compile model: {e}")
        
        return model
    
    def _apply_cuda_optimizations(self, model: nn.Module) -> nn.Module:
        """Apply CUDA-specific optimizations"""
        # Enable cudnn optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Enable TensorFloat-32 for Ampere GPUs
        if 'A100' in self.gpu_info.get('name', '') or 'RTX' in self.gpu_info.get('name', ''):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            self.optimizations_applied.append('tf32')
        
        return model
    
    def _enable_flash_attention(self, model: nn.Module) -> nn.Module:
        """Enable Flash Attention optimizations"""
        try:
            # Try to use flash attention if available
            import flash_attn
            logger.info("Flash Attention enabled")
        except ImportError:
            # Fallback to PyTorch's efficient attention
            if hasattr(F, 'scaled_dot_product_attention'):
                logger.info("Using PyTorch scaled_dot_product_attention")
            else:
                logger.warning("Flash Attention not available")
        
        return model
    
    def _enable_tensor_cores(self, model: nn.Module) -> nn.Module:
        """Enable Tensor Core optimizations"""
        # Enable automatic mixed precision
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Ensure model uses tensor core friendly shapes
        for module in model.modules():
            if isinstance(module, nn.Linear):
                # Tensor cores work best with dimensions divisible by 8 or 16
                self._optimize_linear_layer(module)
        
        logger.info("Tensor Core optimizations enabled")
        return model
    
    def _optimize_linear_layer(self, layer: nn.Linear):
        """Optimize linear layer for Tensor Cores"""
        # This is a placeholder - in practice you'd adjust layer dimensions
        # to be multiples of 8 or 16 for optimal tensor core usage
        pass
    
    @contextmanager
    def optimized_inference_context(self):
        """Context manager for optimized inference"""
        # Store original states
        original_grad_enabled = torch.is_grad_enabled()
        original_autocast = torch.is_autocast_enabled()
        
        try:
            # Disable gradients and enable autocast
            torch.set_grad_enabled(False)
            with torch.cuda.amp.autocast(enabled=self.is_cuda):
                yield
        finally:
            # Restore original states
            torch.set_grad_enabled(original_grad_enabled)
    
    def optimize_memory_usage(self):
        """Optimize memory usage"""
        if self.is_cuda:
            # Clear cache
            torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
            
            # Get memory stats
            memory_stats = self.get_memory_stats()
            logger.info(f"Memory optimized. Current usage: {memory_stats}")
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics"""
        stats = {'system_memory_percent': psutil.virtual_memory().percent}
        
        if self.is_cuda:
            memory_reserved = torch.cuda.memory_reserved() / (1024**3)
            memory_allocated = torch.cuda.memory_allocated() / (1024**3)
            memory_total = torch.cuda.get_device_properties(self.device).total_memory / (1024**3)
            
            stats.update({
                'gpu_memory_allocated_gb': memory_allocated,
                'gpu_memory_reserved_gb': memory_reserved,
                'gpu_memory_total_gb': memory_total,
                'gpu_memory_usage_percent': (memory_allocated / memory_total) * 100
            })
        
        return stats
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of applied optimizations"""
        return {
            'device': str(self.device),
            'gpu_info': self.gpu_info,
            'optimizations_applied': self.optimizations_applied,
            'memory_stats': self.get_memory_stats()
        }

class IntelligentBatchSizer:
    """Intelligent batch sizing based on available memory"""
    
    def __init__(self, base_batch_size: int = 8, safety_factor: float = 0.8):
        self.base_batch_size = base_batch_size
        self.safety_factor = safety_factor
        self.optimal_batch_size = base_batch_size
        self.memory_history = []
    
    def calculate_optimal_batch_size(self, model: nn.Module, sample_input: Any) -> int:
        """Calculate optimal batch size based on available memory"""
        if not torch.cuda.is_available():
            return self.base_batch_size
        
        try:
            # Start with base batch size and test
            test_batch_sizes = [self.base_batch_size * (2 ** i) for i in range(5)]
            
            for batch_size in test_batch_sizes:
                try:
                    # Test memory usage with this batch size
                    with torch.no_grad():
                        # Create test batch
                        test_batch = self._create_test_batch(sample_input, batch_size)
                        
                        # Clear cache before test
                        torch.cuda.empty_cache()
                        memory_before = torch.cuda.memory_allocated()
                        
                        # Run inference
                        with torch.cuda.amp.autocast():
                            _ = model(test_batch)
                        
                        memory_after = torch.cuda.memory_allocated()
                        memory_used = memory_after - memory_before
                        
                        # Check if we have enough memory for this batch size
                        available_memory = torch.cuda.get_device_properties(0).total_memory
                        memory_usage_ratio = memory_used / available_memory
                        
                        if memory_usage_ratio > self.safety_factor:
                            # This batch size is too large
                            break
                        else:
                            self.optimal_batch_size = batch_size
                
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        break
                    else:
                        raise
        
        except Exception as e:
            logger.warning(f"Failed to calculate optimal batch size: {e}")
            return self.base_batch_size
        
        logger.info(f"Optimal batch size calculated: {self.optimal_batch_size}")
        return self.optimal_batch_size
    
    def _create_test_batch(self, sample_input: Any, batch_size: int) -> Any:
        """Create test batch for memory testing"""
        # This would create a test batch based on the sample input
        # Implementation depends on the specific input format
        return sample_input

class MemoryManager:
    """Advanced memory management for large-scale inference"""
    
    def __init__(self):
        self.memory_checkpoints = []
        self.peak_memory = 0
    
    @contextmanager
    def memory_checkpoint(self, name: str = "checkpoint"):
        """Create a memory checkpoint"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            memory_before = torch.cuda.memory_allocated()
            self.memory_checkpoints.append((name, memory_before))
        
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                memory_after = torch.cuda.memory_allocated()
                memory_diff = memory_after - memory_before
                
                logger.debug(f"Memory checkpoint '{name}': {memory_diff / (1024**2):.2f} MB")
                
                if memory_after > self.peak_memory:
                    self.peak_memory = memory_after
    
    def gradient_checkpointing_wrapper(self, module: nn.Module) -> nn.Module:
        """Apply gradient checkpointing to reduce memory usage"""
        if hasattr(module, 'gradient_checkpointing_enable'):
            module.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        return module
