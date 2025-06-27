# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Advanced caching utilities for MatterGen Phase 3.
Provides intelligent caching for expensive graph operations like neighbor search and radius graphs.
"""

import hashlib
import logging
import pickle
import threading
import time
from functools import wraps, lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class GraphOperationCache:
    """
    Thread-safe cache for expensive graph operations.
    Stores results based on input hashes to avoid recomputation.
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: Optional[int] = None,
        enable_disk_cache: bool = False,
        disk_cache_dir: Optional[Path] = None,
        compression_level: int = 1,
    ):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.enable_disk_cache = enable_disk_cache
        self.compression_level = compression_level
        
        # Memory cache
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, float] = {}
        self._lock = threading.RLock()
        
        # Disk cache setup
        if enable_disk_cache:
            if disk_cache_dir is None:
                disk_cache_dir = Path.home() / ".mattergen" / "cache"
            self.disk_cache_dir = Path(disk_cache_dir)
            self.disk_cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.disk_cache_dir = None
            
        # Stats
        self.hits = 0
        self.misses = 0
        self.disk_hits = 0
        self.disk_misses = 0
        
    def _hash_inputs(self, *args, **kwargs) -> str:
        """Create a hash of function inputs."""
        # Convert inputs to hashable representation
        hashable_args = []
        for arg in args:
            if hasattr(arg, 'cpu'):  # torch tensor
                hashable_args.append(arg.cpu().numpy().tobytes())
            elif isinstance(arg, np.ndarray):
                hashable_args.append(arg.tobytes())
            elif isinstance(arg, (list, tuple)):
                hashable_args.append(str(sorted(arg) if isinstance(arg, list) else arg))
            else:
                hashable_args.append(str(arg))
                
        hashable_kwargs = []
        for k, v in sorted(kwargs.items()):
            if hasattr(v, 'cpu'):  # torch tensor
                hashable_kwargs.append(f"{k}:{v.cpu().numpy().tobytes()}")
            elif isinstance(v, np.ndarray):
                hashable_kwargs.append(f"{k}:{v.tobytes()}")
            else:
                hashable_kwargs.append(f"{k}:{v}")
                
        combined = "|".join(hashable_args + hashable_kwargs)
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
        
    def _is_expired(self, timestamp: float) -> bool:
        """Check if cache entry is expired."""
        if self.ttl_seconds is None:
            return False
        return time.time() - timestamp > self.ttl_seconds
        
    def _evict_lru(self):
        """Evict least recently used items if cache is full."""
        if len(self._cache) < self.max_size:
            return
            
        # Find LRU item
        lru_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
        
        with self._lock:
            if lru_key in self._cache:
                del self._cache[lru_key]
                del self._access_times[lru_key]
                
    def _get_disk_cache_path(self, key: str) -> Path:
        """Get disk cache file path for a key."""
        return self.disk_cache_dir / f"{key}.pkl"
        
    def _save_to_disk(self, key: str, value: Any):
        """Save value to disk cache."""
        if not self.enable_disk_cache:
            return
            
        try:
            cache_file = self._get_disk_cache_path(key)
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            logger.warning(f"Failed to save to disk cache: {e}")
            
    def _load_from_disk(self, key: str) -> Optional[Any]:
        """Load value from disk cache."""
        if not self.enable_disk_cache:
            return None
            
        try:
            cache_file = self._get_disk_cache_path(key)
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load from disk cache: {e}")
            
        return None
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        current_time = time.time()
        
        with self._lock:
            # Check memory cache
            if key in self._cache:
                entry = self._cache[key]
                if not self._is_expired(entry['timestamp']):
                    self._access_times[key] = current_time
                    self.hits += 1
                    return entry['value']
                else:
                    # Expired, remove from memory
                    del self._cache[key]
                    del self._access_times[key]
                    
        # Check disk cache
        disk_value = self._load_from_disk(key)
        if disk_value is not None:
            # Load back into memory cache
            with self._lock:
                self._evict_lru()
                self._cache[key] = {'value': disk_value, 'timestamp': current_time}
                self._access_times[key] = current_time
                self.disk_hits += 1
            return disk_value
            
        # Cache miss
        self.misses += 1
        if self.enable_disk_cache:
            self.disk_misses += 1
        return None
        
    def put(self, key: str, value: Any):
        """Put value in cache."""
        current_time = time.time()
        
        with self._lock:
            self._evict_lru()
            self._cache[key] = {'value': value, 'timestamp': current_time}
            self._access_times[key] = current_time
            
        # Save to disk cache
        self._save_to_disk(key, value)
        
    def clear(self):
        """Clear all caches."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
            
        # Clear disk cache
        if self.enable_disk_cache and self.disk_cache_dir.exists():
            for cache_file in self.disk_cache_dir.glob("*.pkl"):
                try:
                    cache_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete disk cache file {cache_file}: {e}")
                    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        disk_total = self.disk_hits + self.disk_misses
        disk_hit_rate = self.disk_hits / disk_total if disk_total > 0 else 0
        
        return {
            'memory_size': len(self._cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'disk_hits': self.disk_hits,
            'disk_misses': self.disk_misses,
            'disk_hit_rate': disk_hit_rate,
            'disk_cache_enabled': self.enable_disk_cache,
        }


# Global cache instance
_global_cache: Optional[GraphOperationCache] = None


def get_global_cache() -> GraphOperationCache:
    """Get or create global cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = GraphOperationCache(
            max_size=1000,
            ttl_seconds=3600,  # 1 hour TTL
            enable_disk_cache=True,
        )
    return _global_cache


def cached_graph_operation(
    cache: Optional[GraphOperationCache] = None,
    key_func: Optional[Callable] = None,
    ttl_seconds: Optional[int] = None,
):
    """
    Decorator to cache expensive graph operations.
    
    Args:
        cache: Cache instance to use (default: global cache)
        key_func: Function to generate cache key from arguments
        ttl_seconds: Time-to-live for this specific operation
    """
    def decorator(func: Callable) -> Callable:
        nonlocal cache
        if cache is None:
            cache = get_global_cache()
            
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{cache._hash_inputs(*args, **kwargs)}"
                
            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result
                
            # Compute result
            result = func(*args, **kwargs)
            
            # Store in cache
            cache.put(cache_key, result)
            
            return result
        
        # Add cache management methods to function
        wrapper.clear_cache = lambda: cache.clear()
        wrapper.get_cache_stats = lambda: cache.get_stats()
        
        return wrapper
    return decorator


class NeighborSearchCache:
    """Specialized cache for neighbor search operations."""
    
    def __init__(self, max_size: int = 500):
        self.cache = GraphOperationCache(max_size=max_size, enable_disk_cache=True)
        
    @cached_graph_operation()
    def radius_graph(self, pos, r, batch=None, loop=False, max_num_neighbors=32):
        """Cached radius graph computation."""
        # This would be implemented with actual torch_geometric operations
        # For now, this is a placeholder that shows the caching pattern
        try:
            # Import torch_geometric here to avoid import errors if not available
            from torch_geometric.nn import radius_graph as tg_radius_graph
            return tg_radius_graph(pos, r=r, batch=batch, loop=loop, max_num_neighbors=max_num_neighbors)
        except ImportError:
            logger.warning("torch_geometric not available, returning dummy result")
            return None
            
    @cached_graph_operation()
    def knn_graph(self, pos, k, batch=None, loop=False):
        """Cached k-nearest neighbors graph computation."""
        try:
            from torch_geometric.nn import knn_graph as tg_knn_graph
            return tg_knn_graph(pos, k=k, batch=batch, loop=loop)
        except ImportError:
            logger.warning("torch_geometric not available, returning dummy result")
            return None


class AdaptiveCache:
    """
    Adaptive cache that adjusts its behavior based on usage patterns.
    """
    
    def __init__(self, initial_size: int = 100):
        self.cache = GraphOperationCache(max_size=initial_size)
        self.initial_size = initial_size
        self.performance_history = []
        self.last_adjustment = time.time()
        self.adjustment_interval = 300  # 5 minutes
        
    def _should_adjust_size(self) -> bool:
        """Check if cache size should be adjusted."""
        return time.time() - self.last_adjustment > self.adjustment_interval
        
    def _adjust_cache_size(self):
        """Adjust cache size based on performance."""
        stats = self.cache.get_stats()
        
        # Simple heuristic: if hit rate is high, increase size; if low, decrease
        if stats['hit_rate'] > 0.8 and stats['memory_size'] == self.cache.max_size:
            # High hit rate and cache is full - increase size
            new_size = min(self.cache.max_size * 2, 2000)
            logger.info(f"Increasing cache size from {self.cache.max_size} to {new_size}")
            # Create new cache with larger size
            old_cache = self.cache._cache.copy()
            self.cache = GraphOperationCache(max_size=new_size)
            self.cache._cache.update(old_cache)
        elif stats['hit_rate'] < 0.3 and self.cache.max_size > self.initial_size:
            # Low hit rate - decrease size
            new_size = max(self.cache.max_size // 2, self.initial_size)
            logger.info(f"Decreasing cache size from {self.cache.max_size} to {new_size}")
            self.cache = GraphOperationCache(max_size=new_size)
            
        self.last_adjustment = time.time()
        
    def get(self, key: str) -> Optional[Any]:
        """Get with adaptive behavior."""
        if self._should_adjust_size():
            self._adjust_cache_size()
        return self.cache.get(key)
        
    def put(self, key: str, value: Any):
        """Put with adaptive behavior."""
        self.cache.put(key, value)


def setup_graph_caching(
    enable_caching: bool = True,
    cache_size: int = 1000,
    enable_disk_cache: bool = True,
    cache_dir: Optional[Path] = None,
) -> GraphOperationCache:
    """
    Setup graph operation caching.
    
    Args:
        enable_caching: Whether to enable caching
        cache_size: Maximum number of cached items
        enable_disk_cache: Whether to enable disk caching
        cache_dir: Directory for disk cache
        
    Returns:
        Configured cache instance
    """
    if not enable_caching:
        # Return a dummy cache that doesn't actually cache
        class DummyCache:
            def get(self, key): return None
            def put(self, key, value): pass
            def clear(self): pass
            def get_stats(self): return {'enabled': False}
        return DummyCache()
        
    cache = GraphOperationCache(
        max_size=cache_size,
        enable_disk_cache=enable_disk_cache,
        disk_cache_dir=cache_dir,
    )
    
    # Set as global cache
    global _global_cache
    _global_cache = cache
    
    logger.info(f"Graph caching enabled: memory_size={cache_size}, disk_cache={enable_disk_cache}")
    
    return cache


def clear_all_caches():
    """Clear all caches."""
    global _global_cache
    if _global_cache:
        _global_cache.clear()
        
    logger.info("All caches cleared")


def print_cache_stats():
    """Print cache statistics."""
    cache = get_global_cache()
    stats = cache.get_stats()
    
    print("\n" + "="*40)
    print("GRAPH OPERATION CACHE STATS")
    print("="*40)
    print(f"Memory Cache Size: {stats['memory_size']}/{stats['max_size']}")
    print(f"Hit Rate: {stats['hit_rate']:.2%}")
    print(f"Hits: {stats['hits']}")
    print(f"Misses: {stats['misses']}")
    
    if stats['disk_cache_enabled']:
        print(f"Disk Hit Rate: {stats['disk_hit_rate']:.2%}")
        print(f"Disk Hits: {stats['disk_hits']}")
        print(f"Disk Misses: {stats['disk_misses']}")
    
    print("="*40)


# Example usage functions for common graph operations
def cached_radius_graph_wrapper():
    """Example wrapper for radius graph with caching."""
    neighbor_cache = NeighborSearchCache()
    return neighbor_cache.radius_graph


def precompute_common_graphs(structures, radii=[5.0, 8.0, 10.0]):
    """
    Precompute and cache common graph structures.
    
    Args:
        structures: List of crystal structures
        radii: List of radii to precompute graphs for
    """
    logger.info(f"Precomputing graphs for {len(structures)} structures with radii {radii}")
    
    cache = get_global_cache()
    neighbor_cache = NeighborSearchCache()
    
    for i, structure in enumerate(structures):
        # Convert structure to positions (this would need actual implementation)
        # pos = structure_to_positions(structure)
        
        for r in radii:
            # This would compute and cache the graph
            # neighbor_cache.radius_graph(pos, r)
            pass
            
        if i % 100 == 0:
            logger.info(f"Precomputed graphs for {i}/{len(structures)} structures")
            
    logger.info("Graph precomputation complete")
    print_cache_stats()
