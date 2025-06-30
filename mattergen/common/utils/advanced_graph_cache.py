"""
MatterGen Phase 3: Advanced Graph Caching Module
Implements LRU cache with memory management and graph preprocessing
"""

import torch
import gc
import hashlib
import pickle
from collections import OrderedDict
from typing import Dict, Any, Optional, Tuple
import logging
from functools import lru_cache
import psutil
import os

logger = logging.getLogger(__name__)

class AdvancedGraphCache:
    """Advanced LRU cache with memory management for graph operations"""
    
    def __init__(self, 
                 max_memory_gb: float = 4.0,
                 max_entries: int = 10000,
                 enable_disk_cache: bool = True,
                 cache_dir: str = "/tmp/mattergen_cache"):
        self.max_memory_bytes = max_memory_gb * 1024**3
        self.max_entries = max_entries
        self.enable_disk_cache = enable_disk_cache
        self.cache_dir = cache_dir
        
        # Memory cache (LRU)
        self.memory_cache: OrderedDict = OrderedDict()
        self.cache_sizes: Dict[str, int] = {}
        self.current_memory = 0
        
        # Disk cache setup
        if enable_disk_cache:
            os.makedirs(cache_dir, exist_ok=True)
        
        # Performance counters
        self.hits = 0
        self.misses = 0
        self.disk_hits = 0
        self.evictions = 0
        
        logger.info(f"Advanced graph cache initialized: {max_memory_gb}GB memory, {max_entries} entries")
    
    def _get_cache_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        key_data = pickle.dumps((args, sorted(kwargs.items())))
        return hashlib.sha256(key_data).hexdigest()
    
    def _get_object_size(self, obj: Any) -> int:
        """Estimate memory size of object"""
        if isinstance(obj, torch.Tensor):
            return obj.element_size() * obj.numel()
        else:
            # Approximate size for other objects
            return len(pickle.dumps(obj))
    
    def _evict_lru(self):
        """Evict least recently used items to free memory"""
        while (self.current_memory > self.max_memory_bytes * 0.8 or 
               len(self.memory_cache) > self.max_entries):
            if not self.memory_cache:
                break
                
            key, value = self.memory_cache.popitem(last=False)
            self.current_memory -= self.cache_sizes.pop(key, 0)
            self.evictions += 1
            
            # Optionally save to disk
            if self.enable_disk_cache:
                self._save_to_disk(key, value)
    
    def _save_to_disk(self, key: str, value: Any):
        """Save item to disk cache"""
        try:
            disk_path = os.path.join(self.cache_dir, f"{key}.pkl")
            with open(disk_path, 'wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            logger.warning(f"Failed to save to disk cache: {e}")
    
    def _load_from_disk(self, key: str) -> Optional[Any]:
        """Load item from disk cache"""
        try:
            disk_path = os.path.join(self.cache_dir, f"{key}.pkl")
            if os.path.exists(disk_path):
                with open(disk_path, 'rb') as f:
                    value = pickle.load(f)
                self.disk_hits += 1
                return value
        except Exception as e:
            logger.warning(f"Failed to load from disk cache: {e}")
        return None
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        # Check memory cache first
        if key in self.memory_cache:
            # Move to end (most recently used)
            value = self.memory_cache.pop(key)
            self.memory_cache[key] = value
            self.hits += 1
            return value
        
        # Check disk cache
        if self.enable_disk_cache:
            value = self._load_from_disk(key)
            if value is not None:
                # Move back to memory cache
                self.put(key, value)
                return value
        
        self.misses += 1
        return None
    
    def put(self, key: str, value: Any):
        """Put item in cache"""
        # Calculate size
        size = self._get_object_size(value)
        
        # Remove existing entry if present
        if key in self.memory_cache:
            self.current_memory -= self.cache_sizes.pop(key, 0)
            del self.memory_cache[key]
        
        # Evict if necessary
        self._evict_lru()
        
        # Add new entry
        self.memory_cache[key] = value
        self.cache_sizes[key] = size
        self.current_memory += size
    
    def cached_call(self, func, *args, **kwargs):
        """Cached function call"""
        key = self._get_cache_key(func.__name__, *args, **kwargs)
        
        result = self.get(key)
        if result is not None:
            return result
        
        # Compute and cache result
        result = func(*args, **kwargs)
        self.put(key, result)
        return result
    
    def clear(self):
        """Clear all caches"""
        self.memory_cache.clear()
        self.cache_sizes.clear()
        self.current_memory = 0
        
        if self.enable_disk_cache:
            try:
                for filename in os.listdir(self.cache_dir):
                    if filename.endswith('.pkl'):
                        os.remove(os.path.join(self.cache_dir, filename))
            except Exception as e:
                logger.warning(f"Failed to clear disk cache: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        hit_rate = self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
        
        return {
            'memory_cache_entries': len(self.memory_cache),
            'memory_usage_mb': self.current_memory / (1024**2),
            'max_memory_mb': self.max_memory_bytes / (1024**2),
            'memory_usage_percent': (self.current_memory / self.max_memory_bytes) * 100,
            'hits': self.hits,
            'misses': self.misses,
            'disk_hits': self.disk_hits,
            'evictions': self.evictions,
            'hit_rate': hit_rate,
            'system_memory_percent': psutil.virtual_memory().percent
        }
    
    def __del__(self):
        """Cleanup when destroyed"""
        try:
            self.clear()
        except:
            pass

# Global cache instance
_global_cache = None

def get_global_cache() -> AdvancedGraphCache:
    """Get or create global cache instance"""
    global _global_cache
    if _global_cache is None:
        _global_cache = AdvancedGraphCache()
    return _global_cache

def cached_graph_operation(func):
    """Decorator for caching graph operations"""
    def wrapper(*args, **kwargs):
        cache = get_global_cache()
        return cache.cached_call(func, *args, **kwargs)
    return wrapper

# Precomputed neighbor list cache
class NeighborListCache:
    """Cache for precomputed neighbor lists"""
    
    def __init__(self):
        self.cache = {}
        self.access_count = {}
    
    @lru_cache(maxsize=1000)
    def get_neighbor_list(self, positions_hash: str, cell_hash: str, cutoff: float):
        """Get cached neighbor list"""
        # This would implement actual neighbor list computation
        # For now, return a placeholder
        return None
    
    def precompute_common_patterns(self, structures: list, cutoff: float):
        """Precompute neighbor lists for common structural patterns"""
        logger.info(f"Precomputing neighbor lists for {len(structures)} structures")
        
        for i, structure in enumerate(structures):
            try:
                # Generate hashes for caching
                pos_hash = hashlib.sha256(structure.get('positions', b'')).hexdigest()
                cell_hash = hashlib.sha256(structure.get('cell', b'')).hexdigest()
                
                # Precompute and cache
                self.get_neighbor_list(pos_hash, cell_hash, cutoff)
                
                if i % 100 == 0:
                    logger.info(f"Precomputed {i}/{len(structures)} neighbor lists")
                    
            except Exception as e:
                logger.warning(f"Failed to precompute neighbor list for structure {i}: {e}")
