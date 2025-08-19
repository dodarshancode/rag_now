"""
Multi-Level Caching System for OpenSCENARIO 2.0 RAG System
Implements disk, memory, and Redis caching with encryption support.
"""

import logging
import json
import pickle
import hashlib
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, asdict
import threading
from concurrent.futures import ThreadPoolExecutor

import redis
import diskcache
from cryptography.fernet import Fernet

from .config_manager import ConfigManager

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    timestamp: float
    ttl: Optional[float] = None
    access_count: int = 0
    size_bytes: int = 0

class CacheManager:
    """Multi-level caching system with encryption support."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.enabled = config.get('cache.enable_disk_cache', True)
        
        # Cache configuration
        self.ttl_seconds = config.get('cache.ttl_seconds', 86400)  # 24 hours
        self.max_cache_size = self._parse_size(config.get('cache.max_cache_size', '10GB'))
        
        # Initialize cache layers
        self.memory_cache = {}
        self.memory_lock = threading.RLock()
        self.disk_cache = None
        self.redis_cache = None
        
        # Encryption
        self.encryption_key = self._get_or_create_encryption_key()
        self.cipher = Fernet(self.encryption_key) if self.encryption_key else None
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'memory_hits': 0,
            'disk_hits': 0,
            'redis_hits': 0,
            'evictions': 0
        }
        
        # Initialize cache layers
        self._initialize_caches()
        
        logger.info(f"CacheManager initialized with TTL={self.ttl_seconds}s")
    
    def _parse_size(self, size_str: str) -> int:
        """Parse size string to bytes."""
        size_str = size_str.upper()
        multipliers = {'B': 1, 'KB': 1024, 'MB': 1024**2, 'GB': 1024**3, 'TB': 1024**4}
        
        for suffix, multiplier in multipliers.items():
            if size_str.endswith(suffix):
                return int(float(size_str[:-len(suffix)]) * multiplier)
        
        return int(size_str)  # Assume bytes if no suffix
    
    def _get_or_create_encryption_key(self) -> Optional[bytes]:
        """Get or create encryption key for cache data."""
        key_file = Path(self.config.get('cache.disk_cache_dir', 'data/cache')) / '.cache_key'
        
        try:
            if key_file.exists():
                with open(key_file, 'rb') as f:
                    return f.read()
            else:
                # Generate new key
                key = Fernet.generate_key()
                key_file.parent.mkdir(parents=True, exist_ok=True)
                with open(key_file, 'wb') as f:
                    f.write(key)
                key_file.chmod(0o600)  # Restrict permissions
                logger.info("Generated new cache encryption key")
                return key
        except Exception as e:
            logger.warning(f"Failed to setup cache encryption: {e}")
            return None
    
    def _initialize_caches(self):
        """Initialize all cache layers."""
        if not self.enabled:
            logger.info("Caching disabled")
            return
        
        # Initialize disk cache
        try:
            cache_dir = Path(self.config.get('cache.disk_cache_dir', 'data/cache'))
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            self.disk_cache = diskcache.Cache(
                str(cache_dir),
                size_limit=self.max_cache_size,
                eviction_policy='least-recently-used'
            )
            logger.info(f"Disk cache initialized at {cache_dir}")
        except Exception as e:
            logger.warning(f"Failed to initialize disk cache: {e}")
        
        # Initialize Redis cache
        try:
            redis_config = self.config.get('cache', {})
            if redis_config.get('type') == 'redis':
                self.redis_cache = redis.Redis(
                    host=redis_config.get('host', 'localhost'),
                    port=redis_config.get('port', 6379),
                    db=redis_config.get('db', 0),
                    decode_responses=False  # Keep binary for encryption
                )
                # Test connection
                self.redis_cache.ping()
                logger.info("Redis cache initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Redis cache: {e}")
            self.redis_cache = None
    
    def _encrypt_data(self, data: bytes) -> bytes:
        """Encrypt data if encryption is enabled."""
        if self.cipher:
            return self.cipher.encrypt(data)
        return data
    
    def _decrypt_data(self, data: bytes) -> bytes:
        """Decrypt data if encryption is enabled."""
        if self.cipher:
            return self.cipher.decrypt(data)
        return data
    
    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for storage."""
        try:
            return pickle.dumps(value)
        except Exception as e:
            logger.error(f"Failed to serialize value: {e}")
            return b''
    
    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize value from storage."""
        try:
            return pickle.loads(data)
        except Exception as e:
            logger.error(f"Failed to deserialize value: {e}")
            return None
    
    def _generate_cache_key(self, key: str, namespace: str = 'default') -> str:
        """Generate cache key with namespace."""
        full_key = f"{namespace}:{key}"
        # Hash long keys
        if len(full_key) > 200:
            full_key = hashlib.sha256(full_key.encode()).hexdigest()
        return full_key
    
    def get(self, key: str, namespace: str = 'default') -> Optional[Any]:
        """Get value from cache with multi-level lookup.
        
        Args:
            key: Cache key
            namespace: Cache namespace
            
        Returns:
            Cached value or None if not found
        """
        if not self.enabled:
            return None
        
        cache_key = self._generate_cache_key(key, namespace)
        current_time = time.time()
        
        # Check memory cache first
        with self.memory_lock:
            if cache_key in self.memory_cache:
                entry = self.memory_cache[cache_key]
                
                # Check TTL
                if entry.ttl and current_time > entry.timestamp + entry.ttl:
                    del self.memory_cache[cache_key]
                else:
                    entry.access_count += 1
                    self.stats['hits'] += 1
                    self.stats['memory_hits'] += 1
                    logger.debug(f"Memory cache hit: {key}")
                    return entry.value
        
        # Check Redis cache
        if self.redis_cache:
            try:
                encrypted_data = self.redis_cache.get(cache_key)
                if encrypted_data:
                    data = self._decrypt_data(encrypted_data)
                    value = self._deserialize_value(data)
                    if value is not None:
                        # Promote to memory cache
                        self._set_memory_cache(cache_key, value, current_time)
                        self.stats['hits'] += 1
                        self.stats['redis_hits'] += 1
                        logger.debug(f"Redis cache hit: {key}")
                        return value
            except Exception as e:
                logger.warning(f"Redis cache get failed: {e}")
        
        # Check disk cache
        if self.disk_cache:
            try:
                encrypted_data = self.disk_cache.get(cache_key)
                if encrypted_data:
                    data = self._decrypt_data(encrypted_data)
                    value = self._deserialize_value(data)
                    if value is not None:
                        # Promote to higher cache levels
                        self._set_memory_cache(cache_key, value, current_time)
                        if self.redis_cache:
                            self._set_redis_cache(cache_key, encrypted_data)
                        
                        self.stats['hits'] += 1
                        self.stats['disk_hits'] += 1
                        logger.debug(f"Disk cache hit: {key}")
                        return value
            except Exception as e:
                logger.warning(f"Disk cache get failed: {e}")
        
        self.stats['misses'] += 1
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None, 
            namespace: str = 'default'):
        """Set value in cache across all levels.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (uses default if None)
            namespace: Cache namespace
        """
        if not self.enabled:
            return
        
        cache_key = self._generate_cache_key(key, namespace)
        current_time = time.time()
        ttl = ttl or self.ttl_seconds
        
        # Serialize and encrypt data
        serialized_data = self._serialize_value(value)
        if not serialized_data:
            return
        
        encrypted_data = self._encrypt_data(serialized_data)
        
        # Set in memory cache
        self._set_memory_cache(cache_key, value, current_time, ttl)
        
        # Set in Redis cache
        if self.redis_cache:
            self._set_redis_cache(cache_key, encrypted_data, ttl)
        
        # Set in disk cache
        if self.disk_cache:
            self._set_disk_cache(cache_key, encrypted_data)
        
        logger.debug(f"Cached value: {key}")
    
    def _set_memory_cache(self, cache_key: str, value: Any, timestamp: float, 
                         ttl: Optional[float] = None):
        """Set value in memory cache."""
        with self.memory_lock:
            # Estimate size
            try:
                size_bytes = len(pickle.dumps(value))
            except:
                size_bytes = 0
            
            entry = CacheEntry(
                key=cache_key,
                value=value,
                timestamp=timestamp,
                ttl=ttl,
                size_bytes=size_bytes
            )
            
            self.memory_cache[cache_key] = entry
            
            # Simple memory management - remove oldest entries if too many
            if len(self.memory_cache) > 10000:  # Max 10k entries in memory
                oldest_key = min(self.memory_cache.keys(), 
                               key=lambda k: self.memory_cache[k].timestamp)
                del self.memory_cache[oldest_key]
                self.stats['evictions'] += 1
    
    def _set_redis_cache(self, cache_key: str, encrypted_data: bytes, 
                        ttl: Optional[float] = None):
        """Set value in Redis cache."""
        try:
            if ttl:
                self.redis_cache.setex(cache_key, int(ttl), encrypted_data)
            else:
                self.redis_cache.set(cache_key, encrypted_data)
        except Exception as e:
            logger.warning(f"Redis cache set failed: {e}")
    
    def _set_disk_cache(self, cache_key: str, encrypted_data: bytes):
        """Set value in disk cache."""
        try:
            self.disk_cache[cache_key] = encrypted_data
        except Exception as e:
            logger.warning(f"Disk cache set failed: {e}")
    
    def delete(self, key: str, namespace: str = 'default'):
        """Delete key from all cache levels.
        
        Args:
            key: Cache key to delete
            namespace: Cache namespace
        """
        cache_key = self._generate_cache_key(key, namespace)
        
        # Delete from memory
        with self.memory_lock:
            self.memory_cache.pop(cache_key, None)
        
        # Delete from Redis
        if self.redis_cache:
            try:
                self.redis_cache.delete(cache_key)
            except Exception as e:
                logger.warning(f"Redis cache delete failed: {e}")
        
        # Delete from disk
        if self.disk_cache:
            try:
                del self.disk_cache[cache_key]
            except Exception as e:
                logger.warning(f"Disk cache delete failed: {e}")
    
    def clear(self, namespace: str = None):
        """Clear cache entries.
        
        Args:
            namespace: If specified, only clear entries in this namespace
        """
        if namespace:
            # Clear specific namespace
            pattern = f"{namespace}:"
            
            # Clear memory cache
            with self.memory_lock:
                keys_to_delete = [k for k in self.memory_cache.keys() if k.startswith(pattern)]
                for key in keys_to_delete:
                    del self.memory_cache[key]
            
            # Clear Redis cache
            if self.redis_cache:
                try:
                    keys = self.redis_cache.keys(f"{pattern}*")
                    if keys:
                        self.redis_cache.delete(*keys)
                except Exception as e:
                    logger.warning(f"Redis cache clear failed: {e}")
            
            # Clear disk cache (more complex, iterate through keys)
            if self.disk_cache:
                try:
                    keys_to_delete = [k for k in self.disk_cache if k.startswith(pattern)]
                    for key in keys_to_delete:
                        del self.disk_cache[key]
                except Exception as e:
                    logger.warning(f"Disk cache clear failed: {e}")
        else:
            # Clear all caches
            with self.memory_lock:
                self.memory_cache.clear()
            
            if self.redis_cache:
                try:
                    self.redis_cache.flushdb()
                except Exception as e:
                    logger.warning(f"Redis cache flush failed: {e}")
            
            if self.disk_cache:
                try:
                    self.disk_cache.clear()
                except Exception as e:
                    logger.warning(f"Disk cache clear failed: {e}")
        
        logger.info(f"Cache cleared: {namespace or 'all'}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0
        
        stats = {
            'enabled': self.enabled,
            'total_requests': total_requests,
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'hit_rate': hit_rate,
            'memory_hits': self.stats['memory_hits'],
            'disk_hits': self.stats['disk_hits'],
            'redis_hits': self.stats['redis_hits'],
            'evictions': self.stats['evictions'],
            'memory_entries': len(self.memory_cache),
            'encryption_enabled': self.cipher is not None
        }
        
        # Add disk cache stats
        if self.disk_cache:
            stats['disk_cache'] = {
                'size': len(self.disk_cache),
                'volume': self.disk_cache.volume(),
                'size_limit': self.max_cache_size
            }
        
        # Add Redis stats
        if self.redis_cache:
            try:
                redis_info = self.redis_cache.info('memory')
                stats['redis_cache'] = {
                    'connected': True,
                    'used_memory': redis_info.get('used_memory', 0),
                    'used_memory_human': redis_info.get('used_memory_human', '0B')
                }
            except:
                stats['redis_cache'] = {'connected': False}
        
        return stats
    
    def cleanup_expired(self):
        """Clean up expired entries from memory cache."""
        if not self.enabled:
            return
        
        current_time = time.time()
        expired_keys = []
        
        with self.memory_lock:
            for key, entry in self.memory_cache.items():
                if entry.ttl and current_time > entry.timestamp + entry.ttl:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.memory_cache[key]
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def get_cache_key_for_query(self, query: str, model_name: str, 
                               generation_params: Dict[str, Any]) -> str:
        """Generate cache key for query results.
        
        Args:
            query: Search query
            model_name: Model used for generation
            generation_params: Generation parameters
            
        Returns:
            Cache key string
        """
        # Create deterministic key from query and parameters
        key_data = {
            'query': query,
            'model': model_name,
            'params': sorted(generation_params.items())
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def get_cache_key_for_embeddings(self, text: str, model_name: str) -> str:
        """Generate cache key for embeddings.
        
        Args:
            text: Text to embed
            model_name: Embedding model name
            
        Returns:
            Cache key string
        """
        key_data = f"{model_name}:{text}"
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    def close(self):
        """Close cache connections."""
        if self.disk_cache:
            self.disk_cache.close()
        
        if self.redis_cache:
            self.redis_cache.close()
        
        logger.info("Cache connections closed")
