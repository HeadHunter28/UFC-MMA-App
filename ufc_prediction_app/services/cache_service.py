"""
Cache Service Module.

Provides caching functionality for LLM responses and other expensive operations.
"""

import hashlib
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CACHE_DIR, CACHE_TTL, CACHE_LLM_RESPONSES, ensure_directories

logger = logging.getLogger(__name__)


class CacheService:
    """
    Cache service for storing and retrieving cached data.

    Supports JSON-based file caching with TTL expiration.
    """

    def __init__(self, cache_dir: Optional[str] = None, ttl: Optional[int] = None):
        """
        Initialize the cache service.

        Args:
            cache_dir: Directory for cache files
            ttl: Time-to-live in seconds
        """
        self.cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR
        self.ttl = ttl if ttl is not None else CACHE_TTL
        self.enabled = CACHE_LLM_RESPONSES

        ensure_directories()
        self._ensure_cache_dir()

    def _ensure_cache_dir(self):
        """Ensure cache directory exists."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, cache_type: str) -> Path:
        """Get the path for a cache file."""
        return self.cache_dir / f"{cache_type}_cache.json"

    def _load_cache(self, cache_type: str) -> dict:
        """Load cache from file."""
        cache_path = self._get_cache_path(cache_type)
        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load cache: {e}")
        return {"entries": {}, "metadata": {"created_at": datetime.now().isoformat()}}

    def _save_cache(self, cache_type: str, cache_data: dict):
        """Save cache to file."""
        cache_path = self._get_cache_path(cache_type)
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2)
        except IOError as e:
            logger.error(f"Failed to save cache: {e}")

    def _generate_key(self, *args) -> str:
        """Generate a cache key from arguments."""
        key_string = "|".join(str(arg) for arg in args)
        return hashlib.md5(key_string.encode()).hexdigest()

    def get(
        self,
        key: str,
        cache_type: str = "llm"
    ) -> Optional[Any]:
        """
        Get a value from the cache.

        Args:
            key: Cache key
            cache_type: Type of cache (llm, scraper, etc.)

        Returns:
            Cached value or None if not found/expired
        """
        if not self.enabled:
            return None

        cache = self._load_cache(cache_type)
        entry = cache.get("entries", {}).get(key)

        if entry is None:
            return None

        # Check expiration
        cached_at = datetime.fromisoformat(entry.get("cached_at", "2000-01-01"))
        if datetime.now() - cached_at > timedelta(seconds=self.ttl):
            logger.debug(f"Cache expired for key: {key[:16]}...")
            return None

        logger.debug(f"Cache hit for key: {key[:16]}...")
        return entry.get("value")

    def set(
        self,
        key: str,
        value: Any,
        cache_type: str = "llm"
    ):
        """
        Set a value in the cache.

        Args:
            key: Cache key
            value: Value to cache
            cache_type: Type of cache
        """
        if not self.enabled:
            return

        cache = self._load_cache(cache_type)

        cache["entries"][key] = {
            "value": value,
            "cached_at": datetime.now().isoformat(),
        }

        cache["metadata"]["updated_at"] = datetime.now().isoformat()

        self._save_cache(cache_type, cache)
        logger.debug(f"Cached value for key: {key[:16]}...")

    def delete(self, key: str, cache_type: str = "llm"):
        """Delete a value from the cache."""
        cache = self._load_cache(cache_type)

        if key in cache.get("entries", {}):
            del cache["entries"][key]
            self._save_cache(cache_type, cache)
            logger.debug(f"Deleted cache key: {key[:16]}...")

    def clear(self, cache_type: Optional[str] = None):
        """
        Clear the cache.

        Args:
            cache_type: Specific cache type to clear, or None for all
        """
        if cache_type:
            cache_path = self._get_cache_path(cache_type)
            if cache_path.exists():
                cache_path.unlink()
                logger.info(f"Cleared cache: {cache_type}")
        else:
            # Clear all cache files
            for cache_file in self.cache_dir.glob("*_cache.json"):
                cache_file.unlink()
                logger.info(f"Cleared cache: {cache_file.name}")

    def get_stats(self, cache_type: str = "llm") -> dict:
        """
        Get cache statistics.

        Args:
            cache_type: Type of cache

        Returns:
            Dict with cache stats
        """
        cache = self._load_cache(cache_type)
        entries = cache.get("entries", {})

        # Count valid (non-expired) entries
        now = datetime.now()
        valid_count = 0
        expired_count = 0

        for key, entry in entries.items():
            cached_at = datetime.fromisoformat(entry.get("cached_at", "2000-01-01"))
            if now - cached_at <= timedelta(seconds=self.ttl):
                valid_count += 1
            else:
                expired_count += 1

        return {
            "cache_type": cache_type,
            "total_entries": len(entries),
            "valid_entries": valid_count,
            "expired_entries": expired_count,
            "ttl_seconds": self.ttl,
            "cache_enabled": self.enabled,
            "metadata": cache.get("metadata", {}),
        }

    def cleanup_expired(self, cache_type: str = "llm") -> int:
        """
        Remove expired entries from the cache.

        Args:
            cache_type: Type of cache to clean

        Returns:
            Number of entries removed
        """
        cache = self._load_cache(cache_type)
        entries = cache.get("entries", {})
        now = datetime.now()
        removed = 0

        keys_to_remove = []
        for key, entry in entries.items():
            cached_at = datetime.fromisoformat(entry.get("cached_at", "2000-01-01"))
            if now - cached_at > timedelta(seconds=self.ttl):
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del entries[key]
            removed += 1

        if removed > 0:
            self._save_cache(cache_type, cache)
            logger.info(f"Cleaned up {removed} expired cache entries")

        return removed


# Convenience functions for LLM caching
_llm_cache = None


def get_llm_cache() -> CacheService:
    """Get the singleton LLM cache instance."""
    global _llm_cache
    if _llm_cache is None:
        _llm_cache = CacheService()
    return _llm_cache


def cache_llm_response(prompt_key: str, response: str):
    """Cache an LLM response."""
    get_llm_cache().set(prompt_key, response, cache_type="llm")


def get_cached_llm_response(prompt_key: str) -> Optional[str]:
    """Get a cached LLM response."""
    return get_llm_cache().get(prompt_key, cache_type="llm")
