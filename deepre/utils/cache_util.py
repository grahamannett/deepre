import os
import hashlib
from typing import Any, Callable

DEFAULT_CACHE_DIR = ".cache"
DEFAULT_HASH_LEN = 10


def hash_str(*texts, hash_len: int = DEFAULT_HASH_LEN) -> str:
    """Generate a hash string from text.

    Args:
        text: Text to hash
        hash_len: Length of the hash to return

    Returns:
        A truncated MD5 hash string
    """
    text = "".join(texts)
    hash_value = hashlib.md5(text.encode()).hexdigest()
    return hash_value[:hash_len]


def create_url_hash(url: str, date: str = "", hash_len: int = DEFAULT_HASH_LEN) -> str:
    """Create a hash from a URL and optional date.

    Args:
        url: URL to hash
        date: Optional date string to include in the hash
        hash_len: Length of the hash to return

    Returns:
        A truncated MD5 hash string
    """
    return hash_str(url, date, hash_len=hash_len)


def ensure_cache_dir(cache_dir: str = DEFAULT_CACHE_DIR) -> None:
    """Ensure the cache directory exists.

    Args:
        cache_dir: Directory to store cache files
    """
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)


def get_cache_path(key: str, cache_dir: str = DEFAULT_CACHE_DIR) -> str:
    """Get the full path to a cache file.

    Args:
        key: Cache key
        cache_dir: Directory to store cache files

    Returns:
        Full path to the cache file
    """
    ensure_cache_dir(cache_dir)
    return f"{cache_dir}/{key}"


def get_from_cache(
    key: str,
    callback: Callable = lambda x: x,
    check_cache: bool = True,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> str | None:
    """Get a value from the cache.

    Args:
        key: Cache key
        callback: Function to process the cached value
        check_cache: Whether to check the cache
        cache_dir: Directory to store cache files

    Returns:
        The cached value, or None if not found
    """
    if not check_cache:
        return None

    cache_path = get_cache_path(key, cache_dir)
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return callback(f.read())
    return None


def save_to_cache(
    key: str,
    value: Any,
    callback: Callable = lambda x: x,
    save_cache: bool = True,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> bool:
    """Save a value to the cache.

    Args:
        key: Cache key
        value: Value to cache
        callback: Function to process the value before caching
        save_cache: Whether to save to the cache
        cache_dir: Directory to store cache files

    Returns:
        True if the value was cached, False otherwise
    """
    if not save_cache:
        return False

    ensure_cache_dir(cache_dir)
    with open(get_cache_path(key, cache_dir), "w") as f:
        f.write(callback(value))
    return True


class CacheManager:
    get_from_cache = staticmethod(get_from_cache)
    save_to_cache = staticmethod(save_to_cache)
    create_hash = staticmethod(hash_str)
