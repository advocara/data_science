import os
import hashlib
from typing import Optional

class FileCache:
    def __init__(self, cache_dir: str = './cache'):
        """Initialize cache with specified directory"""
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def _hash_text(self, text: str) -> str:
        """Create deterministic hash of input text"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()[:8] # shorten for readability vs uniqueness
    
    def _get_cache_path(self, text: str, dataset_name: str) -> str:
        """Generate cache file path from text hash and description"""
        text_hash = self._hash_text(text)
        safe_desc = dataset_name.replace('/', '_').replace('\\', '_')
        return os.path.join(self.cache_dir, f"{text_hash}-{safe_desc}.json")
        
    def get(self, text: str, dataset_name: str) -> Optional[str]:
        """Retrieve cached result if it exists"""
        cache_path = self._get_cache_path(text, dataset_name)
        if os.path.exists(cache_path):
            with open(cache_path, 'r', encoding='utf-8') as f:
                return f.read()
        return None
        
    def put(self, text: str, to_cache: str, dataset_name: str) -> None:
        """Store result in cache"""
        cache_path = self._get_cache_path(text, dataset_name)
        with open(cache_path, 'w', encoding='utf-8') as f:
            f.write(to_cache)
