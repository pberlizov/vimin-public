import secrets
import hashlib
import json
import os
import logging
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)

class SecurityManager:
    """
    Handles API key generation, hashing, and validation for distributed orchestration.
    Supports a Master Key for bootstrapping and individual keys for agents/clients.
    """
    
    def __init__(self, config_path: str = str(Path.home() / ".vimin" / "security_config.json")):
        self.config_path = config_path
        self.master_key = os.environ.get("ORCHESTRATOR_MASTER_KEY")
        self.keys: Dict[str, Dict] = {}
        self._load_keys()
        
        if not self.master_key:
            logger.warning("ORCHESTRATOR_MASTER_KEY not set in environment. Bootstrap registration disabled.")

    def _load_keys(self):
        """Load registered keys from persistent storage"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    data = json.load(f)
                    self.keys = data.get("keys", {})
            except Exception as e:
                logger.error(f"Failed to load security config: {e}")

    def _save_keys(self):
        """Save registered keys to persistent storage"""
        try:
            p = Path(self.config_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.parent.chmod(0o700)
            with open(self.config_path, 'w') as f:
                json.dump({"keys": self.keys, "last_updated": datetime.utcnow().isoformat()}, f, indent=2)
            p.chmod(0o600)
        except Exception as e:
            logger.error(f"Failed to save security config: {e}")

    def generate_key(self, name: str, role: str = "agent") -> str:
        """Generate a new API key and return it (plain text, only shown once)"""
        key = secrets.token_urlsafe(32)
        key_hash = self._hash_key(key)
        
        self.keys[key_hash] = {
            "name": name,
            "role": role,
            "created_at": datetime.utcnow().isoformat(),
            "last_used": None
        }
        self._save_keys()
        return key

    def validate_key(self, key: str) -> Optional[Dict]:
        """Validate an API key and return its associated metadata if valid"""
        if not key:
            return None
            
        # Check Master Key
        if self.master_key and key == self.master_key:
            return {"name": "Master", "role": "admin", "is_master": True}
            
        key_hash = self._hash_key(key)
        if key_hash in self.keys:
            # Update last used
            self.keys[key_hash]["last_used"] = datetime.utcnow().isoformat()
            self._save_keys()
            return self.keys[key_hash]
            
        return None

    def revoke_key(self, key_hash: str) -> bool:
        """Revoke a key by its hash (retrievable via list_keys)"""
        if key_hash in self.keys:
            del self.keys[key_hash]
            self._save_keys()
            return True
        return False

    def list_keys(self) -> List[Dict]:
        """List all active keys (hashes and metadata)"""
        return [{"hash": h, **meta} for h, meta in self.keys.items()]

    def _hash_key(self, key: str) -> str:
        """Deterministic hash of the API key for safe storage"""
        return hashlib.sha256(key.encode()).hexdigest()
