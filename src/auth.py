"""
API Key Authentication & Management Engine for FactorySmart AI™.

Provides:
- Cryptographic API key generation
- JSON persistence in data/api_keys.json
- Multi-channel key extraction (X-API-Key header, Bearer token, or query param)
- Role-based access control (admin, supplier, partner)
- Key usage metrics (request counter, last used timestamp)
"""

import os
import json
import secrets
from datetime import datetime
from typing import Dict, Any, List, Optional
from fastapi import Security, HTTPException, status, Query
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
KEYS_FILE_PATH = os.path.join(PROJECT_ROOT, "data", "api_keys.json")

# Security Schemes for Swagger UI
api_key_header_scheme = APIKeyHeader(name="X-API-Key", auto_error=False, description="Primary API Key header")
http_bearer_scheme = HTTPBearer(auto_error=False, description="Bearer token header: 'Bearer <your_api_key>'")

DEFAULT_KEYS = [
    {
        "key_id": "key_admin_01",
        "key": "fs-admin-master-2026-99x",
        "client_name": "Enterprise Quality Administrator",
        "supplier_id": "ALL",
        "role": "admin",
        "permissions": ["all"],
        "status": "ACTIVE",
        "created_at": "2026-10-26T05:30:00Z",
        "last_used_at": None,
        "request_count": 0
    },
    {
        "key_id": "key_vendor_a",
        "key": "fs-key-vendor-a-8f4b2c1d",
        "client_name": "Vendor A Quality Team",
        "supplier_id": "VENDOR_A",
        "role": "supplier",
        "permissions": ["evaluate_lot", "forecast_impact", "adaptive_recipe", "catalog", "scar"],
        "status": "ACTIVE",
        "created_at": "2026-10-26T05:30:00Z",
        "last_used_at": None,
        "request_count": 0
    },
    {
        "key_id": "key_vendor_b",
        "key": "fs-key-vendor-b-3e7a9d0f",
        "client_name": "Vendor B Quality Team",
        "supplier_id": "VENDOR_B",
        "role": "supplier",
        "permissions": ["evaluate_lot", "forecast_impact", "adaptive_recipe", "catalog", "scar"],
        "status": "ACTIVE",
        "created_at": "2026-10-26T05:30:00Z",
        "last_used_at": None,
        "request_count": 0
    },
    {
        "key_id": "key_vendor_c",
        "key": "fs-key-vendor-c-5b1c8e2a",
        "client_name": "Vendor C Quality Team",
        "supplier_id": "VENDOR_C",
        "role": "supplier",
        "permissions": ["evaluate_lot", "forecast_impact", "adaptive_recipe", "catalog", "scar"],
        "status": "ACTIVE",
        "created_at": "2026-10-26T05:30:00Z",
        "last_used_at": None,
        "request_count": 0
    },
    {
        "key_id": "key_demo_partner",
        "key": "fs-demo-partner-quality-2026",
        "client_name": "Partner Quality Integration (Demo)",
        "supplier_id": "ALL",
        "role": "partner",
        "permissions": ["evaluate_lot", "forecast_impact", "adaptive_recipe", "catalog", "tolerance_gap"],
        "status": "ACTIVE",
        "created_at": "2026-10-26T05:30:00Z",
        "last_used_at": None,
        "request_count": 0
    }
]

class APIKeyManager:
    """Manages creation, validation, and storage of API keys."""

    def __init__(self, filepath: str = KEYS_FILE_PATH):
        self.filepath = filepath
        self._ensure_storage()

    def _ensure_storage(self):
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        if not os.path.exists(self.filepath):
            with open(self.filepath, "w") as f:
                json.dump({"keys": DEFAULT_KEYS}, f, indent=2)

    def _load_keys(self) -> List[Dict[str, Any]]:
        self._ensure_storage()
        try:
            with open(self.filepath, "r") as f:
                data = json.load(f)
                return data.get("keys", [])
        except Exception:
            return DEFAULT_KEYS

    def _save_keys(self, keys: List[Dict[str, Any]]):
        with open(self.filepath, "w") as f:
            json.dump({"keys": keys}, f, indent=2)

    def validate_key(self, raw_key: str) -> Optional[Dict[str, Any]]:
        """Validates key string, increments request count, and updates timestamp."""
        if not raw_key:
            return None

        keys = self._load_keys()
        for k in keys:
            if k["key"] == raw_key:
                if k.get("status") != "ACTIVE":
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"API Key '{raw_key[:10]}...' has been {k.get('status', 'REVOKED')}. Contact quality@factorysmart.ai."
                    )
                # Update metrics
                k["last_used_at"] = datetime.utcnow().isoformat() + "Z"
                k["request_count"] = k.get("request_count", 0) + 1
                self._save_keys(keys)
                return k
        return None

    def create_key(
        self,
        client_name: str,
        supplier_id: str = "ALL",
        role: str = "partner",
        permissions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generates and registers a new secure API key."""
        keys = self._load_keys()
        
        prefix = f"fs-key-{supplier_id.lower()[:8]}" if supplier_id != "ALL" else "fs-partner"
        token = secrets.token_hex(6)
        new_key_str = f"{prefix}-{token}"
        key_id = f"key_{secrets.token_hex(4)}"

        new_entry = {
            "key_id": key_id,
            "key": new_key_str,
            "client_name": client_name,
            "supplier_id": supplier_id.upper(),
            "role": role.lower(),
            "permissions": permissions or ["evaluate_lot", "forecast_impact", "adaptive_recipe"],
            "status": "ACTIVE",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "last_used_at": None,
            "request_count": 0
        }

        keys.append(new_entry)
        self._save_keys(keys)
        return new_entry

    def revoke_key(self, key_str: str) -> bool:
        """Revokes an API key by setting status to REVOKED."""
        keys = self._load_keys()
        for k in keys:
            if k["key"] == key_str or k.get("key_id") == key_str:
                k["status"] = "REVOKED"
                k["revoked_at"] = datetime.utcnow().isoformat() + "Z"
                self._save_keys(keys)
                return True
        return False

    def list_keys(self) -> List[Dict[str, Any]]:
        """Lists registered keys."""
        return self._load_keys()


# Singleton instance
key_manager = APIKeyManager()


# -----------------------------------------------------------------------------
# FASTAPI DEPENDENCY: REQUIRE VALID API KEY
# -----------------------------------------------------------------------------
def get_current_client(
    header_key: Optional[str] = Security(api_key_header_scheme),
    bearer_token: Optional[HTTPAuthorizationCredentials] = Security(http_bearer_scheme),
    query_key: Optional[str] = Query(None, alias="api_key", description="Optional query parameter for API Key")
) -> Dict[str, Any]:
    """
    Validates API key extracted from:
    1. Header: 'X-API-Key: <key>'
    2. Header: 'Authorization: Bearer <key>'
    3. Query parameter: '?api_key=<key>'
    """
    raw_key = header_key or (bearer_token.credentials if bearer_token else None) or query_key

    if not raw_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                "Authentication failed: API Key is required. "
                "Provide an API key via 'X-API-Key' header, 'Authorization: Bearer <key>' header, "
                "or '?api_key=<key>' query param. "
                "Active Test Keys: 'fs-demo-partner-quality-2026' (Demo Partner), "
                "'fs-key-vendor-b-3e7a9d0f' (Vendor B), or 'fs-admin-master-2026-99x' (Admin)."
            ),
            headers={"WWW-Authenticate": "ApiKey, Bearer"}
        )

    client_info = key_manager.validate_key(raw_key.strip())
    if not client_info:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Authentication failed: The provided API Key is invalid or expired. Key provided: '{raw_key[:12]}...'",
            headers={"WWW-Authenticate": "ApiKey, Bearer"}
        )

    return client_info


def require_admin(client: Dict[str, Any] = Security(get_current_client)) -> Dict[str, Any]:
    """Ensures the authenticated client has administrative privileges."""
    if client.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrative privileges required to access key management endpoints. Use admin key 'fs-admin-master-2026-99x'."
        )
    return client
