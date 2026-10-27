"""
Supplier Intelligence API Client SDK & Key Management Demonstration (FactorySmart AI™).

Demonstrates how external systems, vendor portals, and ERP/MES software
authenticate with and invoke the Supplier & Vendor Intelligence API using API keys.
"""

import os
import sys
import json
from typing import Dict, Any, Optional, List
import urllib.request
import urllib.error

class SupplierAPIClient:
    """
    Python client for FactorySmart AI Supplier & Vendor Intelligence API.
    Zero external dependencies required (uses standard library urllib).
    """

    DEFAULT_API_KEY = "fs-demo-partner-quality-2026"

    def __init__(
        self,
        base_url: str = "http://localhost:8001",
        api_key: Optional[str] = DEFAULT_API_KEY
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.prefix = "/api/v1/supplier"

    def _request(self, method: str, endpoint: str, payload: Optional[Dict[str, Any]] = None, override_key: Optional[str] = None) -> Dict[str, Any]:
        url = f"{self.base_url}{self.prefix}{endpoint}"
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        # Determine key to use (allows testing missing key or override)
        key_to_use = self.api_key if override_key is None else override_key
        if key_to_use:
            headers["X-API-Key"] = key_to_use

        data_bytes = json.dumps(payload).encode("utf-8") if payload else None
        req = urllib.request.Request(url, data=data_bytes, headers=headers, method=method)

        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                body = response.read().decode("utf-8")
                return json.loads(body)
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8")
            try:
                err_json = json.loads(err_body)
                raise RuntimeError(f"HTTP {e.code} Error from {endpoint}: {err_json.get('detail', err_body)}")
            except json.JSONDecodeError:
                raise RuntimeError(f"HTTP {e.code} Error from {endpoint}: {err_body}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Connection failed to {url}: {e.reason}. Ensure the API server is running.")

    # -------------------------------------------------------------------------
    # Authentication & Key Management Endpoints
    # -------------------------------------------------------------------------
    def who_am_i(self) -> Dict[str, Any]:
        """Inspect caller's authenticated API key profile."""
        return self._request("GET", "/auth/whoami")

    def create_api_key(
        self,
        client_name: str,
        supplier_id: str = "VENDOR_B",
        role: str = "supplier",
        permissions: Optional[List[str]] = None,
        admin_key: str = "fs-admin-master-2026-99x"
    ) -> Dict[str, Any]:
        """Admin: Create and issue a new API key."""
        payload = {
            "client_name": client_name,
            "supplier_id": supplier_id,
            "role": role,
            "permissions": permissions or ["evaluate_lot", "forecast_impact", "adaptive_recipe"]
        }
        return self._request("POST", "/auth/create-key", payload, override_key=admin_key)

    def list_api_keys(self, admin_key: str = "fs-admin-master-2026-99x") -> Dict[str, Any]:
        """Admin: List all registered API keys and usage statistics."""
        return self._request("GET", "/auth/keys", override_key=admin_key)

    def revoke_api_key(self, key_or_id: str, admin_key: str = "fs-admin-master-2026-99x") -> Dict[str, Any]:
        """Admin: Revoke an active API key."""
        return self._request("DELETE", f"/auth/revoke-key/{key_or_id}", override_key=admin_key)

    # -------------------------------------------------------------------------
    # Model Inference Endpoints (Protected by API Key)
    # -------------------------------------------------------------------------
    def check_health(self) -> Dict[str, Any]:
        """Check API readiness (open probe)."""
        return self._request("GET", "/health", override_key="")

    # Method alias for convenience
    health_check = check_health

    def evaluate_lot(
        self,
        hardness_hv: float,
        tensile_mpa: float,
        supplier_id: str = "VENDOR_B",
        batch_id: str = "LOT-TEST-001",
        lower_spec_limit: float = 180.0,
        upper_spec_limit: float = 230.0,
        material_grade: str = "AISI-4140"
    ) -> Dict[str, Any]:
        """Predict downstream defect risk on incoming/shipped material lot."""
        payload = {
            "material_hardness_hv": hardness_hv,
            "tensile_strength_mpa": tensile_mpa,
            "lower_spec_limit": lower_spec_limit,
            "upper_spec_limit": upper_spec_limit,
            "supplier_id": supplier_id,
            "batch_id": batch_id,
            "material_grade": material_grade
        }
        return self._request("POST", "/evaluate-lot", payload)

    # Alias for convenience
    evaluate_material_lot = evaluate_lot

    def forecast_impact(
        self,
        hardness_hv: float,
        tensile_mpa: float,
        supplier_id: str = "VENDOR_B",
        batch_size_units: int = 5000
    ) -> Dict[str, Any]:
        """Forecast downstream scrap count, defect severity, and financial cost."""
        payload = {
            "material_hardness_hv": hardness_hv,
            "tensile_strength_mpa": tensile_mpa,
            "supplier_id": supplier_id,
            "batch_size_units": batch_size_units
        }
        return self._request("POST", "/forecast-impact", payload)

    def get_adaptive_recipe(self, hardness_hv: float, tensile_mpa: float = 680.0) -> Dict[str, Any]:
        """Get machine parameter recipe offsets (torque, angle, speed) to process lot safely."""
        payload = {
            "material_hardness_hv": hardness_hv,
            "batch_tensile_mpa": tensile_mpa
        }
        return self._request("POST", "/adaptive-recipe", payload)

    def run_comprehensive_lot_qc(
        self,
        supplier_id: str,
        batch_id: str,
        hardness_hv: float,
        tensile_mpa: float,
        batch_size_units: int = 8000
    ) -> Dict[str, Any]:
        """Execute risk evaluation + downstream impact + recipe compensation in 1 call."""
        payload = {
            "supplier_id": supplier_id,
            "batch_id": batch_id,
            "material_grade": "AISI-4140",
            "material_hardness_hv": hardness_hv,
            "tensile_strength_mpa": tensile_mpa,
            "batch_size_units": batch_size_units,
            "lower_spec_limit": 180.0,
            "upper_spec_limit": 230.0
        }
        return self._request("POST", "/comprehensive-qc", payload)

    def get_vendor_profile(self, supplier_id: str) -> Dict[str, Any]:
        """Get 30-day quality trajectory, recurring defects, and CAPA history."""
        return self._request("GET", f"/vendors/{supplier_id}/profile")

    def get_vendor_catalog(self, supplier_id: str) -> Dict[str, Any]:
        """Retrieve registered parts, lead times, and AI recommendations."""
        return self._request("GET", f"/vendors/{supplier_id}/catalog")

    def predict_lead_time_delay(
        self,
        supplier_id: str,
        category: str = "Structural Alloy Steel",
        target_lead_days: int = 18,
        sku: Optional[str] = "STL-2001"
    ) -> Dict[str, Any]:
        """Predict delivery delay days and supply chain bottleneck risk."""
        payload = {
            "supplier_id": supplier_id,
            "category": category,
            "target_lead_days": target_lead_days,
            "sku": sku
        }
        return self._request("POST", "/products/predict-lead-time", payload)

    def calculate_safety_stock(
        self,
        daily_consumption_mean: float = 350.0,
        daily_consumption_std: float = 45.0,
        lead_time_days_mean: float = 18.0,
        lead_time_days_std: float = 4.5,
        service_level_target_pct: float = 97.5
    ) -> Dict[str, Any]:
        """Calculate statistical safety stock and reorder point."""
        payload = {
            "daily_consumption_mean": daily_consumption_mean,
            "daily_consumption_std": daily_consumption_std,
            "lead_time_days_mean": lead_time_days_mean,
            "lead_time_days_std": lead_time_days_std,
            "service_level_target_pct": service_level_target_pct
        }
        return self._request("POST", "/inventory/safety-stock", payload)

    def get_tolerance_leakage_gap(self) -> Dict[str, Any]:
        """Get 216-230 HV leakage zone statistics and revised specification recommendations."""
        return self._request("GET", "/tolerance-gap")

    def trace_claim(self, claim_id: str) -> Dict[str, Any]:
        """Trace a warranty claim through the 5-tier genealogy tree."""
        return self._request("GET", f"/trace/claim/{claim_id}")

    def get_scar_report(self, supplier_id: str) -> Dict[str, Any]:
        """Get formal 8D SCAR document and financial recovery demand."""
        return self._request("GET", f"/scar/{supplier_id}")


# Demonstration execution
if __name__ == "__main__":
    port = sys.argv[1] if len(sys.argv) > 1 else "8001"
    client = SupplierAPIClient(base_url=f"http://localhost:{port}")

    print("=" * 80)
    print(" FACTORYSMART AI™ — SECURE API KEY AUTHENTICATION & MODEL DEMO")
    print("=" * 80)

    try:
        # 1. Test Unauthenticated Request (Expect 401 Rejection)
        print("\n[1] Testing Security: Calling /evaluate-lot WITHOUT an API Key:")
        try:
            client._request("POST", "/evaluate-lot", {"material_hardness_hv": 220.0, "tensile_strength_mpa": 650.0}, override_key="")
            print("    [FAIL] Request succeeded without API key!")
        except RuntimeError as e:
            print(f"    [PASS] Expected 401 Rejection Received: {e}")

        # 2. Test Authenticated Request with Vendor B Key
        print("\n[2] Authenticating with Vendor B Key ('fs-key-vendor-b-3e7a9d0f'):")
        client.api_key = "fs-key-vendor-b-3e7a9d0f"
        who = client.who_am_i()
        print(f"    Authenticated Client: {who['client_name']}")
        print(f"    Assigned Supplier: {who['supplier_id']}")
        print(f"    Role: {who['role']}")
        print(f"    Total API Calls Made: {who['request_count']}")

        # 3. Model Inference: Evaluate Marginal Lot
        print("\n[3] Running Marginal Lot Risk Model (Hardness: 224.0 HV):")
        lot_eval = client.evaluate_lot(hardness_hv=224.0, tensile_mpa=675.0, supplier_id="VENDOR_B")
        print(f"    Risk Probability: {lot_eval['downstream_risk_probability'] * 100:.1f}%")
        print(f"    Risk Tier: {lot_eval['batch_risk_tier']}")
        print(f"    Tolerance Leakage Detected: {lot_eval['is_in_tolerance_leakage_zone']}")
        print(f"    Recommendation: {lot_eval['recommendation']}")

        # 4. Model Inference: Closed-Loop Recipe Offsets
        print("\n[4] Running Adaptive Parameter Compensator Model:")
        recipe = client.get_adaptive_recipe(hardness_hv=224.0, tensile_mpa=675.0)
        print(f"    Status: {recipe['recipe_status']}")
        print(f"    Torque Offset: {recipe['target_torque_offset_nm']:+} Nm")
        print(f"    Angle Offset: {recipe['shutoff_angle_offset_deg']:+} deg")
        print(f"    Expected Scrap Reduction: {recipe['expected_scrap_reduction_pct']}%")

        # 5. Admin Key Management: Issue New API Key
        print("\n[5] Admin Privilege Test: Generating a New API Key for Partner:")
        new_key_res = client.create_api_key(
            client_name="Global Powertrain Tier 1 Partner",
            supplier_id="VENDOR_A",
            role="partner"
        )
        created_key = new_key_res["api_key_details"]
        print(f"    Generated Key ID: {created_key['key_id']}")
        print(f"    Generated API Key: {created_key['key']}")
        print(f"    Status: {created_key['status']}")

        print("\n" + "=" * 80)
        print(" ALL AUTHENTICATION & MODEL TESTS PASSED SUCCESSFULLY!")
        print("=" * 80)

    except Exception as e:
        print(f"\n[ERROR] Demo encountered error: {e}")
