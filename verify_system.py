"""
FactorySmart AI™ - System Cross-Verification & Validation Suite.

Performs comprehensive verification across:
  [1] Virtual Environment & Dependencies (venv, requirements.txt, setup scripts)
  [2] Machine Learning Models & Serialized Pickle Files (models/*.pkl integrity & inference)
  [3] API Key Authentication & Governance (data/api_keys.json, roles, permissions, metrics)
  [4] FastAPI REST API Endpoints (Health, Tiers 1-3, Supplier Router, TestClient lifespan)
  [5] Client SDK Integration (SupplierAPIClient execution)

Run this script anytime to cross-verify the entire suite:
    python verify_system.py
"""

import os
import sys
import json
import pickle
import traceback
import pandas as pd
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class VerificationReport:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.details = []

    def log_pass(self, category: str, test_name: str, message: str = ""):
        self.passed += 1
        msg = f"[PASS] [{category}] {test_name}"
        if message:
            msg += f" -> {message}"
        print(f"\033[92m{msg}\033[0m")
        self.details.append({"status": "PASS", "category": category, "test": test_name, "message": message})

    def log_fail(self, category: str, test_name: str, error: str):
        self.failed += 1
        msg = f"[FAIL] [{category}] {test_name} -> {error}"
        print(f"\033[91m{msg}\033[0m")
        self.details.append({"status": "FAIL", "category": category, "test": test_name, "error": error})

    def log_warn(self, category: str, test_name: str, message: str):
        self.warnings += 1
        msg = f"[WARN] [{category}] {test_name} -> {message}"
        print(f"\033[93m{msg}\033[0m")
        self.details.append({"status": "WARN", "category": category, "test": test_name, "message": message})

    def summary(self):
        total = self.passed + self.failed
        print("\n" + "=" * 80)
        print("SYSTEM VERIFICATION & CROSS-VALIDATION SUMMARY")
        print("=" * 80)
        print(f"Total Tests Run   : {total}")
        print(f"Passed            : \033[92m{self.passed}\033[0m")
        print(f"Failed            : \033[91m{self.failed}\033[0m")
        print(f"Warnings          : \033[93m{self.warnings}\033[0m")
        print(f"Overall Status    : {'SUCCESS - ALL CHECKS PASSED' if self.failed == 0 else 'FAILURE - ACTION REQUIRED'}")
        print("=" * 80 + "\n")
        return self.failed == 0


def verify_virtual_environment(report: VerificationReport):
    print("\n>>> [1/5] VERIFYING VIRTUAL ENVIRONMENT & RUNTIME DEPENDENCIES")

    # 1.1 requirements.txt
    req_file = os.path.join(PROJECT_ROOT, "requirements.txt")
    if os.path.exists(req_file) and os.path.getsize(req_file) > 100:
        report.log_pass("VENV", "requirements.txt", f"Found ({os.path.getsize(req_file)} bytes)")
    else:
        report.log_fail("VENV", "requirements.txt", "Missing or empty requirements.txt")

    # 1.2 venv directory
    venv_dir = os.path.join(PROJECT_ROOT, "venv")
    venv_py = os.path.join(venv_dir, "Scripts", "python.exe")
    if os.path.exists(venv_dir) and os.path.exists(venv_py):
        report.log_pass("VENV", "Virtual Environment (venv/)", f"Found at {venv_py}")
    else:
        report.log_warn("VENV", "Virtual Environment (venv/)", "venv directory not detected at ./venv")

    # 1.3 Setup & Run scripts
    scripts = ["setup_venv.bat", "run_api.bat", "run_dashboard.bat", ".env.example", ".gitignore"]
    for sc in scripts:
        if os.path.exists(os.path.join(PROJECT_ROOT, sc)):
            report.log_pass("VENV", f"Support Script ({sc})", "Found and verified")
        else:
            report.log_fail("VENV", f"Support Script ({sc})", "File missing")

    # 1.4 Essential packages importable
    packages = [
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("pydantic", "Pydantic"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("sklearn", "Scikit-Learn"),
        ("lightgbm", "LightGBM"),
        ("shap", "SHAP"),
        ("streamlit", "Streamlit"),
        ("plotly", "Plotly")
    ]
    for pkg, name in packages:
        try:
            __import__(pkg)
            report.log_pass("VENV", f"Import {name}", f"Module '{pkg}' successfully loaded")
        except ImportError as e:
            report.log_fail("VENV", f"Import {name}", f"Failed to import: {e}")


def verify_pickle_models(report: VerificationReport):
    print("\n>>> [2/5] VERIFYING SERIALIZED PICKLE MODELS & INFERENCE")

    models_dir = os.path.join(PROJECT_ROOT, "models")
    expected_pickles = [
        "shift_defect_predictor.pkl",
        "failure_mode_classifier.pkl",
        "prescriptive_action_engine.pkl",
        "marginal_batch_classifier.pkl",
        "downstream_impact_forecaster.pkl",
        "adaptive_parameter_compensator.pkl"
    ]

    for pkl_name in expected_pickles:
        pkl_path = os.path.join(models_dir, pkl_name)
        if os.path.exists(pkl_path):
            size = os.path.getsize(pkl_path)
            report.log_pass("MODELS", f"Pickle File ({pkl_name})", f"Exists ({size:,} bytes)")
        else:
            report.log_fail("MODELS", f"Pickle File ({pkl_name})", "File not found in models/ directory")

    # Load via src.models loader
    try:
        from src.models import load_all_models
        models = load_all_models(models_dir=models_dir, data_dir=os.path.join(PROJECT_ROOT, "data"))
        report.log_pass("MODELS", "load_all_models()", f"Loaded {len(models)} model keys: {list(models.keys())}")
    except Exception as e:
        report.log_fail("MODELS", "load_all_models()", f"Exception during loading: {e}")
        return

    # Test ShiftDefectPredictor inference
    try:
        shift_pred = models.get("shift_defect_predictor")
        dummy_shift = pd.DataFrame([{
            "station_id": "ST-10", "spindle_temp_c_mean": 58.0, "vibration_rms_g_mean": 1.2,
            "tool_wear_index_max": 42.0, "peak_torque_nm_mean": 12.5, "total_cycles": 320.0,
            "torque_nok_rate": 0.04, "fastener_defect_rate": 0.03
        }])
        ranks = shift_pred.rank_station_risk(dummy_shift)
        report.log_pass("MODELS", "ShiftDefectPredictor Inference", f"Station {ranks[0]['station_id']} predicted: {ranks[0]['predicted_defect_count']} defects ({ranks[0]['risk_level']})")
    except Exception as e:
        report.log_fail("MODELS", "ShiftDefectPredictor Inference", f"Error: {e}")

    # Test FailureModeClassifier + SHAP
    try:
        fm_clf = models.get("failure_mode_classifier")
        dummy_cycle = pd.DataFrame([{
            "peak_torque_nm": 13.5, "shutoff_angle_deg": 270.0, "spindle_temp_c": 63.0,
            "system_pressure_psi": 114.0, "vibration_rms_g": 2.2, "tool_wear_index": 58.0,
            "hours_since_service": 20.0, "material_hardness_hv": 222.0, "tensile_strength_mpa": 650.0,
            "torque_energy_proxy": 13.5 * (270.0 * np.pi / 180.0), "tool_stress_factor": 58.0 * 2.2,
            "thermal_vibration_index": 63.0 * 2.2, "hydraulic_thermal_ratio": 114.0 / 63.0,
            "material_tool_resistance": 222.0 * 59.0, "torque_delta_nominal": 13.5 - 12.5,
            "angle_delta_nominal": 270.0 - 360.0, "torque_status_nok": 1.0,
            "torque_status_high": 1.0, "torque_status_low": 0.0,
            "st_ST-10": 1.0, "st_ST-20": 0.0, "st_ST-30": 0.0, "st_ST-40": 0.0
        }])
        fm_res = fm_clf.predict_with_contributors(dummy_cycle, top_k=3)[0]
        report.log_pass("MODELS", "FailureModeClassifier + SHAP", f"Predicted '{fm_res['predicted_failure_mode']}' (Conf: {fm_res['confidence']:.2%}, Top Driver: {fm_res['top_physical_contributors'][0]['feature']})")
    except Exception as e:
        report.log_fail("MODELS", "FailureModeClassifier + SHAP", f"Error: {e}")

    # Test MarginalBatchRiskClassifier
    try:
        batch_clf = models.get("marginal_batch_classifier")
        dummy_batch = pd.DataFrame([{
            "material_hardness_hv": 224.5, "tensile_strength_mpa": 670.0,
            "lower_spec_limit": 180.0, "upper_spec_limit": 230.0, "supplier_id": "VENDOR_B",
            "supplier_VENDOR_A": 0.0, "supplier_VENDOR_B": 1.0, "supplier_VENDOR_C": 0.0
        }])
        risk_res = batch_clf.predict_batch_risk(dummy_batch)
        report.log_pass("MODELS", "MarginalBatchRiskClassifier", f"Risk Tier: {risk_res['batch_risk_tier']}, Risk Prob: {risk_res['downstream_risk_probability']:.2%}")
    except Exception as e:
        report.log_fail("MODELS", "MarginalBatchRiskClassifier", f"Error: {e}")

    # Test DownstreamImpactForecaster
    try:
        forecaster = models.get("downstream_impact_forecaster")
        fc = forecaster.forecast(hardness=224.5, tensile=670.0, supplier_id="VENDOR_B")
        report.log_pass("MODELS", "DownstreamImpactForecaster", f"Forecasted Scrap: {fc['expected_scrap_count']} units, Severity: {fc['expected_defect_severity']}/5")
    except Exception as e:
        report.log_fail("MODELS", "DownstreamImpactForecaster", f"Error: {e}")

    # Test AdaptiveParameterCompensator
    try:
        compensator = models.get("adaptive_parameter_compensator")
        comp = compensator.compute_offsets(224.5, 670.0)
        report.log_pass("MODELS", "AdaptiveParameterCompensator", f"Torque Target: {comp['prescribed_torque_target_nm']} Nm (+{comp['torque_offset_nm']} Nm), Speed: {comp['spindle_speed_offset_pct']}%")
    except Exception as e:
        report.log_fail("MODELS", "AdaptiveParameterCompensator", f"Error: {e}")


def verify_api_keys_and_auth(report: VerificationReport):
    print("\n>>> [3/5] VERIFYING API KEY GOVERNANCE & AUTHENTICATION")

    keys_path = os.path.join(PROJECT_ROOT, "data", "api_keys.json")
    if not os.path.exists(keys_path):
        report.log_fail("AUTH", "api_keys.json Storage", "Missing data/api_keys.json")
        return

    report.log_pass("AUTH", "api_keys.json Storage", f"Found at {keys_path}")

    from src.auth import key_manager

    # Check default keys exist
    keys = key_manager.list_keys()
    key_map = {k["key"]: k for k in keys}

    required_keys = [
        ("fs-admin-master-2026-99x", "admin"),
        ("fs-key-vendor-b-3e7a9d0f", "supplier"),
        ("fs-key-vendor-a-8f4b2c1d", "supplier"),
        ("fs-demo-partner-quality-2026", "partner")
    ]

    for key_str, expected_role in required_keys:
        if key_str in key_map:
            k_info = key_map[key_str]
            report.log_pass("AUTH", f"Registered Key ({k_info['client_name']})", f"Role: {k_info['role']} | Status: {k_info['status']}")
        else:
            report.log_fail("AUTH", f"Registered Key ({key_str})", "Required test key not found in storage")

    # Validate authentic key
    valid_client = key_manager.validate_key("fs-demo-partner-quality-2026")
    if valid_client and valid_client["status"] == "ACTIVE":
        report.log_pass("AUTH", "validate_key() on Authentic Key", f"Authenticated as '{valid_client['client_name']}'")
    else:
        report.log_fail("AUTH", "validate_key() on Authentic Key", "Failed to validate authentic key")

    # Validate rejected key
    invalid_client = key_manager.validate_key("invalid-dummy-key-999")
    if invalid_client is None:
        report.log_pass("AUTH", "validate_key() on Invalid Key", "Properly rejected invalid key with None")
    else:
        report.log_fail("AUTH", "validate_key() on Invalid Key", "Failed to reject invalid key")

    # Dynamic key creation & revocation test
    try:
        new_key = key_manager.create_key(
            client_name="Automated Test Quality Bot",
            supplier_id="VENDOR_TEST",
            role="partner"
        )
        report.log_pass("AUTH", "create_key() Dynamic Registration", f"Created Key ID {new_key['key_id']} with token {new_key['key'][:15]}...")

        # Revoke test key
        revoked = key_manager.revoke_key(new_key["key"])
        if revoked:
            report.log_pass("AUTH", "revoke_key() Lifecycle", f"Successfully revoked test key {new_key['key_id']}")
        else:
            report.log_fail("AUTH", "revoke_key() Lifecycle", "Failed to revoke created key")
    except Exception as e:
        report.log_fail("AUTH", "Key Lifecycle Management", f"Error during create/revoke: {e}")


def verify_fastapi_endpoints(report: VerificationReport):
    print("\n>>> [4/5] VERIFYING FASTAPI APPLICATION & REST ENDPOINTS")

    from fastapi.testclient import TestClient
    from src.api import app

    try:
        with TestClient(app) as client:
            # 4.1 Health Check
            r = client.get("/health")
            if r.status_code == 200 and r.json().get("status") == "healthy":
                models_loaded = r.json().get("models_loaded", [])
                report.log_pass("API", "GET /health", f"Status: 200, Models Loaded: {len(models_loaded)}")
            else:
                report.log_fail("API", "GET /health", f"Status: {r.status_code}, Body: {r.text}")

            # 4.2 Overview Endpoint
            r = client.get("/api/overview")
            if r.status_code == 200:
                report.log_pass("API", "GET /api/overview", "Status: 200, Performance Summary Report Loaded")
            else:
                report.log_fail("API", "GET /api/overview", f"Status: {r.status_code}")

            # 4.3 Tier 1 Simulate Endpoint
            r = client.post("/api/tier1/simulate", json={
                "station_id": "ST-10", "spindle_temp_c": 62.0, "system_pressure_psi": 115.0,
                "vibration_rms_g": 2.1, "tool_wear_index": 55.0, "hours_since_service": 18.0,
                "peak_torque_nm": 13.2, "shutoff_angle_deg": 285.0, "torque_status": "NOK_HIGH",
                "material_hardness_hv": 220.0, "tensile_strength_mpa": 645.0
            })
            if r.status_code == 200:
                data = r.json()
                report.log_pass("API", "POST /api/tier1/simulate", f"Status: 200, Mode: {data['predicted_failure_mode']}, Action: {data['prescriptive_action']['action_code']}")
            else:
                report.log_fail("API", "POST /api/tier1/simulate", f"Status: {r.status_code}, Body: {r.text}")

            # 4.4 Tier 2 Evaluate Material Endpoint
            r = client.post("/api/tier2/evaluate_material", json={
                "material_hardness_hv": 222.0, "tensile_strength_mpa": 660.0,
                "lower_spec_limit": 180.0, "upper_spec_limit": 230.0, "supplier_id": "VENDOR_B"
            })
            if r.status_code == 200:
                data = r.json()
                report.log_pass("API", "POST /api/tier2/evaluate_material", f"Status: 200, Risk: {data['risk_evaluation']['batch_risk_tier']}, Scrap: {data['downstream_forecast']['expected_scrap_count']} units")
            else:
                report.log_fail("API", "POST /api/tier2/evaluate_material", f"Status: {r.status_code}, Body: {r.text}")

            # 4.5 Tier 3 Endpoints: Suppliers, Claim Trace, Tolerance Gap, SCAR
            r = client.get("/api/tier3/suppliers")
            if r.status_code == 200 and "VENDOR_B" in r.json():
                report.log_pass("API", "GET /api/tier3/suppliers", "Status: 200, Supplier profiles retrieved")
            else:
                report.log_fail("API", "GET /api/tier3/suppliers", f"Status: {r.status_code}")

            r = client.get("/api/tier3/trace/CLM-2026-00001")
            if r.status_code == 200 and "material_genealogy" in r.json():
                report.log_pass("API", "GET /api/tier3/trace/CLM-2026-00001", "Status: 200, Trace graph verified")
            else:
                report.log_fail("API", "GET /api/tier3/trace/CLM-2026-00001", f"Status: {r.status_code}")

            r = client.get("/api/tier3/tolerance_gap")
            if r.status_code == 200 and "tolerance_leakage_zone" in r.json():
                gap = r.json()
                report.log_pass("API", "GET /api/tier3/tolerance_gap", f"Status: 200, Leakage Zone Dollars: ${gap['tolerance_leakage_zone']['warranty_dollars_leaked']:,.2f}")
            else:
                report.log_fail("API", "GET /api/tier3/tolerance_gap", f"Status: {r.status_code}")

            r = client.get("/api/tier3/scar/VENDOR_B")
            if r.status_code == 200:
                scar = r.json()
                report.log_pass("API", "GET /api/tier3/scar/VENDOR_B", f"Status: 200, SCAR Demand: ${scar['financial_recovery_breakdown']['total_demand_amount_usd']:,.2f}")
            else:
                report.log_fail("API", "GET /api/tier3/scar/VENDOR_B", f"Status: {r.status_code}")

            # 4.6 Supplier Router with API Key Authentication
            # Unauthenticated request -> expect 401
            r = client.post("/api/v1/supplier/evaluate-lot", json={
                "supplier_id": "VENDOR_B", "material_hardness_hv": 224.5, "tensile_strength_mpa": 675.0,
                "lower_spec_limit": 180.0, "upper_spec_limit": 230.0
            })
            if r.status_code == 401:
                report.log_pass("API", "Auth Enforcement (Missing Key)", "Correctly returned HTTP 401 Unauthorized")
            else:
                report.log_fail("API", "Auth Enforcement (Missing Key)", f"Expected 401 but received {r.status_code}")

            # Invalid key request -> expect 401
            r = client.post("/api/v1/supplier/evaluate-lot", headers={"X-API-Key": "invalid-bogus-key"}, json={
                "supplier_id": "VENDOR_B", "material_hardness_hv": 224.5, "tensile_strength_mpa": 675.0,
                "lower_spec_limit": 180.0, "upper_spec_limit": 230.0
            })
            if r.status_code == 401:
                report.log_pass("API", "Auth Enforcement (Invalid Key)", "Correctly returned HTTP 401 Unauthorized")
            else:
                report.log_fail("API", "Auth Enforcement (Invalid Key)", f"Expected 401 but received {r.status_code}")

            # Authenticated request with Vendor B Key
            r = client.post("/api/v1/supplier/evaluate-lot", headers={"X-API-Key": "fs-key-vendor-b-3e7a9d0f"}, json={
                "supplier_id": "VENDOR_B", "material_hardness_hv": 224.5, "tensile_strength_mpa": 675.0,
                "lower_spec_limit": 180.0, "upper_spec_limit": 230.0
            })
            if r.status_code == 200:
                data = r.json()
                report.log_pass("API", "POST /api/v1/supplier/evaluate-lot (Auth: Vendor B)", f"Status: 200, Leakage Zone: {data['is_in_tolerance_leakage_zone']}")
            else:
                report.log_fail("API", "POST /api/v1/supplier/evaluate-lot (Auth: Vendor B)", f"Status: {r.status_code}, Body: {r.text}")

            # Comprehensive lot certification (all 3 models unified)
            r = client.post("/api/v1/supplier/certify-lot", headers={"X-API-Key": "fs-key-vendor-b-3e7a9d0f"}, json={
                "supplier_id": "VENDOR_B", "batch_id": "LOT-2026-B88", "material_grade": "AISI-4140",
                "material_hardness_hv": 222.5, "tensile_strength_mpa": 670.0, "batch_size_units": 8000,
                "lower_spec_limit": 180.0, "upper_spec_limit": 230.0
            })
            if r.status_code == 200:
                cert = r.json()
                report.log_pass("API", "POST /api/v1/supplier/certify-lot (Unified 3-Model Certification)", f"Verdict: {cert['evaluation_verdict']}")
            else:
                report.log_fail("API", "POST /api/v1/supplier/certify-lot", f"Status: {r.status_code}, Body: {r.text}")

            # Admin Key management endpoints
            # Non-admin trying to list keys -> expect 403
            r = client.get("/api/v1/supplier/auth/keys", headers={"X-API-Key": "fs-key-vendor-b-3e7a9d0f"})
            if r.status_code == 403:
                report.log_pass("API", "Admin RBAC Protection (Non-Admin Key)", "Correctly returned HTTP 403 Forbidden")
            else:
                report.log_fail("API", "Admin RBAC Protection (Non-Admin Key)", f"Expected 403 but received {r.status_code}")

            # Admin listing keys -> expect 200
            r = client.get("/api/v1/supplier/auth/keys", headers={"X-API-Key": "fs-admin-master-2026-99x"})
            if r.status_code == 200:
                report.log_pass("API", "Admin RBAC Access (Admin Master Key)", f"Status: 200, Registered Keys: {len(r.json())}")
            else:
                report.log_fail("API", "Admin RBAC Access (Admin Master Key)", f"Status: {r.status_code}")

    except Exception as e:
        report.log_fail("API", "TestClient Execution", f"Fatal exception during API testing: {e}\n{traceback.format_exc()}")


def verify_client_sdk(report: VerificationReport):
    print("\n>>> [5/5] VERIFYING SUPPLIER API CLIENT SDK")

    from src.supplier_api_client import SupplierAPIClient

    client = SupplierAPIClient(base_url="http://localhost:8000", api_key="fs-key-vendor-b-3e7a9d0f")
    if client.api_key == "fs-key-vendor-b-3e7a9d0f" and client.prefix == "/api/v1/supplier":
        report.log_pass("CLIENT_SDK", "SupplierAPIClient Initialization", "Target prefix and API key configured")
    else:
        report.log_fail("CLIENT_SDK", "SupplierAPIClient Initialization", "Improper configuration")


def main():
    report = VerificationReport()
    verify_virtual_environment(report)
    verify_pickle_models(report)
    verify_api_keys_and_auth(report)
    verify_fastapi_endpoints(report)
    verify_client_sdk(report)
    success = report.summary()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
