"""
FactorySmart AI™ - Quickstart Example Usage Script.

This script demonstrates two ways for external engineers and data scientists to use the project:
  Method A: Direct Python In-Memory Inference (Loading Pickle Models directly)
  Method B: Remote REST API Client Inference (Invoking API with Authentication Key)
"""

import os
import sys
import json
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models import load_all_models
from src.supplier_api_client import SupplierAPIClient


def demo_method_a_direct_pickle_models():
    print("=" * 80)
    print("METHOD A: DIRECT IN-MEMORY INFERENCE USING SERIALIZED PICKLE MODELS")
    print("=" * 80)

    # 1. Load all serialized models from models/ directory
    print("[*] Loading pickle models from models/ ...")
    models = load_all_models(models_dir="models", data_dir="data")
    print(f"    Loaded {len(models)} models and analytical engines.")

    # 2. Tier 1: Failure Mode Classification with SHAP Physical Attributions
    print("\n--- Tier 1 Demo: Machine Telemetry Failure Mode & Root Cause ---")
    fm_clf = models["failure_mode_classifier"]
    prescriptive = models["prescriptive_action_engine"]

    sample_telemetry = pd.DataFrame([{
        "peak_torque_nm": 13.6,
        "shutoff_angle_deg": 268.0,
        "spindle_temp_c": 64.5,
        "system_pressure_psi": 114.0,
        "vibration_rms_g": 2.2,
        "tool_wear_index": 56.0,
        "hours_since_service": 22.0,
        "material_hardness_hv": 224.0,
        "tensile_strength_mpa": 670.0,
        "torque_energy_proxy": 13.6 * (268.0 * np.pi / 180.0),
        "tool_stress_factor": 56.0 * 2.2,
        "thermal_vibration_index": 64.5 * 2.2,
        "hydraulic_thermal_ratio": 114.0 / 64.5,
        "material_tool_resistance": 224.0 * 57.0,
        "torque_delta_nominal": 13.6 - 12.5,
        "angle_delta_nominal": 268.0 - 360.0,
        "torque_status_nok": 1.0,
        "torque_status_high": 1.0,
        "torque_status_low": 0.0,
        "st_ST-10": 1.0,
        "st_ST-20": 0.0,
        "st_ST-30": 0.0,
        "st_ST-40": 0.0,
    }])

    prediction = fm_clf.predict_with_contributors(sample_telemetry, top_k=3)[0]
    print(f"    Predicted Failure Mode : {prediction['predicted_failure_mode']} (Confidence: {prediction['confidence']:.2%})")
    print(f"    Top Physical Drivers   :")
    for c in prediction["top_physical_contributors"]:
        print(f"      - {c['feature']}: {c['value']} ({c['direction']}, SHAP impact={c['shap_impact']:+.4f})")

    # Recommend prescriptive offsets
    action = prescriptive.recommend_action(
        predicted_failure_mode=prediction["predicted_failure_mode"],
        current_telemetry={"spindle_temp_c": 64.5, "peak_torque_nm": 13.6},
        top_contributors=prediction["top_physical_contributors"]
    )
    print(f"    Prescriptive Task      : {action['recommended_task']}")
    print(f"    Prescribed Offsets     : {action['prescriptive_recipe_offsets']}")

    # 3. Tier 2: Material Lot Downstream Defect Risk & Adaptive Compensation
    print("\n--- Tier 2 Demo: Receiving Material Lot Risk & Adaptive Offsets ---")
    batch_clf = models["marginal_batch_classifier"]
    forecaster = models["downstream_impact_forecaster"]
    compensator = models["adaptive_parameter_compensator"]

    incoming_lot = pd.DataFrame([{
        "material_hardness_hv": 224.5,
        "tensile_strength_mpa": 672.0,
        "lower_spec_limit": 180.0,
        "upper_spec_limit": 230.0,
        "supplier_id": "VENDOR_B",
        "supplier_VENDOR_A": 0.0,
        "supplier_VENDOR_B": 1.0,
        "supplier_VENDOR_C": 0.0,
    }])

    risk_eval = batch_clf.predict_batch_risk(incoming_lot)
    impact_fc = forecaster.forecast(hardness=224.5, tensile=672.0, supplier_id="VENDOR_B")
    comp_offsets = compensator.compute_offsets(batch_hardness_hv=224.5, batch_tensile_mpa=672.0)

    print(f"    Lot Risk Tier          : {risk_eval['batch_risk_tier']} (Downstream Risk Prob: {risk_eval['downstream_risk_probability']:.2%})")
    print(f"    Forecasted Scrap       : {impact_fc['expected_scrap_count']} units (${impact_fc['projected_cost_usd']:,.2f})")
    print(f"    Prescribed Torque Target: {comp_offsets['prescribed_torque_target_nm']} Nm (Offset: +{comp_offsets['torque_offset_nm']} Nm)")
    print(f"    Prescribed Spindle RPM : Speed reduction {comp_offsets['spindle_speed_offset_pct']}%")
    print(f"    Coolant Boost Mode     : {comp_offsets['coolant_lubrication_mode']}")


def demo_method_b_rest_api_client():
    print("\n" + "=" * 80)
    print("METHOD B: REST API CONSUMPTION VIA PYTHON CLIENT WITH API KEY AUTH")
    print("=" * 80)

    # Initialize client targeting unified API port 8000 (or standalone 8001)
    client = SupplierAPIClient(base_url="http://localhost:8000", api_key="fs-key-vendor-b-3e7a9d0f")
    print("[*] Connecting to Supplier API using Vendor B Key ('fs-key-vendor-b-3e7a9d0f')...")

    try:
        health = client.health_check()
        print(f"[OK] Connected! Server Status: {health.get('status')} | Service: {health.get('service')}")

        # Evaluate a border-line hard lot (224.5 HV)
        eval_resp = client.evaluate_material_lot(
            hardness=224.5,
            tensile=675.0,
            supplier_id="VENDOR_B",
            batch_id="LOT-DEMO-2026-X1"
        )
        print("\n--- Material Lot Certification Response ---")
        print(f"    Batch Risk Tier     : {eval_resp.get('batch_risk_tier')}")
        print(f"    Tolerance Leakage   : {eval_resp.get('is_in_tolerance_leakage_zone')}")
        print(f"    Warning Message     : {eval_resp.get('tolerance_leakage_warning')}")

        # Compute adaptive recipe offsets
        recipe_resp = client.get_adaptive_recipe(hardness=224.5, tensile=675.0)
        print("\n--- Prescribed Machine Recipe Offsets ---")
        print(f"    Torque Target       : {recipe_resp.get('prescribed_torque_target_nm')} Nm")
        print(f"    Shutoff Angle       : {recipe_resp.get('prescribed_shutoff_angle_deg')} deg")
        print(f"    Guidance            : {recipe_resp.get('engineering_guidance')}")

    except Exception as e:
        print(f"[INFO] API Server is currently offline ({e}).")
        print("       To test live API endpoints, run 'python run_api.py' in a separate terminal.")


if __name__ == "__main__":
    demo_method_a_direct_pickle_models()
    demo_method_b_rest_api_client()
