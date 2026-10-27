"""
FastAPI Backend for Manufacturing, Material & Supplier Intelligence Suite.
Serves real-time inference, SHAP explanations, prescriptive actions,
genealogy tracing, and the interactive web application.
"""

import os
import sys
import json
from typing import Dict, Any, Optional

# Ensure project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np

from src.models.tier1_manufacturing import (
    ShiftDefectPredictor, FailureModeClassifier, PrescriptiveActionEngine
)
from src.models.tier2_materials import (
    MarginalBatchRiskClassifier, DownstreamImpactForecaster, AdaptiveParameterCompensator
)
from src.models.tier3_supplier import (
    SupplierQualityTrendAnalyzer, WarrantyTraceabilityEngine,
    ToleranceLeakageAnalyzer, SCARGenerator
)

app = FastAPI(
    title="Manufacturing Intelligence API",
    description="End-to-End ML Suite for Manufacturing, Materials, and Supplier Intelligence",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from src.supplier_api import supplier_router, init_models as init_supplier_models
app.include_router(supplier_router)

# Global model state
models: Dict[str, Any] = {}


@app.on_event("startup")
def load_models():
    print("Loading models and analytical engines...")
    init_supplier_models(data_dir=DATA_DIR)
    try:
        # Tier 1
        shift_pred = ShiftDefectPredictor()
        shift_pkl = os.path.join(MODELS_DIR, "shift_defect_predictor.pkl")
        if os.path.exists(shift_pkl):
            shift_pred.load(shift_pkl)
        models["shift_defect"] = shift_pred

        fm_clf = FailureModeClassifier()
        fm_pkl = os.path.join(MODELS_DIR, "failure_mode_classifier.pkl")
        if os.path.exists(fm_pkl):
            fm_clf.load(fm_pkl)
        models["failure_mode"] = fm_clf

        prescriptive = PrescriptiveActionEngine()
        presc_pkl = os.path.join(MODELS_DIR, "prescriptive_action_engine.pkl")
        if os.path.exists(presc_pkl):
            prescriptive.load(presc_pkl)
        models["prescriptive"] = prescriptive

        # Tier 2
        batch_clf = MarginalBatchRiskClassifier()
        batch_pkl = os.path.join(MODELS_DIR, "marginal_batch_classifier.pkl")
        if os.path.exists(batch_pkl):
            batch_clf.load(batch_pkl)
        models["batch_risk"] = batch_clf

        # Downstream forecaster & compensator
        impact_forecaster = DownstreamImpactForecaster()
        impact_pkl = os.path.join(MODELS_DIR, "downstream_impact_forecaster.pkl")
        if os.path.exists(impact_pkl):
            impact_forecaster.load(impact_pkl)
        else:
            from src.data_loader import DataLoader
            dl = DataLoader(data_dir=DATA_DIR)
            batch_summary = dl.get_material_batch_summary()
            impact_forecaster.fit(batch_summary)
        models["impact_forecaster"] = impact_forecaster

        compensator = AdaptiveParameterCompensator()
        comp_pkl = os.path.join(MODELS_DIR, "adaptive_parameter_compensator.pkl")
        if os.path.exists(comp_pkl):
            compensator.load(comp_pkl)
        models["compensator"] = compensator

        # Tier 3
        models["trend_analyzer"] = SupplierQualityTrendAnalyzer(data_dir=DATA_DIR)
        models["trace_engine"] = WarrantyTraceabilityEngine(data_dir=DATA_DIR)
        models["gap_analyzer"] = ToleranceLeakageAnalyzer(data_dir=DATA_DIR)
        models["scar_generator"] = SCARGenerator(data_dir=DATA_DIR)

        print("All models and analytical engines loaded successfully.")
    except Exception as e:
        print(f"Warning during model loading: {e}")


@app.get("/health", tags=["System Health"])
@app.get("/api/health", tags=["System Health"])
def health_check():
    """System health check and loaded model registry inspection."""
    return {
        "status": "healthy",
        "service": "Manufacturing Intelligence API",
        "version": "1.0.0",
        "models_loaded": list(models.keys()),
        "models_directory": MODELS_DIR,
        "data_directory": DATA_DIR
    }


# Pydantic Schemas
class SensorTelemetryInput(BaseModel):
    station_id: str = "ST-10"
    spindle_temp_c: float = 62.0
    system_pressure_psi: float = 115.0
    vibration_rms_g: float = 2.1
    tool_wear_index: float = 55.0
    hours_since_service: float = 18.0
    peak_torque_nm: float = 13.2
    shutoff_angle_deg: float = 285.0
    torque_status: str = "NOK_HIGH"
    material_hardness_hv: float = 220.0
    tensile_strength_mpa: float = 645.0


class MaterialBatchInput(BaseModel):
    material_hardness_hv: float = 222.0
    tensile_strength_mpa: float = 660.0
    lower_spec_limit: float = 180.0
    upper_spec_limit: float = 230.0
    supplier_id: str = "VENDOR_B"


# Endpoints
@app.get("/api/overview")
def get_overview():
    summary_path = os.path.join(REPORTS_DIR, "model_performance_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            data = json.load(f)
            return data
    raise HTTPException(status_code=404, detail="Performance summary report not found.")


@app.post("/api/tier1/simulate")
def simulate_tier1(input_data: SensorTelemetryInput):
    """
    Simulates real-time cycle telemetry:
    Predicts Failure Mode, computes SHAP physical contributors, and recommends prescriptive offsets.
    """
    fm_clf: FailureModeClassifier = models.get("failure_mode")
    prescriptive: PrescriptiveActionEngine = models.get("prescriptive")

    if not fm_clf or not prescriptive:
        raise HTTPException(status_code=500, detail="Tier 1 models not initialized.")

    # Create single row DataFrame
    row = pd.DataFrame([{
        "peak_torque_nm": input_data.peak_torque_nm,
        "shutoff_angle_deg": input_data.shutoff_angle_deg,
        "spindle_temp_c": input_data.spindle_temp_c,
        "system_pressure_psi": input_data.system_pressure_psi,
        "vibration_rms_g": input_data.vibration_rms_g,
        "tool_wear_index": input_data.tool_wear_index,
        "hours_since_service": input_data.hours_since_service,
        "torque_energy_proxy": input_data.peak_torque_nm * (input_data.shutoff_angle_deg * np.pi / 180.0),
        "tool_stress_factor": input_data.tool_wear_index * input_data.vibration_rms_g,
        "thermal_vibration_index": input_data.spindle_temp_c * input_data.vibration_rms_g,
        "hydraulic_thermal_ratio": input_data.system_pressure_psi / (input_data.spindle_temp_c + 1e-3),
        "material_tool_resistance": input_data.material_hardness_hv * (input_data.tool_wear_index + 1.0),
        "torque_delta_nominal": input_data.peak_torque_nm - 12.5,
        "angle_delta_nominal": input_data.shutoff_angle_deg - 360.0,
        "torque_status_nok": 1.0 if input_data.torque_status != "OK" else 0.0,
        "torque_status_high": 1.0 if input_data.torque_status == "NOK_HIGH" else 0.0,
        "torque_status_low": 1.0 if input_data.torque_status == "NOK_LOW" else 0.0,
        "material_hardness_hv": input_data.material_hardness_hv,
        "tensile_strength_mpa": input_data.tensile_strength_mpa,
        "st_ST-10": 1.0 if input_data.station_id == "ST-10" else 0.0,
        "st_ST-20": 1.0 if input_data.station_id == "ST-20" else 0.0,
        "st_ST-30": 1.0 if input_data.station_id == "ST-30" else 0.0,
        "st_ST-40": 1.0 if input_data.station_id == "ST-40" else 0.0,
    }])

    # Predict failure mode & contributors
    prediction_results = fm_clf.predict_with_contributors(row, top_k=5)[0]

    # Prescriptive recommendations
    telemetry_dict = {
        "spindle_temp_c": input_data.spindle_temp_c,
        "peak_torque_nm": input_data.peak_torque_nm,
        "shutoff_angle_deg": input_data.shutoff_angle_deg,
        "vibration_rms_g": input_data.vibration_rms_g,
        "tool_wear_index": input_data.tool_wear_index,
        "system_pressure_psi": input_data.system_pressure_psi,
    }
    action_rec = prescriptive.recommend_action(
        predicted_failure_mode=prediction_results["predicted_failure_mode"],
        current_telemetry=telemetry_dict,
        top_contributors=prediction_results["top_physical_contributors"]
    )

    return {
        "station_id": input_data.station_id,
        "predicted_failure_mode": prediction_results["predicted_failure_mode"],
        "confidence": prediction_results["confidence"],
        "top_physical_contributors": prediction_results["top_physical_contributors"],
        "prescriptive_action": action_rec
    }


@app.post("/api/tier2/evaluate_material")
def evaluate_material(input_data: MaterialBatchInput):
    batch_clf: MarginalBatchRiskClassifier = models.get("batch_risk")
    forecaster: DownstreamImpactForecaster = models.get("impact_forecaster")
    compensator: AdaptiveParameterCompensator = models.get("compensator")

    if not batch_clf or not forecaster or not compensator:
        raise HTTPException(status_code=500, detail="Tier 2 models not initialized.")

    row = pd.DataFrame([{
        "material_hardness_hv": input_data.material_hardness_hv,
        "tensile_strength_mpa": input_data.tensile_strength_mpa,
        "lower_spec_limit": input_data.lower_spec_limit,
        "upper_spec_limit": input_data.upper_spec_limit,
        "supplier_id": input_data.supplier_id,
        "supplier_VENDOR_A": 1.0 if input_data.supplier_id == "VENDOR_A" else 0.0,
        "supplier_VENDOR_B": 1.0 if input_data.supplier_id == "VENDOR_B" else 0.0,
        "supplier_VENDOR_C": 1.0 if input_data.supplier_id == "VENDOR_C" else 0.0,
    }])

    risk = batch_clf.predict_batch_risk(row)
    forecast = forecaster.forecast(
        hardness=input_data.material_hardness_hv,
        tensile=input_data.tensile_strength_mpa,
        supplier_id=input_data.supplier_id
    )
    compensation = compensator.compute_offsets(
        batch_hardness_hv=input_data.material_hardness_hv,
        batch_tensile_mpa=input_data.tensile_strength_mpa
    )

    return {
        "risk_evaluation": risk,
        "downstream_forecast": forecast,
        "adaptive_compensation": compensation
    }


@app.get("/api/tier3/suppliers")
def get_supplier_trends():
    analyzer: SupplierQualityTrendAnalyzer = models.get("trend_analyzer")
    if not analyzer:
        raise HTTPException(status_code=500, detail="Trend analyzer not initialized.")
    return analyzer.analyze_trends()


@app.get("/api/tier3/trace/{claim_id}")
def trace_warranty_claim(claim_id: str):
    trace_engine: WarrantyTraceabilityEngine = models.get("trace_engine")
    if not trace_engine:
        raise HTTPException(status_code=500, detail="Trace engine not initialized.")
    res = trace_engine.trace_claim(claim_id)
    if not res:
        raise HTTPException(status_code=404, detail=f"Claim {claim_id} not found.")
    return res


@app.get("/api/tier3/tolerance_gap")
def get_tolerance_gap():
    analyzer: ToleranceLeakageAnalyzer = models.get("gap_analyzer")
    if not analyzer:
        raise HTTPException(status_code=500, detail="Gap analyzer not initialized.")
    return analyzer.run_gap_analysis()


@app.get("/api/tier3/scar/{supplier_id}")
def get_scar_report(supplier_id: str):
    scar_path = os.path.join(REPORTS_DIR, "scar_reports", f"{supplier_id.upper()}_SCAR.json")
    if os.path.exists(scar_path):
        with open(scar_path, "r") as f:
            return json.load(f)
    gen: SCARGenerator = models.get("scar_generator")
    if gen:
        return gen.generate_scar(supplier_id)
    raise HTTPException(status_code=404, detail="SCAR generator unavailable.")


# Mount web directory if it exists
web_dir = os.path.join(PROJECT_ROOT, "web")
if os.path.exists(web_dir):
    app.mount("/static", StaticFiles(directory=web_dir), name="static")

    @app.get("/")
    def serve_index():
        index_file = os.path.join(web_dir, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        return {"status": "online", "docs": "/docs", "message": "Manufacturing Intelligence API"}


if __name__ == "__main__":
    import uvicorn
    # Default port 8000 for unified API
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
