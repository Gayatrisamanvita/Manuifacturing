"""
FastAPI Backend for Manufacturing, Material & Supplier Intelligence Suite.
Serves real-time inference, SHAP explanations, prescriptive actions,
genealogy tracing, and the interactive web application.
"""

import os
import sys
import json
from functools import lru_cache
from typing import Dict, Any, List, Literal, Optional

# Ensure project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
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
model_load_errors: Dict[str, str] = {}


@lru_cache(maxsize=1)
def load_product_catalog() -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA_DIR, "supplier_products_catalog.csv"))


@lru_cache(maxsize=1)
def load_material_batches() -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA_DIR, "material_batch_master.csv"))


@lru_cache(maxsize=1)
def load_finished_product_records() -> pd.DataFrame:
    inspections = pd.read_csv(os.path.join(DATA_DIR, "part_inspection_defect_logs.csv"))
    torque = pd.read_csv(os.path.join(DATA_DIR, "torque_cycle_logs.csv"))
    materials = pd.read_csv(os.path.join(DATA_DIR, "material_batch_master.csv"))
    production = pd.read_csv(os.path.join(DATA_DIR, "production_shift_logs.csv"))
    records = inspections.merge(
        torque[["fastener_cycle_id", "event_timestamp", "station_id", "peak_torque_nm", "shutoff_angle_deg", "torque_status"]],
        on="fastener_cycle_id",
        how="left",
    )
    records = records.merge(
        materials[["material_batch_id", "supplier_id", "material_hardness_hv", "material_risk_status"]],
        on="material_batch_id",
        how="left",
    )
    return records.merge(
        production[["event_timestamp", "station_id", "shift_id", "line_speed_upm", "units_produced"]],
        on=["event_timestamp", "station_id"],
        how="left",
    )


def json_records(frame: pd.DataFrame) -> list:
    return json.loads(frame.to_json(orient="records"))


@app.on_event("startup")
def load_models():
    print("Loading models and analytical engines...")
    models.clear()
    model_load_errors.clear()

    def register(key: str, builder):
        try:
            models[key] = builder()
        except Exception as exc:
            model_load_errors[key] = f"{type(exc).__name__}: {exc}"
            print(f"Failed to initialize {key}: {model_load_errors[key]}")

    def artifact_model(model_class, filename: str):
        artifact_path = os.path.join(MODELS_DIR, filename)
        if not os.path.isfile(artifact_path):
            raise FileNotFoundError(f"Required model artifact is missing: {artifact_path}")
        model = model_class()
        model.load(artifact_path)
        return model

    def load_impact_forecaster():
        from src.data_loader import DataLoader

        model = DownstreamImpactForecaster()
        artifact_path = os.path.join(MODELS_DIR, "downstream_impact_forecaster.pkl")
        batch_summary = None
        if os.path.isfile(artifact_path):
            model.load(artifact_path)
        else:
            batch_summary = DataLoader(data_dir=DATA_DIR).get_material_batch_summary()
            model.fit(batch_summary)
        if not model.reference_batch_size:
            if batch_summary is None:
                batch_summary = DataLoader(data_dir=DATA_DIR).get_material_batch_summary()
            model.reference_batch_size = float(batch_summary["total_parts"].mean())
        return model

    try:
        init_supplier_models(data_dir=DATA_DIR)
    except Exception as exc:
        model_load_errors["supplier_api"] = f"{type(exc).__name__}: {exc}"
        print(f"Failed to initialize supplier_api: {model_load_errors['supplier_api']}")
    register("shift_defect", lambda: artifact_model(ShiftDefectPredictor, "shift_defect_predictor.pkl"))
    register("failure_mode", lambda: artifact_model(FailureModeClassifier, "failure_mode_classifier.pkl"))
    register("prescriptive", lambda: artifact_model(PrescriptiveActionEngine, "prescriptive_action_engine.pkl"))
    register("batch_risk", lambda: artifact_model(MarginalBatchRiskClassifier, "marginal_batch_classifier.pkl"))
    register("impact_forecaster", load_impact_forecaster)
    register("compensator", lambda: artifact_model(AdaptiveParameterCompensator, "adaptive_parameter_compensator.pkl"))
    register("trend_analyzer", lambda: SupplierQualityTrendAnalyzer(data_dir=DATA_DIR))
    register("trace_engine", lambda: WarrantyTraceabilityEngine(data_dir=DATA_DIR))
    register("gap_analyzer", lambda: ToleranceLeakageAnalyzer(data_dir=DATA_DIR))
    register("scar_generator", lambda: SCARGenerator(data_dir=DATA_DIR))

    if model_load_errors:
        print(f"Model initialization completed with errors: {model_load_errors}")
    else:
        print("All models and analytical engines loaded successfully.")


@app.get("/health", tags=["System Health"])
@app.get("/api/health", tags=["System Health"])
def health_check():
    """System health check and loaded model registry inspection."""
    required_models = {
        "shift_defect", "failure_mode", "prescriptive", "batch_risk",
        "impact_forecaster", "compensator", "trend_analyzer",
        "trace_engine", "gap_analyzer", "scar_generator",
    }
    missing_models = sorted(required_models.difference(models))
    return {
        "status": "healthy" if not model_load_errors and not missing_models else "degraded",
        "service": "Manufacturing Intelligence API",
        "version": "1.0.0",
        "models_loaded": list(models.keys()),
        "models_missing": missing_models,
        "model_load_errors": model_load_errors.copy(),
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


class ShiftStationInput(BaseModel):
    station_id: Literal["ST-10", "ST-20", "ST-30", "ST-40"]
    operator_tenure_yrs: float = 0.0
    spindle_temp_c: float
    vibration_rms_g: float
    system_pressure_psi: float
    tool_wear_index: float
    peak_torque_nm: float = 12.5
    shutoff_angle_deg: float = 360.0
    max_spindle_temp_c: Optional[float] = None
    max_vibration_rms_g: Optional[float] = None
    max_tool_wear_index: Optional[float] = None
    std_peak_torque_nm: float = 0.0


class ShiftPredictionInput(BaseModel):
    shift_id: Literal["SHIFT_1", "SHIFT_2", "SHIFT_3"] = "SHIFT_1"
    planned_units: float = Field(..., ge=0)
    line_speed_upm: float = Field(..., ge=0)
    std_line_speed_upm: float = Field(0.0, ge=0)
    stations: List[ShiftStationInput] = Field(..., min_length=1, max_length=4)


class MaterialBatchInput(BaseModel):
    material_hardness_hv: float = 222.0
    tensile_strength_mpa: float = 660.0
    lower_spec_limit: float = Field(180.0, ge=0)
    upper_spec_limit: float = Field(230.0, gt=0)
    supplier_id: Literal["VENDOR_A", "VENDOR_B", "VENDOR_C"] = "VENDOR_B"
    batch_size_units: Optional[int] = Field(default=None, ge=1)


class PrescriptiveActionInput(BaseModel):
    predicted_failure_mode: str
    current_telemetry: SensorTelemetryInput
    top_contributors: List[Dict[str, Any]] = Field(default_factory=list)


# Endpoints
@app.get("/api/overview")
def get_overview():
    summary_path = os.path.join(REPORTS_DIR, "model_performance_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            data = json.load(f)
            return data
    raise HTTPException(status_code=404, detail="Performance summary report not found.")


@app.get("/api/models", tags=["Model Registry"])
def get_model_registry():
    """Lists every model/engine, its serialized artifact when applicable, and inference routes."""
    registry = [
        {
            "name": "ShiftDefectPredictor",
            "registry_key": "shift_defect",
            "type": "trained_ml",
            "artifact": "shift_defect_predictor.pkl",
            "routes": ["POST /api/tier1/predict_shift"],
        },
        {
            "name": "FailureModeClassifier",
            "registry_key": "failure_mode",
            "type": "trained_ml",
            "artifact": "failure_mode_classifier.pkl",
            "routes": ["POST /api/tier1/predict_failure_mode", "POST /api/tier1/simulate"],
        },
        {
            "name": "PrescriptiveActionEngine",
            "registry_key": "prescriptive",
            "type": "rule_engine",
            "artifact": "prescriptive_action_engine.pkl",
            "routes": ["POST /api/tier1/recommend_action", "POST /api/tier1/simulate"],
        },
        {
            "name": "MarginalBatchRiskClassifier",
            "registry_key": "batch_risk",
            "type": "trained_ml",
            "artifact": "marginal_batch_classifier.pkl",
            "routes": [
                "POST /api/tier2/classify_material",
                "POST /api/tier2/evaluate_material",
                "POST /api/v1/supplier/evaluate-lot",
            ],
        },
        {
            "name": "DownstreamImpactForecaster",
            "registry_key": "impact_forecaster",
            "type": "trained_ml",
            "artifact": "downstream_impact_forecaster.pkl",
            "routes": [
                "POST /api/tier2/forecast_impact",
                "POST /api/tier2/evaluate_material",
                "POST /api/v1/supplier/forecast-impact",
            ],
        },
        {
            "name": "AdaptiveParameterCompensator",
            "registry_key": "compensator",
            "type": "rule_engine",
            "artifact": "adaptive_parameter_compensator.pkl",
            "routes": [
                "POST /api/tier2/compensate",
                "POST /api/tier2/evaluate_material",
                "POST /api/v1/supplier/adaptive-recipe",
            ],
        },
        {
            "name": "SupplierQualityTrendAnalyzer",
            "registry_key": "trend_analyzer",
            "type": "data_analytics_engine",
            "artifact": None,
            "routes": ["GET /api/tier3/suppliers", "GET /api/v1/supplier/vendors"],
        },
        {
            "name": "WarrantyTraceabilityEngine",
            "registry_key": "trace_engine",
            "type": "data_analytics_engine",
            "artifact": None,
            "routes": ["GET /api/tier3/trace/{claim_id}", "GET /api/v1/supplier/trace/claim/{claim_id}"],
        },
        {
            "name": "ToleranceLeakageAnalyzer",
            "registry_key": "gap_analyzer",
            "type": "data_analytics_engine",
            "artifact": None,
            "routes": ["GET /api/tier3/tolerance_gap", "GET /api/v1/supplier/tolerance-gap"],
        },
        {
            "name": "SCARGenerator",
            "registry_key": "scar_generator",
            "type": "data_analytics_engine",
            "artifact": None,
            "routes": ["GET /api/tier3/scar/{supplier_id}", "GET /api/v1/supplier/scar/{supplier_id}"],
        },
    ]
    for model in registry:
        artifact = model["artifact"]
        model["artifact_exists"] = (
            os.path.isfile(os.path.join(MODELS_DIR, artifact)) if artifact else None
        )
        instance = models.get(model["registry_key"])
        model["initialized"] = instance is not None
        if model["type"] == "trained_ml":
            model["ready"] = bool(
                instance
                and (
                    getattr(instance, "feature_names", None)
                    or getattr(instance, "features", None)
                )
            )
        else:
            model["ready"] = instance is not None
        model["artifact"] = f"models/{artifact}" if artifact else None
    return {"models": registry}


def _shift_feature_row(
    shift_id: str,
    planned_units: float,
    line_speed_upm: float,
    std_line_speed_upm: float,
    station: ShiftStationInput,
) -> Dict[str, Any]:
    return {
        "station_id": station.station_id,
        "total_units": planned_units,
        "avg_line_speed": line_speed_upm,
        "std_line_speed": std_line_speed_upm,
        "operator_tenure_yrs": station.operator_tenure_yrs,
        "mean_spindle_temp": station.spindle_temp_c,
        "max_spindle_temp": station.max_spindle_temp_c if station.max_spindle_temp_c is not None else station.spindle_temp_c,
        "mean_vibration": station.vibration_rms_g,
        "max_vibration": station.max_vibration_rms_g if station.max_vibration_rms_g is not None else station.vibration_rms_g,
        "mean_pressure": station.system_pressure_psi,
        "mean_tool_wear": station.tool_wear_index,
        "max_tool_wear": station.max_tool_wear_index if station.max_tool_wear_index is not None else station.tool_wear_index,
        "mean_peak_torque": station.peak_torque_nm,
        "std_peak_torque": station.std_peak_torque_nm,
        "mean_shutoff_angle": station.shutoff_angle_deg,
        **{
            f"station_{station_id}": float(station.station_id == station_id)
            for station_id in ("ST-10", "ST-20", "ST-30", "ST-40")
        },
        **{
            f"shift_{shift}": float(shift_id == shift)
            for shift in ("SHIFT_1", "SHIFT_2", "SHIFT_3")
        },
    }


@app.post("/api/tier1/predict_shift", tags=["Tier 1: Manufacturing"])
def predict_shift(input_data: ShiftPredictionInput):
    predictor: ShiftDefectPredictor = models.get("shift_defect")
    if not predictor or not predictor.feature_names:
        raise HTTPException(status_code=503, detail="Shift defect predictor artifact is not loaded.")
    station_ids = [station.station_id for station in input_data.stations]
    if len(station_ids) != len(set(station_ids)):
        raise HTTPException(status_code=422, detail="Each station_id may appear only once.")

    shift_data = pd.DataFrame([
        _shift_feature_row(
            input_data.shift_id,
            input_data.planned_units,
            input_data.line_speed_upm,
            input_data.std_line_speed_upm,
            station,
        )
        for station in input_data.stations
    ])
    rankings = predictor.rank_station_risk(shift_data)
    return {
        "shift_id": input_data.shift_id,
        "planned_units": input_data.planned_units,
        "line_speed_upm": input_data.line_speed_upm,
        "station_count": len(rankings),
        "station_risk_ranking": rankings,
    }


@app.get("/api/tier3/catalog")
def get_supplier_product_catalog():
    catalog = load_product_catalog().copy()
    supplier_history = pd.read_csv(os.path.join(DATA_DIR, "supplier_quality_history.csv"))
    material_batches = load_material_batches()
    warranty_claims = pd.read_csv(os.path.join(DATA_DIR, "warranty_field_claims.csv")).merge(
        material_batches[["material_batch_id", "supplier_id"]],
        on="material_batch_id",
        how="left",
    )
    catalog["quality_score_pct"] = pd.to_numeric(
        catalog["quality_score"].astype(str).str.rstrip("%"), errors="coerce"
    )
    catalog["monthly_volume_units"] = pd.to_numeric(
        catalog["monthly_vol"].astype(str).str.replace(",", "", regex=False).str.extract(r"(\d+)")[0],
        errors="coerce",
    ).fillna(0).astype(int)
    catalog["risk_level"] = catalog["delay_days"].apply(
        lambda delay: "HIGH" if delay >= 7 else ("MODERATE" if delay >= 3 else "LOW")
    )
    catalog = catalog.sort_values(["delay_days", "quality_score_pct"], ascending=[False, True])

    products_by_supplier = {
        supplier_id: json_records(products)
        for supplier_id, products in catalog.groupby("supplier_id", sort=True)
    }
    supplier_summaries = []
    for supplier_id, products in catalog.groupby("supplier_id", sort=True):
        highest_delay = products.iloc[0]
        history = supplier_history[supplier_history["supplier_id"] == supplier_id].sort_values("snapshot_date")
        claims = warranty_claims[warranty_claims["supplier_id"] == supplier_id]
        batches = material_batches[material_batches["supplier_id"] == supplier_id]
        failure_modes = history["recurring_failure_mode"].value_counts()
        supplier_summaries.append({
            "supplier_id": supplier_id,
            "product_count": int(len(products)),
            "high_risk_products": int((products["risk_level"] == "HIGH").sum()),
            "delayed_products": int((products["delay_days"] > 0).sum()),
            "average_delay_days": round(float(products["delay_days"].mean()), 1),
            "average_quality_score_pct": round(float(products["quality_score_pct"].mean()), 1),
            "monthly_volume_units": int(products["monthly_volume_units"].sum()),
            "worst_product_name": str(highest_delay["product_name"]),
            "worst_product_delay_days": int(highest_delay["delay_days"]),
            "current_defect_rate_pct": float(history.iloc[-1]["historical_defect_rate_pct"]) if not history.empty else 0.0,
            "top_recurring_failure_mode": str(failure_modes.index[0]) if not failure_modes.empty else "Not available",
            "capa_recurrence_count": int(history["capa_recurrence_count"].sum()),
            "warranty_claim_count": int(len(claims)),
            "warranty_claim_cost_usd": round(float(claims["claim_cost_usd"].sum()), 2),
            "batch_chargebacks_usd": round(float(batches["chargeback_amount_usd"].sum()), 2),
            "total_financial_impact_usd": round(
                float(claims["claim_cost_usd"].sum() + batches["chargeback_amount_usd"].sum()), 2
            ),
        })

    return {
        "suppliers": supplier_summaries,
        "products_by_supplier": products_by_supplier,
    }


@app.get("/api/tier2/materials")
def get_material_batch_analysis():
    materials = load_material_batches().copy()
    classifier: MarginalBatchRiskClassifier = models.get("batch_risk")
    if not classifier or not classifier.feature_names:
        raise HTTPException(status_code=503, detail="Marginal batch risk classifier artifact is not loaded.")

    materials["hardness_margin_upper"] = (
        materials["upper_spec_limit"] - materials["material_hardness_hv"]
    )
    materials["hardness_margin_lower"] = (
        materials["material_hardness_hv"] - materials["lower_spec_limit"]
    )
    materials["hardness_nominal_dev"] = np.abs(
        materials["material_hardness_hv"]
        - (materials["upper_spec_limit"] + materials["lower_spec_limit"]) / 2.0
    )
    materials["tensile_to_hardness_ratio"] = materials["tensile_strength_mpa"] / (
        materials["material_hardness_hv"] + 1e-4
    )
    for supplier_id in ("VENDOR_A", "VENDOR_B", "VENDOR_C"):
        materials[f"supplier_{supplier_id}"] = (
            materials["supplier_id"] == supplier_id
        ).astype(float)
    features = materials.reindex(columns=classifier.feature_names, fill_value=0.0)
    materials["model_risk_probability"] = classifier.model.predict_proba(features)[:, 1]
    materials["model_risk_tier"] = np.select(
        [materials["model_risk_probability"] >= 0.6, materials["model_risk_probability"] >= 0.3],
        ["HIGH_RISK_MARGINAL", "BORDERLINE_MONITOR"],
        default="STANDARD_PASS",
    )
    out_of_spec = (
        (materials["material_hardness_hv"] < materials["lower_spec_limit"])
        | (materials["material_hardness_hv"] > materials["upper_spec_limit"])
    )
    materials.loc[out_of_spec, "model_risk_tier"] = "OUT_OF_SPEC_REJECT"
    materials = materials.sort_values(
        ["model_risk_probability", "chargeback_amount_usd", "material_hardness_hv"],
        ascending=[False, False, False],
    )
    supplier_batches = materials.groupby("supplier_id", sort=True).agg(
        recorded_batch_count=("material_batch_id", "count"),
        high_risk_batch_count=("model_risk_tier", lambda tiers: int((tiers == "HIGH_RISK_MARGINAL").sum())),
        out_of_spec_batch_count=("model_risk_tier", lambda tiers: int((tiers == "OUT_OF_SPEC_REJECT").sum())),
        mean_predicted_risk_pct=("model_risk_probability", lambda values: round(float(values.mean()) * 100, 2)),
        recorded_chargebacks_usd=("chargeback_amount_usd", "sum"),
    ).reset_index()
    latest_snapshot = str(materials["snapshot_date"].max())
    return {
        "batch_count": int(len(materials)),
        "latest_recorded_batch_date": latest_snapshot,
        "high_risk_count": int((materials["model_risk_tier"] == "HIGH_RISK_MARGINAL").sum()),
        "recorded_scrap_count": int(materials["scrap_count"].sum()),
        "recorded_chargebacks_usd": round(float(materials["chargeback_amount_usd"].sum()), 2),
        "inspection_summary": {
            "out_of_spec_hardness_batches": int(out_of_spec.sum()),
            "marginal_high_recorded_batches": int((materials["material_risk_status"] == "MARGINAL_HIGH").sum()),
            "marginal_low_recorded_batches": int((materials["material_risk_status"] == "MARGINAL_LOW").sum()),
            "normal_recorded_batches": int((materials["material_risk_status"] == "NORMAL").sum()),
        },
        "supplier_availability_basis": "Recorded incoming-batch counts only; inventory on-hand is not present.",
        "supplier_batches": json_records(supplier_batches),
        "materials": json_records(materials),
    }


@app.get("/api/tier1/production/summary")
def get_production_summary():
    production = pd.read_csv(os.path.join(DATA_DIR, "production_shift_logs.csv"))
    production["units_produced"] = pd.to_numeric(production["units_produced"], errors="coerce").fillna(0)
    production["defect_count"] = pd.to_numeric(production["defect_count"], errors="coerce").fillna(0)
    production["line_speed_upm"] = pd.to_numeric(production["line_speed_upm"], errors="coerce")
    by_station = production.groupby("station_id", sort=True).agg(
        units_produced=("units_produced", "sum"),
        defect_count=("defect_count", "sum"),
        average_line_speed_upm=("line_speed_upm", "mean"),
    ).reset_index()
    total_units = int(production["units_produced"].sum())
    total_defects = int(production["defect_count"].sum())
    return {
        "total_units_produced": total_units,
        "total_defects": total_defects,
        "defect_rate_pct": round(total_defects / total_units * 100, 2) if total_units else 0,
        "average_line_speed_upm": round(float(production["line_speed_upm"].mean()), 1),
        "stations": json_records(by_station),
    }


def load_operations_monitoring() -> Dict[str, Any]:
    production = pd.read_csv(os.path.join(DATA_DIR, "production_shift_logs.csv"))
    sensors = pd.read_csv(os.path.join(DATA_DIR, "machine_sensor_readings.csv"))
    tools = pd.read_csv(os.path.join(DATA_DIR, "tool_machine_health.csv"))
    torque = pd.read_csv(os.path.join(DATA_DIR, "torque_cycle_logs.csv"))
    inspections = pd.read_csv(os.path.join(DATA_DIR, "part_inspection_defect_logs.csv"))

    for frame in (production, sensors, tools, torque):
        frame["event_timestamp"] = pd.to_datetime(frame["event_timestamp"])
        frame["date"] = frame["event_timestamp"].dt.strftime("%Y-%m-%d")
        frame["shift_id"] = np.select(
            [
                (frame["event_timestamp"].dt.hour >= 6) & (frame["event_timestamp"].dt.hour < 14),
                (frame["event_timestamp"].dt.hour >= 14) & (frame["event_timestamp"].dt.hour < 22),
            ],
            ["SHIFT_1", "SHIFT_2"],
            default="SHIFT_3",
        )

    latest_date = str(production["date"].max())
    latest_production = production[production["date"] == latest_date]
    production_by_station = latest_production.groupby(
        ["shift_id", "station_id"], sort=True
    ).agg(
        units_produced=("units_produced", "sum"),
        defect_count=("defect_count", "sum"),
        average_line_speed_upm=("line_speed_upm", "mean"),
        operator_tenure_yrs=("operator_tenure_yrs", "mean"),
    ).reset_index()
    production_by_station["defect_rate_pct"] = (
        production_by_station["defect_count"]
        / production_by_station["units_produced"].replace(0, np.nan)
        * 100
    ).fillna(0).round(2)

    latest_sensors = sensors[sensors["date"] == latest_date]
    sample_intervals = (
        sensors.sort_values(["station_id", "event_timestamp"])
        .groupby("station_id")["event_timestamp"]
        .diff()
        .dt.total_seconds()
    )
    sample_interval_seconds = float(sample_intervals[sample_intervals > 0].median())
    machine_by_station = latest_sensors.groupby("station_id", sort=True).agg(
        telemetry_samples=("machine_state", "size"),
        running_samples=("machine_state", lambda states: int((states == "RUNNING").sum())),
        idle_samples=("machine_state", lambda states: int((states == "IDLE").sum())),
        faulted_samples=("machine_state", lambda states: int((states == "FAULTED").sum())),
        mean_spindle_temp_c=("spindle_temp_c", "mean"),
        mean_system_pressure_psi=("system_pressure_psi", "mean"),
        mean_vibration_rms_g=("vibration_rms_g", "mean"),
    ).reset_index()
    machine_by_station["running_telemetry_pct"] = (
        machine_by_station["running_samples"] / machine_by_station["telemetry_samples"] * 100
    ).round(2)
    machine_by_station["estimated_faulted_hours"] = (
        machine_by_station["faulted_samples"] * sample_interval_seconds / 3600
    ).round(2)
    machine_by_station["estimated_idle_hours"] = (
        machine_by_station["idle_samples"] * sample_interval_seconds / 3600
    ).round(2)

    latest_torque = torque[torque["date"] == latest_date]
    material = load_material_batches()
    inspected_cycles = inspections.merge(
        latest_torque[["fastener_cycle_id", "station_id", "date", "shift_id"]],
        on="fastener_cycle_id",
        how="inner",
    ).merge(
        material[
            [
                "material_batch_id",
                "supplier_id",
                "material_hardness_hv",
                "material_risk_status",
            ]
        ],
        on="material_batch_id",
        how="left",
    )
    inspection_by_station = inspected_cycles.groupby(
        "station_id", sort=True
    ).agg(
        inspected_parts=("part_serial_no", "count"),
        rejected_parts=("defect_flag", "sum"),
        observed_failure_modes=("failure_mode", lambda modes: int((modes != "NONE").sum())),
    ).reset_index()
    inspection_by_station["inspection_rejection_rate_pct"] = (
        inspection_by_station["rejected_parts"]
        / inspection_by_station["inspected_parts"].replace(0, np.nan)
        * 100
    ).fillna(0).round(2)
    supplier_issue_rows = inspected_cycles.groupby(
        ["station_id", "supplier_id"], sort=True
    ).agg(
        supplier_inspected_parts=("part_serial_no", "count"),
        supplier_rejected_parts=("defect_flag", "sum"),
        parts_from_marginal_high_batches=(
            "material_risk_status",
            lambda statuses: int((statuses == "MARGINAL_HIGH").sum()),
        ),
        average_hardness_hv=("material_hardness_hv", "mean"),
    ).reset_index()
    supplier_issue_rows["supplier_rejection_rate_pct"] = (
        supplier_issue_rows["supplier_rejected_parts"]
        / supplier_issue_rows["supplier_inspected_parts"].replace(0, np.nan)
        * 100
    ).fillna(0).round(2)
    supplier_issue_rows = supplier_issue_rows.sort_values(
        ["station_id", "supplier_rejected_parts"], ascending=[True, False]
    ).groupby("station_id", sort=True).head(3)
    top_supplier_issues = {
        station_id: group[
            [
                "supplier_id",
                "supplier_inspected_parts",
                "supplier_rejected_parts",
                "supplier_rejection_rate_pct",
                "parts_from_marginal_high_batches",
                "average_hardness_hv",
            ]
        ].to_dict("records")
        for station_id, group in supplier_issue_rows.groupby("station_id", sort=True)
    }

    torque["date"] = torque["date"].astype(str)
    latest_torque = torque[torque["date"] == latest_date]
    tools["date"] = tools["date"].astype(str)
    latest_tools = tools[tools["date"] == latest_date]
    process_features = latest_torque.merge(
        latest_sensors.drop(columns=["shift_id"]),
        on=["event_timestamp", "station_id", "date"],
        how="left",
        suffixes=("", "_sensor"),
    ).merge(
        latest_tools[["event_timestamp", "station_id", "tool_wear_index", "hours_since_service"]],
        on=["event_timestamp", "station_id"],
        how="left",
    )
    merged_production = latest_production.drop(columns=["date"]).copy()
    process_features = process_features.merge(
        merged_production[["event_timestamp", "station_id", "operator_tenure_yrs"]],
        on=["event_timestamp", "station_id"],
        how="left",
    )
    inspections_latest = inspections.merge(
        latest_torque[["fastener_cycle_id"]],
        on="fastener_cycle_id",
        how="inner",
    )
    process_features = process_features.merge(
        inspections_latest[["fastener_cycle_id", "material_batch_id"]],
        on="fastener_cycle_id",
        how="left",
    ).merge(
        material[["material_batch_id", "material_hardness_hv", "tensile_strength_mpa"]],
        on="material_batch_id",
        how="left",
    )
    process_features["material_hardness_hv"] = process_features["material_hardness_hv"].fillna(
        float(material["material_hardness_hv"].median())
    )
    process_features["tensile_strength_mpa"] = process_features["tensile_strength_mpa"].fillna(
        float(material["tensile_strength_mpa"].median())
    )
    process_features["hours_since_service"] = process_features["hours_since_service"].fillna(0)
    classifier: FailureModeClassifier = models.get("failure_mode")
    if not classifier or not classifier.feature_names:
        raise RuntimeError("Failure mode classifier artifact is not loaded.")
    from src.features import FeatureEngineering

    feature_rows = FeatureEngineering().add_physical_domain_features(process_features)
    feature_rows["torque_status_nok"] = (feature_rows["torque_status"] != "OK").astype(float)
    feature_rows["torque_status_high"] = (feature_rows["torque_status"] == "NOK_HIGH").astype(float)
    feature_rows["torque_status_low"] = (feature_rows["torque_status"] == "NOK_LOW").astype(float)
    for station_id in ("ST-10", "ST-20", "ST-30", "ST-40"):
        feature_rows[f"st_{station_id}"] = (feature_rows["station_id"] == station_id).astype(float)
    feature_rows = feature_rows.reindex(columns=classifier.feature_names, fill_value=0.0)
    predicted = classifier.model.predict_proba(feature_rows)
    process_features["predicted_failure_mode"] = classifier.label_encoder.inverse_transform(
        np.argmax(predicted, axis=1)
    )
    process_features["failure_mode_confidence"] = predicted.max(axis=1)
    predicted_modes = process_features.groupby(
        ["station_id", "predicted_failure_mode"], sort=True
    ).size().rename("predicted_cycle_count").reset_index()
    predicted_modes = predicted_modes.sort_values(
        ["station_id", "predicted_cycle_count"], ascending=[True, False]
    ).groupby("station_id", sort=True).head(3)
    top_modes_by_station = {
        station_id: group[["predicted_failure_mode", "predicted_cycle_count"]].to_dict("records")
        for station_id, group in predicted_modes.groupby("station_id", sort=True)
    }

    station_monitoring = production_by_station.merge(
        machine_by_station, on="station_id", how="left"
    ).merge(
        inspection_by_station, on="station_id", how="left"
    )
    station_monitoring["top_ml_failure_modes"] = station_monitoring["station_id"].map(
        top_modes_by_station
    ).apply(lambda value: value if isinstance(value, list) else [])
    station_monitoring["top_supplier_issues"] = station_monitoring["station_id"].map(
        top_supplier_issues
    ).apply(lambda value: value if isinstance(value, list) else [])
    for column in ("inspected_parts", "rejected_parts", "observed_failure_modes"):
        station_monitoring[column] = station_monitoring[column].fillna(0).astype(int)
    station_monitoring["inspection_rejection_rate_pct"] = station_monitoring[
        "inspection_rejection_rate_pct"
    ].fillna(0)

    latest_summary = {
        "date": latest_date,
        "total_units_produced": int(latest_production["units_produced"].sum()),
        "total_logged_defects": int(latest_production["defect_count"].sum()),
        "logged_defect_rate_pct": round(
            float(latest_production["defect_count"].sum())
            / max(1, int(latest_production["units_produced"].sum()))
            * 100,
            2,
        ),
        "inspected_parts": int(inspection_by_station["inspected_parts"].sum()),
        "rejected_parts": int(inspection_by_station["rejected_parts"].sum()),
        "inspection_rejection_rate_pct": round(
            float(inspection_by_station["rejected_parts"].sum())
            / max(1, int(inspection_by_station["inspected_parts"].sum()))
            * 100,
            2,
        ),
        "faulted_telemetry_samples": int(machine_by_station["faulted_samples"].sum()),
        "estimated_faulted_hours": round(
            float(machine_by_station["estimated_faulted_hours"].sum()), 2
        ),
        "telemetry_sample_interval_seconds": sample_interval_seconds,
    }
    return {
        "as_of_date": latest_date,
        "data_basis": (
            "Recorded CSV history; faulted and idle hours estimate elapsed time from telemetry "
            "sample counts and median sample interval. This is not a live equipment feed."
        ),
        "summary": latest_summary,
        "stations": json_records(station_monitoring),
    }


@app.get("/api/tier1/operations/monitor")
def get_operations_monitor():
    try:
        return load_operations_monitoring()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not build operations monitoring summary: {type(exc).__name__}: {exc}",
        ) from exc


@app.get("/api/tier1/finished-products")
def list_finished_products(
    search: Optional[str] = Query(default=None, min_length=1, max_length=80),
    limit: int = Query(default=50, ge=1, le=100),
):
    records = load_finished_product_records()
    if search:
        records = records[records["part_serial_no"].str.contains(search, case=False, na=False)]
    else:
        records = records.sort_values(
            ["defect_flag", "defect_severity", "material_hardness_hv"],
            ascending=[False, False, False],
        )
    matches = len(records)
    return {
        "match_count": matches,
        "products": json_records(records.head(limit)),
    }


@app.get("/api/tier1/finished-products/{part_serial_no}")
def get_finished_product(part_serial_no: str):
    records = load_finished_product_records()
    product = records[records["part_serial_no"] == part_serial_no]
    if product.empty:
        raise HTTPException(status_code=404, detail=f"Finished product '{part_serial_no}' not found.")
    return json_records(product.head(1))[0]


def _failure_mode_feature_row(input_data: SensorTelemetryInput) -> pd.DataFrame:
    return pd.DataFrame([{
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


@app.post("/api/tier1/predict_failure_mode", tags=["Tier 1: Manufacturing"])
def predict_failure_mode(input_data: SensorTelemetryInput):
    classifier: FailureModeClassifier = models.get("failure_mode")
    if not classifier or not classifier.feature_names:
        raise HTTPException(status_code=503, detail="Failure mode classifier artifact is not loaded.")
    predictions = classifier.predict_with_contributors(_failure_mode_feature_row(input_data), top_k=5)
    return {"station_id": input_data.station_id, **predictions[0]}


@app.post("/api/tier1/simulate")
def simulate_tier1(input_data: SensorTelemetryInput):
    """
    Simulates real-time cycle telemetry:
    Predicts Failure Mode, computes SHAP physical contributors, and recommends prescriptive offsets.
    """
    fm_clf: FailureModeClassifier = models.get("failure_mode")
    prescriptive: PrescriptiveActionEngine = models.get("prescriptive")

    if not fm_clf or not fm_clf.feature_names:
        raise HTTPException(status_code=503, detail="Failure mode classifier artifact is not loaded.")
    if not prescriptive:
        raise HTTPException(status_code=503, detail="Prescriptive action engine is not initialized.")

    row = _failure_mode_feature_row(input_data)

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


@app.post("/api/tier1/recommend_action", tags=["Tier 1: Manufacturing"])
def recommend_action(input_data: PrescriptiveActionInput):
    engine: PrescriptiveActionEngine = models.get("prescriptive")
    if not engine:
        raise HTTPException(status_code=503, detail="Prescriptive action engine is not initialized.")
    telemetry = input_data.current_telemetry
    return engine.recommend_action(
        predicted_failure_mode=input_data.predicted_failure_mode,
        current_telemetry={
            "spindle_temp_c": telemetry.spindle_temp_c,
            "peak_torque_nm": telemetry.peak_torque_nm,
            "shutoff_angle_deg": telemetry.shutoff_angle_deg,
            "vibration_rms_g": telemetry.vibration_rms_g,
            "tool_wear_index": telemetry.tool_wear_index,
            "system_pressure_psi": telemetry.system_pressure_psi,
        },
        top_contributors=input_data.top_contributors,
    )


def _material_feature_row(input_data: MaterialBatchInput) -> pd.DataFrame:
    if input_data.lower_spec_limit >= input_data.upper_spec_limit:
        raise HTTPException(status_code=422, detail="lower_spec_limit must be less than upper_spec_limit.")
    return pd.DataFrame([{
        "material_hardness_hv": input_data.material_hardness_hv,
        "tensile_strength_mpa": input_data.tensile_strength_mpa,
        "lower_spec_limit": input_data.lower_spec_limit,
        "upper_spec_limit": input_data.upper_spec_limit,
        "supplier_id": input_data.supplier_id,
        "supplier_VENDOR_A": 1.0 if input_data.supplier_id == "VENDOR_A" else 0.0,
        "supplier_VENDOR_B": 1.0 if input_data.supplier_id == "VENDOR_B" else 0.0,
        "supplier_VENDOR_C": 1.0 if input_data.supplier_id == "VENDOR_C" else 0.0,
    }])


@app.post("/api/tier2/classify_material", tags=["Tier 2: Raw Materials"])
def classify_material(input_data: MaterialBatchInput):
    classifier: MarginalBatchRiskClassifier = models.get("batch_risk")
    if not classifier or not classifier.feature_names:
        raise HTTPException(status_code=503, detail="Marginal batch risk classifier artifact is not loaded.")
    return classifier.predict_batch_risk(_material_feature_row(input_data))


@app.post("/api/tier2/forecast_impact", tags=["Tier 2: Raw Materials"])
def forecast_material_impact(input_data: MaterialBatchInput):
    forecaster: DownstreamImpactForecaster = models.get("impact_forecaster")
    if not forecaster or not forecaster.features:
        raise HTTPException(status_code=503, detail="Downstream impact forecaster artifact is not loaded.")
    return forecaster.forecast(
        hardness=input_data.material_hardness_hv,
        tensile=input_data.tensile_strength_mpa,
        supplier_id=input_data.supplier_id,
        batch_size_units=input_data.batch_size_units,
    )


@app.post("/api/tier2/compensate", tags=["Tier 2: Raw Materials"])
def compensate_material(input_data: MaterialBatchInput):
    compensator: AdaptiveParameterCompensator = models.get("compensator")
    if not compensator:
        raise HTTPException(status_code=503, detail="Adaptive parameter compensator is not initialized.")
    return compensator.compute_offsets(
        batch_hardness_hv=input_data.material_hardness_hv,
        batch_tensile_mpa=input_data.tensile_strength_mpa,
    )


@app.post("/api/tier2/evaluate_material")
def evaluate_material(input_data: MaterialBatchInput):
    batch_clf: MarginalBatchRiskClassifier = models.get("batch_risk")
    forecaster: DownstreamImpactForecaster = models.get("impact_forecaster")
    compensator: AdaptiveParameterCompensator = models.get("compensator")

    if not batch_clf or not batch_clf.feature_names:
        raise HTTPException(status_code=503, detail="Marginal batch risk classifier artifact is not loaded.")
    if not forecaster or not forecaster.features:
        raise HTTPException(status_code=503, detail="Downstream impact forecaster artifact is not loaded.")
    if not compensator:
        raise HTTPException(status_code=503, detail="Adaptive parameter compensator is not initialized.")

    row = _material_feature_row(input_data)

    risk = batch_clf.predict_batch_risk(row)
    forecast = forecaster.forecast(
        hardness=input_data.material_hardness_hv,
        tensile=input_data.tensile_strength_mpa,
        supplier_id=input_data.supplier_id,
        batch_size_units=input_data.batch_size_units,
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
    supplier_id = supplier_id.upper()
    if supplier_id not in {"VENDOR_A", "VENDOR_B", "VENDOR_C"}:
        raise HTTPException(status_code=404, detail=f"Supplier '{supplier_id}' was not found.")
    gen: SCARGenerator = models.get("scar_generator")
    if gen:
        return gen.generate_scar(supplier_id)
    raise HTTPException(status_code=503, detail="SCAR generator is not initialized.")


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
