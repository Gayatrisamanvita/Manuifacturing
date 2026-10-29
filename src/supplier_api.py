"""
Supplier & Vendor Intelligence API Microservice (FactorySmart AI™).

A dedicated, enterprise-grade REST API enabling external vendors, suppliers,
ERP/MES platforms, and partner quality teams to run machine learning models with
full API Key Authentication & Access Governance:
- Cryptographic API Key Authentication (X-API-Key, Bearer Token, or ?api_key= query)
- Key Management (Create, List, Revoke, Inspect)
- Incoming/Shipped Material Lot Risk Evaluation (Marginal Batch Risk Classifier)
- Downstream Assembly Defect & Scrap Impact Forecasting (GradientBoosting Regressor)
- Closed-Loop Adaptive Machine Parameter Recipe Compensation (Physics-Informed Offset Engine)
- Comprehensive Pre-Shipment Lot Certification (All 3 Models Unified)
- Supplier 30-Day Quality Trends & CAPA Recurrence Analysis
- Product Catalog Lead-Time & Delivery Delay Risk Prediction
- Inventory Safety Stock Statistical Optimization
- End-to-End Warranty Forensic Traceability (Genealogy DAG)
- 216-230 HV Tolerance Leakage Gap Verification
- Automated 8D Supplier Corrective Action Request (SCAR) Generation with $5.44M Recovery Ledger
"""

import os
import sys
import json
from datetime import datetime
from typing import Dict, Any, List, Literal, Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, model_validator

# Ensure project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, APIRouter, HTTPException, Security, Depends, status, Query
from fastapi.middleware.cors import CORSMiddleware

from src.auth import (
    get_current_client, require_admin, key_manager,
    api_key_header_scheme, http_bearer_scheme
)
from src.models.tier2_materials import (
    MarginalBatchRiskClassifier, DownstreamImpactForecaster, AdaptiveParameterCompensator
)
from src.models.tier3_supplier import (
    SupplierQualityTrendAnalyzer, WarrantyTraceabilityEngine,
    ToleranceLeakageAnalyzer, SCARGenerator
)
from src.data_loader import DataLoader

# -----------------------------------------------------------------------------
# 1. API INITIALIZATION & METADATA
# -----------------------------------------------------------------------------

API_TITLE = "FactorySmart AI™ — Supplier & Vendor Intelligence API"
API_DESCRIPTION = """
### Enterprise API for Supplier Quality, Ingot Metallurgy & Vendor Risk

This API allows external suppliers, vendor portals, and ERP/MES integrations to directly consume
FactorySmart AI's trained machine learning models and analytical engines:

🔐 **API Key Required**: All endpoints require authentication via `X-API-Key` header, `Authorization: Bearer <key>`, or `?api_key=<key>` query parameter.
- **Demo Partner Key**: `fs-demo-partner-quality-2026`
- **Vendor B Key**: `fs-key-vendor-b-3e7a9d0f`
- **Admin Master Key**: `fs-admin-master-2026-99x` (used for key creation & management)

1. **Incoming / Shipped Lot Quality**: Predict downstream defect risk on incoming alloy/fastener lots.
2. **Tolerance Leakage Verification**: Detect whether a material lot falls in the **216.0 – 230.0 HV leakage zone**.
3. **Downstream Scrap & Severity Forecasting**: Estimate scrap count and defect severity before batch processing.
4. **Adaptive Recipe Compensation**: Obtain dynamic torque, angle, and feed offsets to prevent scrap on CNC/tightening spindles.
5. **Vendor Performance & Trends**: Access 30-day quality trajectories, CAPA recurrence scores, and product catalogs.
6. **Genealogy & Warranty Traceability**: Forensically trace field failures back to supplier heat lots.
7. **Automated 8D SCAR & Financial Recovery**: Generate legally grounded corrective action requests.
"""

supplier_router = APIRouter(prefix="/api/v1/supplier", tags=["Supplier & Vendor Intelligence"])

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# 2. GLOBAL MODEL SINGLETONS
# -----------------------------------------------------------------------------
class ModelRegistry:
    batch_classifier: Optional[MarginalBatchRiskClassifier] = None
    impact_forecaster: Optional[DownstreamImpactForecaster] = None
    adaptive_compensator: Optional[AdaptiveParameterCompensator] = None
    trend_analyzer: Optional[SupplierQualityTrendAnalyzer] = None
    trace_engine: Optional[WarrantyTraceabilityEngine] = None
    gap_analyzer: Optional[ToleranceLeakageAnalyzer] = None
    scar_generator: Optional[SCARGenerator] = None
    products_catalog_df: Optional[pd.DataFrame] = None
    batch_master_df: Optional[pd.DataFrame] = None

registry = ModelRegistry()

def init_models(data_dir: str = "data"):
    print("[Supplier API] Initializing ML models and analytical engines...")
    # Tier 2 - Marginal Batch Classifier
    clf = MarginalBatchRiskClassifier()
    model_path = os.path.join(PROJECT_ROOT, "models", "marginal_batch_classifier.pkl")
    if os.path.exists(model_path):
        clf.load(model_path)
    else:
        dl = DataLoader(data_dir=os.path.join(PROJECT_ROOT, data_dir))
        clf.fit(dl.get_material_batch_summary())
    registry.batch_classifier = clf

    # Tier 2 - Downstream Forecaster & Compensator
    forecaster = DownstreamImpactForecaster()
    forecaster_path = os.path.join(PROJECT_ROOT, "models", "downstream_impact_forecaster.pkl")
    if os.path.exists(forecaster_path):
        try:
            forecaster.load(forecaster_path)
        except Exception as exc:
            print(f"[Supplier API] Could not load downstream forecaster ({exc}); retraining from source data.")
            dl = DataLoader(data_dir=os.path.join(PROJECT_ROOT, data_dir))
            forecaster.fit(dl.get_material_batch_summary())
            forecaster.save(forecaster_path)
    else:
        dl = DataLoader(data_dir=os.path.join(PROJECT_ROOT, data_dir))
        batch_summary = dl.get_material_batch_summary()
        forecaster.fit(batch_summary)
    if not forecaster.reference_batch_size:
        dl = DataLoader(data_dir=os.path.join(PROJECT_ROOT, data_dir))
        batch_summary = dl.get_material_batch_summary()
        forecaster.reference_batch_size = float(batch_summary["total_parts"].mean())
    registry.impact_forecaster = forecaster

    compensator = AdaptiveParameterCompensator()
    compensator_path = os.path.join(PROJECT_ROOT, "models", "adaptive_parameter_compensator.pkl")
    if os.path.exists(compensator_path):
        compensator.load(compensator_path)
    registry.adaptive_compensator = compensator

    # Tier 3 - Supplier Analytics
    registry.trend_analyzer = SupplierQualityTrendAnalyzer(data_dir=os.path.join(PROJECT_ROOT, data_dir))
    registry.trace_engine = WarrantyTraceabilityEngine(data_dir=os.path.join(PROJECT_ROOT, data_dir))
    registry.gap_analyzer = ToleranceLeakageAnalyzer(data_dir=os.path.join(PROJECT_ROOT, data_dir))
    registry.scar_generator = SCARGenerator(data_dir=os.path.join(PROJECT_ROOT, data_dir))

    # Static / Catalog caches
    catalog_path = os.path.join(PROJECT_ROOT, data_dir, "supplier_products_catalog.csv")
    if os.path.exists(catalog_path):
        registry.products_catalog_df = pd.read_csv(catalog_path)
    
    batch_path = os.path.join(PROJECT_ROOT, data_dir, "material_batch_master.csv")
    if os.path.exists(batch_path):
        registry.batch_master_df = pd.read_csv(batch_path)

    print("[Supplier API] All models loaded and ready for external traffic.")

@app.on_event("startup")
def startup_event():
    init_models()

# -----------------------------------------------------------------------------
# 3. PYDANTIC REQUEST & RESPONSE SCHEMAS
# -----------------------------------------------------------------------------

class CreateKeyRequest(BaseModel):
    client_name: str = Field(..., description="Organization or team name", example="Bosch Precision Fasteners Quality Team")
    supplier_id: str = Field("VENDOR_B", description="Associated supplier ID (e.g. VENDOR_A, VENDOR_B, VENDOR_C, or ALL)", example="VENDOR_B")
    role: str = Field("supplier", description="Role: 'supplier', 'partner', or 'admin'", example="supplier")
    permissions: Optional[List[str]] = Field(None, description="Allowed model permissions", example=["evaluate_lot", "forecast_impact", "adaptive_recipe"])

class LotEvaluationRequest(BaseModel):
    material_hardness_hv: float = Field(..., ge=100.0, le=400.0, description="Vickers Hardness (HV) measured on incoming lot sample", example=223.5)
    tensile_strength_mpa: float = Field(..., ge=200.0, le=1500.0, description="Ultimate Tensile Strength Rm (MPa)", example=675.0)
    lower_spec_limit: float = Field(180.0, description="Contractual Lower Specification Limit (HV)", example=180.0)
    upper_spec_limit: float = Field(230.0, description="Contractual Upper Specification Limit (HV)", example=230.0)
    supplier_id: Literal["VENDOR_A", "VENDOR_B", "VENDOR_C"] = Field("VENDOR_B", description="Supplier identifier (e.g. VENDOR_A, VENDOR_B, VENDOR_C)", example="VENDOR_B")
    batch_id: Optional[str] = Field("BATCH-INBOUND-001", description="Vendor heat/lot identifier", example="LOT-2026-X49")
    material_grade: Optional[str] = Field("AISI-4140", description="Alloy specification grade", example="AISI-4140")

    @model_validator(mode="after")
    def validate_spec_limits(self):
        if self.lower_spec_limit >= self.upper_spec_limit:
            raise ValueError("lower_spec_limit must be less than upper_spec_limit.")
        return self

class LotEvaluationResponse(BaseModel):
    batch_id: str
    supplier_id: str
    material_hardness_hv: float
    tensile_strength_mpa: float
    downstream_risk_probability: float
    batch_risk_tier: str
    passed_nominal_spec: bool
    is_in_tolerance_leakage_zone: bool
    tolerance_leakage_warning: Optional[str]
    recommendation: str

class DownstreamForecastRequest(BaseModel):
    material_hardness_hv: float = Field(..., example=224.0)
    tensile_strength_mpa: float = Field(..., example=680.0)
    supplier_id: Literal["VENDOR_A", "VENDOR_B", "VENDOR_C"] = Field("VENDOR_B", example="VENDOR_B")
    batch_size_units: int = Field(5000, ge=1, description="Number of units in batch shipment", example=5000)

class DownstreamForecastResponse(BaseModel):
    expected_defect_severity: float
    expected_scrap_count: int
    projected_scrap_rate_pct: float
    likely_downstream_failure_mode: str
    physical_root_cause: str
    failure_mode_method: str
    projected_scrap_cost_usd: float

class AdaptiveRecipeRequest(BaseModel):
    material_hardness_hv: float = Field(..., example=224.5)
    batch_tensile_mpa: float = Field(680.0, example=680.0)

class AdaptiveRecipeResponse(BaseModel):
    recipe_status: str
    target_torque_offset_nm: float
    shutoff_angle_offset_deg: float
    spindle_speed_reduction_pct: float
    feed_rate_offset_pct: float
    lubricant_boost_mode: str
    expected_scrap_reduction_pct: float
    engineering_justification: str

class ComprehensiveLotQCRequest(BaseModel):
    supplier_id: Literal["VENDOR_A", "VENDOR_B", "VENDOR_C"] = Field("VENDOR_B", example="VENDOR_B")
    batch_id: str = Field("LOT-2026-B88", example="LOT-2026-B88")
    material_grade: str = Field("AISI-4140", example="AISI-4140")
    material_hardness_hv: float = Field(222.5, example=222.5)
    tensile_strength_mpa: float = Field(670.0, example=670.0)
    batch_size_units: int = Field(8000, ge=1, example=8000)
    lower_spec_limit: float = Field(180.0, example=180.0)
    upper_spec_limit: float = Field(230.0, example=230.0)

    @model_validator(mode="after")
    def validate_spec_limits(self):
        if self.lower_spec_limit >= self.upper_spec_limit:
            raise ValueError("lower_spec_limit must be less than upper_spec_limit.")
        return self

class ComprehensiveLotQCResponse(BaseModel):
    timestamp: str
    batch_id: str
    supplier_id: str
    evaluation_verdict: str
    risk_assessment: LotEvaluationResponse
    downstream_impact: DownstreamForecastResponse
    adaptive_compensation: AdaptiveRecipeResponse
    actionable_shipping_advice: str

class LeadTimePredictionRequest(BaseModel):
    supplier_id: Literal["VENDOR_A", "VENDOR_B", "VENDOR_C"] = Field("VENDOR_B", example="VENDOR_B")
    sku: Optional[str] = Field("STL-2001", example="STL-2001")
    category: str = Field("Structural Alloy Steel", example="Structural Alloy Steel")
    target_lead_days: int = Field(18, ge=1, example=18)
    monthly_vol_units: int = Field(18500, ge=0, example=18500)

class LeadTimePredictionResponse(BaseModel):
    supplier_id: str
    sku: Optional[str]
    category: str
    target_lead_days: int
    predicted_actual_lead_days: int
    predicted_delay_days: int
    delivery_status: str
    risk_classification: str
    recommended_procurement_action: str
    estimation_method: str

class SafetyStockRequest(BaseModel):
    daily_consumption_mean: float = Field(..., ge=1.0, example=350.0)
    daily_consumption_std: float = Field(..., ge=0.0, example=45.0)
    lead_time_days_mean: float = Field(..., ge=1.0, example=18.0)
    lead_time_days_std: float = Field(..., ge=0.0, example=4.5)
    service_level_target_pct: float = Field(97.5, ge=80.0, le=99.9, example=97.5)

class SafetyStockResponse(BaseModel):
    service_level_z_score: float
    recommended_safety_stock_units: int
    buffer_coverage_days: float
    reorder_point_units: int
    formula_basis: str

# -----------------------------------------------------------------------------
# 4. API KEY MANAGEMENT ENDPOINTS
# -----------------------------------------------------------------------------

@supplier_router.post("/auth/create-key", summary="Admin: Generate New API Key")
def create_new_api_key(request: CreateKeyRequest, admin: Dict[str, Any] = Depends(require_admin)):
    """
    Creates and registers a new secure API key for a supplier, vendor, or partner team.
    Requires Admin privileges (Key: 'fs-admin-master-2026-99x').
    """
    new_key = key_manager.create_key(
        client_name=request.client_name,
        supplier_id=request.supplier_id,
        role=request.role,
        permissions=request.permissions
    )
    return {
        "status": "SUCCESS",
        "message": f"API Key successfully created for '{request.client_name}'.",
        "api_key_details": new_key
    }

@supplier_router.get("/auth/keys", summary="Admin: List All Registered API Keys")
def list_registered_keys(admin: Dict[str, Any] = Depends(require_admin)):
    """
    Returns all registered API keys, status, and usage statistics.
    Requires Admin privileges.
    """
    keys = key_manager.list_keys()
    return {
        "total_keys": len(keys),
        "keys": keys
    }

@supplier_router.delete("/auth/revoke-key/{key_or_id}", summary="Admin: Revoke an API Key")
def revoke_api_key(key_or_id: str, admin: Dict[str, Any] = Depends(require_admin)):
    """
    Revokes an active API key, blocking future model access.
    Requires Admin privileges.
    """
    success = key_manager.revoke_key(key_or_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"API Key or Key ID '{key_or_id}' not found.")
    return {
        "status": "REVOKED",
        "message": f"API Key '{key_or_id}' has been permanently revoked."
    }

@supplier_router.get("/auth/whoami", summary="Inspect Authenticated API Key Profile")
def who_am_i(client: Dict[str, Any] = Depends(get_current_client)):
    """
    Returns the authenticated client identity, associated supplier, role, and usage metrics.
    """
    return {
        "authenticated": True,
        "key_id": client.get("key_id"),
        "client_name": client.get("client_name"),
        "supplier_id": client.get("supplier_id"),
        "role": client.get("role"),
        "permissions": client.get("permissions"),
        "request_count": client.get("request_count", 0),
        "created_at": client.get("created_at"),
        "last_used_at": client.get("last_used_at")
    }

# -----------------------------------------------------------------------------
# 5. GENERAL SERVICE HEALTH
# -----------------------------------------------------------------------------

@supplier_router.get("/health", summary="Microservice Health & Model Readiness")
def check_health():
    """Returns the operational readiness of all Supplier & Vendor ML engines (open health probe)."""
    models_ready = {
        "batch_risk_classifier": bool(
            registry.batch_classifier and registry.batch_classifier.feature_names
        ),
        "downstream_impact_forecaster": bool(
            registry.impact_forecaster and registry.impact_forecaster.features
        ),
        "adaptive_parameter_compensator": registry.adaptive_compensator is not None,
        "trend_analyzer": registry.trend_analyzer is not None,
        "warranty_trace_engine": registry.trace_engine is not None,
        "tolerance_gap_analyzer": registry.gap_analyzer is not None,
        "scar_generator": registry.scar_generator is not None,
    }
    return {
        "status": "HEALTHY" if all(models_ready.values()) else "DEGRADED",
        "service": "FactorySmart AI - Supplier Intelligence API",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "models_ready": models_ready,
        "loaded_catalogs": {
            "products_catalog": len(registry.products_catalog_df) if registry.products_catalog_df is not None else 0,
            "material_batches_master": len(registry.batch_master_df) if registry.batch_master_df is not None else 0,
        },
        "authentication_enabled": True,
        "active_keys_registered": len(key_manager.list_keys())
    }

# -----------------------------------------------------------------------------
# 6. PROTECTED MODEL INFERENCE ENDPOINTS (REQUIRES API KEY)
# -----------------------------------------------------------------------------

@supplier_router.post("/evaluate-lot", response_model=LotEvaluationResponse, summary="Predict Downstream Defect Risk on Material Lot (Q5)")
def evaluate_lot(request: LotEvaluationRequest, client: Dict[str, Any] = Depends(get_current_client)):
    """
    Evaluates an incoming or vendor-dispatched material lot against historical quality models.
    Identifies if a batch that nominally passes blueprint specifications ($180 - 230\\text{ HV}$)
    actually poses elevated downstream assembly defect risk (e.g. within the $216 - 230\\text{ HV}$ leakage zone).
    """
    if not registry.batch_classifier:
        raise HTTPException(status_code=500, detail="Batch risk classifier model not loaded.")

    row = pd.DataFrame([{
        "material_hardness_hv": request.material_hardness_hv,
        "tensile_strength_mpa": request.tensile_strength_mpa,
        "lower_spec_limit": request.lower_spec_limit,
        "upper_spec_limit": request.upper_spec_limit,
        "supplier_id": request.supplier_id,
        "supplier_VENDOR_A": 1.0 if request.supplier_id == "VENDOR_A" else 0.0,
        "supplier_VENDOR_B": 1.0 if request.supplier_id == "VENDOR_B" else 0.0,
        "supplier_VENDOR_C": 1.0 if request.supplier_id == "VENDOR_C" else 0.0,
    }])

    res = registry.batch_classifier.predict_batch_risk(row)
    risk_prob = res["downstream_risk_probability"]

    is_leakage = 216.0 <= request.material_hardness_hv <= 230.0
    passed_nominal = request.lower_spec_limit <= request.material_hardness_hv <= request.upper_spec_limit

    warning = None
    if is_leakage:
        warning = (
            f"CRITICAL WARNING: Batch hardness of {request.material_hardness_hv:.1f} HV falls squarely in the "
            f"216.0 - 230.0 HV Tolerance Leakage Zone. While it meets nominal USL (230 HV), historical "
            f"data proves this hardness band causes 78.2% of all 12-month field warranty claims."
        )

    return LotEvaluationResponse(
        batch_id=request.batch_id or "UNKNOWN",
        supplier_id=request.supplier_id,
        material_hardness_hv=request.material_hardness_hv,
        tensile_strength_mpa=request.tensile_strength_mpa,
        downstream_risk_probability=risk_prob,
        batch_risk_tier=res["batch_risk_tier"],
        passed_nominal_spec=passed_nominal,
        is_in_tolerance_leakage_zone=is_leakage,
        tolerance_leakage_warning=warning,
        recommendation=res["recommendation"] if not is_leakage else "MANDATE ADAPTIVE RECIPE COMPENSATION OR REJECT LOT AT DOCK"
    )

@supplier_router.post("/forecast-impact", response_model=DownstreamForecastResponse, summary="Forecast Downstream Assembly Defect & Scrap Impact (Q6)")
def forecast_impact(request: DownstreamForecastRequest, client: Dict[str, Any] = Depends(get_current_client)):
    """
    Simulates downstream manufacturing execution for a given material lot.
    Forecasts expected scrap count, defect severity (1-5), likely failure mode, and financial scrap cost.
    """
    if not registry.impact_forecaster:
        raise HTTPException(status_code=500, detail="Downstream impact forecaster not loaded.")

    res = registry.impact_forecaster.forecast(
        hardness=request.material_hardness_hv,
        tensile=request.tensile_strength_mpa,
        supplier_id=request.supplier_id,
        batch_size_units=request.batch_size_units,
    )

    scrap_units = res["expected_scrap_count"]
    scrap_rate = round((scrap_units / max(1, request.batch_size_units)) * 100.0, 2)
    scrap_cost = round(scrap_units * 55.0, 2)

    return DownstreamForecastResponse(
        expected_defect_severity=res["expected_defect_severity"],
        expected_scrap_count=scrap_units,
        projected_scrap_rate_pct=scrap_rate,
        likely_downstream_failure_mode=res["likely_downstream_failure_mode"],
        physical_root_cause=res["physical_root_cause"],
        failure_mode_method=res["failure_mode_method"],
        projected_scrap_cost_usd=scrap_cost
    )

@supplier_router.post("/adaptive-recipe", response_model=AdaptiveRecipeResponse, summary="Compute Closed-Loop Adaptive Recipe Offsets (Q7)")
def get_adaptive_recipe(request: AdaptiveRecipeRequest, client: Dict[str, Any] = Depends(get_current_client)):
    """
    Calculates precise, machine-executable parameter offsets (Torque, Angle, Speed, Coolant)
    to safely process marginal or borderline vendor material lots without generating scrap.
    """
    if not registry.adaptive_compensator:
        raise HTTPException(status_code=500, detail="Adaptive parameter compensator not loaded.")

    res = registry.adaptive_compensator.compute_offsets(
        batch_hardness_hv=request.material_hardness_hv,
        batch_tensile_mpa=request.batch_tensile_mpa
    )

    scrap_cut = 68.4 if request.material_hardness_hv >= 215.0 else (54.0 if request.material_hardness_hv <= 195.0 else 0.0)

    return AdaptiveRecipeResponse(
        recipe_status=res.get("status", "STANDARD_NOMINAL"),
        target_torque_offset_nm=res.get("torque_offset_nm", 0.0),
        shutoff_angle_offset_deg=res.get("angle_offset_deg", 0.0),
        spindle_speed_reduction_pct=abs(res.get("spindle_speed_offset_pct", 0.0)),
        feed_rate_offset_pct=res.get("feed_rate_offset_pct", 0.0),
        lubricant_boost_mode=res.get("coolant_lubrication_mode", "STANDARD_COOLANT"),
        expected_scrap_reduction_pct=scrap_cut,
        engineering_justification=res.get("engineering_guidance", "Nominal material properties; apply standard baseline recipe.")
    )

@supplier_router.post("/comprehensive-qc", response_model=ComprehensiveLotQCResponse, summary="Unified Pre-Shipment / Receiving QC Audit (Q5 + Q6 + Q7)")
@supplier_router.post("/certify-lot", response_model=ComprehensiveLotQCResponse, summary="Unified Pre-Shipment / Receiving QC Audit (Q5 + Q6 + Q7)")
def comprehensive_lot_qc(request: ComprehensiveLotQCRequest, client: Dict[str, Any] = Depends(get_current_client)):
    """
    Executes the entire Tier 2 model chain in a single unified API call.
    Recommended for Supplier Portals, SAP/Oracle ERP receiving workflows, and pre-shipment certifications.
    """
    # 1. Evaluate Lot Risk
    lot_req = LotEvaluationRequest(
        material_hardness_hv=request.material_hardness_hv,
        tensile_strength_mpa=request.tensile_strength_mpa,
        lower_spec_limit=request.lower_spec_limit,
        upper_spec_limit=request.upper_spec_limit,
        supplier_id=request.supplier_id,
        batch_id=request.batch_id,
        material_grade=request.material_grade
    )
    eval_res = evaluate_lot(lot_req, client)

    # 2. Forecast Impact
    forecast_req = DownstreamForecastRequest(
        material_hardness_hv=request.material_hardness_hv,
        tensile_strength_mpa=request.tensile_strength_mpa,
        supplier_id=request.supplier_id,
        batch_size_units=request.batch_size_units
    )
    forecast_res = forecast_impact(forecast_req, client)

    # 3. Adaptive Recipe
    recipe_req = AdaptiveRecipeRequest(
        material_hardness_hv=request.material_hardness_hv,
        batch_tensile_mpa=request.tensile_strength_mpa
    )
    recipe_res = get_adaptive_recipe(recipe_req, client)

    # Overall Verdict
    if not eval_res.passed_nominal_spec:
        verdict = "OUT_OF_SPEC_REJECT"
        advice = "Reject or quarantine the lot: material hardness is outside the receiving specification."
    elif eval_res.is_in_tolerance_leakage_zone:
        verdict = "MARGINAL_LEAKAGE_RISK_COMPENSATION_MANDATORY"
        advice = "Lot is accepted under concession ONLY if factory controllers apply the returned adaptive torque and angle recipe."
    elif eval_res.downstream_risk_probability > 0.5:
        verdict = "HIGH_DEFECT_RISK_HOLD"
        advice = "Quarantine lot for microstructural grain size analysis before releasing to production floor."
    else:
        verdict = "UNCONDITIONAL_PASS"
        advice = "Lot conforms to optimal metallurgical core band. Release directly to production line."

    return ComprehensiveLotQCResponse(
        timestamp=datetime.utcnow().isoformat() + "Z",
        batch_id=request.batch_id,
        supplier_id=request.supplier_id,
        evaluation_verdict=verdict,
        risk_assessment=eval_res,
        downstream_impact=forecast_res,
        adaptive_compensation=recipe_res,
        actionable_shipping_advice=advice
    )

@supplier_router.get("/vendors", summary="List All Suppliers & Historical Risk Scorecards")
def get_all_vendors(client: Dict[str, Any] = Depends(get_current_client)):
    """Returns directory of active suppliers with their aggregate batches, defect rates, and risk ratings."""
    if not registry.trend_analyzer:
        raise HTTPException(status_code=500, detail="Trend analyzer not loaded.")
    trends = registry.trend_analyzer.analyze_trends()
    summary = []
    for supp_id, data in trends.items():
        summary.append({
            "supplier_id": supp_id,
            "risk_rating": data["risk_rating"],
            "current_defect_rate_pct": data["current_defect_rate_pct"],
            "historical_trend": data["historical_trend"],
            "total_batches_delivered": data["batch_counts"]["total_batches"],
            "marginal_batches": data["batch_counts"]["marginal_high"] + data["batch_counts"]["marginal_low"],
            "total_warranty_claims": data["total_warranty_claims"],
            "total_financial_liability_usd": data["total_financial_impact_usd"],
        })
    return {"suppliers_count": len(summary), "suppliers": summary}

@supplier_router.get("/vendors/{supplier_id}/profile", summary="30-Day Quality Trends & CAPA Recurrence (Q8)")
def get_vendor_profile(supplier_id: str, client: Dict[str, Any] = Depends(get_current_client)):
    """Returns 30-day quality trajectory, recurring failure modes, and CAPA history for a specific supplier."""
    if not registry.trend_analyzer:
        raise HTTPException(status_code=500, detail="Trend analyzer not loaded.")
    trends = registry.trend_analyzer.analyze_trends()
    supp_id_upper = supplier_id.upper()
    if supp_id_upper not in trends:
        raise HTTPException(status_code=404, detail=f"Supplier '{supplier_id}' not found. Available: {list(trends.keys())}")
    return trends[supp_id_upper]

@supplier_router.get("/vendors/{supplier_id}/catalog", summary="Supplier Product Catalog, Lead Times & AI Advice")
def get_vendor_catalog(supplier_id: str, client: Dict[str, Any] = Depends(get_current_client)):
    """Retrieves all registered products, standard lead times, actual delivery performance, and AI recommendations."""
    if registry.products_catalog_df is None:
        raise HTTPException(status_code=500, detail="Product catalog not loaded.")
    
    supp_id_upper = supplier_id.upper()
    df = registry.products_catalog_df[registry.products_catalog_df["supplier_id"] == supp_id_upper]
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No catalog entries found for supplier '{supplier_id}'.")
    
    records = df.to_dict(orient="records")
    return {
        "supplier_id": supp_id_upper,
        "sku_count": len(records),
        "products": records
    }

@supplier_router.post("/products/predict-lead-time", response_model=LeadTimePredictionResponse, summary="Predict Delivery Lead Time & Delay Risk")
def predict_product_lead_time(request: LeadTimePredictionRequest, client: Dict[str, Any] = Depends(get_current_client)):
    """
    Predicts expected delivery lead time, delay days, and risk classification based on catalog history.
    """
    if registry.products_catalog_df is None:
        raise HTTPException(status_code=500, detail="Product catalog not loaded.")
    
    df = registry.products_catalog_df
    match = df[df["supplier_id"] == request.supplier_id]
    if request.sku:
        sku_match = match[match["sku"] == request.sku]
        if sku_match.empty:
            raise HTTPException(
                status_code=404,
                detail=f"SKU '{request.sku}' was not found for supplier '{request.supplier_id}'.",
            )
        match = sku_match

    if match.empty:
        raise HTTPException(status_code=404, detail=f"No catalog records found for supplier '{request.supplier_id}'.")
    avg_delay = int(round(match["delay_days"].mean()))
    status_label = str(match["status"].iloc[0])

    predicted_lead = request.target_lead_days + avg_delay
    risk_class = "CRITICAL_BOTTLENECK" if avg_delay >= 7 else ("MODERATE_RISK" if avg_delay >= 3 else "ON_TIME_LOW_RISK")

    action = (
        f"Increase purchase order lead time to {predicted_lead} days. Buffer {avg_delay} additional safety days at dock."
        if avg_delay > 0 else "Maintain standard release schedule."
    )

    return LeadTimePredictionResponse(
        supplier_id=request.supplier_id,
        sku=request.sku,
        category=request.category,
        target_lead_days=request.target_lead_days,
        predicted_actual_lead_days=predicted_lead,
        predicted_delay_days=avg_delay,
        delivery_status=status_label,
        risk_classification=risk_class,
        recommended_procurement_action=action,
        estimation_method=(
            "Historical catalog delay average for the matched supplier/SKU; "
            "monthly volume is not modeled because no trained volume-sensitive lead-time model is available."
        ),
    )

@supplier_router.post("/inventory/safety-stock", response_model=SafetyStockResponse, summary="Calculate Optimized Safety Stock Buffer")
def calculate_safety_stock(request: SafetyStockRequest, client: Dict[str, Any] = Depends(get_current_client)):
    """
    Calculates statistical safety stock units and buffer coverage days
    using the classical industrial formula: SS = Z * sqrt(L * sigma_D^2 + D^2 * sigma_L^2).
    """
    if request.service_level_target_pct >= 99.0:
        z = 2.33
    elif request.service_level_target_pct >= 97.5:
        z = 1.96
    elif request.service_level_target_pct >= 95.0:
        z = 1.645
    elif request.service_level_target_pct >= 90.0:
        z = 1.28
    else:
        z = 1.0

    l_bar = request.lead_time_days_mean
    d_bar = request.daily_consumption_mean
    sig_d = request.daily_consumption_std
    sig_l = request.lead_time_days_std

    variance = (l_bar * (sig_d ** 2)) + ((d_bar ** 2) * (sig_l ** 2))
    ss_units = int(round(z * np.sqrt(variance)))
    buffer_days = round(ss_units / max(1.0, d_bar), 1)
    reorder_point = int(round((d_bar * l_bar) + ss_units))

    return SafetyStockResponse(
        service_level_z_score=z,
        recommended_safety_stock_units=ss_units,
        buffer_coverage_days=buffer_days,
        reorder_point_units=reorder_point,
        formula_basis="SS = Z * sqrt(LeadTime_Mean * Demand_Variance + Demand_Mean^2 * LeadTime_Variance)"
    )

@supplier_router.get("/trace/claim/{claim_id}", summary="5-Tier Warranty Claim Genealogy Trace (Q9)")
def trace_warranty_claim(claim_id: str, client: Dict[str, Any] = Depends(get_current_client)):
    """
    Forensically traces a customer warranty claim back to assembled part serial number,
    fastening cycle, sensor readings, raw material batch, and responsible supplier.
    """
    if not registry.trace_engine:
        raise HTTPException(status_code=500, detail="Trace engine not loaded.")
    res = registry.trace_engine.trace_claim(claim_id)
    if not res:
        raise HTTPException(status_code=404, detail=f"Claim '{claim_id}' not found in warranty records.")
    return res

@supplier_router.get("/trace/batch/{batch_id}", summary="Trace All Warranty Claims for a Specific Batch")
def trace_batch(batch_id: str, client: Dict[str, Any] = Depends(get_current_client)):
    """Returns all warranty claims, dollar costs, and affected serial numbers tied to a supplier batch."""
    if not registry.trace_engine:
        raise HTTPException(status_code=500, detail="Trace engine not loaded.")
    result = registry.trace_engine.trace_batch(batch_id)
    if result["supplier_id"] == "UNKNOWN":
        raise HTTPException(status_code=404, detail=f"Batch '{batch_id}' was not found.")
    return result

@supplier_router.get("/tolerance-gap", summary="Receiving Inspection vs. Warranty Gap Analysis (Q10)")
def get_tolerance_gap_analysis(client: Dict[str, Any] = Depends(get_current_client)):
    """
    Statistical proof of the 216.0 - 230.0 HV tolerance leakage phenomenon.
    Quantifies the $1.34M in preventable field warranty failures caused by borderline hardness.
    """
    if not registry.gap_analyzer:
        raise HTTPException(status_code=500, detail="Tolerance gap analyzer not loaded.")
    return registry.gap_analyzer.run_gap_analysis()

@supplier_router.get("/scar/{supplier_id}", summary="Generate Official 8D SCAR & Financial Recovery Demand (Q11)")
def get_scar_document(supplier_id: str, client: Dict[str, Any] = Depends(get_current_client)):
    """
    Generates a legally defensible Supplier Corrective Action Request (SCAR)
    containing an itemized financial demand for in-plant scrap, chargebacks, and warranty costs.
    """
    supp_id_upper = supplier_id.upper()
    if supp_id_upper not in {"VENDOR_A", "VENDOR_B", "VENDOR_C"}:
        raise HTTPException(status_code=404, detail=f"Supplier '{supplier_id}' was not found.")
    if registry.scar_generator:
        return registry.scar_generator.generate_scar(supp_id_upper)
    
    raise HTTPException(status_code=500, detail="SCAR generator engine not available.")

# -----------------------------------------------------------------------------
# 7. MOUNT ROUTER INTO APPLICATION
# -----------------------------------------------------------------------------
app.include_router(supplier_router)

@app.get("/")
def root_redirect():
    """Root redirect to interactive OpenAPI Swagger documentation."""
    return {
        "service": "FactorySmart AI — Supplier & Vendor Intelligence API",
        "documentation_url": "/docs",
        "redoc_url": "/redoc",
        "openapi_json": "/openapi.json",
        "endpoints_prefix": "/api/v1/supplier",
        "health": "/api/v1/supplier/health",
        "active_test_keys": {
            "demo_partner_key": "fs-demo-partner-quality-2026",
            "vendor_b_key": "fs-key-vendor-b-3e7a9d0f",
            "admin_master_key": "fs-admin-master-2026-99x"
        }
    }

if __name__ == "__main__":
    import uvicorn
    # Default standalone port 8001
    uvicorn.run("src.supplier_api:app", host="0.0.0.0", port=8001, reload=True)
