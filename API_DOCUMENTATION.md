# FactorySmart AI™ — Manufacturing & Supplier Intelligence API Documentation

Welcome to the **FactorySmart AI™ REST API**. This enterprise API platform serves machine learning inference, SHAP physical parameter attributions, physics-informed prescriptive compensations, warranty forensic genealogy, and automated 8D Supplier Corrective Action Requests (SCAR).

---

## 1. Quick Reference & Base URLs

| Service Mode | Base URL | Interactive Docs (Swagger) | Alternative Docs (ReDoc) |
| :--- | :--- | :--- | :--- |
| **Unified API (All Tiers + Supplier Router + Web)** | `http://localhost:8001` | [`http://localhost:8001/docs`](http://localhost:8001/docs) | [`http://localhost:8001/redoc`](http://localhost:8001/redoc) |
| **Standalone Supplier Microservice** | `http://localhost:8001` | [`http://localhost:8001/docs`](http://localhost:8001/docs) | [`http://localhost:8001/redoc`](http://localhost:8001/redoc) |

### Starting the API Server
```bash
# Launch default Unified API on port 8001
python run_api.py

# Launch on custom port with auto-reload
python run_api.py --port 8001 --reload

# Launch Standalone Supplier microservice on port 8001
python run_api.py --mode supplier --port 8001
```

The Streamlit dashboard runs separately on port `8000`; its API base defaults to
`http://127.0.0.1:8001` and can be overridden with `MANUFACTURING_API_URL`.

Or on Windows:
```cmd
run_api.bat
```

---

## 2. API Key Authentication & Access Governance

All endpoints under `/api/v1/supplier/...` are protected with cryptographic API Key Authentication.

### Supported Authentication Channels
You can pass your API key via any of the following 3 mechanisms:

1. **HTTP Header (Recommended)**:
   ```http
   X-API-Key: fs-key-vendor-b-3e7a9d0f
   ```
2. **Authorization Bearer Header**:
   ```http
   Authorization: Bearer fs-key-vendor-b-3e7a9d0f
   ```
3. **URL Query Parameter**:
   ```http
   GET /api/v1/supplier/health?api_key=fs-key-vendor-b-3e7a9d0f
   ```

### Pre-Configured Test API Keys
| Key Token | Role | Target Organization / Team | Permissions |
| :--- | :--- | :--- | :--- |
| `fs-admin-master-2026-99x` | `admin` | Enterprise Quality Admin | `["all"]` (Can create, list, and revoke keys) |
| `fs-key-vendor-b-3e7a9d0f` | `supplier` | Vendor B Quality Team | `["evaluate_lot", "forecast_impact", "adaptive_recipe", "catalog", "scar"]` |
| `fs-key-vendor-a-8f4b2c1d` | `supplier` | Vendor A Quality Team | `["evaluate_lot", "forecast_impact", "adaptive_recipe", "catalog", "scar"]` |
| `fs-key-vendor-c-5b1c8e2a` | `supplier` | Vendor C Quality Team | `["evaluate_lot", "forecast_impact", "adaptive_recipe", "catalog", "scar"]` |
| `fs-demo-partner-quality-2026` | `partner` | Partner Quality Demo | `["evaluate_lot", "forecast_impact", "adaptive_recipe", "catalog", "tolerance_gap"]` |

---

## 3. Endpoints Catalog

### Model Artifacts & Per-Model Inference

The six files below are the model/configuration artifacts used by the unified API and are stored in `models/`. `GET /api/models` reports artifact presence plus whether each engine is initialized and ready. The supplier analytics engines read project data files directly, so they do not have pickle artifacts.

| Model / Engine | Artifact | Type | Direct inference endpoint(s) |
| :--- | :--- | :--- | :--- |
| `ShiftDefectPredictor` | `models/shift_defect_predictor.pkl` | Trained ML | `POST /api/tier1/predict_shift` |
| `FailureModeClassifier` | `models/failure_mode_classifier.pkl` | Trained ML + SHAP | `POST /api/tier1/predict_failure_mode`, `POST /api/tier1/simulate` |
| `PrescriptiveActionEngine` | `models/prescriptive_action_engine.pkl` | Rule/configuration engine | `POST /api/tier1/recommend_action`, `POST /api/tier1/simulate` |
| `MarginalBatchRiskClassifier` | `models/marginal_batch_classifier.pkl` | Trained ML | `POST /api/tier2/classify_material`, `POST /api/tier2/evaluate_material` |
| `DownstreamImpactForecaster` | `models/downstream_impact_forecaster.pkl` | Trained ML | `POST /api/tier2/forecast_impact`, `POST /api/tier2/evaluate_material` |
| `AdaptiveParameterCompensator` | `models/adaptive_parameter_compensator.pkl` | Rule/configuration engine | `POST /api/tier2/compensate`, `POST /api/tier2/evaluate_material` |
| `SupplierQualityTrendAnalyzer` | Not applicable (reads data files) | Data analytics | `GET /api/tier3/suppliers` |
| `WarrantyTraceabilityEngine` | Not applicable (reads data files) | Data analytics | `GET /api/tier3/trace/{claim_id}` |
| `ToleranceLeakageAnalyzer` | Not applicable (reads data files) | Data analytics | `GET /api/tier3/tolerance_gap` |
| `SCARGenerator` | Not applicable (reads data files) | Data analytics | `GET /api/tier3/scar/{supplier_id}` |

The combined `/api/tier1/simulate` and `/api/tier2/evaluate_material` routes remain available; the direct routes above allow each individual model or engine to be called independently. `/api/models` does not expose filesystem paths outside the project.

### Dashboard monitoring endpoints

- `GET /api/tier2/materials` returns incoming-batch measurements, receiving QC status, scrap and
  chargeback records, and per-batch risk probability/tier from the trained Random Forest.
  Incoming-batch counts describe recorded receipts only; warehouse stock and material purchase
  costs are not available in the source data.
- `GET /api/tier1/operations/monitor` returns the latest recorded date's station/shift production,
  machine-state and inspection summaries, plus LightGBM predicted failure-mode counts. Faulted
  and idle hours are estimates from telemetry sample counts and the median sample interval, not
  maintenance work-order durations or live machine telemetry.
- `GET /api/tier1/production/summary` returns recorded production output, logged defects, rates,
  and station-level production totals.

The Streamlit dashboard calls these APIs and the existing model inference endpoints. Its shift
target is entered by the operator because no work-order target records are included in the data.
Reported target attainment compares that entered target with the latest historical shift and is
not a recorded work-order result.

#### Shift defect prediction — `POST /api/tier1/predict_shift`
Provide shift-level planned volume and telemetry for each station to receive predicted defect counts and station risk rankings. Include the stations you want ranked (each station ID at most once).

```json
{
  "shift_id": "SHIFT_1",
  "planned_units": 1200,
  "line_speed_upm": 47.2,
  "std_line_speed_upm": 1.8,
  "stations": [
    {
      "station_id": "ST-10",
      "spindle_temp_c": 55.1,
      "vibration_rms_g": 1.4,
      "system_pressure_psi": 115.0,
      "tool_wear_index": 42.0,
      "peak_torque_nm": 12.5,
      "shutoff_angle_deg": 360.0
    }
  ]
}
```

#### Failure mode classification — `POST /api/tier1/predict_failure_mode`
Accepts the sensor/material telemetry fields from `/api/tier1/simulate` and returns the predicted failure mode, confidence, and top SHAP contributors without running the prescriptive engine.

#### Prescriptive action — `POST /api/tier1/recommend_action`
Request fields: `predicted_failure_mode`, `current_telemetry` (the same telemetry fields accepted by `/api/tier1/simulate`), and optional `top_contributors` (an array of feature/value/SHAP-impact/direction objects).

#### Individual raw-material engines
The following all accept the same material request shape as `/api/tier2/evaluate_material` (`material_hardness_hv`, `tensile_strength_mpa`, `lower_spec_limit`, `upper_spec_limit`, `supplier_id`):

- `POST /api/tier2/classify_material`: risk probability, risk tier, inspection pass/fail, and recommendation.
- `POST /api/tier2/forecast_impact`: expected severity, scrap count/cost, likely failure mode, and physical root cause.
- `POST /api/tier2/compensate`: adaptive machine recipe adjustments.

`batch_size_units` is optional on these unified API requests. When supplied to `forecast_impact` or `evaluate_material`, the trained scrap estimate is scaled from the average lot size used for training and capped at the requested batch size. An out-of-spec hardness result is returned as `OUT_OF_SPEC_REJECT`; `lower_spec_limit` must be less than `upper_spec_limit`.

The Gradient Boosting regressors predict severity and scrap. The likely failure mode and physical root-cause text are assigned by the documented hardness-threshold rules (`failure_mode_method: "HARDNESS_THRESHOLD_RULES"`), not by a separate failure-mode classifier.

### A. System & Operational Health
- `GET /health`: Unified API health check and registry of loaded models.
- `GET /api/models`: Artifact inventory, endpoint mapping, and engine readiness.
- `GET /api/overview`: JSON summary of model performance, cross-validation metrics, and financial recovery demand.
- `GET /api/v1/supplier/health`: Supplier API health and catalog status.

The unified health response reports `status: "degraded"` and a `model_load_errors` map when model artifacts or analytics engines fail to initialize; failed engines are not silently reported as ready.

### B. Tier 1: Manufacturing Issue Level
#### `POST /api/tier1/simulate`
Simulates real-time cycle telemetry. Predicts failure modes using `failure_mode_classifier.pkl`, extracts top 5 physical sensor drivers using SHAP TreeExplainer, and prescribes parameter offsets via `prescriptive_action_engine.pkl`.

**Sample Request:**
```json
{
  "station_id": "ST-10",
  "spindle_temp_c": 62.0,
  "system_pressure_psi": 115.0,
  "vibration_rms_g": 2.1,
  "tool_wear_index": 55.0,
  "hours_since_service": 18.0,
  "peak_torque_nm": 13.2,
  "shutoff_angle_deg": 285.0,
  "torque_status": "NOK_HIGH",
  "material_hardness_hv": 220.0,
  "tensile_strength_mpa": 645.0
}
```

**Sample Response:**
```json
{
  "station_id": "ST-10",
  "predicted_failure_mode": "INCOMPLETE_FASTENING",
  "confidence": 0.884,
  "top_physical_contributors": [
    {
      "feature": "peak_torque_nm",
      "value": 13.2,
      "shap_impact": 2.0382,
      "direction": "RISK_INFLATING"
    },
    {
      "feature": "spindle_temp_c",
      "value": 62.0,
      "shap_impact": 0.4442,
      "direction": "RISK_INFLATING"
    }
  ],
  "prescriptive_action": {
    "action_code": "RECALIBRATE_TOOL",
    "urgency": "HIGH - BEFORE NEXT BATCH",
    "recommended_task": "Recalibrate torque shutoff sensor and inspect fastener thread engagement depth.",
    "prescriptive_recipe_offsets": {
      "torque_offset_nm": -0.7,
      "shutoff_angle_offset_deg": 75.0,
      "rundown_speed_reduction_pct": 5.0
    },
    "root_driver": "peak_torque_nm"
  }
}
```

---

### C. Tier 2: Raw Material Intelligence
#### `POST /api/tier2/evaluate_material`
Evaluates marginal incoming batches against receiving specs, forecasts downstream defect severity and scrap volume, and computes physics-informed recipe offsets.

**Sample Request:**
```json
{
  "material_hardness_hv": 224.5,
  "tensile_strength_mpa": 670.0,
  "lower_spec_limit": 180.0,
  "upper_spec_limit": 230.0,
  "supplier_id": "VENDOR_B"
}
```

#### `POST /api/v1/supplier/evaluate-lot` *(Requires API Key)*
Specifically flags lots falling within the **216.0 – 230.0 HV tolerance leakage zone** that cause downstream field failures.

#### `POST /api/v1/supplier/certify-lot` (or `/comprehensive-qc`) *(Requires API Key)*
Runs a unified 3-model audit (Marginal Batch Classifier + Downstream Impact Forecaster + Adaptive Parameter Compensator) and returns pre-shipment certification.

The adaptive-recipe response returns the compensator's calculated `torque_offset_nm`, `angle_offset_deg`, speed reduction, lubrication mode, and engineering guidance. Forecast scrap counts and cost are scaled to the requested `batch_size_units`.

---

### D. Tier 3: Supplier & Vendor Intelligence
- `GET /api/tier3/suppliers` or `GET /api/v1/supplier/vendors`: Returns historical quality snapshots and risk ratings. The response includes `data_as_of` and `trend_basis`; the checked-in history does not support a current rolling 30-day trend.
- `GET /api/tier3/trace/{claim_id}`: End-to-end genealogy DAG linking warranty claims back to part serial numbers, spindle telemetry, material heat lots, and vendor IDs.
- `GET /api/tier3/tolerance_gap`: Proof of the 216–230 HV tolerance leakage zone and recommended tightened Upper Specification Limit.
- `GET /api/tier3/scar/{supplier_id}`: Official 8D Supplier Corrective Action Request document and itemized financial claim demand.

---

### E. API Key Administration (Admin Master Key Required)
- `POST /api/v1/supplier/auth/create-key`: Issues a new authenticated API key.
- `GET /api/v1/supplier/auth/keys`: Lists all registered keys, last used timestamps, and request counts.
- `DELETE /api/v1/supplier/auth/revoke-key/{key_or_id}`: Revokes an active key.
- `GET /api/v1/supplier/auth/whoami`: Returns metadata of current authenticated caller.

---

## 4. Client Code Examples

### Python (using requests)
```python
import requests

url = "http://localhost:8000/api/v1/supplier/evaluate-lot"
headers = {
    "X-API-Key": "fs-key-vendor-b-3e7a9d0f",
    "Content-Type": "application/json"
}
payload = {
    "supplier_id": "VENDOR_B",
    "material_hardness_hv": 224.5,
    "tensile_strength_mpa": 670.0,
    "lower_spec_limit": 180.0,
    "upper_spec_limit": 230.0
}

response = requests.post(url, headers=headers, json=payload)
print(response.json())
```

### Python SDK (`src/supplier_api_client.py`)
```python
from src.supplier_api_client import SupplierAPIClient

client = SupplierAPIClient(base_url="http://localhost:8000", api_key="fs-key-vendor-b-3e7a9d0f")

# Certify incoming material lot
result = client.evaluate_material_lot(
    hardness=224.5,
    tensile=670.0,
    supplier_id="VENDOR_B",
    batch_id="LOT-2026-B88"
)
print(result)
```

### cURL
```bash
curl -X POST "http://localhost:8000/api/v1/supplier/evaluate-lot" \
  -H "X-API-Key: fs-key-vendor-b-3e7a9d0f" \
  -H "Content-Type: application/json" \
  -d '{"supplier_id":"VENDOR_B","material_hardness_hv":224.5,"tensile_strength_mpa":670.0,"lower_spec_limit":180.0,"upper_spec_limit":230.0}'
```
