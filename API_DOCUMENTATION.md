# FactorySmart AI™ — Manufacturing & Supplier Intelligence API Documentation

Welcome to the **FactorySmart AI™ REST API**. This enterprise API platform serves machine learning inference, SHAP physical parameter attributions, physics-informed prescriptive compensations, warranty forensic genealogy, and automated 8D Supplier Corrective Action Requests (SCAR).

---

## 1. Quick Reference & Base URLs

| Service Mode | Base URL | Interactive Docs (Swagger) | Alternative Docs (ReDoc) |
| :--- | :--- | :--- | :--- |
| **Unified API (All Tiers + Supplier Router + Web)** | `http://localhost:8000` | [`http://localhost:8000/docs`](http://localhost:8000/docs) | [`http://localhost:8000/redoc`](http://localhost:8000/redoc) |
| **Standalone Supplier Microservice** | `http://localhost:8001` | [`http://localhost:8001/docs`](http://localhost:8001/docs) | [`http://localhost:8001/redoc`](http://localhost:8001/redoc) |

### Starting the API Server
```bash
# Launch default Unified API on port 8000
python run_api.py

# Launch on custom port with auto-reload
python run_api.py --port 8000 --reload

# Launch Standalone Supplier microservice on port 8001
python run_api.py --mode supplier --port 8001
```

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

### A. System & Operational Health
- `GET /health`: Unified API health check and registry of loaded models.
- `GET /api/overview`: JSON summary of model performance, cross-validation metrics, and financial recovery demand.
- `GET /api/v1/supplier/health`: Supplier API health and catalog status.

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

---

### D. Tier 3: Supplier & Vendor Intelligence
- `GET /api/tier3/suppliers` or `GET /api/v1/supplier/vendors`: Returns 30-day quality trajectories and risk ratings.
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
