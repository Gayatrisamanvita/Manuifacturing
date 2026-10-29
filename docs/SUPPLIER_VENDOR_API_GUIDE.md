# FactorySmart AI™ — Supplier & Vendor Intelligence REST API Guide
## External Integration Manual, Model Inference Endpoints & SDK Reference

**Document Identifier:** FS-API-SUPP-2026-V1  
**Classification:** Enterprise Engineering Standard / Partner Integration Guide  
**Compliance Standards:** OpenAPI 3.0, ISO 9001:2015, IATF 16949:2016  
**Target Audience:** External Suppliers, Vendor Quality Teams, MES / ERP Integrators (SAP, Oracle), Machine Control Engineers  
**Microservice File:** [`src/supplier_api.py`](file:///c:/Users/NagaLakshmiGayatriSa/.gemini/antigravity-ide/scratch/manufacturing_ml_suite/src/supplier_api.py)  
**Client SDK File:** [`src/supplier_api_client.py`](file:///c:/Users/NagaLakshmiGayatriSa/.gemini/antigravity-ide/scratch/manufacturing_ml_suite/src/supplier_api_client.py)  

---

## 1. Overview & Business Purpose

The **Supplier & Vendor Intelligence API** exposes FactorySmart AI's trained machine learning models, physics-informed transfer functions, and forensic traceability engines to external stakeholders. 

Using this API, external suppliers and enterprise systems can:
1. **Pre-Certify Inbound Material Lots**: Run metallurgical measurements ($HV$, $R_m$) through the trained `MarginalBatchRiskClassifier` before dispatching shipments.
2. **Detect the 216–230 HV Tolerance Leakage Zone**: Identify batches that nominally meet purchase specifications ($180 - 230\text{ HV}$) but cause downstream micro-galling and field warranty claims.
3. **Forecast Downstream Scrap & Severity**: Predict the exact scrap volume, defect severity ($1-5$), and financial cost ($55/part) before feeding a batch to assembly lines.
4. **Obtain Closed-Loop Machine Recipes**: Retrieve CNC and fastening controller offsets ($\Delta\text{Torque}$, $\Delta\text{Angle}$, Speed Reduction $\%$, Coolant Boost) to safely process marginal lots without scrap.
5. **Access 30-Day Vendor Quality Trajectories**: Monitor defect rate slopes, recurring failure modes, and CAPA recurrence scores.
6. **Query Forensic Warranty Genealogy**: Trace customer warranty claims through a 5-tier graph back to specific vendor heat lots.
7. **Retrieve Automated 8D SCAR Documents**: Access itemized financial non-conformance recovery demands.

---

## 2. Quickstart: Running the API Server

The API can be run in two modes:

### Mode A: Standalone Supplier Microservice (Port 8001)
Ideal for external partner access, DMZ deployment, or dedicated vendor portal instances:
```powershell
python -m uvicorn src.supplier_api:app --host 0.0.0.0 --port 8001 --reload
```

### Mode B: Integrated Factory Suite (Port 8000)
Runs all factory operations (Tier 1, Tier 2, Tier 3) together:
```powershell
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

### Interactive Documentation & Swagger UI
Once running, open your browser to:
- **Interactive Swagger UI**: `http://localhost:8001/docs` (or `http://localhost:8000/docs`)
- **Alternative ReDoc UI**: `http://localhost:8001/redoc`
- **Raw OpenAPI JSON**: `http://localhost:8001/openapi.json`

---

## 3. Endpoints Reference & Schemas

### Base URL Prefix
All supplier endpoints are versioned under:
```
/api/v1/supplier
```

---

### 3.1 Material & Lot Quality Evaluation

#### `POST /api/v1/supplier/evaluate-lot`
Evaluates an incoming or proposed material lot against historical quality models.

**Request Payload:**
```json
{
  "material_hardness_hv": 223.5,
  "tensile_strength_mpa": 675.0,
  "lower_spec_limit": 180.0,
  "upper_spec_limit": 230.0,
  "supplier_id": "VENDOR_B",
  "batch_id": "LOT-2026-X49",
  "material_grade": "AISI-4140"
}
```

**Response (HTTP 200 OK):**
```json
{
  "batch_id": "LOT-2026-X49",
  "supplier_id": "VENDOR_B",
  "material_hardness_hv": 223.5,
  "tensile_strength_mpa": 675.0,
  "downstream_risk_probability": 0.7241,
  "batch_risk_tier": "HIGH_RISK_MARGINAL",
  "passed_nominal_spec": true,
  "is_in_tolerance_leakage_zone": true,
  "tolerance_leakage_warning": "CRITICAL WARNING: Batch hardness of 223.5 HV falls squarely in the 216.0 - 230.0 HV Tolerance Leakage Zone...",
  "recommendation": "MANDATE ADAPTIVE RECIPE COMPENSATION OR REJECT LOT AT DOCK"
}
```

---

#### `POST /api/v1/supplier/forecast-impact`
Forecasts downstream assembly scrap count, defect severity (1-5), and financial scrap cost.

**Request Payload:**
```json
{
  "material_hardness_hv": 224.0,
  "tensile_strength_mpa": 680.0,
  "supplier_id": "VENDOR_B",
  "batch_size_units": 5000
}
```

**Response (HTTP 200 OK):**
```json
{
  "expected_defect_severity": 2.35,
  "expected_scrap_count": 316,
  "projected_scrap_rate_pct": 6.32,
  "likely_downstream_failure_mode": "INCOMPLETE_FASTENING",
  "physical_root_cause": "Excessive material hardness causing high fastening torque resistance before target shutoff angle is reached.",
  "projected_scrap_cost_usd": 17380.00
}
```

---

#### `POST /api/v1/supplier/adaptive-recipe`
Generates machine controller recipe offsets to safely process marginal material stock.

**Request Payload:**
```json
{
  "material_hardness_hv": 224.5,
  "batch_tensile_mpa": 680.0
}
```

**Response (HTTP 200 OK):**
```json
{
  "recipe_status": "ACTIVE_COMPENSATION_HARD_STOCK",
  "target_torque_offset_nm": 0.38,
  "shutoff_angle_offset_deg": 11.4,
  "spindle_speed_reduction_pct": 7.6,
  "feed_rate_offset_pct": -5.0,
  "lubricant_boost_mode": "ENABLE_HIGH_PRESSURE_MIST",
  "expected_scrap_reduction_pct": 68.4,
  "engineering_justification": "Batch hardness is +19.5 HV above nominal. Applying torque offset of +0.38 Nm and speed reduction of -7.6%..."
}
```

---

#### `POST /api/v1/supplier/comprehensive-qc`
**One-stop pre-shipment audit.** Executes Lot Evaluation + Impact Forecast + Adaptive Recipe in a single roundtrip.

**Request Payload:**
```json
{
  "supplier_id": "VENDOR_B",
  "batch_id": "LOT-2026-B88",
  "material_grade": "AISI-4140",
  "material_hardness_hv": 222.5,
  "tensile_strength_mpa": 670.0,
  "batch_size_units": 8000,
  "lower_spec_limit": 180.0,
  "upper_spec_limit": 230.0
}
```

**Response (HTTP 200 OK):**
```json
{
  "timestamp": "2026-10-26T05:30:00Z",
  "batch_id": "LOT-2026-B88",
  "supplier_id": "VENDOR_B",
  "evaluation_verdict": "MARGINAL_LEAKAGE_RISK_COMPENSATION_MANDATORY",
  "risk_assessment": { ... },
  "downstream_impact": { ... },
  "adaptive_compensation": { ... },
  "actionable_shipping_advice": "Lot is accepted under concession ONLY if factory controllers apply the returned adaptive torque and angle recipe."
}
```

---

### 3.2 Vendor Performance, Product Catalog & Inventory

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/supplier/vendors` | Returns list of all vendors with defect rates, batch counts, and risk tiers. |
| `GET` | `/api/v1/supplier/vendors/{supplier_id}/profile` | 30-day defect rate trajectory, recurring failure modes, and CAPA history. |
| `GET` | `/api/v1/supplier/vendors/{supplier_id}/catalog` | Complete product catalog with SKUs, lead days, delay status, and root causes. |
| `POST`| `/api/v1/supplier/products/predict-lead-time` | Predicts delivery delay days and bottleneck risk based on SKU and volume. |
| `POST`| `/api/v1/supplier/inventory/safety-stock` | Computes dynamic safety stock units ($SS$) and reorder points. |

The lead-time endpoint currently uses historical catalog delay averages for the matched supplier/SKU. It is not a trained lead-time ML model, and the supplied monthly volume is not used in the estimate; this limitation is returned in `estimation_method`. An unknown supplier/SKU returns `404` rather than an invented fallback prediction. The checked-in supplier quality records are historical snapshots through their `data_as_of` date, not a live rolling 30-day series.

---

### 3.3 Warranty Forensics, Tolerance Gap & 8D SCAR

#### `GET /api/v1/supplier/trace/claim/{claim_id}`
Returns the complete 5-hop genealogy tree for a field warranty claim:
```
Claim ID (CLM-84920) 
  --> Part Serial (SN-2026-84920) 
  --> Fastening Cycle (CYC-2026-99382) 
  --> Material Batch (BATCH-042) 
  --> Supplier (VENDOR_B)
```

#### `GET /api/v1/supplier/tolerance-gap`
Returns statistical evidence of the 216–230 HV leakage zone and the justification for revising receiving inspection USL to 216.0 HV.

#### `GET /api/v1/supplier/scar/{supplier_id}`
Returns the formal 8D Supplier Corrective Action Request (SCAR) document with line-item financial recovery ledger (Scrap + Chargebacks + Warranty Payouts).

---

## 4. Python Client SDK Integration

External developers can consume the API directly using the lightweight [`src/supplier_api_client.py`](file:///c:/Users/NagaLakshmiGayatriSa/.gemini/antigravity-ide/scratch/manufacturing_ml_suite/src/supplier_api_client.py) (uses Python standard library, zero third-party dependencies required):

```python
from src.supplier_api_client import SupplierAPIClient

# Initialize client pointing to your server
client = SupplierAPIClient(base_url="http://localhost:8001")

# 1. Pre-screen an incoming lot
lot = client.evaluate_lot(
    hardness_hv=223.5,
    tensile_mpa=675.0,
    supplier_id="VENDOR_B",
    batch_id="LOT-4091"
)

print(f"Risk Tier: {lot['batch_risk_tier']}")
print(f"In Leakage Zone: {lot['is_in_tolerance_leakage_zone']}")

# 2. If marginal, generate controller offsets
if lot['is_in_tolerance_leakage_zone']:
    recipe = client.get_adaptive_recipe(hardness_hv=223.5, tensile_mpa=675.0)
    print(f"Torque Offset: {recipe['target_torque_offset_nm']:+} Nm")
    print(f"Angle Offset: {recipe['shutoff_angle_offset_deg']:+} deg")
    print(f"Speed Cut: -{recipe['spindle_speed_reduction_pct']}%")

# 3. Retrieve supplier 8D SCAR claim
scar = client.get_scar_report("VENDOR_B")
print(f"Recovery Claim Demand: ${scar['financial_recovery_breakdown']['total_demand_amount_usd']:,.2f}")
```

---

## 5. cURL Integration Examples

### Example 1: Check Inbound Lot Risk
```bash
curl -X POST "http://localhost:8001/api/v1/supplier/evaluate-lot" \
     -H "Content-Type: application/json" \
     -d '{
       "material_hardness_hv": 224.0,
       "tensile_strength_mpa": 680.0,
       "supplier_id": "VENDOR_B",
       "lower_spec_limit": 180.0,
       "upper_spec_limit": 230.0
     }'
```

### Example 2: Retrieve 8D SCAR Report
```bash
curl -X GET "http://localhost:8001/api/v1/supplier/scar/VENDOR_B" \
     -H "Accept: application/json"
```

---

## 6. Security, Authentication & Key Management

All model inference and analytical endpoints are protected by FactorySmart AI's cryptographic API Key system.

### 6.1 Authentication Methods Supported
Clients can authenticate via any of the following 3 channels:
1. **Request Header (Recommended)**:
   ```http
   X-API-Key: fs-key-vendor-b-3e7a9d0f
   ```
2. **Bearer Token Header**:
   ```http
   Authorization: Bearer fs-key-vendor-b-3e7a9d0f
   ```
3. **URL Query Parameter (Convenient for quick browser/script testing)**:
   ```
   http://localhost:8001/api/v1/supplier/vendors?api_key=fs-key-vendor-b-3e7a9d0f
   ```

### 6.2 Pre-Configured Active API Keys
The API comes pre-loaded with active keys stored securely in [`data/api_keys.json`](file:///c:/Users/NagaLakshmiGayatriSa/.gemini/antigravity-ide/scratch/manufacturing_ml_suite/data/api_keys.json):

| Role / Client | API Key Token | Permissions / Scope |
|---|---|---|
| **Admin Master Key** | `fs-admin-master-2026-99x` | Full administrative access; can issue and revoke API keys |
| **Vendor A Key** | `fs-key-vendor-a-8f4b2c1d` | Dedicated to `VENDOR_A` material lots, catalog, and SCAR claims |
| **Vendor B Key** | `fs-key-vendor-b-3e7a9d0f` | Dedicated to `VENDOR_B` material lots, catalog, and SCAR claims |
| **Vendor C Key** | `fs-key-vendor-c-5b1c8e2a` | Dedicated to `VENDOR_C` material lots, catalog, and SCAR claims |
| **Demo Partner Key** | `fs-demo-partner-quality-2026` | General evaluation, downstream forecast, and recipe compensation |

### 6.3 Using the Swagger UI "Authorize" Button
1. Open **[http://localhost:8001/docs](http://localhost:8001/docs)** in your browser.
2. Click the green **Authorize 🔓** button at the top right of the page.
3. In the modal, paste one of the active keys (e.g. `fs-key-vendor-b-3e7a9d0f` or `fs-admin-master-2026-99x`).
4. Click **Authorize** and then **Close**.
5. All endpoint requests will now automatically include the key and execute with **200 OK**!

### 6.4 Key Management Endpoints (Admin)
Administrators using `fs-admin-master-2026-99x` can manage keys dynamically:
- **`POST /api/v1/supplier/auth/create-key`**: Issues a new API key for any new supplier or partner.
- **`GET /api/v1/supplier/auth/keys`**: Lists all active keys, roles, and real-time request counts.
- **`DELETE /api/v1/supplier/auth/revoke-key/{key_or_id}`**: Revokes an existing key immediately.
- **`GET /api/v1/supplier/auth/whoami`**: Returns identity and usage profile of the calling key.
