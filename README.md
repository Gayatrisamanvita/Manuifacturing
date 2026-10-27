# Manufacturing, Material & Supplier Intelligence Suite (FactorySmart AI™)

An end-to-end, enterprise-grade AI/ML intelligence platform designed for manufacturing quality control, raw material receiving inspection, adaptive machine compensation, warranty forensic traceability, and vendor financial recovery.

---

## 1. Quickstart & Setup Guide

### Option A: Automatic 1-Click Setup (Windows)
Double-click `setup_venv.bat` or run:
```cmd
setup_venv.bat
```
This automatically verifies Python 3.10+, provisions the virtual environment in `.\venv`, upgrades `pip`, and installs all requirements from `requirements.txt`.

### Option B: Manual Setup (Windows / macOS / Linux)

1. **Clone and navigate to the project directory:**
   ```bash
   cd manufacturing_ml_suite
   ```

2. **Create and activate the virtual environment (`venv`):**
   - **Windows:**
     ```cmd
     python -m venv venv
     venv\Scripts\activate
     ```
   - **macOS / Linux:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify the installation:**
   ```bash
   python verify_system.py
   ```
   *(Runs automated cross-verification on all models, pickle files, API keys, and endpoints).*

---

## 2. Running the REST API

Launch the unified REST API backend:
```bash
python run_api.py
```
Or on Windows:
```cmd
run_api.bat
```

- **Unified API Base URL:** `http://localhost:8000`
- **Interactive Swagger UI:** [`http://localhost:8000/docs`](http://localhost:8000/docs)
- **Alternative ReDoc UI:** [`http://localhost:8000/redoc`](http://localhost:8000/redoc)
- **Health Check Endpoint:** [`http://localhost:8000/health`](http://localhost:8000/health)
- **Web Dashboard:** [`http://localhost:8000/`](http://localhost:8000/)

### Standalone Supplier Microservice Mode (Port 8001)
```bash
python run_api.py --mode supplier --port 8001
```

---

## 3. Serialized Pickle Models (`models/*.pkl`)

All trained machine learning models are serialized in `models/` as standard Python pickle files for fast, in-memory inference without needing to re-train on raw datasets:

| Model File | Class / Engine | Tier & Core Question Addressed |
| :--- | :--- | :--- |
| `models/shift_defect_predictor.pkl` | `ShiftDefectPredictor` | **Tier 1 (Q1):** Shift Defect Count Regressor & Station Risk Ranking |
| `models/failure_mode_classifier.pkl` | `FailureModeClassifier` | **Tier 1 (Q2 & Q4):** Multi-class Failure Mode Classifier & SHAP Attributions |
| `models/prescriptive_action_engine.pkl` | `PrescriptiveActionEngine` | **Tier 1 (Q3):** Machine Recipe Offsets & Preventive Maintenance Actions |
| `models/marginal_batch_classifier.pkl` | `MarginalBatchRiskClassifier` | **Tier 2 (Q5):** Marginal Material Batch Downstream Defect Risk Prediction |
| `models/downstream_impact_forecaster.pkl` | `DownstreamImpactForecaster` | **Tier 2 (Q6):** Downstream Scrap Volume & Defect Severity Regressors |
| `models/adaptive_parameter_compensator.pkl` | `AdaptiveParameterCompensator` | **Tier 2 (Q7):** Physics-Informed Closed-Loop Parameter Offsets |

### Using Pickle Models Directly in Python
You can load all models in 3 lines of code:
```python
from src.models import load_all_models

# Load all 6 serialized models and 4 analytical engines
models = load_all_models(models_dir="models", data_dir="data")

# 1. Failure Mode Classification with SHAP Contributors
fm_clf = models["failure_mode_classifier"]
# fm_clf.predict_with_contributors(telemetry_df)

# 2. Marginal Batch Downstream Risk
batch_clf = models["marginal_batch_classifier"]
# batch_clf.predict_batch_risk(batch_df)

# 3. Downstream Scrap & Severity Forecast
forecaster = models["downstream_impact_forecaster"]
# forecaster.forecast(hardness=224.5, tensile=670.0, supplier_id="VENDOR_B")

# 4. Adaptive Recipe Offsets
compensator = models["adaptive_parameter_compensator"]
# compensator.compute_offsets(batch_hardness_hv=224.5, batch_tensile_mpa=670.0)
```
*(See `example_usage.py` for a full runnable demonstration).*

---

## 4. API Key Authentication & Governance

All supplier and vendor endpoints (`/api/v1/supplier/...`) require authentication using API keys stored securely in `data/api_keys.json`.

### Active Pre-Configured Test Keys:
- **Enterprise Admin Master Key:** `fs-admin-master-2026-99x` (Role: `admin` — allows key creation, listing, and revoking)
- **Vendor B Key:** `fs-key-vendor-b-3e7a9d0f` (Role: `supplier` — associated with VENDOR_B)
- **Vendor A Key:** `fs-key-vendor-a-8f4b2c1d` (Role: `supplier` — associated with VENDOR_A)
- **Vendor C Key:** `fs-key-vendor-c-5b1c8e2a` (Role: `supplier` — associated with VENDOR_C)
- **Demo Partner Key:** `fs-demo-partner-quality-2026` (Role: `partner`)

### Sending Authentication:
Pass via header `X-API-Key: <key>`, `Authorization: Bearer <key>`, or query parameter `?api_key=<key>`.

See [`API_DOCUMENTATION.md`](API_DOCUMENTATION.md) for complete details on every endpoint, schemas, and payload examples.

---

## 5. Running the Interactive Streamlit Dashboard

To launch the multi-tier Six Sigma quality control dashboard:
```bash
streamlit run streamlit_app.py
```
Or on Windows:
```cmd
run_dashboard.bat
```
Access the dashboard at `http://localhost:8501`.

---

## 6. End-to-End System Cross-Verification

Run the automated verification test suite:
```bash
python verify_system.py
```
Checks:
- Virtual environment (`venv/`) and required package dependencies
- Integrity and inference of all 6 serialized pickle models
- API key validation, security rejection, role-based access control, and dynamic key lifecycles
- All FastAPI endpoints (health, simulation, material evaluation, supplier profiles, claim traceability, tolerance leakage gap, and 8D SCAR reports)
- Client SDK integration (`SupplierAPIClient`)
