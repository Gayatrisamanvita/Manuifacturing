---
title: Manufacturing Intelligence API
emoji: 🏭
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

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

1. **Navigate to the project directory:**
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
python run_api.py --port 8001
```
Or on Windows:
```cmd
python run_api.py --port 8001
```

The unified API defaults to port `8001` so it can run alongside the Streamlit dashboard on port `8000`. Set `PORT` or pass `--port` to override it.

- **Unified API Base URL:** `http://localhost:8001`
- **Interactive Swagger UI:** [`http://localhost:8001/docs`](http://localhost:8001/docs)
- **Alternative ReDoc UI:** [`http://localhost:8001/redoc`](http://localhost:8001/redoc)
- **Health Check Endpoint:** [`http://localhost:8001/health`](http://localhost:8001/health)
- **Streamlit Dashboard:** [`http://localhost:8000/`](http://localhost:8000/)

### Streamlit model coverage and data limits

The dashboard's predictive outputs use the trained Random Forest batch-risk classifier,
Gradient Boosting material forecaster, LightGBM failure-mode classifier, and LightGBM
shift-defect predictor. Machine recommendations and adaptive recipe offsets are explicitly
identified as rule-based; supplier traceability and quality history are data analytics.

The dashboard does not present inventory forecasts, factory-unit/OEE predictions, or
work-order forecasts because the repository has no inventory time series, factory-unit
identifiers, OEE definitions, or recorded work-order targets to train or validate those
models. Machine state is available by station; faulted time is estimated from sampled state
telemetry and is not a maintenance-log duration. The supplier catalog is shown only as
recorded source data; its small sample is not sufficient for a reliable lead-time model.
Provide those operational datasets before adding the corresponding predictions.

On the supplier tab, choose a vendor to see its most-delayed catalog product, the vendor's
full recorded catalog, and an ML evaluation form for a selected product's measured incoming
batch. The product form requires actual hardness and tensile measurements; the catalog does
not link SKUs to batch measurements. On the raw-material tab, batch-risk probabilities are
color-coded green-to-red. They represent material quality risk, not stock depletion: inventory
levels and consumption history are not available. Material cost metrics are recorded scrap and
chargebacks, not procurement cost. Production charts show recorded output and defect/inspection
rates, an operator-entered shift target, a LightGBM defect forecast, and estimated station fault
time. The dashboard calls the REST API; set `MANUFACTURING_API_URL` if the API is not at
`http://127.0.0.1:8001`. The Streamlit UI uses a white/light theme.

### Publish a Shareable UI and API

The repository includes a Render Blueprint in [`render.yaml`](render.yaml). To create a public deployment:

1. Sign in to [Render](https://render.com), choose **New → Blueprint**, and connect this GitHub repository.
2. Deploy the `manufacturing-intelligence-api` web service defined by the Blueprint.
3. After the deploy succeeds, Render will provide the service's public URL. Share that URL for the dashboard; the API documentation is at `<your-public-url>/docs`, and the health check is at `<your-public-url>/health`.

The dashboard and REST API are served by the same service, so its `/api/...` endpoints use that same public domain. There is no live deployment URL configured in this repository yet; the public URL is assigned by the hosting provider after deployment. Before sharing a deployment, configure production API keys and secrets in the host's environment settings rather than exposing local credentials.

### Standalone Supplier Microservice Mode (Port 8001)
```bash
python run_api.py --mode supplier --port 8001
```

---

## 3. Serialized Machine Learning Models (`models/*.pkl`)

All trained machine learning models are serialized in `models/` as standard Python pickle files for fast, in-memory inference without needing to re-train on raw datasets:

| Model File | Class / Engine | Tier & Core Question Addressed |
| :--- | :--- | :--- |
| `models/shift_defect_predictor.pkl` | `ShiftDefectPredictor` | **Tier 1 (Q1):** Shift Defect Count Regressor & Station Risk Ranking |
| `models/failure_mode_classifier.pkl` | `FailureModeClassifier` | **Tier 1 (Q2 & Q4):** Multi-class Failure Mode Classifier & SHAP Attributions |
| `models/prescriptive_action_engine.pkl` | `PrescriptiveActionEngine` | **Tier 1 (Q3):** Machine Recipe Offsets & Preventive Maintenance Actions |
| `models/marginal_batch_classifier.pkl` | `MarginalBatchRiskClassifier` | **Tier 2 (Q5):** Marginal Material Batch Downstream Defect Risk Prediction |
| `models/downstream_impact_forecaster.pkl` | `DownstreamImpactForecaster` | **Tier 2 (Q6):** Downstream Scrap Volume & Defect Severity Regressors |
| `models/adaptive_parameter_compensator.pkl` | `AdaptiveParameterCompensator` | **Tier 2 (Q7):** Physics-Informed Closed-Loop Parameter Offsets |

### Loading and Using Models in Python
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

Supplier and vendor endpoints (`/api/v1/supplier/...`) require authentication using API keys configured in the application.

### Authentication Methods
The API key can be passed using any of the following supported methods:
- **Header:** `X-API-Key: <your_key>`
- **Authorization Header:** `Authorization: Bearer <your_key>`
- **Query Parameter:** `?api_key=<your_key>`

Refer to [`API_DOCUMENTATION.md`](API_DOCUMENTATION.md) for complete details on every endpoint, schemas, and payload examples.

> [!NOTE]
> **Security Note:** Do not commit real API keys or production credentials to a public GitHub repository. Store production secrets in environment variables or a dedicated secrets manager. For local testing, pre-configured local sandbox keys are defined in `data/api_keys.json`.

---

## 5. Running the Interactive Streamlit Dashboard

To launch the multi-tier Six Sigma quality control dashboard:
```bash
streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8000
```
Or on Windows:
```cmd
run_dashboard.bat
```
Access the dashboard in your web browser at:
```
http://localhost:8000
```
The dashboard uses port `8000`; when running it alongside the REST API, launch the API on port `8001` as shown above.

The dashboard loads the serialized models from `models/` and runs inference directly for:
- Shift defect prediction and station ranking (LightGBM)
- Failure-mode classification, SHAP contributors, and prescriptive actions
- Material batch risk (Random Forest), downstream impact (Gradient Boosting), and adaptive compensation
- Supplier trend, warranty trace, tolerance-gap, and SCAR analytics

These are deterministic ML models and data/rule-based analytics; the dashboard does not use a generative AI/LLM. Catalog, inventory, factory-unit, and work-order panels that are identified as reference/demo data are not live predictions. The repository does not currently include trained models or source data for live unit availability/OEE, order-completion forecasting, or production scheduling.

---

## 6. End-to-End System Verification

Run the automated verification test suite:
```bash
python verify_system.py
```

Automated verification suite checks:
- Virtual environment (`venv/`) and required package dependencies
- Integrity and inference of all 6 serialized machine learning models
- API key validation, security rejection, and role-based access control (RBAC)
- FastAPI application and health endpoints
- Simulation and material evaluation pipeline
- Supplier profiles and warranty claim forensic traceability
- Tolerance leakage gap analysis
- 8D Supplier Corrective Action Request (SCAR) reports
- Client SDK integration (`SupplierAPIClient`)

---

## 7. How Others Can Access This API From Their System

There are 4 ways other engineers, teammates, or external systems can connect to this project:

### Method 1: Interactive Swagger UI in a Web Browser
Once the API is running (locally or on cloud), anyone can test all endpoints without writing code by opening:
```
http://<HOST_IP_OR_DOMAIN>:8001/docs
```
*(Provides interactive "Try it out" buttons, parameter forms, and live responses directly in their browser).*

### Method 2: Accessing Over the Same Local Network (Wi-Fi / LAN)
Since `run_api.py` binds to `0.0.0.0`, any device on the same local network can access the API:
1. Find your machine's IP (e.g., run `ipconfig` on Windows or `ifconfig` on Linux/Mac).
2. Teammates open: `http://<YOUR_LOCAL_IP>:8001/docs`

### Method 3: Instant Public Tunnel (Worldwide Access in 30 Seconds)
To share with remote colleagues immediately without deploying to the cloud:
```bash
# Terminal 1: Run the API alongside the dashboard
python run_api.py --port 8001

# Terminal 2: Expose the Streamlit dashboard
npx localtunnel --port 8000
# OR: ngrok http 8000
```
Share the generated public HTTPS URL for the dashboard; expose API port `8001` separately to share the API docs.

### Method 4: 1-Click Containerized Launch (Docker / Docker Compose)
Any teammate with Docker installed on Windows, Mac, or Linux can run the complete project with zero setup:
```bash
# Start both Streamlit dashboard (port 8000) and REST API (port 8001)
docker compose up --build
```
The dashboard is at `http://localhost:8000`; API docs are at `http://localhost:8001/docs`.

### Method 5: Calling the API from Remote Code (Python / cURL)
From another machine or backend system:

**cURL:**
```bash
curl -X POST "http://<HOST>:8001/api/tier1/simulate" \
     -H "Content-Type: application/json" \
     -d '{"station_id":"ST-10","spindle_temp_c":62.0,"system_pressure_psi":115.0,"vibration_rms_g":2.1,"tool_wear_index":55.0,"hours_since_service":18.0,"peak_torque_nm":13.2,"shutoff_angle_deg":285.0,"torque_status":"NOK_HIGH","material_hardness_hv":220.0,"tensile_strength_mpa":645.0}'
```

**Python (Remote Client):**
```python
import requests

API_URL = "http://<HOST>:8001/api/tier1/simulate"
payload = {
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
res = requests.post(API_URL, json=payload)
print(res.json())
```

---

## 8. Project Repository

- **GitHub:** [https://github.com/Gayatrisamanvita/Manuifacturing](https://github.com/Gayatrisamanvita/Manuifacturing)