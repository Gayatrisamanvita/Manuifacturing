"""
Hugging Face Spaces Entry Point (Gradio SDK).
Mounts the FactorySmart AI FastAPI REST API onto Gradio.
Provides both the interactive Web UI and full REST API (/docs, /api/...)
without requiring any credit card or paid Docker subscription.
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import gradio as gr
from src.api import app as fastapi_app
import httpx

# ---------------------------------------------------------------------------
# Gradio Interactive Interface
# ---------------------------------------------------------------------------
with gr.Blocks(title="FactorySmart AI™ - Manufacturing Intelligence API") as demo:
    gr.Markdown(
        """
        # 🏭 FactorySmart AI™ — Manufacturing & Supplier Intelligence API
        
        Welcome to the cloud endpoint for **FactorySmart AI™**. This instance serves machine learning models for:
        - **Tier 1:** Cycle telemetry failure mode prediction & SHAP physical sensor drivers
        - **Tier 2:** Raw material lot defect risk forecasting & adaptive CNC offsets
        - **Tier 3:** Supplier quality trends, warranty forensic trace, and 8D SCAR reports
        
        ---
        
        ### 📡 Remote REST API Access for Other Systems
        Anyone on any system can call this API directly using standard HTTP requests:
        - 📖 **Interactive Swagger UI**: [Open /docs](./docs) *(Test all endpoints with parameter forms)*
        - 📄 **Alternative ReDoc UI**: [Open /redoc](./redoc)
        - 🩺 **System Health Check**: [Open /health](./health)
        
        **Active Authentication Keys for Testing:**
        - Admin Master Key: `fs-admin-master-2026-99x`
        - Vendor B Key: `fs-key-vendor-b-3e7a9d0f`
        - Vendor A Key: `fs-key-vendor-a-8f4b2c1d`
        ---
        """
    )
    
    with gr.Tab("⚡ Quick Model Inference (Tier 1)"):
        gr.Markdown("Submit telemetry directly to test `/api/tier1/simulate`:")
        with gr.Row():
            spindle_temp = gr.Number(value=62.0, label="Spindle Temp (°C)")
            system_pressure = gr.Number(value=115.0, label="System Pressure (psi)")
            vibration = gr.Number(value=2.1, label="Vibration RMS (g)")
            peak_torque = gr.Number(value=13.2, label="Peak Torque (Nm)")
            shutoff_angle = gr.Number(value=285.0, label="Shutoff Angle (deg)")
        
        btn = gr.Button("🚀 Run AI Inference via REST API", variant="primary")
        output_json = gr.JSON(label="API Response & SHAP Attribution Drivers")

        def run_inference(temp, pressure, vib, torque, angle):
            payload = {
                "station_id": "ST-10",
                "spindle_temp_c": float(temp),
                "system_pressure_psi": float(pressure),
                "vibration_rms_g": float(vib),
                "tool_wear_index": 55.0,
                "hours_since_service": 18.0,
                "peak_torque_nm": float(torque),
                "shutoff_angle_deg": float(angle),
                "torque_status": "NOK_HIGH",
                "material_hardness_hv": 220.0,
                "tensile_strength_mpa": 645.0
            }
            # Query FastAPI application directly in-process
            with httpx.Client(transport=httpx.ASGITransport(app=fastapi_app), base_url="http://localhost") as client:
                res = client.post("/api/tier1/simulate", json=payload)
                return res.json()

        btn.click(
            run_inference,
            inputs=[spindle_temp, system_pressure, vibration, peak_torque, shutoff_angle],
            outputs=output_json
        )

# Mount Gradio onto the existing FastAPI application
# - Root "/" serves the Gradio UI
# - "/docs" serves the full interactive OpenAPI / Swagger UI
# - "/api/..." serves all ML inference endpoints
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
