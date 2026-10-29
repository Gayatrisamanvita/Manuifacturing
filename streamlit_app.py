"""Streamlit dashboard for the repository's trained manufacturing ML models."""

import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
API_BASE_URL = os.environ.get("MANUFACTURING_API_URL", "http://127.0.0.1:8001").rstrip("/")


@st.cache_resource
def load_dashboard_models():
    from src.data_loader import DataLoader
    from src.models import load_all_models

    models = load_all_models(models_dir=MODELS_DIR, data_dir=DATA_DIR)
    forecaster = models.get("downstream_impact_forecaster")
    if forecaster is not None and not forecaster.reference_batch_size:
        batches = DataLoader(data_dir=DATA_DIR).get_material_batch_summary()
        forecaster.reference_batch_size = float(batches["total_parts"].mean())
    return models


@st.cache_data
def load_dashboard_data():
    from src.data_loader import DataLoader

    data_loader = DataLoader(data_dir=DATA_DIR)
    material = data_loader.load_raw("material_batches")
    supplier_history = data_loader.load_raw("supplier_history")
    warranty = data_loader.load_raw("warranty_claims")
    catalog = data_loader.load_raw("supplier_products")
    warranty = warranty.merge(
        material[["material_batch_id", "supplier_id"]],
        on="material_batch_id",
        how="left",
    )
    return {
        "material": material,
        "supplier_history": supplier_history,
        "warranty": warranty,
        "catalog": catalog,
    }


def _models():
    try:
        return load_dashboard_models()
    except Exception as exc:
        st.error(f"Could not load the inference models: {type(exc).__name__}: {exc}")
        return None


@st.cache_data(ttl=60)
def _api_get(path: str):
    response = requests.get(f"{API_BASE_URL}{path}", timeout=30)
    response.raise_for_status()
    return response.json()


def _api_post(path: str, payload: dict):
    response = requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=60)
    response.raise_for_status()
    return response.json()


def _api_error_message(exc: Exception) -> str:
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        try:
            detail = exc.response.json().get("detail", exc.response.text)
        except ValueError:
            detail = exc.response.text
        return f"API request failed ({exc.response.status_code}): {detail}"
    return f"Could not reach the Manufacturing API at {API_BASE_URL}: {exc}"


def _batch_feature_frame(material: pd.DataFrame) -> pd.DataFrame:
    frame = material.copy()
    frame["hardness_margin_upper"] = frame["upper_spec_limit"] - frame["material_hardness_hv"]
    frame["hardness_margin_lower"] = frame["material_hardness_hv"] - frame["lower_spec_limit"]
    frame["hardness_nominal_dev"] = np.abs(
        frame["material_hardness_hv"]
        - (frame["upper_spec_limit"] + frame["lower_spec_limit"]) / 2.0
    )
    frame["tensile_to_hardness_ratio"] = frame["tensile_strength_mpa"] / (
        frame["material_hardness_hv"] + 1e-4
    )
    for supplier in ("VENDOR_A", "VENDOR_B", "VENDOR_C"):
        frame[f"supplier_{supplier}"] = (frame["supplier_id"] == supplier).astype(float)
    return frame


def _historical_batch_scores(material: pd.DataFrame, model) -> pd.DataFrame:
    frame = _batch_feature_frame(material)
    features = frame.reindex(columns=model.feature_names, fill_value=0.0)
    probabilities = model.model.predict_proba(features)[:, 1]
    frame["model_risk_probability"] = probabilities
    frame["model_risk_tier"] = np.select(
        [probabilities >= 0.6, probabilities >= 0.3],
        ["HIGH_RISK_MARGINAL", "BORDERLINE_MONITOR"],
        default="STANDARD_PASS",
    )
    frame.loc[
        (frame["material_hardness_hv"] < frame["lower_spec_limit"])
        | (frame["material_hardness_hv"] > frame["upper_spec_limit"]),
        "model_risk_tier",
    ] = "OUT_OF_SPEC_REJECT"
    return frame


def _evaluate_material_batch(
    supplier_id: str,
    hardness: float,
    tensile: float,
    batch_size: int,
    lower_limit: float = 180.0,
    upper_limit: float = 230.0,
):
    return _api_post("/api/tier2/evaluate_material", {
        "material_hardness_hv": hardness,
        "tensile_strength_mpa": tensile,
        "lower_spec_limit": lower_limit,
        "upper_spec_limit": upper_limit,
        "supplier_id": supplier_id,
        "batch_size_units": batch_size,
    })


def _risk_probability_styles(values: pd.Series) -> list[str]:
    styles = []
    for probability in values:
        high, moderate = (60, 30) if probability > 1 else (0.6, 0.3)
        if probability >= high:
            color = "#FECACA"
        elif probability >= moderate:
            color = "#FEF3C7"
        else:
            color = "#DCFCE7"
        styles.append(f"background-color: {color}; color: #1F2937;")
    return styles


def _failure_mode_features(
    station_id: str,
    spindle_temp_c: float,
    system_pressure_psi: float,
    vibration_rms_g: float,
    tool_wear_index: float,
    hours_since_service: float,
    peak_torque_nm: float,
    shutoff_angle_deg: float,
    torque_status: str,
    material_hardness_hv: float,
    tensile_strength_mpa: float,
) -> pd.DataFrame:
    row = {
        "peak_torque_nm": peak_torque_nm,
        "shutoff_angle_deg": shutoff_angle_deg,
        "spindle_temp_c": spindle_temp_c,
        "system_pressure_psi": system_pressure_psi,
        "vibration_rms_g": vibration_rms_g,
        "tool_wear_index": tool_wear_index,
        "hours_since_service": hours_since_service,
        "torque_energy_proxy": peak_torque_nm * np.deg2rad(shutoff_angle_deg),
        "tool_stress_factor": tool_wear_index * vibration_rms_g,
        "thermal_vibration_index": spindle_temp_c * vibration_rms_g,
        "hydraulic_thermal_ratio": system_pressure_psi / (spindle_temp_c + 1e-3),
        "material_tool_resistance": material_hardness_hv * (tool_wear_index + 1.0),
        "torque_delta_nominal": peak_torque_nm - 12.5,
        "angle_delta_nominal": shutoff_angle_deg - 360.0,
        "torque_status_nok": float(torque_status != "OK"),
        "torque_status_high": float(torque_status == "NOK_HIGH"),
        "torque_status_low": float(torque_status == "NOK_LOW"),
        "material_hardness_hv": material_hardness_hv,
        "tensile_strength_mpa": tensile_strength_mpa,
    }
    row.update({
        f"st_{candidate}": float(station_id == candidate)
        for candidate in ("ST-10", "ST-20", "ST-30", "ST-40")
    })
    return pd.DataFrame([row])


def _render_supplier_tab(models, data):
    st.header("Supplier / Vendor")
    suppliers = sorted(data["supplier_history"]["supplier_id"].unique())
    supplier_id = st.selectbox("Supplier", suppliers, key="supplier_selector")
    try:
        profiles = _api_get("/api/tier3/suppliers")
        catalog_response = _api_get("/api/tier3/catalog")
    except Exception as exc:
        st.error(_api_error_message(exc))
        return
    profile = profiles.get(supplier_id)
    if profile is None:
        st.error(f"The supplier API returned no profile for {supplier_id}.")
        return
    supplier_batches = data["material"].query("supplier_id == @supplier_id")
    supplier_catalog = pd.DataFrame(
        catalog_response.get("products_by_supplier", {}).get(supplier_id, [])
    )
    if supplier_catalog.empty:
        st.warning(f"The supplier API returned no catalog products for {supplier_id}.")
        return
    supplier_catalog["delay_days"] = pd.to_numeric(
        supplier_catalog["delay_days"], errors="raise"
    )
    most_delayed = supplier_catalog.sort_values(
        ["delay_days", "actual_lead_days"], ascending=False
    ).iloc[0]
    scored_batches = _historical_batch_scores(
        supplier_batches, models["marginal_batch_classifier"]
    )
    claims = data["warranty"].query("supplier_id == @supplier_id")

    st.subheader("Most-delayed product")
    product_cols = st.columns([2, 1, 1, 1])
    product_cols[0].markdown(
        f"**Product**  \n{most_delayed['product_name']}  \n"
        f"`{most_delayed['sku']}`"
    )
    product_cols[1].metric("Delay", f"{most_delayed['delay_days']:.0f} days")
    product_cols[2].metric(
        "Actual / target lead time",
        f"{most_delayed['actual_lead_days']} / {most_delayed['target_lead_days']} days",
    )
    product_cols[3].metric("Delivery risk", most_delayed["risk_level"])

    metric_cols = st.columns(4)
    metric_cols[0].metric(
        "Latest recorded defect rate",
        f"{profile['latest_recorded_defect_rate_pct']:.2f}%",
    )
    metric_cols[1].metric("Historical trend", profile["historical_trend"])
    metric_cols[2].metric("Risk rating", profile["risk_rating"])
    metric_cols[3].metric("Warranty claim cost", f"${claims['claim_cost_usd'].sum():,.0f}")
    st.caption(
        f"Supplier trend history ends {profile['data_as_of']}; this compares the first and last "
        "available snapshots and is not a current rolling 30-day forecast."
    )

    st.subheader("Recorded product catalog")
    st.caption(
        "Catalog and lead times are recorded source data, not ML predictions. "
        "Products are sorted by longest recorded delivery delay."
    )
    st.dataframe(
        supplier_catalog[
            [
                "sku",
                "product_name",
                "category",
                "target_lead_days",
                "actual_lead_days",
                "delay_days",
                "quality_score_pct",
                "risk_level",
            ]
        ].sort_values(
            ["delay_days", "actual_lead_days"], ascending=False
        ).rename(columns={
            "sku": "SKU",
            "product_name": "Product",
            "category": "Category",
            "target_lead_days": "Target lead (days)",
            "actual_lead_days": "Actual lead (days)",
            "delay_days": "Delay (days)",
            "quality_score_pct": "Recorded quality score (%)",
            "risk_level": "Recorded delivery risk",
        }),
        width="stretch",
        hide_index=True,
    )

    st.subheader("Individual product material-risk analysis")
    st.caption(
        "The material models use hardness, tensile strength, specification limits, and supplier. "
        "The catalog has no SKU-to-batch measurement link, so enter the receiving measurements "
        "for this product's batch. Apply these models only to material comparable to the model's "
        "metal-batch training data."
    )
    product_options = supplier_catalog["sku"].tolist()
    default_hardness = float(supplier_batches["material_hardness_hv"].median())
    default_tensile = float(supplier_batches["tensile_strength_mpa"].median())
    with st.form("supplier_product_material_analysis"):
        product_sku = st.selectbox(
            "Product to evaluate",
            product_options,
            format_func=lambda sku: (
                f"{sku} — "
                f"{supplier_catalog.loc[supplier_catalog['sku'] == sku, 'product_name'].iloc[0]}"
            ),
        )
        input_cols = st.columns(4)
        with input_cols[0]:
            hardness = st.number_input(
                "Incoming hardness (HV)",
                min_value=100.0,
                max_value=400.0,
                value=default_hardness,
            )
        with input_cols[1]:
            tensile = st.number_input(
                "Incoming tensile strength (MPa)",
                min_value=200.0,
                max_value=1500.0,
                value=default_tensile,
            )
        with input_cols[2]:
            batch_size = st.number_input(
                "Batch size (units)", min_value=1, value=1200, step=100
            )
        with input_cols[3]:
            st.markdown("**Specification limits (HV)**")
            lower_limit = st.number_input(
                "Lower limit", min_value=0.0, value=180.0, key="supplier_product_lsl"
            )
            upper_limit = st.number_input(
                "Upper limit", min_value=0.1, value=230.0, key="supplier_product_usl"
            )
        evaluate_product = st.form_submit_button("Run product ML analysis")

    if evaluate_product:
        if lower_limit >= upper_limit:
            st.error("The lower specification limit must be less than the upper specification limit.")
        else:
            try:
                result = _evaluate_material_batch(
                    supplier_id,
                    hardness,
                    tensile,
                    int(batch_size),
                    lower_limit,
                    upper_limit,
                )
                st.markdown(f"**ML evaluation for {product_sku}**")
                result_cols = st.columns(3)
                result_cols[0].write("**Random Forest risk classifier**")
                result_cols[0].json(result["risk_evaluation"])
                result_cols[1].write("**Gradient Boosting impact forecast**")
                result_cols[1].json(result["downstream_forecast"])
                result_cols[2].write("**Adaptive compensation rules**")
                result_cols[2].json(result["adaptive_compensation"])
            except Exception as exc:
                st.error(f"Product material model inference failed: {type(exc).__name__}: {exc}")

    st.subheader("Supplier batches scored by the Random Forest")
    st.caption(
        "Historical batch predictions use recorded measurements and are retrospective inference, "
        "not a prospective supplier-quality forecast."
    )
    st.metric(
        "Mean predicted batch risk",
        f"{scored_batches['model_risk_probability'].mean():.1%}",
    )
    supplier_score_view = scored_batches[
        [
            "material_batch_id",
            "material_hardness_hv",
            "tensile_strength_mpa",
            "model_risk_probability",
            "model_risk_tier",
        ]
    ].rename(
        columns={
            "material_batch_id": "Batch ID",
            "material_hardness_hv": "Hardness (HV)",
            "tensile_strength_mpa": "Tensile (MPa)",
            "model_risk_probability": "Predicted risk probability",
            "model_risk_tier": "Predicted tier",
        }
    )
    st.dataframe(
        supplier_score_view.style.apply(
            _risk_probability_styles,
            subset=["Predicted risk probability"],
        ).format({"Predicted risk probability": "{:.1%}"}),
        width="stretch",
        hide_index=True,
    )

    with st.expander("Warranty trace, batch trace, SCAR, and tolerance analysis"):
        with st.form("supplier_analysis"):
            claim_id = st.text_input("Warranty claim ID")
            batch_id = st.text_input("Material batch ID")
            generate_scar = st.checkbox("Generate supplier SCAR")
            submit = st.form_submit_button("Run analysis")
        if submit:
            if claim_id.strip():
                trace = models["trace_engine"].trace_claim(claim_id.strip())
                if trace is None:
                    st.warning(f"No claim found for {claim_id.strip()}.")
                else:
                    st.subheader("Warranty genealogy")
                    st.json(trace)
            if batch_id.strip():
                trace = models["trace_engine"].trace_batch(batch_id.strip())
                if trace.get("supplier_id") == "UNKNOWN":
                    st.warning(f"No material batch found for {batch_id.strip()}.")
                else:
                    st.subheader("Batch warranty trace")
                    st.json(trace)
            if generate_scar:
                st.subheader("Generated SCAR")
                st.json(models["scar_generator"].generate_scar(supplier_id))
        st.subheader("Tolerance leakage analysis")
        st.json(models["gap_analyzer"].run_gap_analysis())


def _render_material_tab(models, data):
    st.header("Raw Material")
    try:
        material_monitor = _api_get("/api/tier2/materials")
    except Exception as exc:
        st.error(_api_error_message(exc))
        return
    st.caption(
        "Quality risk is predicted by the trained Random Forest through the API. Incoming lot "
        "counts, inspection status, scrap, and chargebacks are recorded values. No on-hand "
        "inventory or usage history is available to forecast stock exhaustion."
    )
    summary = material_monitor["inspection_summary"]
    supplier_batches = pd.DataFrame(material_monitor["supplier_batches"])
    material_metrics = st.columns(5)
    material_metrics[0].metric(
        "Recorded incoming batches", f"{material_monitor['batch_count']:,}"
    )
    material_metrics[1].metric(
        "Latest recorded receipt", material_monitor["latest_recorded_batch_date"]
    )
    material_metrics[2].metric(
        "Model high-risk batches", f"{material_monitor['high_risk_count']:,}"
    )
    material_metrics[3].metric(
        "Recorded scrap parts", f"{material_monitor['recorded_scrap_count']:,}"
    )
    material_metrics[4].metric(
        "Recorded chargebacks", f"${material_monitor['recorded_chargebacks_usd']:,.0f}"
    )
    st.info(material_monitor["supplier_availability_basis"])

    st.subheader("Incoming material availability by supplier")
    st.caption("Availability here means recorded incoming-batch volume; it is not warehouse stock on hand.")
    if not supplier_batches.empty:
        st.bar_chart(
            supplier_batches.set_index("supplier_id")[["recorded_batch_count"]],
            x_label="Supplier",
            y_label="Recorded incoming batches",
            color="#2563EB",
        )
        st.dataframe(
            supplier_batches.rename(
                columns={
                    "supplier_id": "Supplier",
                    "recorded_batch_count": "Recorded incoming batches",
                    "high_risk_batch_count": "Model high-risk batches",
                    "out_of_spec_batch_count": "Out-of-spec batches",
                    "mean_predicted_risk_pct": "Mean predicted quality risk (%)",
                    "recorded_chargebacks_usd": "Recorded chargebacks (USD)",
                }
            ).style.apply(
                _risk_probability_styles,
                subset=["Mean predicted quality risk (%)"],
            ).format({
                "Mean predicted quality risk (%)": "{:.1f}%",
                "Recorded chargebacks (USD)": "${:,.2f}",
            }),
            width="stretch",
            hide_index=True,
        )

    st.subheader("Incoming inspection and quality")
    inspection_cols = st.columns(4)
    inspection_cols[0].metric(
        "Out-of-spec hardness batches",
        f"{summary['out_of_spec_hardness_batches']:,}",
    )
    inspection_cols[1].metric(
        "Recorded marginal-high lots",
        f"{summary['marginal_high_recorded_batches']:,}",
    )
    inspection_cols[2].metric(
        "Recorded marginal-low lots",
        f"{summary['marginal_low_recorded_batches']:,}",
    )
    inspection_cols[3].metric(
        "Recorded normal lots",
        f"{summary['normal_recorded_batches']:,}",
    )

    st.subheader("Live batch evaluation")
    st.caption("Calls POST /api/tier2/evaluate_material: Random Forest, Gradient Boosting, and rule-based compensation.")
    with st.form("material_model_inference"):
        col1, col2, col3 = st.columns(3)
        with col1:
            batch_id = st.text_input("Batch ID", value="INBOUND-BATCH")
            supplier_id = st.selectbox("Supplier", ("VENDOR_A", "VENDOR_B", "VENDOR_C"), index=1)
            batch_size = st.number_input("Batch size (units)", min_value=1, value=1200, step=100)
        with col2:
            hardness = st.number_input("Hardness (HV)", min_value=100.0, max_value=400.0, value=222.0)
            tensile = st.number_input("Tensile strength (MPa)", min_value=200.0, max_value=1500.0, value=660.0)
        with col3:
            lower_limit = st.number_input("Lower specification limit (HV)", min_value=0.0, value=180.0)
            upper_limit = st.number_input("Upper specification limit (HV)", min_value=0.1, value=230.0)
        submit = st.form_submit_button("Evaluate batch")

    if submit:
        if lower_limit >= upper_limit:
            st.error("The lower specification limit must be less than the upper specification limit.")
        else:
            try:
                result = _evaluate_material_batch(
                    supplier_id,
                    hardness,
                    tensile,
                    int(batch_size),
                    lower_limit,
                    upper_limit,
                )
                st.subheader(f"Evaluation for {batch_id}")
                st.write("Random Forest batch-risk prediction")
                st.json(result["risk_evaluation"])
                st.write("Gradient Boosting downstream forecast")
                st.json(result["downstream_forecast"])
                st.write("Adaptive recipe compensation (rules, not ML)")
                st.json(result["adaptive_compensation"])
            except Exception as exc:
                st.error(_api_error_message(exc))

    st.subheader("Historical incoming batches scored by the Random Forest")
    scored = pd.DataFrame(material_monitor["materials"])
    st.caption(
        "Color indicates model-predicted batch quality risk (green low, amber medium, red high). "
        "Cost is recorded supplier chargeback, not purchase or carrying cost."
    )
    display = scored[
            [
                "material_batch_id",
                "supplier_id",
                "material_hardness_hv",
                "tensile_strength_mpa",
                "material_risk_status",
                "model_risk_tier",
                "model_risk_probability",
                "scrap_count",
                "chargeback_amount_usd",
            ]
        ].rename(
            columns={
                "material_batch_id": "Batch ID",
                "supplier_id": "Supplier",
                "material_hardness_hv": "Hardness (HV)",
                "tensile_strength_mpa": "Tensile (MPa)",
                "material_risk_status": "Recorded QC status",
                "model_risk_tier": "Predicted tier",
                "model_risk_probability": "Predicted quality risk (%)",
                "scrap_count": "Recorded scrap",
                "chargeback_amount_usd": "Recorded chargeback (USD)",
            }
        )
    display["Predicted quality risk (%)"] *= 100
    st.dataframe(
        display.style.apply(
            _risk_probability_styles,
            subset=["Predicted quality risk (%)"],
        ).format({
            "Predicted quality risk (%)": "{:.1f}%",
            "Recorded chargeback (USD)": "${:,.2f}",
        }),
        width="stretch",
        hide_index=True,
    )


def _render_manufacturing_tab(models):
    st.header("Manufacturing: Stations and Machines")
    try:
        monitoring = _api_get("/api/tier1/operations/monitor")
    except Exception as exc:
        st.error(_api_error_message(exc))
        return
    st.caption(f"Recorded manufacturing monitor as of {monitoring['as_of_date']}.")
    summary = monitoring["summary"]
    kpis = st.columns(5)
    kpis[0].metric("Recorded output", f"{summary['total_units_produced']:,} units")
    kpis[1].metric("Production-log defect rate", f"{summary['logged_defect_rate_pct']:.2f}%")
    kpis[2].metric("Inspection rejection rate", f"{summary['inspection_rejection_rate_pct']:.2f}%")
    kpis[3].metric("Estimated faulted time", f"{summary['estimated_faulted_hours']:.2f} h")
    kpis[4].metric("Faulted telemetry samples", f"{summary['faulted_telemetry_samples']:,}")
    st.info(monitoring["data_basis"])
    shift_station_frame = pd.DataFrame(monitoring["stations"])
    station_frame = shift_station_frame.groupby("station_id", as_index=False).agg(
        units_produced=("units_produced", "sum"),
        defect_count=("defect_count", "sum"),
        inspected_parts=("inspected_parts", "sum"),
        rejected_parts=("rejected_parts", "sum"),
        telemetry_samples=("telemetry_samples", "sum"),
        running_samples=("running_samples", "sum"),
        faulted_samples=("faulted_samples", "sum"),
        estimated_faulted_hours=("estimated_faulted_hours", "sum"),
        mean_spindle_temp_c=("mean_spindle_temp_c", "mean"),
        mean_system_pressure_psi=("mean_system_pressure_psi", "mean"),
        mean_vibration_rms_g=("mean_vibration_rms_g", "mean"),
    )
    station_frame["defect_rate_pct"] = (
        station_frame["defect_count"] / station_frame["units_produced"].replace(0, np.nan) * 100
    ).fillna(0)
    station_frame["inspection_rejection_rate_pct"] = (
        station_frame["rejected_parts"]
        / station_frame["inspected_parts"].replace(0, np.nan)
        * 100
    ).fillna(0)
    station_frame["running_telemetry_pct"] = (
        station_frame["running_samples"]
        / station_frame["telemetry_samples"].replace(0, np.nan)
        * 100
    ).fillna(0)
    station_failure_modes = {}
    station_supplier_issues = {}
    for row in shift_station_frame.itertuples(index=False):
        counts = station_failure_modes.setdefault(row.station_id, {})
        for mode in row.top_ml_failure_modes:
            label = mode["predicted_failure_mode"]
            counts[label] = counts.get(label, 0) + mode["predicted_cycle_count"]
        suppliers = station_supplier_issues.setdefault(row.station_id, {})
        for issue in row.top_supplier_issues:
            supplier = issue["supplier_id"]
            aggregate = suppliers.setdefault(
                supplier,
                {
                    "supplier_id": supplier,
                    "inspected_parts": 0,
                    "rejected_parts": 0,
                    "parts_from_marginal_high_batches": 0,
                    "hardness_sum": 0.0,
                    "hardness_count": 0,
                },
            )
            aggregate["inspected_parts"] += issue["supplier_inspected_parts"]
            aggregate["rejected_parts"] += issue["supplier_rejected_parts"]
            aggregate["parts_from_marginal_high_batches"] += issue[
                "parts_from_marginal_high_batches"
            ]
            aggregate["hardness_sum"] += (
                issue["average_hardness_hv"] * issue["supplier_inspected_parts"]
            )
            aggregate["hardness_count"] += issue["supplier_inspected_parts"]
    station_frame["top_ml_failure_modes"] = station_frame["station_id"].map(
        lambda station_id: [
            {"predicted_failure_mode": mode, "predicted_cycle_count": count}
            for mode, count in sorted(
                station_failure_modes.get(station_id, {}).items(),
                key=lambda item: item[1],
                reverse=True,
            )[:3]
        ]
    )
    def supplier_issue_summary(station_id):
        issues = station_supplier_issues.get(station_id, {}).values()
        ranked = sorted(issues, key=lambda item: item["rejected_parts"], reverse=True)[:3]
        for issue in ranked:
            issue["rejection_rate_pct"] = (
                issue["rejected_parts"] / max(1, issue["inspected_parts"]) * 100
            )
            issue["average_hardness_hv"] = (
                issue["hardness_sum"] / max(1, issue["hardness_count"])
            )
            issue.pop("hardness_sum")
            issue.pop("hardness_count")
        return ranked

    station_frame["top_supplier_issues"] = station_frame["station_id"].map(
        supplier_issue_summary
    )
    st.subheader("Machine, quality, and supplier-linked station monitor")
    chart = px.bar(
        station_frame,
        x="station_id",
        y="estimated_faulted_hours",
        color="inspection_rejection_rate_pct",
        color_continuous_scale=["#34D399", "#FBBF24", "#EF4444"],
        hover_data=[
            "running_telemetry_pct",
            "units_produced",
            "defect_rate_pct",
            "inspection_rejection_rate_pct",
            "mean_spindle_temp_c",
            "mean_vibration_rms_g",
        ],
        labels={
            "station_id": "Station",
            "estimated_faulted_hours": "Estimated faulted hours",
            "inspection_rejection_rate_pct": "Inspection rejection (%)",
        },
        title="Estimated machine fault time by station; color shows inspection rejection",
    )
    st.plotly_chart(chart, width="stretch")
    st.dataframe(
        station_frame[
            [
                "station_id",
                "units_produced",
                "defect_count",
                "defect_rate_pct",
                "inspected_parts",
                "rejected_parts",
                "inspection_rejection_rate_pct",
                "running_telemetry_pct",
                "estimated_faulted_hours",
                "mean_spindle_temp_c",
                "mean_system_pressure_psi",
                "mean_vibration_rms_g",
                "top_ml_failure_modes",
                "top_supplier_issues",
            ]
        ].assign(
            top_ml_failure_modes=lambda frame: frame["top_ml_failure_modes"].apply(
                lambda modes: ", ".join(
                    f"{mode['predicted_failure_mode']} ({mode['predicted_cycle_count']})"
                    for mode in modes
                )
            ),
            top_supplier_issues=lambda frame: frame["top_supplier_issues"].apply(
                lambda issues: ", ".join(
                    f"{issue['supplier_id']} ({issue['rejected_parts']} rejects)"
                    for issue in issues
                )
            ),
        ).rename(
            columns={
                "station_id": "Station",
                "units_produced": "Recorded output",
                "defect_count": "Logged defects",
                "defect_rate_pct": "Production-log defect rate (%)",
                "inspected_parts": "Inspected parts",
                "rejected_parts": "Rejected parts",
                "inspection_rejection_rate_pct": "Inspection rejection (%)",
                "running_telemetry_pct": "Running telemetry (%)",
                "estimated_faulted_hours": "Estimated faulted hours",
                "mean_spindle_temp_c": "Mean temperature (°C)",
                "mean_system_pressure_psi": "Mean pressure (psi)",
                "mean_vibration_rms_g": "Mean vibration (g)",
                "top_ml_failure_modes": "Top LightGBM predicted modes",
                "top_supplier_issues": "Top supplier-linked issues",
            }
        ).style.apply(
            _risk_probability_styles,
            subset=["Inspection rejection (%)"],
        ).format({
            "Production-log defect rate (%)": "{:.2f}%",
            "Inspection rejection (%)": "{:.2f}%",
            "Running telemetry (%)": "{:.1f}%",
            "Estimated faulted hours": "{:.2f}",
        }),
        width="stretch",
        hide_index=True,
    )
    st.subheader("Live station diagnosis")
    st.caption("Calls POST /api/tier1/simulate for LightGBM + SHAP and the rule-based action engine.")
    with st.form("station_model_inference"):
        col1, col2, col3 = st.columns(3)
        with col1:
            station_id = st.selectbox("Station", ("ST-10", "ST-20", "ST-30", "ST-40"))
            temp = st.number_input("Spindle temperature (°C)", value=62.0)
            pressure = st.number_input("System pressure (psi)", value=115.0)
            vibration = st.number_input("Vibration RMS (g)", value=2.1)
        with col2:
            wear = st.number_input("Tool wear index", value=55.0)
            service_hours = st.number_input("Hours since service", value=18.0)
            torque = st.number_input("Peak torque (Nm)", value=13.2)
            angle = st.number_input("Shutoff angle (degrees)", value=285.0)
        with col3:
            torque_status = st.selectbox("Torque status", ("OK", "NOK_HIGH", "NOK_LOW"), index=1)
            hardness = st.number_input("Material hardness (HV)", value=220.0)
            tensile = st.number_input("Tensile strength (MPa)", value=645.0)
        submit = st.form_submit_button("Predict failure mode and prescribe action")

    if submit:
        telemetry = {
            "spindle_temp_c": temp,
            "system_pressure_psi": pressure,
            "vibration_rms_g": vibration,
            "tool_wear_index": wear,
            "hours_since_service": service_hours,
            "peak_torque_nm": torque,
            "shutoff_angle_deg": angle,
            "torque_status": torque_status,
            "material_hardness_hv": hardness,
            "tensile_strength_mpa": tensile,
        }
        try:
            result = _api_post("/api/tier1/simulate", {
                "station_id": station_id,
                **telemetry,
            })
            left, right = st.columns(2)
            left.metric("Predicted failure mode", result["predicted_failure_mode"])
            right.metric("Model confidence", f"{result['confidence']:.1%}")
            st.write("Top SHAP contributors")
            st.dataframe(
                pd.DataFrame(result["top_physical_contributors"]),
                width="stretch",
                hide_index=True,
            )
            st.write("Prescriptive action (rule-based engine)")
            st.json(result["prescriptive_action"])
        except Exception as exc:
            st.error(_api_error_message(exc))


def _render_production_tab(models):
    try:
        monitoring = _api_get("/api/tier1/operations/monitor")
        production_summary = _api_get("/api/tier1/production/summary")
    except Exception as exc:
        st.error(_api_error_message(exc))
        return

    st.header("Production: Shifts and Lines")
    st.caption(
        f"Recorded production history through {monitoring['as_of_date']}. Enter a planned shift "
        "target below to compare against the latest recorded shift and get an ML defect-risk forecast."
    )
    station_frame = pd.DataFrame(monitoring["stations"])
    shifts = sorted(station_frame["shift_id"].dropna().unique())
    selected_shift = st.selectbox("Shift to review", shifts, key="production_shift_review")
    selected = station_frame[station_frame["shift_id"] == selected_shift].copy()
    actual_units = int(selected["units_produced"].sum())
    actual_defects = int(selected["defect_count"].sum())
    actual_inspected = int(selected["inspected_parts"].sum())
    actual_rejected = int(selected["rejected_parts"].sum())
    shift_kpis = st.columns(5)
    shift_kpis[0].metric("Recorded output", f"{actual_units:,} units")
    shift_kpis[1].metric("Logged defects", f"{actual_defects:,}")
    shift_kpis[2].metric(
        "Production-log defect rate",
        f"{actual_defects / max(1, actual_units) * 100:.2f}%",
    )
    shift_kpis[3].metric("Inspection rejects", f"{actual_rejected:,} / {actual_inspected:,}")
    shift_kpis[4].metric(
        "Inspection rejection rate",
        f"{actual_rejected / max(1, actual_inspected) * 100:.2f}%",
    )
    st.caption(
        f"All-period production summary: {production_summary['total_units_produced']:,} units, "
        f"{production_summary['total_defects']:,} logged defects, "
        f"{production_summary['defect_rate_pct']:.2f}% production-log defect rate."
    )
    st.subheader("Output and downtime by station")
    prod_chart = px.bar(
        selected,
        x="station_id",
        y="units_produced",
        color="estimated_faulted_hours",
        color_continuous_scale=["#34D399", "#FBBF24", "#EF4444"],
        hover_data=[
            "defect_count",
            "defect_rate_pct",
            "inspection_rejection_rate_pct",
            "estimated_faulted_hours",
            "running_telemetry_pct",
        ],
        labels={
            "station_id": "Station",
            "units_produced": "Recorded output (units)",
            "estimated_faulted_hours": "Estimated faulted hours",
        },
        title=f"{selected_shift}: recorded output; color shows estimated faulted hours",
    )
    st.plotly_chart(prod_chart, width="stretch")
    st.dataframe(
        selected[
            [
                "station_id",
                "units_produced",
                "defect_count",
                "defect_rate_pct",
                "inspected_parts",
                "rejected_parts",
                "inspection_rejection_rate_pct",
                "estimated_faulted_hours",
            ]
        ].rename(
            columns={
                "station_id": "Station",
                "units_produced": "Recorded output",
                "defect_count": "Logged defects",
                "defect_rate_pct": "Production-log defect rate (%)",
                "inspected_parts": "Inspected parts",
                "rejected_parts": "Rejected parts",
                "inspection_rejection_rate_pct": "Inspection rejection (%)",
                "estimated_faulted_hours": "Estimated faulted hours",
            }
        ).style.apply(
            _risk_probability_styles,
            subset=["Inspection rejection (%)"],
        ).format({
            "Production-log defect rate (%)": "{:.2f}%",
            "Inspection rejection (%)": "{:.2f}%",
            "Estimated faulted hours": "{:.2f}",
        }),
        width="stretch",
        hide_index=True,
    )
    st.subheader("Production target and next-shift defect forecast")
    st.caption("Calls POST /api/tier1/predict_shift for the trained Poisson LightGBM model.")
    with st.form("shift_model_inference"):
        top1, top2, top3 = st.columns(3)
        with top1:
            shift_id = st.selectbox("Forecast shift", ("SHIFT_1", "SHIFT_2", "SHIFT_3"))
            planned_units = st.number_input(
                "Planned target per station (units)",
                min_value=0.0,
                value=float(max(1, int(selected["units_produced"].mean()))),
                step=50.0,
            )
        with top2:
            line_speed = st.number_input("Line speed (units/min)", min_value=0.0, value=47.2)
            speed_variation = st.number_input("Line speed standard deviation", min_value=0.0, value=1.8)
        with top3:
            operator_tenure = st.number_input("Operator tenure (years)", min_value=0.0, value=5.0)

        station_rows = []
        station_columns = st.columns(4)
        for index, station_id in enumerate(("ST-10", "ST-20", "ST-30", "ST-40")):
            with station_columns[index]:
                st.markdown(f"**{station_id} telemetry**")
                station_rows.append({
                    "station_id": station_id,
                    "operator_tenure_yrs": operator_tenure,
                    "spindle_temp_c": st.number_input("Temperature (°C)", value=58.0, key=f"shift_temp_{station_id}"),
                    "max_spindle_temp_c": st.number_input("Peak temperature (°C)", value=62.0, key=f"shift_peak_temp_{station_id}"),
                    "vibration_rms_g": st.number_input("Vibration (g)", value=1.8, key=f"shift_vibration_{station_id}"),
                    "max_vibration_rms_g": st.number_input("Peak vibration (g)", value=2.1, key=f"shift_peak_vibration_{station_id}"),
                    "system_pressure_psi": st.number_input("Pressure (psi)", value=115.0, key=f"shift_pressure_{station_id}"),
                    "tool_wear_index": st.number_input("Tool wear", value=55.0, key=f"shift_wear_{station_id}"),
                    "max_tool_wear_index": st.number_input("Peak tool wear", value=58.0, key=f"shift_peak_wear_{station_id}"),
                    "peak_torque_nm": st.number_input("Peak torque (Nm)", value=12.5, key=f"shift_torque_{station_id}"),
                    "std_peak_torque_nm": st.number_input("Torque standard deviation", value=0.2, key=f"shift_torque_std_{station_id}"),
                    "shutoff_angle_deg": st.number_input("Shutoff angle (degrees)", value=360.0, key=f"shift_angle_{station_id}"),
                })
        submit = st.form_submit_button("Predict shift defects and rank stations")

    if submit:
        try:
            result = _api_post("/api/tier1/predict_shift", {
                "shift_id": shift_id,
                "planned_units": planned_units,
                "line_speed_upm": line_speed,
                "std_line_speed_upm": speed_variation,
                "stations": station_rows,
            })
            target_cols = st.columns(3)
            forecast_shift_actual = int(
                station_frame.loc[station_frame["shift_id"] == shift_id, "units_produced"].sum()
            )
            target_total = planned_units * len(station_rows)
            predicted_defects = sum(
                entry["predicted_defect_count"]
                for entry in result["station_risk_ranking"]
            )
            target_cols[0].metric("Planned line target", f"{target_total:,.0f} units")
            target_cols[1].metric(
                "Forecast-shift recorded output", f"{forecast_shift_actual:,} units"
            )
            target_cols[2].metric(
                "Historical target attainment",
                f"{forecast_shift_actual / target_total:.1%}" if target_total else "Target is zero",
            )
            st.caption(
                f"Poisson LightGBM predicted approximately {predicted_defects:.1f} defects across "
                "the four stations. Target attainment compares your target with this recorded "
                "historical shift; the target is not a stored work-order plan."
            )
            ranking = pd.DataFrame(result["station_risk_ranking"])
            st.dataframe(
                ranking.style.apply(
                    _risk_probability_styles, subset=["predicted_defect_count"]
                ),
                width="stretch",
                hide_index=True,
            )
        except Exception as exc:
            st.error(_api_error_message(exc))


def main():
    st.set_page_config(
        page_title="Manufacturing Quality Intelligence",
        page_icon="🏭",
        layout="wide",
    )
    st.title("Manufacturing Quality Intelligence")
    st.caption("Color-coded operations monitoring powered by trained ML inference and project API endpoints.")
    st.markdown(
        """
        <style>
        .stApp { background: #F7FAFF; color: #172033; }
        [data-testid="stMetricValue"] { font-size: 1.35rem !important; }
        [data-testid="stMetricLabel"] { font-size: 0.78rem !important; }
        [data-testid="stCaptionContainer"] { font-size: 0.82rem !important; }
        [data-testid="stDataFrame"] { font-size: 0.82rem !important; }
        div[data-testid="stMetric"] {
            background: linear-gradient(135deg, #FFFFFF 0%, #EFF6FF 100%);
            border: 1px solid #DBEAFE;
            border-radius: 0.65rem;
            padding: 0.65rem 0.8rem;
        }
        h1, h2, h3 { color: #1D4ED8 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.caption(f"Manufacturing API: {API_BASE_URL}")
    try:
        api_health = _api_get("/health")
        if api_health["status"] == "healthy":
            st.sidebar.success("API connected; all models are ready.")
        else:
            st.sidebar.error(
                f"API status: {api_health['status']}; unavailable models: "
                f"{', '.join(api_health.get('models_missing', [])) or 'see API health'}"
            )
    except Exception as exc:
        st.sidebar.error(_api_error_message(exc))
    models = _models()
    if not models:
        st.stop()

    required = {
        "trend_analyzer",
        "trace_engine",
        "gap_analyzer",
        "scar_generator",
        "marginal_batch_classifier",
        "downstream_impact_forecaster",
        "adaptive_parameter_compensator",
        "failure_mode_classifier",
        "prescriptive_action_engine",
        "shift_defect_predictor",
    }
    missing = sorted(required.difference(models))
    if missing:
        st.error(f"Required models or engines are unavailable: {', '.join(missing)}")
        st.stop()

    try:
        data = load_dashboard_data()
    except Exception as exc:
        st.error(f"Could not load dashboard source data: {type(exc).__name__}: {exc}")
        st.stop()

    supplier_tab, material_tab, manufacturing_tab, production_tab = st.tabs(
        ["Supplier / Vendor", "Raw Material", "Manufacturing", "Production"]
    )
    with supplier_tab:
        _render_supplier_tab(models, data)
    with material_tab:
        _render_material_tab(models, data)
    with manufacturing_tab:
        _render_manufacturing_tab(models)
    with production_tab:
        _render_production_tab(models)


if __name__ == "__main__":
    main()
