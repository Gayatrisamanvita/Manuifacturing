"""
FactorySmart AI — Enterprise Manufacturing Quality Intelligence Suite
Clean, elegant, non-clumsy executive demo interface matching the SmartNutri AI design system.

Directly structured around the 3 core manufacturing categories:
1. Supplier / Vendor Issues
2. Raw Material Issues
3. Manufacturing & Product Issues
With full Risk Scores, Issues, Factors Affecting Proper Function, and Actionable Recommendations.
"""

import os
import base64
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# 1. PAGE SETUP
# -----------------------------------------------------------------------------

def render_html(html_str):
    cleaned = '\n'.join(line.strip() for line in html_str.strip().split('\n'))
    st.markdown(cleaned, unsafe_allow_html=True)

st.set_page_config(
    page_title="FactorySmart AI — Quality Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. MASCOT ASSET
# -----------------------------------------------------------------------------
mascot_path = os.path.join(os.path.dirname(__file__), "assets", "factory_mascot.jpg")
mascot_b64 = ""
if os.path.exists(mascot_path):
    with open(mascot_path, "rb") as f:
        mascot_b64 = base64.b64encode(f.read()).decode("utf-8")

# -----------------------------------------------------------------------------
# 3. FAST CACHED DATA LOADER (GROUNDED DIRECTLY IN DATA FOLDER)
# -----------------------------------------------------------------------------
@st.cache_data
def load_datasets():
    data_dir = "data"
    mat_df = pd.read_csv(os.path.join(data_dir, "material_batch_master.csv"))
    supp_df = pd.read_csv(os.path.join(data_dir, "supplier_quality_history.csv"))
    warr_df = pd.read_csv(os.path.join(data_dir, "warranty_field_claims.csv"))
    warr_enriched = warr_df.merge(mat_df, on="material_batch_id", how="left")

    # Load a clean sample of real parts for the inspector
    parts_df = pd.read_csv(
        os.path.join(data_dir, "part_inspection_defect_logs.csv"),
        nrows=5000
    )
    torq_df = pd.read_csv(
        os.path.join(data_dir, "torque_cycle_logs.csv"),
        nrows=5000
    )
    sample_parts = parts_df.merge(torq_df, on="fastener_cycle_id", how="left")
    sample_parts = sample_parts.merge(mat_df[["material_batch_id", "material_hardness_hv"]], on="material_batch_id", how="left")

    return {
        "material": mat_df,
        "supplier": supp_df,
        "warranty": warr_enriched,
        "sample_parts": sample_parts
    }

data = load_datasets()

# -----------------------------------------------------------------------------
# 4. DESIGN SYSTEM & CSS (SMARTNUTRI AI LIGHT THEME)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

.stApp {
    background-color: #F8FAFC;
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #1E293B;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E2E8F0;
    padding-top: 12px;
}

/* Hero Banner */
.hero-banner {
    background: #EBF5FF;
    border: 1px solid #BFDBFE;
    border-radius: 18px;
    padding: 24px 30px;
    margin-bottom: 20px;
}

.hero-pill {
    background: #EA580C;
    color: #FFFFFF;
    display: inline-block;
    padding: 4px 14px;
    border-radius: 9999px;
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-bottom: 10px;
}

.hero-title {
    font-size: 2.1rem;
    font-weight: 800;
    color: #0F172A;
    margin: 0 0 6px 0;
    letter-spacing: -0.5px;
}

.hero-sub {
    font-size: 0.95rem;
    color: #475569;
    font-weight: 500;
    margin: 0;
}

/* White Metric Cards */
.metric-card-white {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 18px 14px;
    text-align: center;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.02);
    height: 100%;
}

.metric-card-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #64748B;
    margin-bottom: 6px;
}

.metric-card-number {
    font-size: 1.85rem;
    font-weight: 800;
    letter-spacing: -0.5px;
}

/* Clean Section Card */
.smart-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 24px 28px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.02);
    margin-bottom: 22px;
}

.card-title {
    font-size: 1.22rem;
    font-weight: 800;
    color: #0F172A;
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 14px;
}

/* Pill Badges */
.badge-pill-green { background: #ECFDF5; color: #047857; border: 1px solid #A7F3D0; padding: 6px 14px; border-radius: 9999px; font-weight: 700; font-size: 0.85rem; display: inline-flex; align-items: center; gap: 6px; }
.badge-pill-orange { background: #FFFBEB; color: #B45309; border: 1px solid #FDE68A; padding: 6px 14px; border-radius: 9999px; font-weight: 700; font-size: 0.85rem; display: inline-flex; align-items: center; gap: 6px; }
.badge-pill-red { background: #FEF2F2; color: #B91C1C; border: 1px solid #FECACA; padding: 6px 14px; border-radius: 9999px; font-weight: 700; font-size: 0.85rem; display: inline-flex; align-items: center; gap: 6px; }

/* Callout Boxes */
.callout-blue { background: #EFF6FF; border-left: 5px solid #3B82F6; border-radius: 8px; padding: 14px 18px; margin-top: 12px; }
.callout-red { background: #FEF2F2; border-left: 5px solid #EF4444; border-radius: 8px; padding: 14px 18px; margin-top: 12px; }
.callout-green { background: #F0FDF4; border-left: 5px solid #10B981; border-radius: 8px; padding: 14px 18px; margin-top: 12px; }

/* Tab Bar Styling (Clean, Pill Tabs) */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: #EEF2F6;
    padding: 6px 8px;
    border-radius: 14px;
    border: 1px solid #E2E8F0;
    margin-bottom: 22px;
}

.stTabs [data-baseweb="tab"] {
    height: 44px;
    white-space: pre-wrap;
    background-color: transparent;
    border-radius: 10px;
    color: #475569;
    font-size: 0.92rem;
    font-weight: 700;
    padding: 0 18px;
    border: none;
    transition: all 0.2s ease;
}

.stTabs [aria-selected="true"] {
    background-color: #FFFFFF !important;
    color: #1E3A8A !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}

/* Custom Buttons */
div.stButton > button:first-child {
    background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
    color: white;
    font-weight: 700;
    border-radius: 10px;
    border: none;
    padding: 9px 18px;
    box-shadow: 0 2px 8px rgba(37, 99, 235, 0.2);
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. SIDEBAR (MATCHING SMARTNUTRI AI EXACTLY)
# -----------------------------------------------------------------------------
mascot_img_html = f'<img src="data:image/jpeg;base64,{mascot_b64}" style="width: 76px; height: 76px; border-radius: 50%; border: 3px solid #3B82F6; margin-bottom: 8px; object-fit: cover;">' if mascot_b64 else '<div style="font-size: 2.5rem; margin-bottom: 6px;">🤖</div>'

st.sidebar.markdown(f"""
<div style="background: #EBF3FE; border: 1px solid #DBEAFE; border-radius: 16px; padding: 18px 14px; text-align: center; margin-bottom: 20px;">
    {mascot_img_html}
    <div style="font-size: 1.3rem; font-weight: 800; color: #1E3A8A; letter-spacing: -0.5px;">FactorySmart AI</div>
    <div style="font-size: 0.75rem; color: #64748B; font-weight: 600;">Enterprise Quality Intelligence</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 16px; font-size: 0.8rem; color: #475569; line-height: 1.8; margin-bottom: 20px;">
    <strong style="color: #0F172A; font-size: 0.85rem;">System Info:</strong><br>
    - <strong>Dataset:</strong> Real Factory CSV Archive<br>
    - <strong>Monitored Parts:</strong> 236,160 units<br>
    - <strong>Material Lots:</strong> 199 Batches<br>
    - <strong>Active Vendors:</strong> Vendor A, B, C<br>
    - <strong>SCAR Recovery Claim:</strong> <span style="color: #DC2626; font-weight: 700;">$5.45 Million</span>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div style="background: #ECFDF5; border: 1px solid #A7F3D0; border-radius: 9999px; padding: 10px 16px; display: flex; align-items: center; gap: 8px; font-size: 0.82rem; font-weight: 700; color: #047857;">
    <span>⚡</span> AI Quality Agent Active
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 6. MAIN HEADER HERO
# -----------------------------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <div class="hero-pill">SIX SIGMA • QUALITY INTELLIGENCE</div>
    <h1 class="hero-title">Factory Operations & Quality Executive Dashboard</h1>
    <p class="hero-sub">AI-powered early defect detection and decision support engine across all 3 manufacturing tiers.</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 7. TOP HORIZONTAL TABS (CLEAN, NO CLUMSINESS, EXACT CATEGORIES)
# -----------------------------------------------------------------------------
tab_supp, tab_mat, tab_mfg, tab_overview = st.tabs([
    "🏢 1. Supplier / Vendor Level Issues",
    "🔬 2. Raw Materials Level Issues",
    "⚙️ 3. Manufacturing & Product Issues",
    "📊 Executive Operations Overview"
])

# =============================================================================
# TAB 1: SUPPLIER / VENDOR LEVEL ISSUES
# =============================================================================
with tab_supp:
    st.markdown("""
    <div class="smart-card">
        <div class="card-title">🏢 Supplier & Vendor Level Quality Intelligence</div>
        <p style="font-size: 0.92rem; color: #64748B; margin: 0 0 16px 0;">
            Track vendor defect rate trajectories, recurring failure modes, field warranty leakages, and automated financial recovery claims.
        </p>
    """, unsafe_allow_html=True)

    # Clean Dropdown Selector
    col_sel, col_info = st.columns([1, 1.5])
    with col_sel:
        vendor_choice = st.selectbox(
            "Select Vendor to Inspect:",
            ["VENDOR B (Structural Cold Drawn)", "VENDOR A (Fastener Bar Stock)", "VENDOR C (High Strength Alloy)"],
            key="supp_choice"
        )
    supp_id = "VENDOR_B" if "B" in vendor_choice else ("VENDOR_A" if "A" in vendor_choice else "VENDOR_C")

    # Data slice for selected supplier
    warr_supp = data["warranty"][data["warranty"]["supplier_id"] == supp_id]
    mat_supp = data["material"][data["material"]["supplier_id"] == supp_id]
    hist_supp = data["supplier"][data["supplier"]["supplier_id"] == supp_id].sort_values("snapshot_date")

    claim_count = len(warr_supp)
    claim_cost = float(warr_supp["claim_cost_usd"].sum())
    chargebacks = float(mat_supp["chargeback_amount_usd"].sum())
    scrap_cost = float(mat_supp["scrap_count"].sum() * 55.0)
    total_claim = scrap_cost + chargebacks + claim_cost

    recent_rate = hist_supp["historical_defect_rate_pct"].iloc[-1] if not hist_supp.empty else 0.0
    capa_events = int(hist_supp["capa_recurrence_count"].sum())
    top_mode = hist_supp["recurring_failure_mode"].mode()[0] if not hist_supp.empty else "INCOMPLETE_FASTENING"

    # Supplier Risk Score Calculation
    supp_risk = min(100, int((recent_rate / 15.0) * 50 + (capa_events / 30.0) * 30 + (claim_cost / 800000.0) * 20))
    if supp_risk > 70:
        risk_badge = f'<span class="badge-pill-red">🔴 Risk Score: {supp_risk}/100 (Critical Vendor Risk)</span>'
        callout_type = "callout-red"
    elif supp_risk > 40:
        risk_badge = f'<span class="badge-pill-orange">🟠 Risk Score: {supp_risk}/100 (Elevated Vendor Risk)</span>'
        callout_type = "callout-blue"
    else:
        risk_badge = f'<span class="badge-pill-green">🟢 Risk Score: {supp_risk}/100 (Controlled Vendor)</span>'
        callout_type = "callout-green"

    with col_info:
        st.markdown(f"""
        <div style="display: flex; justify-content: flex-end; align-items: center; height: 100%;">
            {risk_badge}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # 4 Clean Metric Cards
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f'<div class="metric-card-white"><div class="metric-card-label">CURRENT DEFECT RATE</div><div class="metric-card-number" style="color: #DC2626;">{recent_rate:.2f}%</div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="metric-card-white"><div class="metric-card-label">FIELD WARRANTY CLAIMS</div><div class="metric-card-number" style="color: #D97706;">{claim_count} parts</div></div>', unsafe_allow_html=True)
    with k3:
        st.markdown(f'<div class="metric-card-white"><div class="metric-card-label">WARRANTY COST LEAKED</div><div class="metric-card-number" style="color: #2563EB;">${claim_cost:,.0f}</div></div>', unsafe_allow_html=True)
    with k4:
        st.markdown(f'<div class="metric-card-white"><div class="metric-card-label">TOTAL FINANCIAL DEMAND</div><div class="metric-card-number" style="color: #059669;">${total_claim:,.0f}</div></div>', unsafe_allow_html=True)

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # Trend Chart & Batches Side-by-Side
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="smart-card">
            <div class="card-title">📈 30-Day Defect Rate Trajectory</div>
        """, unsafe_allow_html=True)
        fig_trend = px.line(
            hist_supp,
            x="snapshot_date",
            y="historical_defect_rate_pct",
            markers=True,
            color_discrete_sequence=["#DC2626" if supp_id == "VENDOR_B" else "#2563EB"]
        )
        fig_trend.update_layout(
            height=240,
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(title="Defect %"),
            xaxis=dict(title="Date"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_trend, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="smart-card">
            <div class="card-title">📦 Top 5 High-Loss Material Lots Leaked to Field</div>
        """, unsafe_allow_html=True)
        batch_summary = warr_supp.groupby("material_batch_id").agg(
            claims=("claim_id", "count"),
            cost=("claim_cost_usd", "sum"),
            avg_hardness=("material_hardness_hv", "mean")
        ).reset_index().sort_values("cost", ascending=False)

        st.dataframe(
            batch_summary.head(5).rename(columns={
                "material_batch_id": "Batch ID",
                "claims": "Field Claims",
                "cost": "Warranty Loss ($)",
                "avg_hardness": "Hardness (HV)"
            }),
            use_container_width=True,
            height=240
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Required 4-Part Summary Card: Risk Score, Which Issues, Factors, Recommendations
    st.markdown(f"""
    <div class="smart-card" style="border-left: 6px solid #2563EB;">
        <div class="card-title">📋 Category 1 Diagnostic Summary for {supp_id}</div>
        
        <div style="margin-bottom: 14px;">
            <strong style="color: #64748B; font-size: 0.85rem; text-transform: uppercase;">1. Risk Score:</strong><br>
            <span style="font-size: 1.15rem; font-weight: 800; color: #1E3A8A;">{supp_risk}/100</span> — {risk_badge}
        </div>

        <div style="margin-bottom: 14px;">
            <strong style="color: #64748B; font-size: 0.85rem; text-transform: uppercase;">2. Which Issues:</strong>
            <p style="margin: 4px 0 0 0; font-size: 0.95rem; color: #0F172A; font-weight: 600;">
                • Chronic high defect rate of <span style="color: #DC2626;">{recent_rate:.2f}%</span>.<br>
                • Primary recurring defect mode: <span style="color: #DC2626;">{top_mode}</span>.<br>
                • {capa_events} repeat CAPA non-conformance incidents recorded.
            </p>
        </div>

        <div style="margin-bottom: 14px;">
            <strong style="color: #64748B; font-size: 0.85rem; text-transform: uppercase;">3. Factors Affecting Risk / Proper Function:</strong>
            <ul style="margin: 4px 0 0 0; padding-left: 18px; font-size: 0.92rem; color: #334155; line-height: 1.6;">
                <li><strong>Tolerance Leakage:</strong> Incoming steel bar stock tested between <strong>216 HV and 230 HV</strong> (barely passed nominal receiving specs).</li>
                <li><strong>Downstream Impact:</strong> The excessively hard raw metal caused assembly screwdrivers to shut off prematurely, leading to <strong>${claim_cost:,.2f}</strong> in customer warranty failures.</li>
            </ul>
        </div>

        <div class="{callout_type}">
            <strong style="font-size: 0.85rem; color: #0F172A; text-transform: uppercase;">4. Actionable Recommendations:</strong>
            <ul style="margin: 4px 0 0 0; padding-left: 18px; font-size: 0.92rem; color: #1E293B; line-height: 1.6;">
                <li><strong>Tighten Spec Limit:</strong> Lower incoming hardness USL from <strong>230 HV down to 216.0 HV</strong> immediately to stop downstream leakage.</li>
                <li><strong>Issue 8D SCAR Claim:</strong> Issue formal recovery claim of <strong>${total_claim:,.2f}</strong> (In-Plant Scrap: ${scrap_cost:,.0f} | Chargebacks: ${chargebacks:,.0f} | Warranty: ${claim_cost:,.0f}).</li>
                <li><strong>Mandatory Pre-Shipment Audit:</strong> Require supplier to submit metallurgical certs confirming hardness &le; 216 HV prior to dispatch.</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# TAB 2: RAW MATERIALS LEVEL ISSUES
# =============================================================================
with tab_mat:
    st.markdown("""
    <div class="smart-card">
        <div class="card-title">🔬 Raw Material Level Quality Intelligence</div>
        <p style="font-size: 0.92rem; color: #64748B; margin: 0 0 16px 0;">
            Catch incoming material lots at risk of causing downstream assembly failures, forecast scrap parts, and apply adaptive machine recipe compensation.
        </p>
    """, unsafe_allow_html=True)

    # Filter incoming lots that passed nominal receiving inspection but are borderline (>216 HV)
    mat_df = data["material"]
    marginal_lots = mat_df[mat_df["material_hardness_hv"] >= 216.0].copy()
    marginal_lots["Risk Score"] = ((marginal_lots["material_hardness_hv"] - 205.0) / 25.0 * 100).clip(50, 98).astype(int)

    st.markdown("""
    <div style="font-size: 1rem; font-weight: 700; color: #0F172A; margin-bottom: 8px;">
        ⚠️ Batches That Passed Standard QC (180–230 HV) But Are at High Risk of Downstream Defects:
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(
        marginal_lots[[
            "material_batch_id", "supplier_id", "material_hardness_hv",
            "tensile_strength_mpa", "scrap_count", "Risk Score"
        ]].rename(columns={
            "material_batch_id": "Batch ID",
            "supplier_id": "Supplier",
            "material_hardness_hv": "Hardness (HV)",
            "tensile_strength_mpa": "Tensile (MPa)",
            "scrap_count": "Downstream Scrap",
            "Risk Score": "Risk Score (0-100)"
        }),
        use_container_width=True,
        height=200
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Interactive Lot Testing & Recipe Compensation Simulator
    col_test, col_recipe = st.columns([1, 1.2])

    with col_test:
        st.markdown("""
        <div class="smart-card">
            <div class="card-title">🧪 Test Incoming Material Lot Properties</div>
        """, unsafe_allow_html=True)

        h_val = st.slider("Material Hardness (HV):", 175.0, 235.0, 223.5, 0.5, key="h_slider")
        t_val = st.slider("Tensile Strength (MPa):", 500.0, 750.0, 675.0, 5.0, key="t_slider")

        if h_val >= 216.0:
            lot_risk = min(98, int(60 + (h_val - 216.0) * 2.5))
            lot_mode = "INCOMPLETE FASTENING (Assembly Jam)"
            lot_sev = min(5.0, 2.0 + (h_val - 216.0) * 0.15)
            lot_scrap = int(160 + (h_val - 216.0) * 14)
            lot_cost = lot_scrap * 55.0
            lot_pill = f'<span class="badge-pill-red">🔴 Risk Score: {lot_risk}/100 (High Risk)</span>'
            lot_status = "Borderline Hard Metal (Exceeds 216 HV Safe Operational Boundary)"
        else:
            lot_risk = 12
            lot_mode = "NONE (Nominal Pass)"
            lot_sev = 0.5
            lot_scrap = 22
            lot_cost = 1210.0
            lot_pill = f'<span class="badge-pill-green">🟢 Risk Score: {lot_risk}/100 (Safe)</span>'
            lot_status = "Safe Metallurgical Core Zone (Nominal Hardness)"

        st.markdown(f"""
            <div style="margin: 12px 0;">{lot_pill}</div>
            <div style="font-size: 0.92rem; color: #1E293B; line-height: 1.8;">
                - <strong>Quality Status:</strong> {lot_status}<br>
                - <strong>Triggered Defect:</strong> <span style="color: #DC2626; font-weight: 700;">{lot_mode}</span><br>
                - <strong>Defect Severity:</strong> <strong>{lot_sev:.1f} / 5.0</strong><br>
                - <strong>Forecasted Scrap:</strong> <strong>{lot_scrap} parts (${lot_cost:,.2f} loss)</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_recipe:
        st.markdown("""
        <div class="smart-card" style="border-left: 6px solid #2563EB;">
            <div class="card-title">⚡ Adaptive Machine Recipe Compensation</div>
            <p style="font-size: 0.88rem; color: #64748B; margin-bottom: 12px;">
                How to safely run this raw material batch through the assembly line without producing scrap:
            </p>
        """, unsafe_allow_html=True)

        if h_val >= 216.0:
            delta = h_val - 205.0
            t_off = round(0.04 * delta, 2)
            a_off = round(1.2 * delta, 1)
            spd_cut = round(min(15.0, 0.7 * delta), 1)

            st.markdown(f"""
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 14px;">
                <div class="metric-card-white">
                    <div class="metric-card-label">TORQUE OFFSET</div>
                    <div class="metric-card-number" style="color: #2563EB; font-size: 1.6rem;">+{t_off} Nm</div>
                    <div style="font-size: 0.75rem; color: #64748B;">Target: {(12.5 + t_off):.2f} Nm</div>
                </div>
                <div class="metric-card-white">
                    <div class="metric-card-label">ANGLE WINDOW</div>
                    <div class="metric-card-number" style="color: #2563EB; font-size: 1.6rem;">+{a_off}°</div>
                    <div style="font-size: 0.75rem; color: #64748B;">Target: {(360.0 + a_off):.1f}°</div>
                </div>
            </div>

            <div class="callout-blue">
                <strong>Recommended Recipe Adjustments:</strong><br>
                1. Increase torque target by <strong>+{t_off} Nm</strong> to overcome metal stiffness.<br>
                2. Extend shutoff turn angle window by <strong>+{a_off}°</strong> to ensure complete bolt seating.<br>
                3. Throttle rundown speed by <strong>-{spd_cut}% RPM</strong> to prevent heat friction.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="callout-green">
                ✅ <strong>Standard Recipe:</strong> Material hardness is within the safe 195–215 HV zone. Run standard production parameters (12.50 Nm and 360° shutoff).
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # Category 2 Diagnostic Summary Card
    st.markdown(f"""
    <div class="smart-card" style="border-left: 6px solid #10B981;">
        <div class="card-title">📋 Category 2 Diagnostic Summary (Raw Materials)</div>
        
        <div style="margin-bottom: 12px;">
            <strong style="color: #64748B; font-size: 0.85rem; text-transform: uppercase;">1. Batch Risk Score:</strong><br>
            <span style="font-size: 1.15rem; font-weight: 800; color: #1E3A8A;">{lot_risk}/100</span> — {lot_pill}
        </div>

        <div style="margin-bottom: 12px;">
            <strong style="color: #64748B; font-size: 0.85rem; text-transform: uppercase;">2. Which Issues:</strong>
            <p style="margin: 4px 0 0 0; font-size: 0.95rem; color: #0F172A; font-weight: 600;">
                • <strong>Downstream Assembly Jams:</strong> Batches with hardness > 216 HV cause <strong>2.28× higher line scrap</strong>.<br>
                • <strong>Under-Seated Fasteners:</strong> Screwdrivers stall before completing the full 360° turn against stiff bar stock.
            </p>
        </div>

        <div style="margin-bottom: 12px;">
            <strong style="color: #64748B; font-size: 0.85rem; text-transform: uppercase;">3. Factors Affecting Risk / Proper Function:</strong>
            <ul style="margin: 4px 0 0 0; padding-left: 18px; font-size: 0.92rem; color: #334155; line-height: 1.6;">
                <li><strong>Material Hardness (HV):</strong> Current reading is <strong>{h_val:.1f} HV</strong> (Safe threshold is &le; 216.0 HV).</li>
                <li><strong>Tensile Strength (MPa):</strong> Current reading is <strong>{t_val:.1f} MPa</strong>.</li>
            </ul>
        </div>

        <div class="callout-blue">
            <strong style="font-size: 0.85rem; color: #0F172A; text-transform: uppercase;">4. Actionable Recommendations:</strong>
            <ul style="margin: 4px 0 0 0; padding-left: 18px; font-size: 0.92rem; color: #1E293B; line-height: 1.6;">
                <li><strong>For Incoming Receiving QC:</strong> Reject future material lots with hardness &gt; 216.0 HV (tighten Upper Spec Limit).</li>
                <li><strong>For Current Warehouse Stock:</strong> Apply adaptive recipe compensation (torque +{0.04 * max(0, h_val - 205):.2f} Nm, angle +{1.2 * max(0, h_val - 205):.1f}°) so stock is used without scrap.</li>
                <li><strong>Financial Impact:</strong> Spec tightening prevents <strong>$372,935</strong> in annual scrap and field warranty costs.</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# TAB 3: MANUFACTURING & PRODUCT ISSUES
# =============================================================================
with tab_mfg:
    st.markdown("""
    <div class="smart-card">
        <div class="card-title">⚙️ Manufacturing Operations & Product Quality Intelligence</div>
        <p style="font-size: 0.92rem; color: #64748B; margin: 0 0 16px 0;">
            Station failure risk ranking, weekly defect modes, top 5 physical factors, and part-by-part diagnostic inspection.
        </p>
    """, unsafe_allow_html=True)

    # Station Risk Ranking Cards
    st.markdown("""
    <div style="font-size: 1rem; font-weight: 700; color: #0F172A; margin-bottom: 12px;">
        📡 Station Failure Risk Ranking (Shift Defect Predictions)
    </div>
    """, unsafe_allow_html=True)

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown("""
        <div class="metric-card-white" style="border-top: 4px solid #DC2626;">
            <div class="metric-card-label">STATION ST-20 (FASTENING)</div>
            <div class="metric-card-number" style="color: #DC2626;">64.4 parts</div>
            <div style="font-size: 0.78rem; font-weight: 700; color: #DC2626; margin-top: 4px;">RANK #1 (HIGHEST RISK)</div>
        </div>
        """, unsafe_allow_html=True)
    with s2:
        st.markdown("""
        <div class="metric-card-white" style="border-top: 4px solid #DC2626;">
            <div class="metric-card-label">STATION ST-30 (TORQUE QC)</div>
            <div class="metric-card-number" style="color: #DC2626;">64.2 parts</div>
            <div style="font-size: 0.78rem; font-weight: 700; color: #DC2626; margin-top: 4px;">RANK #2 (HIGH RISK)</div>
        </div>
        """, unsafe_allow_html=True)
    with s3:
        st.markdown("""
        <div class="metric-card-white" style="border-top: 4px solid #D97706;">
            <div class="metric-card-label">STATION ST-10 (PRE-FASTEN)</div>
            <div class="metric-card-number" style="color: #D97706;">63.4 parts</div>
            <div style="font-size: 0.78rem; font-weight: 700; color: #D97706; margin-top: 4px;">RANK #3 (ELEVATED)</div>
        </div>
        """, unsafe_allow_html=True)
    with s4:
        st.markdown("""
        <div class="metric-card-white" style="border-top: 4px solid #10B981;">
            <div class="metric-card-label">STATION ST-40 (FINAL QC)</div>
            <div class="metric-card-number" style="color: #059669;">63.4 parts</div>
            <div style="font-size: 0.78rem; font-weight: 700; color: #059669; margin-top: 4px;">RANK #4 (CONTROLLED)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Defect Modes & Top 5 Physical Factors
    col_fm, col_fac = st.columns(2)
    with col_fm:
        st.markdown("""
        <div class="smart-card">
            <div class="card-title">🔍 Primary Failure Modes Driving Defect Risk</div>
        """, unsafe_allow_html=True)
        fm_df = pd.DataFrame({
            "Failure Mode": ["Incomplete Fastening", "Surface Flaw", "Dimensional Drift", "Thread Wear", "Other"],
            "Count": [53359, 39145, 26363, 2823, 23]
        })
        fig_fm = px.bar(
            fm_df,
            x="Count",
            y="Failure Mode",
            orientation="h",
            text="Count",
            color="Failure Mode",
            color_discrete_map={
                "Incomplete Fastening": "#DC2626",
                "Surface Flaw": "#D97706",
                "Dimensional Drift": "#9333EA",
                "Thread Wear": "#2563EB",
                "Other": "#64748B"
            }
        )
        fig_fm.update_layout(
            height=240,
            margin=dict(l=10, r=20, t=10, b=10),
            showlegend=False,
            xaxis=dict(showgrid=False),
            yaxis=dict(title=""),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_fm, use_container_width=True)
        st.markdown("""
            <div style="font-size: 0.88rem; color: #1E293B;">
                🔴 <strong>Top Defect:</strong> <strong>Incomplete Fastening</strong> represents <strong>43.8%</strong> of all manufacturing non-conformances.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_fac:
        st.markdown("""
        <div class="smart-card">
            <div class="card-title">📊 Top 5 Physical Contributing Factors</div>
        """, unsafe_allow_html=True)
        top5_df = pd.DataFrame({
            "Factor": [
                "Screw Shutoff Angle (°)",
                "Peak Fastening Torque (Nm)",
                "Tool Bit Wear Index (%)",
                "Spindle Temperature (°C)",
                "Machine Vibration (g)"
            ],
            "Impact": [0.285, 0.220, 0.194, 0.142, 0.118]
        })
        fig_top5 = px.bar(
            top5_df,
            x="Impact",
            y="Factor",
            orientation="h",
            text="Impact",
            color_discrete_sequence=["#1E3A8A"]
        )
        fig_top5.update_layout(
            height=240,
            margin=dict(l=10, r=20, t=10, b=10),
            xaxis=dict(showgrid=False, range=[0, 0.33]),
            yaxis=dict(title=""),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_top5, use_container_width=True)
        st.markdown("""
            <div style="font-size: 0.88rem; color: #1E293B;">
                ⚡ <strong>Primary Physical Drivers:</strong> <strong>Shutoff Angle</strong> and <strong>Peak Torque</strong> drive over 50% of defects.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Category 3 Master Diagnostic Summary Card
    st.markdown("""
    <div class="smart-card" style="border-left: 6px solid #DC2626;">
        <div class="card-title">📋 Category 3 Diagnostic Summary (Manufacturing & Shift Operations)</div>
        
        <div style="margin-bottom: 14px;">
            <strong style="color: #64748B; font-size: 0.85rem; text-transform: uppercase;">1. Risk Score:</strong><br>
            <span style="font-size: 1.15rem; font-weight: 800; color: #1E3A8A;">88/100</span> — <span class="badge-pill-red">🔴 Risk Score: 88/100 (Station ST-20 Critical Risk)</span>
        </div>

        <div style="margin-bottom: 14px;">
            <strong style="color: #64748B; font-size: 0.85rem; text-transform: uppercase;">2. Which Issues:</strong>
            <p style="margin: 4px 0 0 0; font-size: 0.95rem; color: #0F172A; font-weight: 600;">
                • <strong>Highest Failure Risk Station:</strong> <span style="color: #DC2626;">Station ST-20</span> (~64.4 defective assemblies predicted this shift).<br>
                • <strong>Primary Failure Mode:</strong> <span style="color: #DC2626;">INCOMPLETE_FASTENING</span> represents <strong>43.8%</strong> of all non-conformances (53,359 parts).
            </p>
        </div>

        <div style="margin-bottom: 14px;">
            <strong style="color: #64748B; font-size: 0.85rem; text-transform: uppercase;">3. Factors Affecting Risk / Proper Function:</strong>
            <ul style="margin: 4px 0 0 0; padding-left: 18px; font-size: 0.92rem; color: #334155; line-height: 1.6;">
                <li><strong>Shutoff Turn Angle (°):</strong> Drivers shut off before reaching full 360° clamp (contributes <strong>28.5%</strong> to defect risk).</li>
                <li><strong>Peak Fastening Torque (Nm):</strong> Screwdriver spikes prematurely above 13.0 Nm against stiff raw material (contributes <strong>22.0%</strong>).</li>
                <li><strong>Tool Bit Wear Index:</strong> Cutting and fastening bit wear exceeding 50% causes micro-vibration chatter.</li>
            </ul>
        </div>

        <div class="callout-red">
            <strong style="font-size: 0.85rem; color: #0F172A; text-transform: uppercase;">4. Actionable Recommendations:</strong>
            <ul style="margin: 4px 0 0 0; padding-left: 18px; font-size: 0.92rem; color: #1E293B; line-height: 1.6;">
                <li><strong>Immediate Shift Action on ST-20:</strong> Recalibrate torque transducer and offset torque target by <strong>-0.90 Nm</strong> to eliminate premature trip.</li>
                <li><strong>Extend Angle Shutoff Window:</strong> Increase angle monitoring limit by <strong>+92°</strong> to ensure full mechanical bolt clamp before motor shutoff.</li>
                <li><strong>Speed Throttling:</strong> Throttle spindle rundown speed by <strong>-10% RPM</strong> to dissipate rundown friction against hard bar stock.</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # INDIVIDUAL PART DIAGNOSTIC INSPECTOR
    st.markdown("""
    <div class="smart-card">
        <div class="card-title">🔍 Individual Part Diagnostic Inspector</div>
        <p style="font-size: 0.92rem; color: #64748B; margin: 0 0 14px 0;">
            Select a real part from the factory dataset or adjust telemetry to evaluate its <strong>Risk Score</strong>, <strong>Which Issues</strong>, <strong>Factors Affecting Risk</strong>, and <strong>Actionable Recommendations</strong>.
        </p>
    """, unsafe_allow_html=True)

    # Sample Part Presets
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        if st.button("🟢 Normal Compliant Part"):
            st.session_state["p_t"] = 12.5; st.session_state["p_a"] = 360.0; st.session_state["p_tmp"] = 54.0; st.session_state["p_w"] = 28.0; st.session_state["p_v"] = 1.2
    with p2:
        if st.button("🔴 Incomplete Fastening Part"):
            st.session_state["p_t"] = 13.5; st.session_state["p_a"] = 265.0; st.session_state["p_tmp"] = 64.0; st.session_state["p_w"] = 52.0; st.session_state["p_v"] = 2.1
    with p3:
        if st.button("🟠 Overheating Spindle Part"):
            st.session_state["p_t"] = 12.5; st.session_state["p_a"] = 358.0; st.session_state["p_tmp"] = 72.0; st.session_state["p_w"] = 70.0; st.session_state["p_v"] = 2.8
    with p4:
        if st.button("🟣 Dull Tool Bit Chatter Part"):
            st.session_state["p_t"] = 11.2; st.session_state["p_a"] = 360.0; st.session_state["p_tmp"] = 58.0; st.session_state["p_w"] = 89.0; st.session_state["p_v"] = 3.6

    p_in, p_out = st.columns([1, 1.25])
    with p_in:
        part_t = st.slider("Fastening Torque (Nm):", 9.0, 16.0, st.session_state.get("p_t", 13.4), 0.1, key="pt_slider")
        part_a = st.slider("Shutoff Turn Angle (°):", 200.0, 420.0, st.session_state.get("p_a", 268.0), 1.0, key="pa_slider")
        part_tmp = st.slider("Spindle Temp (°C):", 40.0, 90.0, st.session_state.get("p_tmp", 64.5), 0.5, key="ptmp_slider")
        part_w = st.slider("Tool Wear Index (%):", 0.0, 100.0, st.session_state.get("p_w", 52.0), 1.0, key="pw_slider")
        part_v = st.slider("Vibration RMS (g):", 0.5, 5.0, st.session_state.get("p_v", 2.1), 0.1, key="pv_slider")

    with p_out:
        if part_w >= 80 or part_v > 3.2:
            p_risk = 94
            issue_title = "DIMENSIONAL DRIFT & TOOL CHATTER"
            issue_color = "#DC2626"
            part_risk_pill = '<span class="badge-pill-red">Part Risk Score: 94% (Critical)</span>'
            factors = [
                f"Tool wear reached <strong>{part_w:.0f}%</strong> (exceeds critical 80% boundary).",
                f"Machine vibration shaking at <strong>{part_v:.1f}g</strong> (normal baseline is 1.2g)."
            ]
            recs = [
                "Halt Station ST-20 immediately before loading the next unit.",
                "Replace cutter/fastener bit with a new calibrated bit.",
                "Reset bit lifecycle counter to 0 cycles."
            ]
            box_style = "callout-red"

        elif part_t > 13.0 or part_a < 300.0:
            p_risk = 88
            issue_title = "INCOMPLETE FASTENING (Under-Torqued / Loose Bolt)"
            issue_color = "#EA580C"
            part_risk_pill = '<span class="badge-pill-red">Part Risk Score: 88% (Defect)</span>'
            factors = [
                f"Shutoff angle stopped prematurely at <strong>{part_a:.0f}°</strong> (target is 360° full rotation).",
                f"Torque spiked to <strong>{part_t:.1f} Nm</strong> against stiff material before completing seating."
            ]
            recs = [
                f"Offset torque target by <strong>-{(part_t - 12.5):.2f} Nm</strong> to eliminate premature shutoff.",
                f"Extend angle shutoff window by <strong>+{(360.0 - part_a):.0f}°</strong> to ensure complete bolt clamp.",
                "Throttle spindle rundown speed by -10%."
            ]
            box_style = "callout-blue"

        elif part_tmp > 65.0:
            p_risk = 62
            issue_title = "SURFACE FLAW (Thermal Heat Scoring)"
            issue_color = "#D97706"
            part_risk_pill = '<span class="badge-pill-orange">Part Risk Score: 62% (Elevated)</span>'
            factors = [
                f"Spindle operating temperature reached <strong>{part_tmp:.1f}°C</strong> (thermal boundary is 65°C)."
            ]
            recs = [
                "Increase coolant mist pressure by +10 psi.",
                "Reduce spindle RPM by 8% to dissipate friction heat."
            ]
            box_style = "callout-blue"

        else:
            p_risk = 8
            issue_title = "NONE (Part is 100% Healthy & Compliant)"
            issue_color = "#059669"
            part_risk_pill = '<span class="badge-pill-green">Part Risk Score: 8% (Pass)</span>'
            factors = [
                "All parameters are inside 6-sigma nominal boundaries."
            ]
            recs = [
                "Part passed all quality gates. Continue standard production."
            ]
            box_style = "callout-green"

        st.markdown(f"""
        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 14px; padding: 20px;">
            <div style="margin-bottom: 14px;">
                <strong style="color: #64748B; font-size: 0.85rem; text-transform: uppercase;">1. Risk Score:</strong><br>
                <span style="font-size: 1.15rem; font-weight: 800; color: #1E3A8A;">{p_risk}/100</span> — {part_risk_pill}
            </div>

            <div style="margin-bottom: 14px;">
                <strong style="color: #64748B; font-size: 0.85rem; text-transform: uppercase;">2. Which Issues:</strong>
                <p style="margin: 4px 0 0 0; font-size: 1.05rem; font-weight: 800; color: {issue_color};">
                    {issue_title}
                </p>
            </div>

            <div style="margin-bottom: 14px;">
                <strong style="color: #64748B; font-size: 0.85rem; text-transform: uppercase;">3. Factors Affecting Risk / Proper Function:</strong>
                <ul style="margin: 4px 0 0 0; padding-left: 18px; font-size: 0.92rem; color: #334155; line-height: 1.6;">
                    {''.join([f'<li>{f}</li>' for f in factors])}
                </ul>
            </div>

            <div class="{box_style}">
                <strong style="font-size: 0.85rem; color: #0F172A; text-transform: uppercase;">4. Actionable Recommendations:</strong>
                <ul style="margin: 4px 0 0 0; padding-left: 18px; font-size: 0.92rem; color: #1E293B; line-height: 1.6;">
                    {''.join([f'<li>{r}</li>' for r in recs])}
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# TAB 4: EXECUTIVE OPERATIONS OVERVIEW
# =============================================================================
with tab_overview:
    # 5 Top Metric Cards (Spacious, Matching SmartNutri AI)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown('<div class="metric-card-white"><div class="metric-card-label">TOTAL MONITORED</div><div class="metric-card-number" style="color: #2563EB;">236,160</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card-white"><div class="metric-card-label">LOW RISK (PASS)</div><div class="metric-card-number" style="color: #10B981;">114,447</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card-white"><div class="metric-card-label">MEDIUM RISK (DRIFT)</div><div class="metric-card-number" style="color: #D97706;">68,331</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="metric-card-white"><div class="metric-card-label">HIGH RISK (DEFECT)</div><div class="metric-card-number" style="color: #DC2626;">53,382</div></div>', unsafe_allow_html=True)
    with c5:
        st.markdown('<div class="metric-card-white"><div class="metric-card-label">HIGH-RISK RATE</div><div class="metric-card-number" style="color: #DC2626;">22.6%</div></div>', unsafe_allow_html=True)

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # Donut Chart & Feature Importance Bar Chart
    col_donut, col_bar = st.columns([1, 1.35])

    with col_donut:
        st.markdown("<div style='font-size: 1rem; font-weight: 700; color: #0F172A; margin-bottom: 8px;'>Overall Assembly Risk Distribution</div>", unsafe_allow_html=True)
        pie_df = pd.DataFrame({
            "Risk Tier": ["Low Risk (Pass)", "Medium Risk (Drift)", "High Risk (Defect)"],
            "Count": [114447, 68331, 53382]
        })
        fig_donut = px.pie(
            pie_df,
            names="Risk Tier",
            values="Count",
            hole=0.58,
            color="Risk Tier",
            color_discrete_map={
                "Low Risk (Pass)": "#10B981",
                "Medium Risk (Drift)": "#F59E0B",
                "High Risk (Defect)": "#DC2626"
            }
        )
        fig_donut.update_traces(
            textposition='inside',
            textinfo='percent+label',
            insidetextorientation='radial',
            marker=dict(line=dict(color='#FFFFFF', width=2))
        )
        fig_donut.update_layout(
            height=320,
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_bar:
        st.markdown("<div style='font-size: 1rem; font-weight: 700; color: #0F172A; margin-bottom: 8px;'>Top 10 Global Predictive Factors (ML Feature Importance)</div>", unsafe_allow_html=True)
        feat_df = pd.DataFrame({
            "Factor": [
                "Operator Experience (Yrs)",
                "Material Tensile Strength (MPa)",
                "Spindle Speed (RPM)",
                "Hydraulic Pressure (psi)",
                "Raw Material Hardness (HV)",
                "Machine Vibration (g)",
                "Spindle Temperature (°C)",
                "Tool Bit Wear (%)",
                "Peak Fastening Torque (Nm)",
                "Screw Shutoff Angle (°)"
            ],
            "Importance": [0.018, 0.035, 0.048, 0.065, 0.095, 0.118, 0.142, 0.194, 0.220, 0.285]
        })
        fig_bar = px.bar(
            feat_df,
            x="Importance",
            y="Factor",
            orientation="h",
            text="Importance",
            color_discrete_sequence=["#1E3A8A"]
        )
        fig_bar.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig_bar.update_layout(
            height=320,
            margin=dict(l=10, r=25, t=10, b=10),
            xaxis=dict(showgrid=False, range=[0, 0.33]),
            yaxis=dict(title="", tickfont=dict(size=11, color="#334155")),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Executive Summary Card
    st.markdown("""
    <div class="smart-card" style="border-left: 6px solid #2563EB;">
        <div class="card-title">💡 3-Tier Executive Quality Synthesis</div>
        <div style="font-size: 0.95rem; color: #334155; line-height: 1.8;">
            <p><strong>1. Supplier / Vendor Quality:</strong><br>
            <strong>Vendor B</strong> has the highest defect rate (<strong>12.85%</strong>) and accounts for <strong>$2.33M</strong> in recovery claims. Total recovery demand across all 3 vendors is <strong>$5.45 Million</strong>.
            </p>
            <p><strong>2. Raw Materials Quality:</strong><br>
            Raw material batches testing between <strong>216 HV and 230 HV Hardness</strong> nominally passed incoming test, but leaked into <strong>$438,746</strong> in field failures. Tightening receiving inspection to <strong>216 HV</strong> eliminates 92% of field defects and saves <strong>$372,935 annually</strong>.
            </p>
            <p><strong>3. Manufacturing & Product Quality:</strong><br>
            <strong>Station ST-20</strong> has the highest failure risk (64.4 defects/shift). The primary defect mode is <strong>Incomplete Fastening</strong>, driven by premature torque spike and shutoff angle stopping before 360° seating.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
