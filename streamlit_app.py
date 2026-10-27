"""
FactorySmart AI  -  Enterprise Manufacturing Quality Intelligence Suite
Clean, elegant executive demo interface matching the Six Sigma design system.

Structured across 4 comprehensive operational tiers:
1. Supplier / Vendor Level Issues (with product catalog, lead times, delayed products, and per-product recommendations)
2. Raw Materials Level Issues (Full 199-batch master, risk assessment, safety stock models, and shortage mitigation without excess inventory)
3. Manufacturing Level (Factories & Units: meaningful targets, machine health, downtime vs target, root causes, resolution & preventive actions)
4. Production Level (Lines & Orders: planned vs actual quantities, completion times, bottlenecks, individual order recommendations)
"""

import os
import re
import base64
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# 1. PAGE SETUP & HTML CLEANER (ELIMINATES RAW HTML LEAKS)
# -----------------------------------------------------------------------------

def render_html(html_str):
    """
    Renders HTML without letting Markdown treat 4-space indentation as a code block.
    Strips leading and trailing spaces from every line.
    """
    cleaned = "\n".join(line.strip() for line in str(html_str).strip().split("\n"))
    st.markdown(cleaned, unsafe_allow_html=True)

st.set_page_config(
    page_title="FactorySmart AI  -  Quality Intelligence Suite",
    page_icon="[Tier 4]",
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
# 3. FAST CACHED DATA LOADER
# -----------------------------------------------------------------------------
@st.cache_data
def load_datasets():
    data_dir = "data"
    mat_df = pd.read_csv(os.path.join(data_dir, "material_batch_master.csv"))
    supp_df = pd.read_csv(os.path.join(data_dir, "supplier_quality_history.csv"))
    warr_df = pd.read_csv(os.path.join(data_dir, "warranty_field_claims.csv"))
    warr_enriched = warr_df.merge(mat_df, on="material_batch_id", how="left")

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

    # Full 199 Material Batches Enriched with Risk Levels
    mat_enriched = mat_df.copy()
    def classify_lot_risk(row):
        h = row["material_hardness_hv"]
        scrap = row["scrap_count"]
        status = row["material_risk_status"]
        if h >= 216.0 or status == "MARGINAL_HIGH" or scrap >= 200:
            return "[Critical] CRITICAL RISK", "badge-pill-red", "Hardness >216 HV tolerance leakage or scrap >200 parts; high risk of screwdriver stalls and downstream defects"
        elif h >= 210.0 or scrap >= 100 or status == "MARGINAL_LOW":
            return "[Elevated] ELEVATED RISK", "badge-pill-orange", "Hardness approaching 216 HV boundary; requires active torque recipe offset and line monitoring"
        else:
            return "[Optimal] NORMAL / SAFE", "badge-pill-green", "Fully within nominal safe core metallurgical envelope (195-210 HV); standard production recipe"

    risk_info = [classify_lot_risk(r) for _, r in mat_enriched.iterrows()]
    mat_enriched["Risk_Level"] = [r[0] for r in risk_info]
    mat_enriched["Risk_Badge"] = [r[1] for r in risk_info]
    mat_enriched["Risk_Explanation"] = [r[2] for r in risk_info]
    mat_enriched["Safe_Margin_HV"] = (216.0 - mat_enriched["material_hardness_hv"]).round(2)
    mat_enriched["parent_sku"] = mat_enriched["supplier_id"].map({
        "VENDOR_B": "RAW-4140 (AISI 4140 Steel)",
        "VENDOR_A": "RAW-ST88 / M10 (Fastener Bar)",
        "VENDOR_C": "RAW-NICR / TI01 (Bushings & Ti)"
    })

    # Supplier Products Master
    supplier_products = {
        "VENDOR_A": [
            {
                "sku": "FST-1001",
                "name": "M10x1.5 Grade 10.9 Fastener Bar Stock",
                "category": "Raw Bar Stock",
                "target_lead_days": 12,
                "actual_lead_days": 14,
                "delay_days": 2,
                "status": "[Optimal] On Schedule (Normal Freight Buffer)",
                "status_badge": "status-badge-green",
                "risk_tier": "[Optimal] Low Risk (Optimal)",
                "monthly_vol": "12,000 units",
                "quality_score": "96.2%",
                "defect_mode": "Dimensional Diameter Drift (+/-0.04mm)",
                "root_cause": "Cold-drawing coil cooling cycle prolonged at supplier mill (within normal freight buffer).",
                "recs": "Maintain standard 2-day PO lead time buffer; keep 3-day safety stock at dock; routine certified micrometer inspection.",
                "is_delayed": False
            },
            {
                "sku": "FST-1002",
                "name": "High-Tensile Threaded Studs ST-88",
                "category": "Threaded Fasteners",
                "target_lead_days": 15,
                "actual_lead_days": 21,
                "delay_days": 6,
                "status": "[Critical] Critical Delay (+6d Bottleneck)",
                "status_badge": "status-badge-red",
                "risk_tier": "[Critical] Critical Delay Risk",
                "monthly_vol": "8,500 units",
                "quality_score": "92.1%",
                "defect_mode": "Thread Pitch Micro-Wear & Debur Burrs",
                "root_cause": "Thread-rolling die changeover bottleneck and worn rolling dies at supplier facility.",
                "recs": "Reserve dedicated tooling slot with supplier; authorize batch pre-staging of 2,000 blank studs; enforce die replacement every 50,000 cycles.",
                "is_delayed": True
            },
            {
                "sku": "FST-1003",
                "name": "Cold-Forged Flange Bolts CB-12",
                "category": "Bolts & Studs",
                "target_lead_days": 10,
                "actual_lead_days": 10,
                "delay_days": 0,
                "status": "[Optimal] On Schedule (Zero Delay)",
                "status_badge": "status-badge-green",
                "risk_tier": "[Optimal] Zero Risk (Benchmark)",
                "monthly_vol": "25,000 units",
                "quality_score": "98.4%",
                "defect_mode": "None (Nominal Pass)",
                "root_cause": "Stable automated multi-station cold heading press with zero downtime.",
                "recs": "Maintain current release schedule; qualify Vendor A as benchmark preferred supplier for flange fasteners.",
                "is_delayed": False
            },
            {
                "sku": "FST-1004",
                "name": "Zinc-Plated Socket Head Cap Screws",
                "category": "Fastener Screws",
                "target_lead_days": 11,
                "actual_lead_days": 12,
                "delay_days": 1,
                "status": "[Optimal] On Schedule (Nominal Buffer)",
                "status_badge": "status-badge-green",
                "risk_tier": "[Optimal] Low Risk (Optimal)",
                "monthly_vol": "14,000 units",
                "quality_score": "97.0%",
                "defect_mode": "Plating Thickness Non-Uniformity (<5µm)",
                "root_cause": "Normal electroplating barrel batch curing cycle turnaround.",
                "recs": "Implement rolling release blanket POs to smooth lot sizes; verify hydrogen embrittlement bake certs.",
                "is_delayed": False
            }
        ],
        "VENDOR_B": [
            {
                "sku": "STL-2001",
                "name": "AISI 4140 Cold-Drawn Alloy Steel Bar",
                "category": "Structural Alloy Steel",
                "target_lead_days": 18,
                "actual_lead_days": 28,
                "delay_days": 10,
                "status": "[Critical] Severe Bottleneck (+10d)",
                "status_badge": "status-badge-red",
                "risk_tier": "[Critical] Critical Risk (Hardness Drift)",
                "monthly_vol": "18,500 units",
                "quality_score": "87.1%",
                "defect_mode": "Excess Hardness (>216 HV) & Tightness Variation",
                "root_cause": "Induction hardening furnace queue backlog and mill heat-treat quenching inconsistency.",
                "recs": "Immediate Action: Split 40% allocation to secondary qualified supplier; enforce receiving rejection threshold at 216.0 HV; issue formal $2.33M SCAR recovery claim.",
                "is_delayed": True
            },
            {
                "sku": "STL-2002",
                "name": "Heavy-Duty Spline Shaft Blanks",
                "category": "Shafts & Rotors",
                "target_lead_days": 20,
                "actual_lead_days": 31,
                "delay_days": 11,
                "status": "[Critical] Severe Bottleneck (+11d)",
                "status_badge": "status-badge-red",
                "risk_tier": "[Critical] Critical Risk (Machining Bottleneck)",
                "monthly_vol": "6,200 units",
                "quality_score": "85.3%",
                "defect_mode": "Concentricity Runout & Surface Pitting",
                "root_cause": "5-axis roughing machine downtime and carbide insert shortages at supplier shop.",
                "recs": "Shift procurement to pre-machined blanks; conduct on-site supplier process audit; mandate dual-sourcing contract clause.",
                "is_delayed": True
            },
            {
                "sku": "STL-2003",
                "name": "Precision Ground Pivot Pins PP-04",
                "category": "Pins & Sleeves",
                "target_lead_days": 16,
                "actual_lead_days": 24,
                "delay_days": 8,
                "status": "[Critical] High Delay (+8d)",
                "status_badge": "status-badge-red",
                "risk_tier": "[Critical] High Delay Risk",
                "monthly_vol": "9,000 units",
                "quality_score": "89.5%",
                "defect_mode": "OD Taper & Surface Roughness (>Ra 0.8)",
                "root_cause": "Centerless grinding wheel glazing and manual micrometer calibration drift.",
                "recs": "Mandate in-process automated laser gauging at supplier site before batch dispatch; buffer 8 days in warehouse.",
                "is_delayed": True
            },
            {
                "sku": "STL-2004",
                "name": "Torque Reaction Sleeves TRS-20",
                "category": "Coupling Sleeves",
                "target_lead_days": 14,
                "actual_lead_days": 19,
                "delay_days": 5,
                "status": "[Elevated] Moderate Delay (+5d)",
                "status_badge": "status-badge-orange",
                "risk_tier": "[Elevated] Moderate Risk (Watchlist)",
                "monthly_vol": "7,800 units",
                "quality_score": "91.0%",
                "defect_mode": "Keyway Width Out of Tolerance",
                "root_cause": "Subcontracted heat-treatment logistics turnaround latency.",
                "recs": "Approve internal finish induction hardening at Detroit hub to eliminate third-party logistics latency.",
                "is_delayed": False
            }
        ],
        "VENDOR_C": [
            {
                "sku": "ALY-3001",
                "name": "Titanium-Alloy Structural Brackets TB-01",
                "category": "Aerospace & Lightweight Alloy",
                "target_lead_days": 14,
                "actual_lead_days": 16,
                "delay_days": 2,
                "status": "[Optimal] On Schedule (Port Customs Buffer)",
                "status_badge": "status-badge-green",
                "risk_tier": "[Optimal] Low Risk (Optimal)",
                "monthly_vol": "5,400 units",
                "quality_score": "98.0%",
                "defect_mode": "Minor Surface Anodize Color Variation",
                "root_cause": "Raw titanium ingot customs clearance transit buffer (normal international shipping window).",
                "recs": "Authorize 30-day bonded warehouse reserve at port of entry; establish electronic pre-clearance.",
                "is_delayed": False
            },
            {
                "sku": "ALY-3002",
                "name": "Nickel-Chromium High-Wear Bushings",
                "category": "Sintered Bushings",
                "target_lead_days": 15,
                "actual_lead_days": 22,
                "delay_days": 7,
                "status": "[Critical] Bottleneck Delay (+7d)",
                "status_badge": "status-badge-red",
                "risk_tier": "[Critical] High Delay Risk",
                "monthly_vol": "11,200 units",
                "quality_score": "93.8%",
                "defect_mode": "Sintering Porosity Density Variance",
                "root_cause": "Sintering vacuum furnace batch cycle capacity bottleneck at supplier plant.",
                "recs": "Request vendor allocate second sintering furnace chamber during peak demand; increase plant safety stock from 5 to 9 days.",
                "is_delayed": True
            },
            {
                "sku": "ALY-3003",
                "name": "Pre-Hardened Tool Steel Inserts TI-09",
                "category": "Tooling & Carbide",
                "target_lead_days": 12,
                "actual_lead_days": 13,
                "delay_days": 1,
                "status": "[Optimal] On Schedule (Nominal Buffer)",
                "status_badge": "status-badge-green",
                "risk_tier": "[Optimal] Low Risk (Optimal)",
                "monthly_vol": "8,000 units",
                "quality_score": "97.6%",
                "defect_mode": "None (Nominal Pass)",
                "root_cause": "High CNC machining repeatability, minor freight buffer within acceptable tolerance.",
                "recs": "Maintain Kanban pull signals; lock in multi-year volume pricing with Vendor C.",
                "is_delayed": False
            },
            {
                "sku": "ALY-3004",
                "name": "Micro-Alloyed Connecting Link Pins",
                "category": "Linkage Hardware",
                "target_lead_days": 10,
                "actual_lead_days": 10,
                "delay_days": 0,
                "status": "[Optimal] On Schedule (Benchmark Zero-Defect)",
                "status_badge": "status-badge-green",
                "risk_tier": "[Optimal] Zero Risk (Benchmark)",
                "monthly_vol": "16,500 units",
                "quality_score": "99.1%",
                "defect_mode": "Zero Defect Baseline",
                "root_cause": "Fully automated continuous wire-feed cold heading line with in-line laser gauging.",
                "recs": "Adopt as benchmark supplier standard across company plants.",
                "is_delayed": False
            }
        ]
    }

    # Raw Material Inventory Model (Safety stock & shortage avoidance)
    raw_materials_inventory = [
        {
            "sku": "RAW-4140",
            "name": "AISI 4140 Cold-Drawn Alloy Steel Bar",
            "supplier": "VENDOR_B",
            "current_stock": 1850,
            "unit": "kg",
            "daily_demand": 450,
            "days_cover": 4.1,
            "lead_time_days": 28,
            "safety_stock": 3250,
            "reorder_point": 15850,
            "eoq": 9500,
            "risk_tier": "CRITICAL RISK",
            "risk_badge": "status-badge-red",
            "risk_factors": "Stock below safety stock (4.1 days vs 7.2 days needed); supplier lead time slipped to 28 days; incoming hardness drift >216 HV.",
            "actions_to_avoid_shortage": "Expedite 4,000 kg split shipment via expedited freight; apply adaptive torque offset (+0.38 Nm) to safely process existing 1,850 kg lot without scrap.",
            "actions_to_avoid_excess": "Cap max order quantity at 9,500 kg (EOQ) to prevent holding excess dormant inventory and tied-up capital."
        },
        {
            "sku": "RAW-ST88",
            "name": "High-Tensile Threaded Stud Stock (ST-88)",
            "supplier": "VENDOR_A",
            "current_stock": 3100,
            "unit": "units",
            "daily_demand": 500,
            "days_cover": 6.2,
            "lead_time_days": 21,
            "safety_stock": 4200,
            "reorder_point": 14700,
            "eoq": 8000,
            "risk_tier": "HIGH RISK",
            "risk_badge": "status-badge-orange",
            "risk_factors": "Current stock (3,100) below required safety stock (4,200); thread rolling delay adds 6 days.",
            "actions_to_avoid_shortage": "Trigger immediate PO release at 14,700 ROP; request 2-batch staggered delivery (4,000 units every 10 days).",
            "actions_to_avoid_excess": "Avoid bulk ordering above 8,000 units to minimize shelf corrosion risk and holding expenses."
        },
        {
            "sku": "RAW-M10",
            "name": "Fastener Bar Stock M10x1.5 Grade 10.9",
            "supplier": "VENDOR_A",
            "current_stock": 9400,
            "unit": "units",
            "daily_demand": 800,
            "days_cover": 11.7,
            "lead_time_days": 14,
            "safety_stock": 4800,
            "reorder_point": 16000,
            "eoq": 12000,
            "risk_tier": "CONTROLLED (OPTIMAL)",
            "risk_badge": "status-badge-green",
            "risk_factors": "Healthy buffer (11.7 days cover); low scrap rate (1.2%); delivery on 14-day schedule.",
            "actions_to_avoid_shortage": "Maintain automated ERP trigger at 16,000 ROP; weekly supplier visibility sync.",
            "actions_to_avoid_excess": "Keep strict pull replenishment aligned with master production schedule."
        },
        {
            "sku": "RAW-NICR",
            "name": "Nickel-Chromium Sintered Bushing Alloy",
            "supplier": "VENDOR_C",
            "current_stock": 2400,
            "unit": "units",
            "daily_demand": 350,
            "days_cover": 6.8,
            "lead_time_days": 22,
            "safety_stock": 3150,
            "reorder_point": 10850,
            "eoq": 5500,
            "risk_tier": "ELEVATED RISK",
            "risk_badge": "status-badge-orange",
            "risk_factors": "Lead time volatility (+7 days delay); sintering furnace batch constraints at vendor.",
            "actions_to_avoid_shortage": "Establish 1,000 unit Vendor-Managed Inventory (VMI) consignment buffer at local freight hub.",
            "actions_to_avoid_excess": "Do not pre-pay stock; VMI consignment ensures inventory is only paid for when pulled to assembly floor."
        },
        {
            "sku": "RAW-TI01",
            "name": "Titanium-Alloy Structural Plate TB-01",
            "supplier": "VENDOR_C",
            "current_stock": 3800,
            "unit": "units",
            "daily_demand": 200,
            "days_cover": 19.0,
            "lead_time_days": 16,
            "safety_stock": 1600,
            "reorder_point": 4800,
            "eoq": 3000,
            "risk_tier": "EXCESS STOCK WARNING",
            "risk_badge": "status-badge-blue",
            "risk_factors": "Stock is 2.38x above safety stock ceiling; $48,000 in tied-up working capital.",
            "actions_to_avoid_shortage": "Zero stockout risk. Safe for the next 19 operating days.",
            "actions_to_avoid_excess": "Pause upcoming replenishment PO for 12 days to burn off excess buffer down to 1,600 units; release capital."
        }
    ]

    # Manufacturing Units Master with Meaningful Targets & Clear Variance
    mfg_units = {
        "UNIT_1": {
            "name": "[Tier 3] Factory Unit 1  -  Detroit Precision Hub (Assembly & Fastening)",
            "location": "Detroit, MI",
            "focus": "Powertrain Fastening & Automated Assembly Cells",
            "total_machines": 28,
            "active_machines": 25,
            "avail_target": "100% Operational Readiness",
            "avail_status": "3 machines offline in maintenance (89.3% availability)",
            "avail_color": "#D97706",
            "downtime_hours": 4.8,
            "downtime_pct": "6.7%",
            "downtime_target": "Target: < 1.5h per 8h shift (Max 2.0% loss)",
            "downtime_status": "[Critical] +3.3h Over Allowed Downtime Ceiling",
            "downtime_color": "#DC2626",
            "mfg_delays_hrs": 3.2,
            "delays_target": "Target: 0.0h (Zero-Delay On-Time SLA)",
            "delays_status": "[Critical] 3.2h Backlog Behind Customer Dispatch Window",
            "delays_color": "#DC2626",
            "material_shortages_count": 2,
            "shortages_target": "Target: 0 Restricted SKUs (100% Line Fill)",
            "shortages_status": "[Critical] 2 Constrained Materials: AISI 4140 & ST-88 Studs",
            "shortages_color": "#DC2626",
            "defect_rate": "4.12%",
            "defect_target": "Target: ≤ 1.50% (Six Sigma Standard)",
            "defect_status": "[Critical] +2.62% Above Allowed Defect Tolerance",
            "defect_color": "#DC2626",
            "oee_efficiency": "78.4%",
            "oee_target": "Target: ≥ 85.0% (World-Class OEE Benchmark)",
            "oee_status": "[Elevated] -6.6% Below Operating Target",
            "oee_color": "#D97706",
            "overall_status": "High Risk / Bottlenecked",
            "status_badge": "status-badge-red",
            "issues": [
                {
                    "type": "Machine Breakdown",
                    "title": "Station ST-20 Fastener Spindle Bearing Seizure & Thermal Spike",
                    "severity": "[Critical] Critical",
                    "root_cause": "Spindle continuous run-time exceeded 18 hours without lubrication purge; vibration chattered to 2.85g under stiff material load.",
                    "resolve_action": "Halt Station ST-20 for a 25-minute emergency maintenance window; flush lubricant, swap cartridge bearing, and replace worn tool bit.",
                    "prevent_action": "Implement automated vibration threshold interlock at 2.5g and schedule dynamic PM cycles every 120 operating hours."
                },
                {
                    "type": "Raw Material Shortage",
                    "title": "AISI 4140 Alloy Bar Stock Depleted Below Safety Threshold",
                    "severity": "[Elevated] High",
                    "root_cause": "Vendor B shipment delayed by 10 days due to mill heat-treat quenching furnace backlog.",
                    "resolve_action": "Transfer 350 emergency units from Nagoya central buffer; prioritize high-margin work orders.",
                    "prevent_action": "Enforce 7-day safety stock buffer and trigger automated PO release when warehouse stock hits 15,850 kg."
                },
                {
                    "type": "High Material Wastage",
                    "title": "142 Scrap Parts Generated per Shift on Fastening Operations",
                    "severity": "[Critical] Critical",
                    "root_cause": "Metal hardness from Vendor B exceeds 216 HV, causing screwdriver stalls and stripped heads before reaching full seating torque.",
                    "resolve_action": "Deploy adaptive machine recipe compensation (+0.45 Nm torque offset, +12° angle window) to salvage current stock without stalling.",
                    "prevent_action": "Enforce incoming receiving inspection Upper Spec Limit (USL) at 216.0 HV with automated CMM test gate."
                },
                {
                    "type": "Quality Problems",
                    "title": "Dimensional Drift and Surface Scuff Rate at 3.9%",
                    "severity": " Medium",
                    "root_cause": "Tool bit wear index reached 78% on cutting station, causing micro-chatter during fastener seating.",
                    "resolve_action": "Index cutter inserts; increase coolant mist pressure by +10 psi to dissipate frictional heat.",
                    "prevent_action": "Program automatic tool wear tracking that flags cutter replacement at 70% wear life."
                }
            ]
        },
        "UNIT_2": {
            "name": "[Tier 3] Factory Unit 2  -  Stuttgart Powertrain Plant (Machining & Tooling)",
            "location": "Stuttgart, Germany",
            "focus": "Precision CNC Milling, Honing & Hydraulic Components",
            "total_machines": 34,
            "active_machines": 32,
            "avail_target": "100% Operational Readiness",
            "avail_status": "2 machines offline for scheduled maintenance (94.1% availability)",
            "avail_color": "#059669",
            "downtime_hours": 2.1,
            "downtime_pct": "2.9%",
            "downtime_target": "Target: < 1.5h per 8h shift (Max 2.0% loss)",
            "downtime_status": "[Elevated] +0.6h Over Allowed Ceiling",
            "downtime_color": "#D97706",
            "mfg_delays_hrs": 1.4,
            "delays_target": "Target: 0.0h (Zero-Delay On-Time SLA)",
            "delays_status": "[Elevated] Minor 1.4h Queue Backlog",
            "delays_color": "#D97706",
            "material_shortages_count": 1,
            "shortages_target": "Target: 0 Restricted SKUs (100% Line Fill)",
            "shortages_status": "[Elevated] 1 Constrained: Ni-Cr Sintered Bushings",
            "shortages_color": "#D97706",
            "defect_rate": "1.85%",
            "defect_target": "Target: ≤ 1.50% (Six Sigma Standard)",
            "defect_status": "[Elevated] +0.35% Above Allowed Six Sigma Standard",
            "defect_color": "#D97706",
            "oee_efficiency": "88.6%",
            "oee_target": "Target: ≥ 85.0% (World-Class OEE Benchmark)",
            "oee_status": "[Optimal] +3.6% Exceeds World-Class Benchmark",
            "oee_color": "#059669",
            "overall_status": "Controlled / Elevated",
            "status_badge": "status-badge-orange",
            "issues": [
                {
                    "type": "Machine Breakdown",
                    "title": "CNC Mill #04 Hydraulic Pressure Fluctuation (95 psi vs 115 psi)",
                    "severity": "[Elevated] High",
                    "root_cause": "Hydraulic proportional valve filter clogged with micro-particulates and seal micro-leakage.",
                    "resolve_action": "Replace valve filter element and bleed hydraulic accumulator during scheduled shift change.",
                    "prevent_action": "Install differential pressure sensor across filter housing for real-time predictive clogging alerts."
                },
                {
                    "type": "Raw Material Shortage",
                    "title": "Nickel-Chromium Bushing Stock Lead Time Extended to 22 Days",
                    "severity": " Medium",
                    "root_cause": "Sintering vacuum furnace batch bottleneck at Vendor C plant.",
                    "resolve_action": "Reallocate 500 bushings from reserve batch; expedite air freight shipment.",
                    "prevent_action": "Establish local consignment buffer of 1,000 units at regional freight hub."
                },
                {
                    "type": "High Material Wastage",
                    "title": "Swarf Chip Accumulation Causing Valve Face Scoring (18 scrapped parts)",
                    "severity": " Medium",
                    "root_cause": "Chip evacuation nozzle pressure dropped to 3.8 bar.",
                    "resolve_action": "Clear swarf flutes and boost coolant flush pump pressure to 6.5 bar.",
                    "prevent_action": "Program automated high-pressure pulse flush cycle every 50 machining cycles."
                },
                {
                    "type": "Quality Problems",
                    "title": "Thread Pitch Micro-Wear on Tapping Spindle",
                    "severity": " Medium",
                    "root_cause": "Tap tool life reached 72% cycle limit against high-alloy valve blocks.",
                    "resolve_action": "Swap tapping tool insert with calibrated TiN-coated bit.",
                    "prevent_action": "Set tap retirement threshold at 65% life."
                }
            ]
        },
        "UNIT_3": {
            "name": "[Tier 3] Factory Unit 3  -  Nagoya Advanced Chassis Facility (Forging & Heat Treatment)",
            "location": "Nagoya, Japan",
            "focus": "Heavy Forging, Heat-Treatment & Structural Chassis Hubs",
            "total_machines": 22,
            "active_machines": 20,
            "avail_target": "100% Operational Readiness",
            "avail_status": "2 heavy presses offline for quench recalibration (90.9% availability)",
            "avail_color": "#D97706",
            "downtime_hours": 3.4,
            "downtime_pct": "4.7%",
            "downtime_target": "Target: < 1.5h per 8h shift (Max 2.0% loss)",
            "downtime_status": "[Critical] +1.9h Over Allowed Ceiling (4.7% loss vs 2.0% allowed)",
            "downtime_color": "#DC2626",
            "mfg_delays_hrs": 2.0,
            "delays_target": "Target: 0.0h (Zero-Delay On-Time SLA)",
            "delays_status": "[Elevated] Line Backlog: 2.0h Behind Scheduled Dispatch",
            "delays_color": "#D97706",
            "material_shortages_count": 0,
            "shortages_target": "Target: 0 Restricted SKUs (100% Line Fill)",
            "shortages_status": "[Optimal] 100% BOM In Stock & Fully Compliant (Zero Shortages)",
            "shortages_color": "#059669",
            "defect_rate": "2.45%",
            "defect_target": "Target: ≤ 1.50% (Six Sigma Standard)",
            "defect_status": "[Elevated] +0.95% Above Allowed Six Sigma Standard",
            "defect_color": "#D97706",
            "oee_efficiency": "84.2%",
            "oee_target": "Target: ≥ 85.0% (World-Class OEE Benchmark)",
            "oee_status": "[Elevated] -0.8% to World-Class Benchmark",
            "oee_color": "#D97706",
            "overall_status": "Controlled",
            "status_badge": "status-badge-blue",
            "issues": [
                {
                    "type": "Machine Breakdown",
                    "title": "Induction Quench Coil Temperature Sensor Drift (+12°C offset)",
                    "severity": "[Elevated] High",
                    "root_cause": "Thermocouple calibration degradation from extreme thermal cycling.",
                    "resolve_action": "Re-zero optical pyrometer and replace thermocouple probe #2.",
                    "prevent_action": "Bi-weekly dual-pyrometer cross-validation audit."
                },
                {
                    "type": "High Material Wastage",
                    "title": "Scale Decarburization on Billet Ends (35 kg scrap)",
                    "severity": " Medium",
                    "root_cause": "Atmosphere purge nitrogen gas flow reduced by 15% due to regulator freeze.",
                    "resolve_action": "Increase nitrogen blanket purge rate to 40 SCFH; thaw heater jacket.",
                    "prevent_action": "Install continuous oxygen sensor in preheat tunnel with automated gas regulation."
                },
                {
                    "type": "Quality Problems",
                    "title": "Hardness Variance Between Core and Skin (Δ14 HV)",
                    "severity": "[Elevated] High",
                    "root_cause": "Quench spray ring nozzles partially calcified, causing uneven cooling.",
                    "resolve_action": "Acid wash quench nozzles and inspect spray coverage cone.",
                    "prevent_action": "Add water de-ionizer to quench water recirculation loop."
                }
            ]
        },
        "UNIT_4": {
            "name": "[Tier 3] Factory Unit 4  -  Austin Gigafactory (Final Integration & Quality Gate)",
            "location": "Austin, TX",
            "focus": "Robotic Enclosure Assembly, Battery & Final Quality Gates",
            "total_machines": 42,
            "active_machines": 41,
            "avail_target": "100% Operational Readiness",
            "avail_status": "1 robotic cell offline for tool change (97.6% availability)",
            "avail_color": "#059669",
            "downtime_hours": 1.5,
            "downtime_pct": "2.1%",
            "downtime_target": "Target: < 1.5h per 8h shift (Max 2.0% loss)",
            "downtime_status": "[Optimal] Met Downtime Target (≤ 1.5h)",
            "downtime_color": "#059669",
            "mfg_delays_hrs": 0.9,
            "delays_target": "Target: 0.0h (Zero-Delay On-Time SLA)",
            "delays_status": "[Optimal] Controlled (Near Zero Delay Backlog)",
            "delays_color": "#059669",
            "material_shortages_count": 1,
            "shortages_target": "Target: 0 Restricted SKUs (100% Line Fill)",
            "shortages_status": " Cap screws buffer approaching 2-shift minimum",
            "shortages_color": "#D97706",
            "defect_rate": "1.20%",
            "defect_target": "Target: ≤ 1.50% (Six Sigma Standard)",
            "defect_status": "[Optimal] Within Six Sigma Tolerance (< 1.50%)",
            "defect_color": "#059669",
            "oee_efficiency": "92.4%",
            "oee_target": "Target: ≥ 85.0% (World-Class OEE Benchmark)",
            "oee_status": "[Optimal] +7.4% Outstanding World-Class Leader",
            "oee_color": "#059669",
            "overall_status": "Benchmark Leader",
            "status_badge": "status-badge-green",
            "issues": [
                {
                    "type": "Machine Breakdown",
                    "title": "Robotic Fastener Vision Camera Lens Coolant Mist Blur on Cell R-08",
                    "severity": " Medium",
                    "root_cause": "Aerosol coolant vapor deposition on camera optical shield.",
                    "resolve_action": "Wipe camera shield and activate pneumatic air-knife curtain.",
                    "prevent_action": "Install continuous positive-pressure air purge around optical sensors."
                },
                {
                    "type": "Raw Material Shortage",
                    "title": "Socket Head Cap Screws Buffer Approaching 2-Shift Minimum",
                    "severity": " Medium",
                    "root_cause": "Upstream Detroit plant delayed shipment of sub-assembly batch.",
                    "resolve_action": "Direct-ship 5,000 screws via hot-shot freight from supplier warehouse.",
                    "prevent_action": "Enable inter-plant ERP auto-requisitioning when stock hits 4-shift mark."
                },
                {
                    "type": "Quality Problems",
                    "title": "Vision Gate False Reject Rate Elevated at 1.8%",
                    "severity": "[Optimal] Low",
                    "root_cause": "Ambient lighting shift during sunset hours through facility skylights.",
                    "resolve_action": "Enclose vision inspection hood with standardized LED backlighting.",
                    "prevent_action": "Multi-spectral polarized illumination to eliminate specular glare."
                }
            ]
        }
    }

    # Production Lines & Orders Master
    prod_orders = [
        {
            "order_id": "PO-9802",
            "line_id": "Line L1: Powertrain Fastening Cell",
            "product_name": "M10-Flange Shaft Assembly (SKU: PWT-4402)",
            "category": "Powertrain Fastening",
            "mfg_unit": "Factory Unit 1 (Detroit)",
            "planned_qty": 2500,
            "actual_qty": 1890,
            "planned_completion": "14:00 (Shift 1)",
            "actual_completion": "18:30 (Shift 1)",
            "delay_hrs": 4.5,
            "status": "Severely Delayed",
            "status_badge": "status-badge-red",
            "rejection_rate": "6.2% (124 rejected parts)",
            "raw_mat_status": "Critical Shortage (AISI 4140 Stock < 1.8 shifts)",
            "bottlenecks": "Station ST-20 Fastening Cell (cycle time 48.2s vs target 32.0s due to stall retries)",
            "issues": [
                "Delayed Order (+4.5 hours behind schedule)",
                "Low Production Output (75.6% attainment)",
                "Severe Line Bottleneck at Station ST-20",
                "High Rejection Rate (6.2% scrap from stripped heads)",
                "Insufficient Raw Material Stock"
            ],
            "recommendations": {
                "prioritizing_urgent_orders": "Mark PO-9802 as Priority P1 rush batch to meet OEM customer shipping window; reschedule non-urgent PO-9807 to night shift.",
                "reallocating_capacity": "Route 35% of fastener sub-assemblies to Line L2 auxiliary cell to recover 3.5 hours of backlog.",
                "resolving_bottlenecks": "Deploy adaptive torque recipe offset (-0.90 Nm, +92° angle window) and swap worn ST-20 tool bit to bring cycle time from 48s back to 32s.",
                "replenishing_materials": "Execute emergency stock transfer of 350 kg 4140 steel from Nagoya warehouse; expedite Vendor B delivery."
            }
        },
        {
            "order_id": "PO-9801",
            "line_id": "Line L1: Powertrain Fastening Cell",
            "product_name": "Heavy-Duty Spline Shaft Assembly (SKU: PWT-5501)",
            "category": "Powertrain Fastening",
            "mfg_unit": "Factory Unit 1 (Detroit)",
            "planned_qty": 3000,
            "actual_qty": 2820,
            "planned_completion": "14:00 (Shift 1)",
            "actual_completion": "14:45 (Shift 1)",
            "delay_hrs": 0.75,
            "status": "On Schedule / Minor Delay",
            "status_badge": "status-badge-green",
            "rejection_rate": "1.8% (54 rejected parts)",
            "raw_mat_status": "Adequate (4.5 days buffer)",
            "bottlenecks": "Station ST-30 torque verification queue (minor)",
            "issues": [
                "Minor queue delay (+45 min)",
                "High production output (94.0% attainment)"
            ],
            "recommendations": {
                "prioritizing_urgent_orders": "Maintain current production cadence; ready batch for heat treatment queue.",
                "reallocating_capacity": "No capacity reallocation needed.",
                "resolving_bottlenecks": "Perform minor tool cleaning on ST-30 during shift handover.",
                "replenishing_materials": "Stock is adequate. Reorder trigger set at 15,850 kg."
            }
        },
        {
            "order_id": "PO-9803",
            "line_id": "Line L2: Chassis Structural Riveting",
            "product_name": "High-Tensile Stud Hub Mount (SKU: CHS-8810)",
            "category": "Chassis Structural",
            "mfg_unit": "Factory Unit 1 (Detroit)",
            "planned_qty": 1800,
            "actual_qty": 1420,
            "planned_completion": "15:30 (Shift 1)",
            "actual_completion": "17:45 (Shift 1)",
            "delay_hrs": 2.25,
            "status": "Delayed with Bottleneck",
            "status_badge": "status-badge-orange",
            "rejection_rate": "4.1% (61 rejected parts)",
            "raw_mat_status": "Low Buffer (ST-88 Studs down to 2.8 days)",
            "bottlenecks": "Riveter anvil feed slide friction and thread burr stalls",
            "issues": [
                "Order Delayed (+2.25 hours)",
                "Output Attainment 78.9%",
                "Riveter feed chute jam from stud burrs",
                "Raw material approaching safety stock limit"
            ],
            "recommendations": {
                "prioritizing_urgent_orders": "Keep batch moving; assign secondary operator to clear feed track jams.",
                "reallocating_capacity": "Line L2 running at maximum single-riveter capacity.",
                "resolving_bottlenecks": "Clear anvil guide tracks and apply synthetic PTFE dry lube; instruct vendor to eliminate deburring burrs.",
                "replenishing_materials": "Release PO for 8,000 ST-88 units immediately at ROP trigger."
            }
        },
        {
            "order_id": "PO-9804",
            "line_id": "Line L2: Chassis Structural Riveting",
            "product_name": "Titanium Support Bracket Sub-Assembly (SKU: CHS-9920)",
            "category": "Chassis Structural",
            "mfg_unit": "Factory Unit 1 (Detroit)",
            "planned_qty": 1200,
            "actual_qty": 1180,
            "planned_completion": "13:00 (Shift 1)",
            "actual_completion": "13:10 (Shift 1)",
            "delay_hrs": 0.15,
            "status": "On Schedule",
            "status_badge": "status-badge-green",
            "rejection_rate": "0.9% (11 rejected parts)",
            "raw_mat_status": "Healthy (Titanium stock > 19 days)",
            "bottlenecks": "None (Smooth automated riveting)",
            "issues": [
                "Optimal performance (98.3% attainment)"
            ],
            "recommendations": {
                "prioritizing_urgent_orders": "Batch completed on schedule; stage for packaging.",
                "reallocating_capacity": "Allow Line L2 operators to assist with clearing PO-9803 backlog.",
                "resolving_bottlenecks": "None.",
                "replenishing_materials": "Hold PO release for 12 days to burn excess stock."
            }
        },
        {
            "order_id": "PO-9805",
            "line_id": "Line L3: Hydraulic Valve Assembly",
            "product_name": "Precision Pressure Regulator Block (SKU: HYD-3301)",
            "category": "Hydraulic Components",
            "mfg_unit": "Factory Unit 2 (Stuttgart)",
            "planned_qty": 2000,
            "actual_qty": 1510,
            "planned_completion": "16:00 (Shift 1)",
            "actual_completion": "19:15 (Shift 1)",
            "delay_hrs": 3.25,
            "status": "Delayed with Bottleneck",
            "status_badge": "status-badge-orange",
            "rejection_rate": "5.4% (86 rejected parts)",
            "raw_mat_status": "Elevated Hardness Drift (>220 HV on valve body)",
            "bottlenecks": "Valve spool bore honing station (thermal taper drift)",
            "issues": [
                "Order Delayed (+3.25 hours)",
                "Low output (75.5% attainment)",
                "Bore taper rejects due to mill hydraulic pressure drop (95 psi)"
            ],
            "recommendations": {
                "prioritizing_urgent_orders": "Prioritize OEM tier-1 orders; defer internal stock replenishments.",
                "reallocating_capacity": "Split final pressure decay testing across 2 test benches.",
                "resolving_bottlenecks": "Replace hydraulic filter on CNC Mill #04 to restore 115 psi clamp; slow hone stroke speed by -8%.",
                "replenishing_materials": "Expedite shipment of 500 normalized Ni-Cr bushings from reserve lot."
            }
        },
        {
            "order_id": "PO-9806",
            "line_id": "Line L4: Electronic Actuator Enclosure",
            "product_name": "Motor Housing Sub-Assembly (SKU: ACT-2204)",
            "category": "Actuator Enclosures",
            "mfg_unit": "Factory Unit 4 (Austin)",
            "planned_qty": 4000,
            "actual_qty": 3850,
            "planned_completion": "17:00 (Shift 1)",
            "actual_completion": "17:20 (Shift 1)",
            "delay_hrs": 0.33,
            "status": "On Schedule",
            "status_badge": "status-badge-green",
            "rejection_rate": "1.4% (55 rejected parts)",
            "raw_mat_status": "Adequate (3.8 days)",
            "bottlenecks": "Seal dispensing nozzle purge (minor)",
            "issues": [
                "High output (96.2% attainment)"
            ],
            "recommendations": {
                "prioritizing_urgent_orders": "Maintain standard flow.",
                "reallocating_capacity": "Keep dedicated robotic cell operating continuously.",
                "resolving_bottlenecks": "Pre-stage adhesive cartridges for shift 2.",
                "replenishing_materials": "Trigger auto-requisitioning for socket head cap screws."
            }
        },
        {
            "order_id": "PO-9807",
            "line_id": "Line L4: Electronic Actuator Enclosure",
            "product_name": "Sensor Mounting Bracket (SKU: ACT-1102)",
            "category": "Actuator Enclosures",
            "mfg_unit": "Factory Unit 4 (Austin)",
            "planned_qty": 2200,
            "actual_qty": 1650,
            "planned_completion": "15:00 (Shift 1)",
            "actual_completion": "17:30 (Shift 1)",
            "delay_hrs": 2.5,
            "status": "Delayed (De-prioritized)",
            "status_badge": "status-badge-orange",
            "rejection_rate": "2.2% (37 rejected parts)",
            "raw_mat_status": "Sufficient",
            "bottlenecks": "Cell capacity temporarily diverted to rush order PO-9806",
            "issues": [
                "Delayed by 2.5 hours due to scheduled capacity pause"
            ],
            "recommendations": {
                "prioritizing_urgent_orders": "Defer non-critical remaining 550 units to Shift 2 night run; keep Line L4 focused on PO-9806 delivery.",
                "reallocating_capacity": "Resume production on Cell R-08 at 18:00.",
                "resolving_bottlenecks": "Wipe vision camera lens to prevent false reject pauses.",
                "replenishing_materials": "Material levels are sufficient."
            }
        },
        {
            "order_id": "PO-9808",
            "line_id": "Line L3: Hydraulic Valve Assembly",
            "product_name": "Spline Coupling Hub (SKU: HYD-7705)",
            "category": "Hydraulic Components",
            "mfg_unit": "Factory Unit 2 (Stuttgart)",
            "planned_qty": 1500,
            "actual_qty": 1470,
            "planned_completion": "13:30 (Shift 1)",
            "actual_completion": "13:40 (Shift 1)",
            "delay_hrs": 0.17,
            "status": "On Schedule",
            "status_badge": "status-badge-green",
            "rejection_rate": "1.1% (16 rejected parts)",
            "raw_mat_status": "Adequate",
            "bottlenecks": "None",
            "issues": [
                "Optimal throughput (98.0% attainment)"
            ],
            "recommendations": {
                "prioritizing_urgent_orders": "Transfer completed batch to Detroit packaging hub.",
                "reallocating_capacity": "Release honing fixture for PO-9805.",
                "resolving_bottlenecks": "None.",
                "replenishing_materials": "Standard replenishment."
            }
        }
    ]

    # Cross-Tier AI Cause-and-Effect Matrix (Connecting Manufacturing to Production)
    ai_cross_tier_matrix = [
        {
            "priority": "P1 (Immediate Critical)",
            "priority_badge": "status-badge-red",
            "mfg_problem": "Station ST-20 Spindle Thermal Chatter & Tool Bit Wear (78%)",
            "mfg_unit": "Factory Unit 1 (Detroit)  -  Cell Fastening",
            "prod_effect": "Line L1 Order PO-9802 delayed by 4.5 hours; 124 rejected parts; cycle time spiked from 32s to 48.2s",
            "prod_order": "PO-9802 (M10-Flange Shaft)",
            "severity": "[Critical] Critical",
            "root_cause": "Tool bit wear + stiff steel (>216 HV) causes screwdriver to slip and trip prematurely before 360° seating clamp.",
            "resolve_action": "Halt line for 20 minutes; replace cutter bit; deploy adaptive torque offset (-0.90 Nm, +92° angle window).",
            "prevent_action": "Automated vibration lockout at 2.5g; auto tool-life countdown flagged at 70% wear life."
        },
        {
            "priority": "P1 (Immediate Critical)",
            "priority_badge": "status-badge-red",
            "mfg_problem": "AISI 4140 Stockout & Metallurgical Hardness Drift (>218 HV)",
            "mfg_unit": "Factory Unit 1 (Detroit)  -  Raw Materials",
            "prod_effect": "Line L1 Order PO-9802 stockout risk in 1.8 shifts; high scrap rate (6.2%)",
            "prod_order": "PO-9802 (M10-Flange Shaft)",
            "severity": "[Critical] Critical",
            "root_cause": "Vendor B mill furnace backlog caused 10-day lead time slip + inconsistent quenching hardness.",
            "resolve_action": "Transfer 350 kg emergency buffer from Nagoya; activate dual-sourcing to Vendor A; apply recipe offset.",
            "prevent_action": "Tighten receiving USL to 216.0 HV; enforce 7-day safety stock buffer; demand $2.33M SCAR recovery."
        },
        {
            "priority": "P2 (High Priority)",
            "priority_badge": "status-badge-orange",
            "mfg_problem": "Hydraulic Pressure Decay on CNC Mill #04 (95 psi vs 115 psi target)",
            "mfg_unit": "Factory Unit 2 (Stuttgart)  -  Machining",
            "prod_effect": "Line L3 Order PO-9805 delayed by 3.25 hours; bore taper dimensional drift (86 scrapped parts)",
            "prod_order": "PO-9805 (Pressure Regulator)",
            "severity": "[Elevated] High",
            "root_cause": "Proportional valve filter clogged with micro-particulates and seal micro-leakage.",
            "resolve_action": "Clean filter element and bleed accumulator during shift handover; reduce hone speed by -8%.",
            "prevent_action": "Install continuous differential pressure sensor across filter housing for predictive maintenance alert."
        },
        {
            "priority": "P3 (Medium Priority)",
            "priority_badge": "status-badge-orange",
            "mfg_problem": "Riveter Anvil Guide Track Friction & Thread Burr Stalls",
            "mfg_unit": "Factory Unit 1 (Detroit)  -  Chassis Cell",
            "prod_effect": "Line L2 Order PO-9803 delayed by 2.25 hours; stud stock approaching safety threshold (2.8 days)",
            "prod_order": "PO-9803 (High-Tensile Stud Hub)",
            "severity": "[Elevated] High",
            "root_cause": "Supplier die wear generated micro-burrs on stud heads, jamming riveter feed chute.",
            "resolve_action": "Apply PTFE dry lube to anvil guides; mandate vendor inspect burr height before shipment.",
            "prevent_action": "Mandate die replacement schedule every 50,000 cycles with Vendor A."
        },
        {
            "priority": "P4 (Scheduled Maintenance)",
            "priority_badge": "status-badge-blue",
            "mfg_problem": "Robotic Fastener Vision Optical Shield Coolant Mist Blur",
            "mfg_unit": "Factory Unit 4 (Austin)  -  Final QC",
            "prod_effect": "Line L4 Order PO-9807 delayed by 2.5 hours; vision gate false rejects elevated to 1.8%",
            "prod_order": "PO-9807 (Sensor Bracket)",
            "severity": " Medium",
            "root_cause": "Aerosol coolant vapor deposition on camera optical shield without continuous positive purge.",
            "resolve_action": "Wipe lens shield and activate positive-pressure air-knife curtain.",
            "prevent_action": "Enclose optical inspection portal with multi-spectral polarized illumination."
        }
    ]

    return {
        "material": mat_df,
        "material_enriched": mat_enriched,
        "supplier": supp_df,
        "warranty": warr_enriched,
        "sample_parts": sample_parts,
        "supplier_products": supplier_products,
        "raw_materials_inventory": raw_materials_inventory,
        "mfg_units": mfg_units,
        "prod_orders": prod_orders,
        "ai_cross_tier_matrix": ai_cross_tier_matrix
    }

data = load_datasets()

# -----------------------------------------------------------------------------
# 4. DESIGN SYSTEM & CSS (SMARTNUTRI AI LIGHT THEME)
# -----------------------------------------------------------------------------
render_html("""
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
    padding: 16px 14px;
    text-align: center;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.02);
    height: 100%;
}

.metric-card-label {
    font-size: 0.70rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #64748B;
    margin-bottom: 4px;
}

.metric-card-number {
    font-size: 1.75rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin-bottom: 4px;
}

.metric-card-target {
    font-size: 0.74rem;
    font-weight: 700;
    color: #1E3A8A;
    background: #EFF6FF;
    padding: 2px 6px;
    border-radius: 4px;
    display: inline-block;
    margin-bottom: 4px;
}

.metric-card-delta {
    font-size: 0.72rem;
    font-weight: 600;
    line-height: 1.3;
}

/* Smart Cards */
.smart-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 22px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.02);
    margin-bottom: 18px;
}

.card-title {
    font-size: 1.15rem;
    font-weight: 800;
    color: #0F172A;
    margin: 0 0 12px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Badge Pills */
.badge-pill-red {
    background: #FEE2E2;
    color: #991B1B;
    border: 1px solid #FCA5A5;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 700;
    display: inline-block;
}

.badge-pill-orange {
    background: #FFEDD5;
    color: #9A3412;
    border: 1px solid #FDBA74;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 700;
    display: inline-block;
}

.badge-pill-green {
    background: #DCFCE7;
    color: #166534;
    border: 1px solid #86EFAC;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 700;
    display: inline-block;
}

.badge-pill-blue {
    background: #DBEAFE;
    color: #1E40AF;
    border: 1px solid #93C5FD;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 700;
    display: inline-block;
}

/* Status Badges */
.status-badge-red {
    background: #FEF2F2;
    color: #B91C1C;
    border: 1px solid #FCA5A5;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 700;
    display: inline-block;
}

.status-badge-orange {
    background: #FFFBEB;
    color: #B45309;
    border: 1px solid #FDE68A;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 700;
    display: inline-block;
}

.status-badge-green {
    background: #ECFDF5;
    color: #047857;
    border: 1px solid #A7F3D0;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 700;
    display: inline-block;
}

.status-badge-blue {
    background: #EFF6FF;
    color: #1D4ED8;
    border: 1px solid #BFDBFE;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 700;
    display: inline-block;
}

/* Callouts */
.callout-blue {
    background: #EFF6FF;
    border-left: 4px solid #2563EB;
    border-radius: 8px;
    padding: 14px 18px;
    font-size: 0.92rem;
    color: #1E40AF;
    margin-top: 10px;
}

.callout-red {
    background: #FEF2F2;
    border-left: 4px solid #DC2626;
    border-radius: 8px;
    padding: 14px 18px;
    font-size: 0.92rem;
    color: #991B1B;
    margin-top: 10px;
}

.callout-green {
    background: #ECFDF5;
    border-left: 4px solid #059669;
    border-radius: 8px;
    padding: 14px 18px;
    font-size: 0.92rem;
    color: #065F46;
    margin-top: 10px;
}

/* Issue Card Grid */
.issue-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.02);
}
</style>
""")

# -----------------------------------------------------------------------------
# 5. MAIN HEADER HERO
# -----------------------------------------------------------------------------
render_html("""
<div class="hero-banner">
    <h1 class="hero-title">Factory Operations & Quality Executive Dashboard</h1>
    <p class="hero-sub">AI-powered defect prevention, 199-batch raw material tracking, production line schedule alignment, and root cause mitigation across all manufacturing tiers.</p>
</div>
""")

# -----------------------------------------------------------------------------
# 7. TOP HORIZONTAL TABS (STRUCTURED & INTUITIVE FOR MANAGERS)
# -----------------------------------------------------------------------------
tab_supp, tab_mat, tab_mfg_unit, tab_prod_line = st.tabs([
    "[Tier 1] 1. Supplier / Vendor Level",
    "[Tier 2] 2. Raw Material & Inventory Level",
    "[Tier 3] 3. Manufacturing Level (Factories & Units)",
    "[Tier 4] 4. Production Level (Lines & Orders)"
])

# =============================================================================
# TAB 1: SUPPLIER / VENDOR LEVEL ISSUES
# =============================================================================
with tab_supp:
    # 1. INDIVIDUAL VENDOR SELECTION & LIVE PROFILE HEADER
    col_sel, col_info = st.columns([1.5, 1.0])
    with col_sel:
        vendor_choice = st.selectbox(
            "[Tier 1] Select Individual Supplier / Vendor to Analyze:",
            [
                "VENDOR A  -  Fastener Bar Stock & Threaded Hardware Co.",
                "VENDOR B  -  Structural Cold Drawn & Steel Technologies",
                "VENDOR C  -  High Strength Alloy & Precision Metallurgy"
            ],
            key="supp_choice"
        )
    supp_id = "VENDOR_A" if "VENDOR A" in vendor_choice else ("VENDOR_B" if "VENDOR B" in vendor_choice else "VENDOR_C")

    # Data slice for selected supplier
    warr_supp = data["warranty"][data["warranty"]["supplier_id"] == supp_id]
    mat_supp = data["material"][data["material"]["supplier_id"] == supp_id]
    hist_supp = data["supplier"][data["supplier"]["supplier_id"] == supp_id].sort_values("snapshot_date")
    supp_prods = data["supplier_products"].get(supp_id, [])

    claim_count = len(warr_supp)
    claim_cost = float(warr_supp["claim_cost_usd"].sum())
    chargebacks = float(mat_supp["chargeback_amount_usd"].sum())
    scrap_cost = float(mat_supp["scrap_count"].sum() * 55.0)
    total_claim = scrap_cost + chargebacks + claim_cost

    recent_rate = hist_supp["historical_defect_rate_pct"].iloc[-1] if not hist_supp.empty else 0.0
    recent_capa = int(hist_supp["capa_recurrence_count"].iloc[-1]) if not hist_supp.empty else 0
    capa_events = int(hist_supp["capa_recurrence_count"].sum()) if not hist_supp.empty else 0
    top_mode = hist_supp["recurring_failure_mode"].mode()[0] if not hist_supp.empty else "INCOMPLETE_FASTENING"

    delayed_count = len([p for p in supp_prods if p.get("is_delayed", False)])
    tolerance_leak = 18 if supp_id == "VENDOR_B" else 0

    # Balanced Enterprise Vendor Risk Score (0 - 100)
    supp_risk = int(min(100, (recent_rate / 15.0) * 35 + (recent_capa / 10.0) * 20 + (delayed_count / len(supp_prods)) * 25 + (claim_cost / 800000.0) * 10 + tolerance_leak))
    if supp_risk > 70:
        risk_badge = f'<span class="badge-pill-red">[Critical] Risk Score: {supp_risk}/100 (Critical Chronic Vendor Risk)</span>'
        callout_type = "callout-red"
    elif supp_risk > 35:
        risk_badge = f'<span class="badge-pill-orange">[Elevated] Risk Score: {supp_risk}/100 (Controlled / Moderate Watchlist)</span>'
        callout_type = "callout-blue"
    else:
        risk_badge = f'<span class="badge-pill-green">[Optimal] Risk Score: {supp_risk}/100 (Controlled Benchmark Vendor - Low Risk)</span>'
        callout_type = "callout-green"

    with col_info:
        render_html(f"""
        <div style="display: flex; justify-content: flex-end; align-items: center; height: 100%;">
            {risk_badge}
        </div>
        """)

    # 4 Clean Metric Cards
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_html(f'<div class="metric-card-white"><div class="metric-card-label">CURRENT DEFECT RATE</div><div class="metric-card-number" style="color: {"#DC2626" if recent_rate > 8 else ("#D97706" if recent_rate > 3 else "#059669")};">{recent_rate:.2f}%</div><div class="metric-card-target">Target: < 1.50%</div><div class="metric-card-delta" style="color: {"#DC2626" if recent_rate > 8 else "#64748B"};">{"+" + f"{recent_rate - 1.5:.2f}%" if recent_rate > 1.5 else "Nominal"} Six Sigma envelope</div></div>')
    with k2:
        render_html(f'<div class="metric-card-white"><div class="metric-card-label">FIELD WARRANTY CLAIMS</div><div class="metric-card-number" style="color: {"#DC2626" if claim_count > 400 else ("#D97706" if claim_count > 300 else "#2563EB")};">{claim_count} parts</div><div class="metric-card-target">Target: 0 Parts</div><div class="metric-card-delta" style="color: #64748B;">Monitored field returns</div></div>')
    with k3:
        render_html(f'<div class="metric-card-white"><div class="metric-card-label">WARRANTY COST LEAKED</div><div class="metric-card-number" style="color: #2563EB;">${claim_cost:,.0f}</div><div class="metric-card-target">Target: $0.00</div><div class="metric-card-delta" style="color: #2563EB;">Eligible for chargeback</div></div>')
    with k4:
        render_html(f'<div class="metric-card-white"><div class="metric-card-label">TOTAL FINANCIAL DEMAND</div><div class="metric-card-number" style="color: #059669;">${total_claim:,.0f}</div><div class="metric-card-target">8D SCAR Claim</div><div class="metric-card-delta" style="color: #059669;">Scrap + Chargebacks + Warranty</div></div>')

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 2. FEATURE A: WHAT ALL PRODUCTS THAT VENDOR WILL SUPPLY
    # -------------------------------------------------------------------------
    render_html(f"""
    <div class="smart-card">
        <div class="card-title" style="display: flex; align-items: center; gap: 8px;">
            <span>[Catalog]</span> 1. What All Products {supp_id} Supplies & Delivery Turnaround
        </div>
        <p style="font-size: 0.90rem; color: #64748B; margin-bottom: 12px;">
            Complete catalog of parts and materials procured from <strong>{supp_id}</strong>, including target lead times, actual turnaround times, delivery variance, quality scores, and health risk tiers.
        </p>
    </div>
    """)

    prod_rows = []
    for p in supp_prods:
        delay_val = p.get('delay_days', 0)
        delay_str = f"+{delay_val} days delay" if delay_val >= 3 else ("Normal Buffer (+1-2d)" if delay_val > 0 else "0d (On Schedule)")
        prod_rows.append({
            "SKU": p["sku"],
            "Product / Material Name": p["name"],
            "Category": p["category"],
            "Monthly Demand": p.get("monthly_vol", "N/A"),
            "Target Lead Time": f"{p['target_lead_days']} days",
            "Actual Turnaround": f"{p['actual_lead_days']} days",
            "Delivery Variance": delay_str,
            "Quality Conformance": p["quality_score"],
            "Risk & Health Tier": p.get("risk_tier", "[Optimal] Low Risk (Optimal)"),
            "Status": p["status"]
        })
    df_prods = pd.DataFrame(prod_rows)
    st.dataframe(df_prods, use_container_width=True, height=190)

    # Visual Comparison Bar Chart for Lead Times with color-coding
    chart_df = pd.DataFrame([
        {
            "Product": p["name"][:25] + "...",
            "Target Lead Time (Days)": p["target_lead_days"],
            "Actual Turnaround (Days)": p["actual_lead_days"],
            "Color": "#DC2626" if p.get("is_delayed", False) else ("#D97706" if p.get("delay_days", 0) >= 4 else "#10B981")
        } for p in supp_prods
    ])

    fig_lead = go.Figure()
    fig_lead.add_trace(go.Bar(
        x=chart_df["Product"],
        y=chart_df["Target Lead Time (Days)"],
        name="Target Lead Time",
        marker_color="#94A3B8"
    ))
    fig_lead.add_trace(go.Bar(
        x=chart_df["Product"],
        y=chart_df["Actual Turnaround (Days)"],
        name="Actual Turnaround (Green=On Schedule, Red=Critical Delay)",
        marker_color=chart_df["Color"]
    ))
    fig_lead.update_layout(
        barmode='group',
        height=260,
        margin=dict(l=10, r=10, t=25, b=10),
        yaxis=dict(title="Days"),
        xaxis=dict(title=""),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_lead, use_container_width=True)

    # -------------------------------------------------------------------------
    # 3. FEATURE B: AMONG THEM WHICH PRODUCT TAKING MORE TIME & MORE DELAY
    # -------------------------------------------------------------------------
    critical_delayed = [p for p in supp_prods if p.get("is_delayed", False) or p.get("delay_days", 0) >= 6]
    moderate_delayed = [p for p in supp_prods if p not in critical_delayed and p.get("delay_days", 0) >= 3]
    healthy_prods = [p for p in supp_prods if p not in critical_delayed and p not in moderate_delayed]

    render_html(f"""
    <div class="smart-card" style="border-left: 6px solid {"#DC2626" if critical_delayed else "#10B981"}; margin-top: 18px;">
        <div class="card-title" style="display: flex; align-items: center; gap: 8px; color: {"#991B1B" if critical_delayed else "#065F46"};">
            <span>{"[Bottleneck Alert]" if critical_delayed else "[Controlled]"}</span> 2. Delay & Bottleneck Analysis: Which Products Are Taking More Time?
        </div>
        <p style="font-size: 0.90rem; color: #64748B; margin-bottom: 12px;">
            Accurately distinguishing healthy, on-schedule inventory from genuine mill bottlenecks and excessive supplier lead time delays.
        </p>
    </div>
    """)

    # High-Level Portfolio Health Summary Banner
    render_html(f"""
    <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 14px 18px; margin-bottom: 14px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.02);">
        <div style="font-size: 0.95rem; font-weight: 700; color: #0F172A;">
            Portfolio Health Breakdown: 
            <span style="color: #059669; font-weight: 800;">{len(healthy_prods)} of {len(supp_prods)} Products Healthy & On Schedule ({int(len(healthy_prods)/len(supp_prods)*100)}%)</span>
            {f' &nbsp;*&nbsp; <span style=\"color: #DC2626; font-weight: 800;\">{len(critical_delayed)} Delayed Bottleneck(s)</span>' if critical_delayed else ''}
            {f' &nbsp;*&nbsp; <span style=\"color: #D97706; font-weight: 800;\">{len(moderate_delayed)} Moderate Watchlist</span>' if moderate_delayed else ''}
        </div>
        <div>
            <span class="{'badge-pill-green' if len(critical_delayed) == 0 else ('badge-pill-orange' if len(critical_delayed) <= 1 else 'badge-pill-red')}">
                {'[Optimal] High Operational Stability' if len(critical_delayed) == 0 else ('[Elevated] Contained Single-Component Delay' if len(critical_delayed) <= 1 else '[Critical] Severe Multi-Product Mill Bottlenecks')}
            </span>
        </div>
    </div>
    """)

    if critical_delayed:
        # Sort delayed items by delay_days descending
        sorted_delayed = sorted(critical_delayed, key=lambda x: x.get('delay_days', 0), reverse=True)
        worst_p = sorted_delayed[0]

        # Highlight the single product taking the most time / delay
        render_html(f"""
        <div style="background: #FEF2F2; border: 2px solid #EF4444; border-radius: 14px; padding: 18px; margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px;">
                <div>
                    <span style="background: #DC2626; color: #FFFFFF; font-size: 0.74rem; font-weight: 800; padding: 4px 10px; border-radius: 6px; text-transform: uppercase; letter-spacing: 0.5px;">
                        [Warning] #1 HIGHEST DELAY BOTTLENECK PRODUCT FOR {supp_id}
                    </span>
                    <div style="font-size: 1.35rem; font-weight: 800; color: #991B1B; margin-top: 8px;">
                        {worst_p['name']}
                    </div>
                    <div style="font-size: 0.88rem; color: #7F1D1D; margin-top: 3px;">
                        SKU: <strong style="font-family: monospace;">{worst_p['sku']}</strong> &nbsp;|&nbsp; Category: <strong>{worst_p['category']}</strong> &nbsp;|&nbsp; Monthly Demand: <strong>{worst_p.get('monthly_vol', 'N/A')}</strong>
                    </div>
                </div>
                <div style="text-align: right; background: #FFFFFF; border: 1px solid #FCA5A5; border-radius: 10px; padding: 10px 18px;">
                    <div style="font-size: 0.75rem; font-weight: 800; color: #991B1B; text-transform: uppercase;">Excess Turnaround Delay</div>
                    <div style="font-size: 1.85rem; font-weight: 900; color: #DC2626; line-height: 1.1;">+{worst_p['delay_days']} DAYS</div>
                    <div style="font-size: 0.80rem; color: #7F1D1D; margin-top: 2px;">Takes <strong>{worst_p['actual_lead_days']}d</strong> (Target: {worst_p['target_lead_days']}d)</div>
                </div>
            </div>
            <div style="margin-top: 14px; padding-top: 12px; border-top: 1px dashed #FCA5A5; font-size: 0.90rem; color: #450A0A; line-height: 1.6;">
                <strong>Mill Process Bottleneck & Root Cause:</strong> {worst_p['root_cause']}
            </div>
        </div>
        """)

        # Grid of all delayed products for this vendor
        st.markdown(f"<div style='font-size: 0.95rem; font-weight: 700; color: #0F172A; margin-bottom: 8px;'>Critical Delayed Components ({len(critical_delayed)} items):</div>", unsafe_allow_html=True)
        cols_d = st.columns(len(critical_delayed))
        for idx, dp in enumerate(critical_delayed):
            with cols_d[idx]:
                render_html(f"""
                <div style="background: #FFFFFF; border: 1px solid #FCA5A5; border-top: 4px solid #DC2626; border-radius: 12px; padding: 14px; height: 100%; box-shadow: 0 2px 8px rgba(0,0,0,0.03);">
                    <div style="font-weight: 800; font-size: 0.95rem; color: #991B1B;">{dp['name']}</div>
                    <div style="font-size: 0.78rem; color: #7F1D1D; margin: 4px 0 8px 0;">SKU: {dp['sku']} | {dp['category']}</div>
                    <div style="font-size: 0.88rem; color: #1E293B; line-height: 1.6;">
                        * <strong>Actual Turnaround:</strong> <span style="color: #DC2626; font-weight: 700;">{dp['actual_lead_days']} days</span><br>
                        * <strong>Target Lead Time:</strong> {dp['target_lead_days']} days<br>
                        * <strong>Excess Delay:</strong> <span style="background: #FEE2E2; color: #B91C1C; padding: 2px 7px; border-radius: 4px; font-weight: 800;">+{dp['delay_days']} days</span><br>
                        * <strong>Cause:</strong> {dp['root_cause']}
                    </div>
                </div>
                """)

        # Reassuring list of healthy products for this vendor
        if healthy_prods:
            healthy_items_html = " &nbsp;*&nbsp; ".join([f"<strong>{p['name']}</strong> ({p['sku']} * {p['actual_lead_days']}d * Conformance: {p['quality_score']})" for p in healthy_prods])
            render_html(f"""
            <div style="background: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 10px; padding: 12px 16px; margin-top: 14px;">
                <div style="font-size: 0.82rem; font-weight: 800; color: #15803D; text-transform: uppercase;">
                    [Controlled] {len(healthy_prods)} Healthy Products Arriving On Schedule (Low Risk):
                </div>
                <div style="font-size: 0.88rem; color: #166534; margin-top: 4px; line-height: 1.6;">
                    {healthy_items_html}
                </div>
            </div>
            """)
    else:
        render_html(f"""
        <div class="callout-green">
            [Controlled] <strong>All {len(supp_prods)} products supplied by {supp_id} are arriving on schedule</strong> within established target tolerances with zero bottleneck delays.
        </div>
        """)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 4. FEATURE C: RECOMMENDATIONS FOR THAT INDIVIDUAL PRODUCT
    # -------------------------------------------------------------------------
    render_html(f"""
    <div class="smart-card" style="border-left: 6px solid #2563EB;">
        <div class="card-title" style="display: flex; align-items: center; gap: 8px;">
            <span>[AI Recommendations]</span> 3. Actionable AI Recommendations for Individual Products
        </div>
        <p style="font-size: 0.90rem; color: #64748B; margin-bottom: 12px;">
            Select any individual product supplied by <strong>{supp_id}</strong> to review specific defect modes, physical symptoms, mill bottlenecks, and prescriptive AI recommendations.
        </p>
    </div>
    """)

    # Dropdown to select individual product with clear Risk & Status tags
    prod_options = [
        f"{p['sku']}  -  {p['name']} [{p.get('risk_tier', '[Optimal] Low Risk')}]"
        for p in supp_prods
    ]

    # Pre-select highest delayed product by default if one exists, else index 0
    default_p_idx = 0
    if critical_delayed:
        worst_sku = sorted(critical_delayed, key=lambda x: x.get('delay_days', 0), reverse=True)[0]['sku']
        for idx, p in enumerate(supp_prods):
            if p['sku'] == worst_sku:
                default_p_idx = idx
                break

    col_p_sel, col_p_badge = st.columns([1.6, 1.0])
    with col_p_sel:
        selected_prod_label = st.selectbox(
            f"Select Individual Product from {supp_id} for Specific AI Recommendations:",
            prod_options,
            index=default_p_idx,
            key=f"prod_select_{supp_id}"
        )

    selected_p = next((p for p in supp_prods if f"{p['sku']}  -  {p['name']}" in selected_prod_label), supp_prods[0])
    status_border_color = "#DC2626" if selected_p.get("is_delayed", False) else ("#D97706" if selected_p.get("delay_days", 0) >= 3 else "#059669")
    delay_indicator = f"+{selected_p['delay_days']} days delay" if selected_p["delay_days"] >= 3 else ("Normal Buffer (+1-2d)" if selected_p["delay_days"] > 0 else "On Schedule (0d)")

    with col_p_badge:
        render_html(f"""
        <div style="display: flex; justify-content: flex-end; align-items: center; height: 100%; padding-top: 14px;">
            <span class="{selected_p['status_badge']}" style="font-size: 0.85rem; padding: 6px 14px; font-weight: 700;">
                {selected_p['status']}
            </span>
        </div>
        """)

    # Detailed Explanation Card for Selected Product
    rec_items = [r.strip() for r in selected_p['recs'].split(";") if r.strip()]
    rec_bullets_html = "".join([f'<li style="margin-bottom: 6px;">{item}</li>' for item in rec_items])
    quality_num = float(selected_p['quality_score'].replace('%', '').strip())
    quality_color = "#059669" if quality_num >= 95 else ("#D97706" if quality_num >= 90 else "#DC2626")

    render_html(f"""
    <div class="issue-card" style="border-left: 6px solid {status_border_color}; background: #FFFFFF; border-radius: 14px; padding: 22px; box-shadow: 0 4px 14px rgba(0,0,0,0.03); margin-top: 10px; margin-bottom: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 14px; border-bottom: 1px solid #F1F5F9; padding-bottom: 14px;">
            <div>
                <span style="font-size: 0.76rem; font-weight: 800; color: #2563EB; text-transform: uppercase; letter-spacing: 0.5px;">INDIVIDUAL PRODUCT DEEP-DIVE SPECIFICATION</span>
                <div style="font-size: 1.35rem; font-weight: 800; color: #0F172A; margin: 3px 0;">{selected_p['name']}</div>
                <div style="font-size: 0.88rem; color: #64748B;">
                    <strong>SKU:</strong> <span style="font-family: monospace; font-weight: 700; color: #334155;">{selected_p['sku']}</span> &nbsp;*&nbsp; 
                    <strong>Category:</strong> <span style="font-weight: 600; color: #334155;">{selected_p['category']}</span> &nbsp;*&nbsp; 
                    <strong>Supplier:</strong> <span style="font-weight: 600; color: #334155;">{supp_id}</span>
                </div>
            </div>
            <div style="text-align: right;">
                <span style="font-size: 0.72rem; font-weight: 800; color: #64748B; text-transform: uppercase;">QUALITY CONFORMANCE</span>
                <div style="font-size: 1.55rem; font-weight: 800; color: {quality_color};">
                    {selected_p['quality_score']}
                </div>
            </div>
        </div>

        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 18px;">
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 10px 14px; text-align: center;">
                <div style="font-size: 0.70rem; font-weight: 700; color: #64748B; text-transform: uppercase;">Monthly Volume</div>
                <div style="font-size: 1.1rem; font-weight: 800; color: #0F172A; margin-top: 2px;">{selected_p['monthly_vol']}</div>
            </div>
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 10px 14px; text-align: center;">
                <div style="font-size: 0.70rem; font-weight: 700; color: #64748B; text-transform: uppercase;">Target Lead Time</div>
                <div style="font-size: 1.1rem; font-weight: 800; color: #0F172A; margin-top: 2px;">{selected_p['target_lead_days']} Days</div>
            </div>
            <div style="background: {'#FEF2F2' if selected_p['is_delayed'] else '#F8FAFC'}; border: 1px solid {'#FCA5A5' if selected_p['is_delayed'] else '#E2E8F0'}; border-radius: 10px; padding: 10px 14px; text-align: center;">
                <div style="font-size: 0.70rem; font-weight: 700; color: {'#B91C1C' if selected_p['is_delayed'] else '#64748B'}; text-transform: uppercase;">Actual Turnaround</div>
                <div style="font-size: 1.1rem; font-weight: 800; color: {'#DC2626' if selected_p['is_delayed'] else '#0F172A'}; margin-top: 2px;">{selected_p['actual_lead_days']} Days</div>
            </div>
            <div style="background: {'#FEF2F2' if selected_p['delay_days'] > 0 else '#ECFDF5'}; border: 1px solid {'#FCA5A5' if selected_p['delay_days'] > 0 else '#A7F3D0'}; border-radius: 10px; padding: 10px 14px; text-align: center;">
                <div style="font-size: 0.70rem; font-weight: 700; color: {'#B91C1C' if selected_p['delay_days'] > 0 else '#047857'}; text-transform: uppercase;">Lead Time Variance</div>
                <div style="font-size: 1.1rem; font-weight: 800; color: {'#DC2626' if selected_p['delay_days'] > 0 else '#059669'}; margin-top: 2px;">{delay_indicator}</div>
            </div>
        </div>

        <div style="display: flex; flex-direction: column; gap: 12px;">
            <div style="background: #FFFBEB; border: 1px solid #FDE68A; border-radius: 10px; padding: 12px 16px;">
                <div style="font-size: 0.80rem; font-weight: 800; color: #B45309; text-transform: uppercase; margin-bottom: 4px; display: flex; align-items: center; gap: 6px;">
                    <span>[Warning]</span> Defect Mode & Physical Symptoms
                </div>
                <div style="font-size: 0.92rem; color: #78350F; font-weight: 600;">
                    {selected_p['defect_mode']}
                </div>
            </div>

            <div style="background: #F1F5F9; border: 1px solid #E2E8F0; border-radius: 10px; padding: 12px 16px;">
                <div style="font-size: 0.80rem; font-weight: 800; color: #475569; text-transform: uppercase; margin-bottom: 4px; display: flex; align-items: center; gap: 6px;">
                    <span></span> Mill Process Bottleneck & Root Cause
                </div>
                <div style="font-size: 0.90rem; color: #1E293B; line-height: 1.5;">
                    {selected_p['root_cause']}
                </div>
            </div>

            <div style="background: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 10px; padding: 14px 18px;">
                <div style="font-size: 0.82rem; font-weight: 800; color: #1D4ED8; text-transform: uppercase; margin-bottom: 8px; display: flex; align-items: center; gap: 6px;">
                    <span>[AI Recommendations]</span> Actionable AI Mitigation & Engineering Recommendations
                </div>
                <ul style="margin: 0; padding-left: 20px; font-size: 0.90rem; color: #1E3A8A; line-height: 1.6;">
                    {rec_bullets_html}
                </ul>
            </div>
        </div>
    </div>
    """)

    # Collapsible overview for all products
    with st.expander(f"[Summary] View Summary Cards for All {len(supp_prods)} Products Supplied by {supp_id}", expanded=False):
        for p in supp_prods:
            status_color = "#DC2626" if p["is_delayed"] else ("#059669" if p["delay_days"] == 0 else "#2563EB")
            render_html(f"""
            <div class="issue-card" style="border-left: 5px solid {status_color}; margin-bottom: 10px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <div style="font-weight: 800; font-size: 0.95rem; color: #0F172A;">
                        {p['name']} <span style="font-size: 0.8rem; font-weight: 600; color: #64748B;">({p['sku']})</span>
                    </div>
                    <span class="{p['status_badge']}">{p['status']}</span>
                </div>
                <div style="font-size: 0.84rem; color: #475569; margin-bottom: 6px;">
                    <strong>Category:</strong> {p['category']} | <strong>Lead Time:</strong> {p['actual_lead_days']}d (Target: {p['target_lead_days']}d, Delay: +{p['delay_days']}d) | <strong>Quality Score:</strong> {p['quality_score']}
                </div>
                <div style="background: #F8FAFC; border-radius: 8px; padding: 8px 12px; font-size: 0.86rem; color: #1E293B; line-height: 1.5;">
                    <strong>Recommended Actions:</strong> {p['recs']}
                </div>
            </div>
            """)

    # Category 1 Master Diagnostic Summary
    render_html(f"""
    <div class="smart-card" style="border-left: 6px solid #2563EB;">
        <div class="card-title">[Summary] Category 1 Diagnostic Summary for {supp_id}</div>
        
        <div style="margin-bottom: 14px;">
            <strong style="color: #64748B; font-size: 0.85rem; text-transform: uppercase;">1. Risk Score:</strong><br>
            <span style="font-size: 1.15rem; font-weight: 800; color: #1E3A8A;">{supp_risk}/100</span>  -  {risk_badge}
        </div>

        <div style="margin-bottom: 14px;">
            <strong style="color: #64748B; font-size: 0.85rem; text-transform: uppercase;">2. Which Issues:</strong>
            <p style="margin: 4px 0 0 0; font-size: 0.95rem; color: #0F172A; font-weight: 600;">
                * Chronic high defect rate of <span style="color: #DC2626;">{recent_rate:.2f}%</span>.<br>
                * Primary recurring defect mode: <span style="color: #DC2626;">{top_mode}</span>.<br>
                * {capa_events} repeat CAPA non-conformance incidents recorded.
            </p>
        </div>

        <div style="margin-bottom: 14px;">
            <strong style="color: #64748B; font-size: 0.85rem; text-transform: uppercase;">3. Factors Affecting Risk / Proper Function:</strong>
            <ul style="margin: 4px 0 0 0; padding-left: 18px; font-size: 0.92rem; color: #334155; line-height: 1.6;">
                <li><strong>Tolerance Leakage:</strong> Incoming steel bar stock tested between <strong>216 HV and 230 HV</strong> (barely passed nominal receiving specs).</li>
                <li><strong>Downstream Impact:</strong> Excessively stiff raw metal caused assembly screwdrivers to shut off prematurely, leading to <strong>${claim_cost:,.2f}</strong> in customer warranty failures.</li>
            </ul>
        </div>

        <div class="{callout_type}">
            <strong style="font-size: 0.85rem; color: #0F172A; text-transform: uppercase;">4. Actionable Supplier Strategy:</strong>
            <ul style="margin: 4px 0 0 0; padding-left: 18px; font-size: 0.92rem; color: #1E293B; line-height: 1.6;">
                <li><strong>Tighten Spec Limit:</strong> Lower incoming hardness USL from <strong>230 HV down to 216.0 HV</strong> immediately to stop downstream leakage.</li>
                <li><strong>Issue 8D SCAR Claim:</strong> Issue formal recovery claim of <strong>${total_claim:,.2f}</strong> (In-Plant Scrap: ${scrap_cost:,.0f} | Chargebacks: ${chargebacks:,.0f} | Warranty: ${claim_cost:,.0f}).</li>
                <li><strong>Mandatory Pre-Shipment Audit:</strong> Require supplier to submit metallurgical certs confirming hardness &le; 216 HV prior to dispatch.</li>
            </ul>
        </div>
    </div>
    """)


# =============================================================================
# TAB 2: RAW MATERIALS & INVENTORY LEVEL ISSUES
# =============================================================================
with tab_mat:
    render_html("""
    <div class="smart-card">
        <div class="card-title">[Tier 2] Raw Material Level & Inventory Optimization Intelligence</div>
        <p style="font-size: 0.92rem; color: #64748B; margin: 0 0 16px 0;">
            Real-time evaluation: <strong>Which raw materials are at risk</strong>, <strong>how much inventory to maintain</strong>, and <strong>actions to avoid shortages without holding excess stock</strong>.
        </p>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 14px;">
            <div style="background: #F0FDF4; border: 1px solid #BBF7D0; border-left: 4px solid #16A34A; border-radius: 8px; padding: 12px 14px;">
                <span style="background: #DCFCE7; color: #15803D; font-size: 0.72rem; font-weight: 800; padding: 2px 7px; border-radius: 4px; text-transform: uppercase;">LEVEL 1: 5 WAREHOUSE SKUs (TOP SECTION)</span>
                <div style="font-size: 0.92rem; font-weight: 700; color: #0F172A; margin-top: 4px;">Macro Inventory & Replenishment Planning</div>
                <div style="font-size: 0.82rem; color: #475569; margin-top: 3px; line-height: 1.5;">
                    Tracks aggregate inventory across the <strong>5 core raw material families</strong> (AISI 4140, ST-88, M10, Ni-Cr, Titanium). Evaluates stockout risks, buffer days, safety stock, and reorder points (ROP).
                </div>
            </div>
            <div style="background: #EFF6FF; border: 1px solid #BFDBFE; border-left: 4px solid #2563EB; border-radius: 8px; padding: 12px 14px;">
                <span style="background: #DBEAFE; color: #1E40AF; font-size: 0.72rem; font-weight: 800; padding: 2px 7px; border-radius: 4px; text-transform: uppercase;">LEVEL 2: 199 RECEIVING LOTS (BOTTOM SECTION)</span>
                <div style="font-size: 0.92rem; font-weight: 700; color: #0F172A; margin-top: 4px;">Physical Heat-Treat Batches & Metallurgical QC</div>
                <div style="font-size: 0.82rem; color: #475569; margin-top: 3px; line-height: 1.5;">
                    Inspects each individual physical shipment delivered across the year to replenish the 5 SKUs. Records Vickers hardness (>216 HV tolerance leakage), tensile strength, scrap count, and vendor chargebacks.
                </div>
            </div>
        </div>
    </div>
    """)

    # 1. WHICH RAW MATERIALS ARE AT RISK (INVENTORY MODEL)
    render_html("""
    <div style="font-size: 1.05rem; font-weight: 800; color: #0F172A; margin-bottom: 8px;">
        1. LEVEL 1  -  Critical Raw Material SKUs at Risk (Stockout & Consumption Velocity Analysis):
    </div>
    """)

    raw_inv = data["raw_materials_inventory"]
    df_raw = pd.DataFrame([
        {
            "SKU": r["sku"],
            "Material Name": r["name"],
            "Supplier": r["supplier"],
            "Current Stock": f"{r['current_stock']:,} {r['unit']}",
            "Daily Demand": f"{r['daily_demand']} {r['unit']}/day",
            "Days Cover": f"{r['days_cover']} days",
            "Lead Time": f"{r['lead_time_days']} days",
            "Risk Tier": r["risk_tier"],
            "Key Vulnerabilities": r["risk_factors"]
        } for r in raw_inv
    ])
    st.dataframe(df_raw, use_container_width=True, height=200)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # 2. HOW MUCH INVENTORY SHOULD WE MAINTAIN (SAFETY STOCK & REORDER MODEL)
    render_html("""
    <div style="font-size: 1.05rem; font-weight: 800; color: #0F172A; margin-bottom: 8px;">
        2. How Much Inventory Should We Maintain (Safety Stock & Reorder Model):
    </div>
    <p style="font-size: 0.88rem; color: #64748B; margin-bottom: 12px;">
        Six Sigma safety stock calculation factoring in daily consumption variance and supplier lead time standard deviation:
    </p>
    """)

    inv_calc_df = pd.DataFrame([
        {
            "SKU": r["sku"],
            "Material Name": r["name"],
            "Current Stock": f"{r['current_stock']:,} {r['unit']}",
            "Recommended Safety Stock": f"{r['safety_stock']:,} {r['unit']}",
            "Reorder Point (ROP)": f"{r['reorder_point']:,} {r['unit']}",
            "Economic Order Qty (EOQ)": f"{r['eoq']:,} {r['unit']}",
            "Status": r["risk_tier"]
        } for r in raw_inv
    ])
    st.dataframe(inv_calc_df, use_container_width=True, height=200)

    # Chart comparing Current Stock vs Safety Stock
    chart_inv = pd.DataFrame([
        {
            "Material": r["sku"] + " (" + r["name"][:18] + "...)",
            "Current Stock": r["current_stock"],
            "Safety Stock": r["safety_stock"],
            "Optimal Reorder Level": r["reorder_point"]
        } for r in raw_inv
    ])
    fig_inv = go.Figure()
    fig_inv.add_trace(go.Bar(
        x=chart_inv["Material"],
        y=chart_inv["Current Stock"],
        name="Current Warehouse Stock",
        marker_color="#2563EB"
    ))
    fig_inv.add_trace(go.Bar(
        x=chart_inv["Material"],
        y=chart_inv["Safety Stock"],
        name="Recommended Safety Stock (Buffer)",
        marker_color="#F59E0B"
    ))
    fig_inv.update_layout(
        barmode='group',
        height=250,
        margin=dict(l=10, r=10, t=25, b=10),
        yaxis=dict(title="Units / kg"),
        xaxis=dict(title=""),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_inv, use_container_width=True)

    # 3. ACTIONS TO AVOID SHORTAGES WITHOUT HOLDING EXCESS STOCK
    render_html("""
    <div style="font-size: 1.05rem; font-weight: 800; color: #0F172A; margin: 14px 0 8px 0;">
        3. Strategic Actions to Avoid Shortages Without Holding Excess Stock:
    </div>
    """)

    a1, a2 = st.columns(2)
    with a1:
        render_html("""
        <div style="background: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 12px; padding: 16px; height: 100%;">
            <div style="font-weight: 800; color: #1E40AF; font-size: 0.95rem; margin-bottom: 6px;">
                [Tier 4] Actions to Prevent Material Shortages:
            </div>
            <ul style="margin: 0; padding-left: 18px; font-size: 0.88rem; color: #1E293B; line-height: 1.6;">
                <li><strong>Dynamic ROP Triggering:</strong> Automate replenishment orders exactly when stock hits the calculated <strong>Reorder Point (ROP)</strong> rather than relying on fixed monthly schedules.</li>
                <li><strong>Dual-Sourcing Critical Steel (AISI 4140):</strong> Split allocations (70% primary, 30% secondary) so that Vendor B's 10-day heat-treatment backlog cannot halt line production.</li>
                <li><strong>Adaptive Recipe Compensation:</strong> Safely process borderline raw stock (216 - 225 HV) by offsetting torque (+0.38 Nm) and angle (+11.4°), preventing scrap and avoiding sudden replenishment panics.</li>
            </ul>
        </div>
        """)

    with a2:
        render_html("""
        <div style="background: #FEF3C7; border: 1px solid #FDE68A; border-radius: 12px; padding: 16px; height: 100%;">
            <div style="font-weight: 800; color: #92400E; font-size: 0.95rem; margin-bottom: 6px;">
                [Preventive Standard] Actions to Avoid Holding Excess Stock:
            </div>
            <ul style="margin: 0; padding-left: 18px; font-size: 0.88rem; color: #1E293B; line-height: 1.6;">
                <li><strong>Strict EOQ Batch Caps:</strong> Restrict purchase order batch sizes to the Economic Order Quantity (e.g. 9,500 kg cap on 4140 steel) to prevent working capital bloat.</li>
                <li><strong>Pause Titanium Replenishment:</strong> Freeze PO-9804 replenishment for 12 days to burn off 2.38x surplus inventory, liberating <strong>$48,000 in tied-up cash</strong>.</li>
                <li><strong>Vendor-Managed Inventory (VMI):</strong> Require Vendor C to maintain a 1,000-unit consignment buffer at local freight hub, billed only when consumed on the line.</li>
            </ul>
        </div>
        """)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # =========================================================================
    # COMPLETE 199 RAW MATERIAL LOTS MASTER & RISK LEVEL ANALYZER
    # =========================================================================
    render_html("""
    <div class="smart-card">
        <div class="card-title">[Summary] LEVEL 2  -  Receiving Inspection Master & Physical Delivery Lots (All 199 Batches)</div>
        <p style="font-size: 0.9rem; color: #64748B; margin-bottom: 12px;">
            Full metallurgical inspection history across all 199 receiving shipments from <code>data/material_batch_master.csv</code> delivered to supply the 5 SKUs above. Evaluates hardness tolerance leakage (&gt;216 HV), scrap induction, and financial liability.
        </p>
        <div style="background: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 10px; padding: 14px 18px; margin-top: 10px; font-size: 0.88rem; color: #1E3A8A; line-height: 1.6;">
            <strong>[AI Recommendations] Why Are There 5 SKUs Above vs. 199 Batches Below?</strong><br>
            * <strong>Level 1 (Top  -  5 Items):</strong> Represents the <strong>5 high-level Warehouse Material Families / SKUs</strong> (AISI 4140 Steel, ST-88 Studs, M10 Fasteners, Ni-Cr Bushings, Titanium Plates) for inventory planning, safety stock, and reorder points.<br>
            * <strong>Level 2 (Bottom  -  199 Items):</strong> Represents the <strong>199 individual physical delivery truckloads & heat-treat lots</strong> received from suppliers across the year to fulfill those 5 materials. For example, <strong>Vendor B delivered 70 individual lots of AISI 4140 Steel</strong>, where 35 lots exceeded 216 HV hardness, causing automated screwdriver stalls and scrap.
        </div>
    </div>
    """)

    all_mat = data["material_enriched"]
    
    # Summary Metrics across all 199 Lots
    tot_lots = len(all_mat)
    crit_lots = len(all_mat[all_mat["Risk_Level"].str.contains("CRITICAL")])
    elev_lots = len(all_mat[all_mat["Risk_Level"].str.contains("ELEVATED")])
    norm_lots = len(all_mat[all_mat["Risk_Level"].str.contains("NORMAL")])
    tot_scrap = all_mat["scrap_count"].sum()
    tot_cb = all_mat["chargeback_amount_usd"].sum()

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        render_html(f'<div class="metric-card-white"><div class="metric-card-label">TOTAL RAW LOTS</div><div class="metric-card-number" style="color: #2563EB;">{tot_lots}</div><div class="metric-card-delta">Physical Shipments</div></div>')
    with m2:
        render_html(f'<div class="metric-card-white"><div class="metric-card-label">CRITICAL RISK LOTS</div><div class="metric-card-number" style="color: #DC2626;">{crit_lots}</div><div class="metric-card-delta" style="color: #DC2626;">{(crit_lots/tot_lots)*100:.1f}% of batches</div></div>')
    with m3:
        render_html(f'<div class="metric-card-white"><div class="metric-card-label">ELEVATED RISK LOTS</div><div class="metric-card-number" style="color: #D97706;">{elev_lots}</div><div class="metric-card-delta" style="color: #D97706;">{(elev_lots/tot_lots)*100:.1f}% of batches</div></div>')
    with m4:
        render_html(f'<div class="metric-card-white"><div class="metric-card-label">NORMAL / SAFE LOTS</div><div class="metric-card-number" style="color: #059669;">{norm_lots}</div><div class="metric-card-delta" style="color: #059669;">{(norm_lots/tot_lots)*100:.1f}% compliant</div></div>')
    with m5:
        render_html(f'<div class="metric-card-white"><div class="metric-card-label">TOTAL SCRAP CAUSED</div><div class="metric-card-number" style="color: #DC2626;">{tot_scrap:,}</div><div class="metric-card-delta">Scrapped Assemblies</div></div>')
    with m6:
        render_html(f'<div class="metric-card-white"><div class="metric-card-label">CHARGEBACK LIABILITY</div><div class="metric-card-number" style="color: #2563EB;">${tot_cb:,.0f}</div><div class="metric-card-delta">Vendor Demand Claim</div></div>')

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # Interactive Filters for All 199 Batches
    f_col1, f_col2, f_col3, f_col4 = st.columns([1.2, 0.9, 1, 1.1])
    with f_col1:
        mat_filter = st.selectbox(
            "Filter by Parent Material SKU:",
            ["All Materials (199 Lots)", "RAW-4140: AISI 4140 Steel (Vendor B)", "RAW-ST88 / M10: Fasteners (Vendor A)", "RAW-NICR / TI01: Bushings & Ti (Vendor C)"],
            key="raw_mat_filter"
        )
    with f_col2:
        supp_filter = st.selectbox(
            "Filter by Supplier:",
            ["All Suppliers", "VENDOR_A", "VENDOR_B", "VENDOR_C"],
            key="raw_supp_filter"
        )
    with f_col3:
        risk_filter = st.selectbox(
            "Filter by Risk Level:",
            ["All Risk Tiers", "[Critical] CRITICAL RISK", "[Elevated] ELEVATED RISK", "[Optimal] NORMAL / SAFE"],
            key="raw_risk_filter"
        )
    with f_col4:
        search_lot = st.text_input("Search Batch ID:", "", placeholder="e.g. LOT-2026-RAW-004", key="raw_search")

    # Apply filters
    filtered_mat = all_mat.copy()
    if mat_filter != "All Materials (199 Lots)":
        if "RAW-4140" in mat_filter:
            filtered_mat = filtered_mat[filtered_mat["supplier_id"] == "VENDOR_B"]
        elif "RAW-ST88" in mat_filter:
            filtered_mat = filtered_mat[filtered_mat["supplier_id"] == "VENDOR_A"]
        elif "RAW-NICR" in mat_filter:
            filtered_mat = filtered_mat[filtered_mat["supplier_id"] == "VENDOR_C"]
    if supp_filter != "All Suppliers":
        filtered_mat = filtered_mat[filtered_mat["supplier_id"] == supp_filter]
    if risk_filter != "All Risk Tiers":
        filtered_mat = filtered_mat[filtered_mat["Risk_Level"] == risk_filter]
    if search_lot:
        filtered_mat = filtered_mat[filtered_mat["material_batch_id"].str.contains(search_lot, case=False)]

    # Display All 199 Batches in Formatted Table with Parent SKU
    display_df = filtered_mat[[
        "material_batch_id", "parent_sku", "supplier_id", "material_hardness_hv",
        "tensile_strength_mpa", "Safe_Margin_HV", "Risk_Level",
        "material_risk_status", "scrap_count", "chargeback_amount_usd"
    ]].rename(columns={
        "material_batch_id": "Batch ID",
        "parent_sku": "Parent Material SKU",
        "supplier_id": "Supplier",
        "material_hardness_hv": "Hardness (HV)",
        "tensile_strength_mpa": "Tensile (MPa)",
        "Safe_Margin_HV": "Margin to Safe HV",
        "Risk_Level": "Risk Level",
        "material_risk_status": "QC Status",
        "scrap_count": "Downstream Scrap",
        "chargeback_amount_usd": "Chargeback ($)"
    })

    st.markdown(f"**Showing {len(display_df)} of {tot_lots} Raw Material Lots:**")
    st.dataframe(display_df, use_container_width=True, height=280)

    # GENERAL INFORMATION & METALLURGICAL RISK BASELINE GUIDE
    render_html("""
    <div class="smart-card" style="border-left: 6px solid #2563EB; margin-top: 14px;">
        <div class="card-title"> General Information & Metallurgical Risk Baseline Guide</div>
        <div style="font-size: 0.90rem; color: #334155; line-height: 1.7;">
            <p><strong>* The 216.0 HV Tolerance Leakage Phenomenon:</strong><br>
            While nominal engineering blueprints define incoming material acceptance between <strong>180.0 HV and 230.0 HV</strong>, machine screwdrivers and assembly bits are calibrated for nominal 200 HV. Raw metal batches measuring between <strong>216.0 HV and 230.0 HV</strong> nominally pass receiving inspection, but cause automated torque drivers to spike prematurely, stalling before completing the 360° seating turn.
            </p>
            <p><strong>* Risk Classification Criteria:</strong><br>
            - <span class="status-badge-red">[Critical] CRITICAL RISK</span>: Hardness &ge; 216.0 HV, marked <code>MARGINAL_HIGH</code>, or generated &gt; 200 scrap parts. <em>Quarantine immediately or apply adaptive torque offset (+0.38 Nm, +11.4° shutoff angle).</em><br>
            - <span class="status-badge-orange">[Elevated] ELEVATED RISK</span>: Hardness 210.0 to 215.9 HV or generated 100 - 200 scrap units. <em>Continuous in-process torque monitoring required.</em><br>
            - <span class="status-badge-green">[Optimal] NORMAL / SAFE</span>: Hardness &lt; 210.0 HV and minimal scrap. <em>Run standard production recipe without modification.</em>
            </p>
            <p><strong>* Financial Protection Protocol:</strong><br>
            All scrap generated from critical lots is automatically aggregated into the quarterly <strong>8D Supplier Corrective Action Request (SCAR)</strong>, recovering <strong>$55 per scrapped unit</strong> plus direct warranty chargebacks.
            </p>
        </div>
    </div>
    """)


# =============================================================================
# TAB 3: MANUFACTURING LEVEL (FACTORIES & UNITS WITH MEANINGFUL TARGETS)
# =============================================================================
with tab_mfg_unit:
    render_html("""
    <div class="smart-card">
        <div class="card-title">[Tier 3] Manufacturing Level  -  Factory Units & Facility Operations</div>
        <p style="font-size: 0.92rem; color: #64748B; margin: 0 0 16px 0;">
            Select any manufacturing unit or factory facility to inspect operational performance measured against Six Sigma benchmark targets, downtime variances, material shortages, defect rates, and diagnostic root causes with resolution actions.
        </p>
    </div>
    """)

    col_unit_sel, col_unit_badge = st.columns([1.5, 1])
    with col_unit_sel:
        unit_keys = list(data["mfg_units"].keys())
        unit_names = [data["mfg_units"][k]["name"] for k in unit_keys]
        selected_unit_name = st.selectbox(
            "Select Manufacturing Unit or Factory:",
            unit_names,
            key="mfg_unit_select"
        )
        selected_unit_key = unit_keys[unit_names.index(selected_unit_name)]
        unit_data = data["mfg_units"][selected_unit_key]

    with col_unit_badge:
        render_html(f"""
        <div style="display: flex; justify-content: flex-end; align-items: center; height: 100%;">
            <span class="{unit_data['status_badge']}" style="font-size: 0.9rem; padding: 6px 14px;">
                {unit_data['overall_status']}
            </span>
        </div>
        """)

    render_html(f"""
    <div style="font-size: 0.88rem; color: #64748B; margin-top: 4px;">
         <strong>Location:</strong> {unit_data['location']} | <strong>Specialization:</strong> {unit_data['focus']}
    </div>
    """)

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # 6 OVERALL PERFORMANCE KPI CARDS WITH MEANINGFUL TARGETS
    u1, u2, u3, u4, u5, u6 = st.columns(6)
    with u1:
        render_html(f"""
        <div class="metric-card-white">
            <div class="metric-card-label">TOTAL MACHINES</div>
            <div class="metric-card-number" style="color: #2563EB;">{unit_data["active_machines"]}/{unit_data["total_machines"]}</div>
            <div class="metric-card-target">{unit_data["avail_target"]}</div>
            <div class="metric-card-delta" style="color: {unit_data['avail_color']};">{unit_data["avail_status"]}</div>
        </div>
        """)
    with u2:
        render_html(f"""
        <div class="metric-card-white">
            <div class="metric-card-label">MACHINE DOWNTIME</div>
            <div class="metric-card-number" style="color: {unit_data['downtime_color']};">{unit_data["downtime_hours"]}h</div>
            <div class="metric-card-target">{unit_data["downtime_target"]}</div>
            <div class="metric-card-delta" style="color: {unit_data['downtime_color']};">{unit_data["downtime_status"]}</div>
        </div>
        """)
    with u3:
        render_html(f"""
        <div class="metric-card-white">
            <div class="metric-card-label">MFG DELAYS</div>
            <div class="metric-card-number" style="color: {unit_data['delays_color']};">{unit_data["mfg_delays_hrs"]}h</div>
            <div class="metric-card-target">{unit_data["delays_target"]}</div>
            <div class="metric-card-delta" style="color: {unit_data['delays_color']};">{unit_data["delays_status"]}</div>
        </div>
        """)
    with u4:
        render_html(f"""
        <div class="metric-card-white">
            <div class="metric-card-label">MATERIAL SHORTAGES</div>
            <div class="metric-card-number" style="color: {unit_data['shortages_color']};">{unit_data["material_shortages_count"]}</div>
            <div class="metric-card-target">{unit_data["shortages_target"]}</div>
            <div class="metric-card-delta" style="color: {unit_data['shortages_color']};">{unit_data["shortages_status"]}</div>
        </div>
        """)
    with u5:
        render_html(f"""
        <div class="metric-card-white">
            <div class="metric-card-label">DEFECT RATE</div>
            <div class="metric-card-number" style="color: {unit_data['defect_color']};">{unit_data["defect_rate"]}</div>
            <div class="metric-card-target">{unit_data["defect_target"]}</div>
            <div class="metric-card-delta" style="color: {unit_data['defect_color']};">{unit_data["defect_status"]}</div>
        </div>
        """)
    with u6:
        render_html(f"""
        <div class="metric-card-white">
            <div class="metric-card-label">OEE EFFICIENCY</div>
            <div class="metric-card-number" style="color: {unit_data['oee_color']};">{unit_data["oee_efficiency"]}</div>
            <div class="metric-card-target">{unit_data["oee_target"]}</div>
            <div class="metric-card-delta" style="color: {unit_data['oee_color']};">{unit_data["oee_status"]}</div>
        </div>
        """)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # IDENTIFIED MANUFACTURING ISSUES
    render_html(f"""
    <div class="smart-card">
        <div class="card-title">[Warning] Identified Manufacturing Issues for {unit_data['name'].split(' - ')[1].strip()}</div>
        <p style="font-size: 0.9rem; color: #64748B; margin-bottom: 16px;">
            Detailed root cause failure analysis, severity rankings, and prescribed resolution & preventive actions:
        </p>
    </div>
    """)

    for issue in unit_data["issues"]:
        render_html(f"""
        <div class="issue-card" style="border-left: 5px solid #2563EB;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <div style="font-weight: 800; font-size: 1rem; color: #0F172A;">
                    {issue['title']}
                </div>
                <div style="display: flex; gap: 8px; align-items: center;">
                    <span style="font-size: 0.8rem; font-weight: 700; color: #64748B;">{issue['type']}</span>
                    <span>{issue['severity']}</span>
                </div>
            </div>
            
            <div style="margin: 8px 0; font-size: 0.9rem; color: #334155;">
                <strong style="color: #64748B; text-transform: uppercase; font-size: 0.78rem;">Possible Root Cause:</strong><br>
                {issue['root_cause']}
            </div>

            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 10px;">
                <div style="background: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 8px; padding: 10px 14px; font-size: 0.86rem; color: #1E40AF;">
                    <strong>[Immediate Action] Recommended Action to Resolve:</strong><br>
                    {issue['resolve_action']}
                </div>
                <div style="background: #ECFDF5; border: 1px solid #A7F3D0; border-radius: 8px; padding: 10px 14px; font-size: 0.86rem; color: #065F46;">
                    <strong>[Preventive Standard] Recommended Preventive Action:</strong><br>
                    {issue['prevent_action']}
                </div>
            </div>
        </div>
        """)


# =============================================================================
# TAB 4: PRODUCTION LEVEL (LINES & ORDERS)
# =============================================================================
with tab_prod_line:
    render_html("""
    <div class="smart-card">
        <div class="card-title">[Tier 4] Production Level  -  Lines, Work Orders & Throughput Intelligence</div>
        <p style="font-size: 0.92rem; color: #64748B; margin: 0 0 16px 0;">
            Select any production line or production work order to inspect planned vs actual output, completion schedules, line bottlenecks, rejection rates, and order-specific recommendations.
        </p>
    </div>
    """)

    prod_orders_list = data["prod_orders"]
    order_labels = [f"[Catalog] {o['order_id']} | {o['line_id']} ({o['product_name'].split('(')[0].strip()})" for o in prod_orders_list]
    
    col_ord_sel, col_ord_stat = st.columns([1.5, 1])
    with col_ord_sel:
        selected_order_label = st.selectbox(
            "Select Production Line or Production Order:",
            order_labels,
            key="prod_order_select"
        )
        selected_order_idx = order_labels.index(selected_order_label)
        ord_data = prod_orders_list[selected_order_idx]

    with col_ord_stat:
        render_html(f"""
        <div style="display: flex; justify-content: flex-end; align-items: center; height: 100%;">
            <span class="{ord_data['status_badge']}" style="font-size: 0.9rem; padding: 6px 14px;">
                {ord_data['status']}
            </span>
        </div>
        """)

    render_html(f"""
    <div style="font-size: 0.88rem; color: #64748B; margin-top: 4px;">
        [Tier 3] <strong>Facility:</strong> {ord_data['mfg_unit']} | <strong>Line:</strong> {ord_data['line_id']} | <strong>Product:</strong> {ord_data['product_name']}
    </div>
    """)

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # 5 PRODUCTION ORDER KPI CARDS
    attainment_pct = (ord_data['actual_qty'] / ord_data['planned_qty']) * 100
    p1, p2, p3, p4, p5 = st.columns(5)
    with p1:
        render_html(f'<div class="metric-card-white"><div class="metric-card-label">PLANNED QUANTITY</div><div class="metric-card-number" style="color: #64748B;">{ord_data["planned_qty"]:,}</div><div class="metric-card-target">Target Run</div><div class="metric-card-delta">Scheduled Batch Size</div></div>')
    with p2:
        render_html(f'<div class="metric-card-white"><div class="metric-card-label">ACTUAL COMPLETED</div><div class="metric-card-number" style="color: #2563EB;">{ord_data["actual_qty"]:,}</div><div class="metric-card-target">{attainment_pct:.1f}% Attainment</div><div class="metric-card-delta">Net Good Units Produced</div></div>')
    with p3:
        render_html(f'<div class="metric-card-white"><div class="metric-card-label">COMPLETION SCHEDULE</div><div class="metric-card-number" style="color: #DC2626; font-size: 1.5rem;">+{ord_data["delay_hrs"]} hrs</div><div class="metric-card-target">Planned: {ord_data["planned_completion"]}</div><div class="metric-card-delta" style="color: #DC2626;">Projected: {ord_data["actual_completion"]}</div></div>')
    with p4:
        render_html(f'<div class="metric-card-white"><div class="metric-card-label">REJECTION RATE</div><div class="metric-card-number" style="color: #DC2626; font-size: 1.5rem;">{ord_data["rejection_rate"].split("(")[0].strip()}</div><div class="metric-card-target">Tolerance: < 1.5%</div><div class="metric-card-delta" style="color: #DC2626;">{ord_data["rejection_rate"].split("(")[1].replace(")", "")}</div></div>')
    with p5:
        render_html(f'<div class="metric-card-white"><div class="metric-card-label">RAW MATERIAL FEED</div><div class="metric-card-number" style="color: #D97706; font-size: 1.25rem;">ALERT</div><div class="metric-card-target">Stock Health</div><div class="metric-card-delta" style="color: #D97706;">{ord_data["raw_mat_status"][:25]}...</div></div>')

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # VISUAL PRODUCTION ATTAINMENT PROGRESS
    col_prog, col_bottle = st.columns([1.2, 1])
    with col_prog:
        render_html("""
        <div class="smart-card">
            <div class="card-title"> Production Output Attainment</div>
        </div>
        """)
        
        scrap_units = int(ord_data["planned_qty"] * float(ord_data["rejection_rate"].split("%")[0]) / 100)
        remaining_units = max(0, ord_data["planned_qty"] - ord_data["actual_qty"])

        fig_prog = go.Figure()
        fig_prog.add_trace(go.Bar(
            name='Good Completed',
            y=['Units'],
            x=[ord_data['actual_qty'] - scrap_units],
            orientation='h',
            marker=dict(color='#059669')
        ))
        fig_prog.add_trace(go.Bar(
            name='Scrap / Rejected',
            y=['Units'],
            x=[scrap_units],
            orientation='h',
            marker=dict(color='#DC2626')
        ))
        fig_prog.add_trace(go.Bar(
            name='Deficit Remaining',
            y=['Units'],
            x=[remaining_units],
            orientation='h',
            marker=dict(color='#E2E8F0')
        ))
        fig_prog.update_layout(
            barmode='stack',
            height=140,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(title="Quantity (Parts)", range=[0, ord_data['planned_qty'] * 1.05]),
            yaxis=dict(showticklabels=False),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_prog, use_container_width=True)
        render_html(f"""
            <div style="font-size: 0.85rem; color: #475569;">
                Planned: <strong>{ord_data['planned_qty']:,}</strong> | Actual Good: <strong>{ord_data['actual_qty'] - scrap_units:,}</strong> | Scrapped: <strong>{scrap_units}</strong> | Behind Target: <strong>{remaining_units}</strong>
            </div>
        """)

    with col_bottle:
        render_html(f"""
        <div class="smart-card" style="border-left: 4px solid #DC2626;">
            <div class="card-title">[Warning] Identified Production Issues & Bottlenecks</div>
            <div style="font-size: 0.88rem; color: #1E293B; line-height: 1.6;">
                * <strong>Line Choke Point:</strong> {ord_data['bottlenecks']}<br>
                * <strong>Raw Material Status:</strong> {ord_data['raw_mat_status']}<br>
                * <strong>Order Schedule Drift:</strong> Running <strong>{ord_data['delay_hrs']} hours behind</strong> shift target schedule.
            </div>
        </div>
        """)

    # INDIVIDUAL PRODUCT / PRODUCTION ORDER RECOMMENDATIONS
    render_html(f"""
    <div class="smart-card">
        <div class="card-title">[Summary] AI-Powered Recommendations for {ord_data['order_id']} ({ord_data['product_name']})</div>
        <p style="font-size: 0.9rem; color: #64748B; margin-bottom: 14px;">
            Targeted operational directives across 4 core manufacturing levers:
        </p>
    </div>
    """)

    r1, r2 = st.columns(2)
    with r1:
        render_html(f"""
        <div class="issue-card" style="border-left: 4px solid #DC2626;">
            <div style="font-weight: 800; font-size: 0.92rem; color: #991B1B; margin-bottom: 4px;">
                1. Prioritizing Urgent Orders:
            </div>
            <div style="font-size: 0.88rem; color: #1E293B; line-height: 1.6;">
                {ord_data['recommendations']['prioritizing_urgent_orders']}
            </div>
        </div>

        <div class="issue-card" style="border-left: 4px solid #2563EB;">
            <div style="font-weight: 800; font-size: 0.92rem; color: #1E40AF; margin-bottom: 4px;">
                2. Reallocating Production Capacity:
            </div>
            <div style="font-size: 0.88rem; color: #1E293B; line-height: 1.6;">
                {ord_data['recommendations']['reallocating_capacity']}
            </div>
        </div>
        """)

    with r2:
        render_html(f"""
        <div class="issue-card" style="border-left: 4px solid #D97706;">
            <div style="font-weight: 800; font-size: 0.92rem; color: #B45309; margin-bottom: 4px;">
                3. Resolving Line Bottlenecks:
            </div>
            <div style="font-size: 0.88rem; color: #1E293B; line-height: 1.6;">
                {ord_data['recommendations']['resolving_bottlenecks']}
            </div>
        </div>

        <div class="issue-card" style="border-left: 4px solid #059669;">
            <div style="font-weight: 800; font-size: 0.92rem; color: #065F46; margin-bottom: 4px;">
                4. Replenishing Materials:
            </div>
            <div style="font-size: 0.88rem; color: #1E293B; line-height: 1.6;">
                {ord_data['recommendations']['replenishing_materials']}
            </div>
        </div>
        """)

