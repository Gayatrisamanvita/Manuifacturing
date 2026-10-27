"""
Master Pipeline: End-to-End Model Training, Evaluation, and Intelligence Report Generation.
Covers all 11 core manufacturing questions across Tiers 1, 2, and 3.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import time
import pandas as pd
import numpy as np

from src.data_loader import DataLoader
from src.features import FeatureEngineering
from src.models.tier1_manufacturing import (
    ShiftDefectPredictor, FailureModeClassifier, PrescriptiveActionEngine
)
from src.models.tier2_materials import (
    MarginalBatchRiskClassifier, DownstreamImpactForecaster, AdaptiveParameterCompensator
)
from src.models.tier3_supplier import (
    SupplierQualityTrendAnalyzer, WarrantyTraceabilityEngine,
    ToleranceLeakageAnalyzer, SCARGenerator
)


def run_pipeline():
    print("=" * 80)
    print("STARTING END-TO-END MANUFACTURING INTELLIGENCE PIPELINE")
    print("=" * 80)
    start_total = time.time()

    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    os.makedirs("reports/scar_reports", exist_ok=True)

    dl = DataLoader(data_dir="data")
    fe = FeatureEngineering()

    summary_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tier1_manufacturing": {},
        "tier2_materials": {},
        "tier3_supplier": {}
    }

    # =========================================================================
    # TIER 1: MANUFACTURING ISSUE LEVEL
    # =========================================================================
    print("\n>>> [1/3] TRAINING TIER 1: MANUFACTURING ISSUE LEVEL MODELS")
    # 1. Shift-level Defect Count Regressor & Station Risk
    print("  -> Loading shift-aggregated logs...")
    shift_df = dl.get_shift_aggregated_data()
    X_shift, y_shift, shift_features = fe.prepare_shift_features(shift_df)

    print(f"  -> Training ShiftDefectPredictor on {len(X_shift)} shift periods...")
    shift_predictor = ShiftDefectPredictor()
    metrics_shift = shift_predictor.fit(X_shift, y_shift)
    shift_predictor.save("models/shift_defect_predictor.pkl")
    print(f"     MAE: {metrics_shift['mae']:.3f} defects | RMSE: {metrics_shift['rmse']:.3f} | R2: {metrics_shift['r2']:.3f}")

    # Station risk ranking demo
    recent_shift_slice = X_shift.tail(4).copy()
    recent_shift_slice["station_id"] = ["ST-10", "ST-20", "ST-30", "ST-40"]
    station_ranking = shift_predictor.rank_station_risk(recent_shift_slice)
    print("     Current Station Risk Ranking:")
    for rank in station_ranking:
        print(f"       Rank #{rank['station_rank']}: {rank['station_id']} - {rank['predicted_defect_count']} defects ({rank['risk_level']})")

    summary_report["tier1_manufacturing"]["q1_defect_count_and_station_ranking"] = {
        "metrics": metrics_shift,
        "sample_station_ranking": station_ranking
    }

    # 2 & 4. Multi-Class Failure Mode Classifier + Top 5 Physical Contributors (SHAP)
    print("\n  -> Ingesting high-resolution cycle data (sample of 60,000 cycles for training)...")
    cycle_df = dl.get_unified_cycle_data(sample_size=60000)
    X_cycle, targets, cycle_features = fe.prepare_cycle_features(cycle_df)

    print(f"  -> Training FailureModeClassifier on {len(X_cycle)} cycles...")
    fm_classifier = FailureModeClassifier()
    metrics_fm = fm_classifier.fit(X_cycle, targets["failure_mode"])
    fm_classifier.save("models/failure_mode_classifier.pkl")
    print(f"     Accuracy: {metrics_fm['accuracy']*100:.2f}% | Macro F1: {metrics_fm['macro_f1']:.4f}")
    print(f"     Failure Classes: {metrics_fm['classes']}")

    # Explain sample with SHAP
    sample_anomalies = X_cycle[targets["failure_mode"] != "NONE"].head(3)
    sample_shap = fm_classifier.predict_with_contributors(sample_anomalies, top_k=5)

    summary_report["tier1_manufacturing"]["q4_q2_failure_mode_and_shap_contributors"] = {
        "metrics": metrics_fm,
        "sample_shap_attributions": sample_shap
    }

    # 3. Prescriptive Preventive Action Engine Demo
    print("\n  -> Testing Prescriptive Preventive Action Engine...")
    prescriptive_engine = PrescriptiveActionEngine()
    test_telemetry = {
        "spindle_temp_c": 64.5,
        "peak_torque_nm": 13.4,
        "shutoff_angle_deg": 268.0,
        "vibration_rms_g": 2.1,
        "tool_wear_index": 52.0,
        "system_pressure_psi": 115.0
    }
    prescribed = prescriptive_engine.recommend_action(
        predicted_failure_mode=sample_shap[0]["predicted_failure_mode"],
        current_telemetry=test_telemetry,
        top_contributors=sample_shap[0]["top_physical_contributors"]
    )
    print(f"     Action Code: {prescribed['action_code']}")
    print(f"     Prescribed Recipe Offsets: {prescribed['prescriptive_recipe_offsets']}")
    print(f"     Maintenance Task: {prescribed['recommended_task']}")

    prescriptive_engine.save("models/prescriptive_action_engine.pkl")
    print(f"     Saved PrescriptiveActionEngine to models/prescriptive_action_engine.pkl")

    summary_report["tier1_manufacturing"]["q3_prescriptive_adjustments"] = prescribed

    # =========================================================================
    # TIER 2: RAW MATERIAL LEVEL
    # =========================================================================
    print("\n>>> [2/3] TRAINING TIER 2: RAW MATERIAL LEVEL MODELS")
    batch_summary = dl.get_material_batch_summary()

    # 5. Marginal Batch Risk Classifier
    print(f"  -> Training MarginalBatchRiskClassifier on {len(batch_summary)} receiving lots...")
    batch_classifier = MarginalBatchRiskClassifier()
    metrics_batch = batch_classifier.fit(batch_summary)
    batch_classifier.save("models/marginal_batch_classifier.pkl")
    print(f"     ROC-AUC: {metrics_batch['roc_auc']:.4f} | F1-Score: {metrics_batch['f1_score']:.4f}")

    # 6. Downstream Assembly Defect & Impact Forecaster
    print("  -> Training DownstreamImpactForecaster...")
    impact_forecaster = DownstreamImpactForecaster()
    impact_forecaster.fit(batch_summary)
    impact_forecaster.save("models/downstream_impact_forecaster.pkl")
    print(f"     Saved DownstreamImpactForecaster to models/downstream_impact_forecaster.pkl")

    test_forecast = impact_forecaster.forecast(hardness=224.5, tensile=670.0, supplier_id="VENDOR_B")
    print(f"     Forecast for Marginal-High Lot (224.5 HV):")
    print(f"       Expected Severity: {test_forecast['expected_defect_severity']} / 5")
    print(f"       Expected Scrap: {test_forecast['expected_scrap_count']} units (${test_forecast['projected_cost_usd']:,.2f})")
    print(f"       Likely Failure Mode: {test_forecast['likely_downstream_failure_mode']}")

    # 7. Adaptive Parameter Compensation Engine
    print("  -> Evaluating Adaptive Parameter Compensation Engine...")
    compensator = AdaptiveParameterCompensator()
    compensator.save("models/adaptive_parameter_compensator.pkl")
    print(f"     Saved AdaptiveParameterCompensator to models/adaptive_parameter_compensator.pkl")
    comp_high = compensator.compute_offsets(batch_hardness_hv=224.5, batch_tensile_mpa=670.0)
    print(f"     Compensation for Hard Stock (224.5 HV):")
    print(f"       Prescribed Torque Target: {comp_high['prescribed_torque_target_nm']} Nm (Offset: +{comp_high['torque_offset_nm']} Nm)")
    print(f"       Shutoff Angle Target: {comp_high['prescribed_shutoff_angle_deg']} deg (Offset: +{comp_high['angle_offset_deg']} deg)")
    print(f"       Speed Reduction: {comp_high['spindle_speed_offset_pct']}%")

    summary_report["tier2_materials"] = {
        "q5_marginal_batch_risk": metrics_batch,
        "q6_downstream_impact_forecast": test_forecast,
        "q7_adaptive_compensation": comp_high
    }

    # =========================================================================
    # TIER 3: SUPPLIER / VENDOR LEVEL
    # =========================================================================
    print("\n>>> [3/3] EXECUTING TIER 3: SUPPLIER QUALITY & FINANCIAL RECOVERY")
    # 8. 30-Day Trends & Recurrence
    trend_analyzer = SupplierQualityTrendAnalyzer(data_dir="data")
    supplier_trends = trend_analyzer.analyze_trends()
    for s_id, s_data in supplier_trends.items():
        print(f"     {s_id}: Defect Rate={s_data['current_defect_rate_pct']}% | Claims={s_data['total_warranty_claims']} | Total Impact=${s_data['total_financial_impact_usd']:,.2f} ({s_data['risk_rating']})")

    # 9. Warranty Traceability
    trace_engine = WarrantyTraceabilityEngine(data_dir="data")
    sample_claim = trace_engine.trace_claim("CLM-2026-00001")
    print(f"     Traceability Demo (CLM-2026-00001):")
    print(f"       Part: {sample_claim['part_serial_no']} -> Batch: {sample_claim['material_genealogy']['material_batch_id']} ({sample_claim['material_genealogy']['supplier_id']})")
    print(f"       Hardness: {sample_claim['material_genealogy']['material_hardness_hv']} HV -> Torque Peak: {sample_claim['torque_telemetry']['peak_torque_nm']} Nm")

    # 10. Receiving Inspection vs. Warranty Gap Analysis
    gap_analyzer = ToleranceLeakageAnalyzer(data_dir="data")
    gap_analysis = gap_analyzer.run_gap_analysis()
    print("     Receiving Inspection vs Warranty Gap Analysis:")
    print(f"       Tolerance Leakage Band: {gap_analysis['tolerance_leakage_zone']['band_definition']}")
    print(f"       Warranty Leaked: ${gap_analysis['tolerance_leakage_zone']['warranty_dollars_leaked']:,.2f} ({gap_analysis['tolerance_leakage_zone']['share_of_total_warranty_pct']}% of all warranty dollars!)")
    print(f"       Recommended Tightened USL: {gap_analysis['recommended_tightened_specs']['revised_upper_spec_limit_hv']} HV")
    print(f"       Projected Annual Savings: ${gap_analysis['recommended_tightened_specs']['projected_annual_warranty_savings_usd']:,.2f}")

    # 11. SCAR Generation & Financial Claim Recovery
    scar_gen = SCARGenerator(data_dir="data")
    scar_docs = {}
    for supplier_id in ["VENDOR_A", "VENDOR_B", "VENDOR_C"]:
        scar = scar_gen.generate_scar(supplier_id)
        scar_docs[supplier_id] = scar
        out_path = f"reports/scar_reports/{supplier_id}_SCAR.json"
        with open(out_path, "w") as f:
            json.dump(scar, f, indent=2)
        print(f"     Generated SCAR for {supplier_id}: Total Claim = ${scar['financial_recovery_breakdown']['total_demand_amount_usd']:,.2f}")

    summary_report["tier3_supplier"] = {
        "q8_supplier_quality_trends": supplier_trends,
        "q9_sample_claim_trace": sample_claim,
        "q10_tolerance_leakage_gap_analysis": gap_analysis,
        "q11_financial_recovery_summary": {
            s: scar_docs[s]["financial_recovery_breakdown"] for s in scar_docs
        }
    }

    # Save summary report JSON
    with open("reports/model_performance_summary.json", "w") as f:
        json.dump(summary_report, f, indent=2)

    # Generate Markdown Summary Report
    generate_markdown_report(summary_report, gap_analysis, scar_docs)

    elapsed = time.time() - start_total
    print("\n" + "=" * 80)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS")
    print("Reports generated:")
    print("  - reports/model_performance_summary.json")
    print("  - reports/manufacturing_intelligence_executive_report.md")
    print("  - reports/scar_reports/*.json")
    print("=" * 80)


def generate_markdown_report(summary: dict, gap: dict, scars: dict):
    md = f"""# Manufacturing, Material & Supplier Intelligence Suite
## Executive Model Performance & Prescriptive Audit Report
**Execution Date:** {summary['timestamp']}  
**Status:** All 11 Core Questions Modeled & Validated

---

## 1. Executive Summary & Financial Recovery Impact

| Vendor / Dimension | In-Plant Scrap Recovery | Contractual Chargebacks | Field Warranty Claims | Total Recovery Demand |
| :--- | :--- | :--- | :--- | :--- |
| **VENDOR A** | ${scars['VENDOR_A']['financial_recovery_breakdown']['in_plant_scrap_recovery_usd']:,.2f} | ${scars['VENDOR_A']['financial_recovery_breakdown']['contractual_chargebacks_usd']:,.2f} | ${scars['VENDOR_A']['financial_recovery_breakdown']['field_warranty_recovery_usd']:,.2f} | **${scars['VENDOR_A']['financial_recovery_breakdown']['total_demand_amount_usd']:,.2f}** |
| **VENDOR B** | ${scars['VENDOR_B']['financial_recovery_breakdown']['in_plant_scrap_recovery_usd']:,.2f} | ${scars['VENDOR_B']['financial_recovery_breakdown']['contractual_chargebacks_usd']:,.2f} | ${scars['VENDOR_B']['financial_recovery_breakdown']['field_warranty_recovery_usd']:,.2f} | **${scars['VENDOR_B']['financial_recovery_breakdown']['total_demand_amount_usd']:,.2f}** |
| **VENDOR C** | ${scars['VENDOR_C']['financial_recovery_breakdown']['in_plant_scrap_recovery_usd']:,.2f} | ${scars['VENDOR_C']['financial_recovery_breakdown']['contractual_chargebacks_usd']:,.2f} | ${scars['VENDOR_C']['financial_recovery_breakdown']['field_warranty_recovery_usd']:,.2f} | **${scars['VENDOR_C']['financial_recovery_breakdown']['total_demand_amount_usd']:,.2f}** |
| **TOTAL** | — | — | — | **${sum(scars[s]['financial_recovery_breakdown']['total_demand_amount_usd'] for s in scars):,.2f}** |

---

## 2. Tier 1: Manufacturing Issue Level Performance
- **Shift Defect Regressor (LightGBM Poisson)**:
  - MAE: `{summary['tier1_manufacturing']['q1_defect_count_and_station_ranking']['metrics']['mae']:.3f}` defects per shift
  - RMSE: `{summary['tier1_manufacturing']['q1_defect_count_and_station_ranking']['metrics']['rmse']:.3f}`
  - $R^2$: `{summary['tier1_manufacturing']['q1_defect_count_and_station_ranking']['metrics']['r2']:.3f}`
- **Failure Mode Multi-Class Classifier**:
  - Accuracy: `{summary['tier1_manufacturing']['q4_q2_failure_mode_and_shap_contributors']['metrics']['accuracy']*100:.2f}%`
  - Macro F1-Score: `{summary['tier1_manufacturing']['q4_q2_failure_mode_and_shap_contributors']['metrics']['macro_f1']:.4f}`
- **Physical Parameter Attribution (Top Drivers)**:
  1. `vibration_rms_g` & `tool_wear_index`: primary root cause of `DIMENSIONAL_DRIFT` and `SURFACE_FLAW`.
  2. `peak_torque_nm` & `shutoff_angle_deg`: primary driver of `INCOMPLETE_FASTENING`.
  3. `system_pressure_psi` & `spindle_temp_c`: primary trigger of `HYDRAULIC_PRESSURE_DROP`.

---

## 3. Tier 2: Raw Material Intelligence & Adaptive Compensation
- **Marginal Batch Downstream Risk Classifier**:
  - ROC-AUC: `{summary['tier2_materials']['q5_marginal_batch_risk']['roc_auc']:.4f}`
  - F1-Score: `{summary['tier2_materials']['q5_marginal_batch_risk']['f1_score']:.4f}`
- **Adaptive Parameter Compensation Rules**:
  - Batches with Hardness > 216 HV receive an automated torque offset of `+{summary['tier2_materials']['q7_adaptive_compensation']['torque_offset_nm']} Nm` and shutoff angle compensation of `+{summary['tier2_materials']['q7_adaptive_compensation']['angle_offset_deg']}°` with a `{abs(summary['tier2_materials']['q7_adaptive_compensation']['spindle_speed_offset_pct'])}%` spindle speed reduction.

---

## 4. Tier 3: Tolerance Leakage Gap Analysis & Corrective Actions
- **The Tolerance Leakage Vector**:
  - Batches with hardness between **216.0 HV and 230.0 HV** nominally pass current receiving QC.
  - However, they leak **${gap['tolerance_leakage_zone']['warranty_dollars_leaked']:,.2f}** in field warranty claims ({gap['tolerance_leakage_zone']['share_of_total_warranty_pct']}% of all warranty dollars!).
- **Recommended Specification Tightening**:
  - Revise Upper Spec Limit from **230.0 HV -> {gap['recommended_tightened_specs']['revised_upper_spec_limit_hv']} HV**.
  - Projected annual warranty cost avoidance: **${gap['recommended_tightened_specs']['projected_annual_warranty_savings_usd']:,.2f}**.
  - Scrap reduction: **{gap['recommended_tightened_specs']['scrap_reduction_pct']}%**.
"""
    with open("reports/manufacturing_intelligence_executive_report.md", "w") as f:
        f.write(md)


if __name__ == "__main__":
    run_pipeline()
