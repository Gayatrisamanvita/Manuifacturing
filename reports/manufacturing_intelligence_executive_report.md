# Manufacturing, Material & Supplier Intelligence Suite
## Executive Model Performance & Prescriptive Audit Report
**Execution Date:** 2026-10-14 05:25:44  
**Status:** All 11 Core Questions Modeled & Validated

---

## 1. Executive Summary & Financial Recovery Impact

| Vendor / Dimension | In-Plant Scrap Recovery | Contractual Chargebacks | Field Warranty Claims | Total Recovery Demand |
| :--- | :--- | :--- | :--- | :--- |
| **VENDOR A** | $666,710.00 | $559,505.03 | $559,505.03 | **$1,785,720.06** |
| **VENDOR B** | $832,095.00 | $750,852.70 | $750,852.70 | **$2,333,800.40** |
| **VENDOR C** | $503,635.00 | $411,485.34 | $411,485.34 | **$1,326,605.68** |
| **TOTAL** | — | — | — | **$5,446,126.14** |

---

## 2. Tier 1: Manufacturing Issue Level Performance
- **Shift Defect Regressor (LightGBM Poisson)**:
  - MAE: `8.820` defects per shift
  - RMSE: `12.805`
  - $R^2$: `0.203`
- **Failure Mode Multi-Class Classifier**:
  - Accuracy: `99.69%`
  - Macro F1-Score: `0.8311`
- **Physical Parameter Attribution (Top Drivers)**:
  1. `vibration_rms_g` & `tool_wear_index`: primary root cause of `DIMENSIONAL_DRIFT` and `SURFACE_FLAW`.
  2. `peak_torque_nm` & `shutoff_angle_deg`: primary driver of `INCOMPLETE_FASTENING`.
  3. `system_pressure_psi` & `spindle_temp_c`: primary trigger of `HYDRAULIC_PRESSURE_DROP`.

---

## 3. Tier 2: Raw Material Intelligence & Adaptive Compensation
- **Marginal Batch Downstream Risk Classifier**:
  - ROC-AUC: `0.8030`
  - F1-Score: `0.8000`
- **Adaptive Parameter Compensation Rules**:
  - Batches with Hardness > 216 HV receive an automated torque offset of `+0.38 Nm` and shutoff angle compensation of `+11.4°` with a `7.6%` spindle speed reduction.

---

## 4. Tier 3: Tolerance Leakage Gap Analysis & Corrective Actions
- **The Tolerance Leakage Vector**:
  - Batches with hardness between **216.0 HV and 230.0 HV** nominally pass current receiving QC.
  - However, they leak **$438,746.58** in field warranty claims (25.48% of all warranty dollars!).
- **Recommended Specification Tightening**:
  - Revise Upper Spec Limit from **230.0 HV -> 216.0 HV**.
  - Projected annual warranty cost avoidance: **$372,934.59**.
  - Scrap reduction: **52.4%**.
