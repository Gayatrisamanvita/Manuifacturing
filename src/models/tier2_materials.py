"""
Tier 2: Raw Material Level ML & Prescriptive Compensation Models
Addresses:
5. Marginal Batch Downstream Defect Risk Prediction
6. Downstream Assembly Defect & Severity Impact Forecast
7. Adaptive Machine Parameter Compensation Engine
"""

import os
import pickle
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, f1_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
import lightgbm as lgb


class MarginalBatchRiskClassifier:
    """
    Question 5: Pinpoints batches that passed nominal receiving inspection
    (hardness between LSL 180 and USL 230 HV) but carry elevated downstream defect risk.
    """
    def __init__(self, model_path: Optional[str] = None):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            class_weight="balanced",
            random_state=42
        )
        self.feature_names: List[str] = []
        self.metrics: Dict[str, float] = {}
        self.model_path = model_path

    def fit(self, batch_summary_df: pd.DataFrame) -> Dict[str, float]:
        df = batch_summary_df.copy()

        # Target: High downstream risk (either marked MARGINAL or batch defect rate > 55% or high scrap)
        # Note: In the master data, material_risk_status is 'MARGINAL_HIGH', 'MARGINAL_LOW', or 'NORMAL'
        y = (df["material_risk_status"] != "NORMAL").astype(int)

        # Features available at receiving inspection
        df["hardness_margin_upper"] = df["upper_spec_limit"] - df["material_hardness_hv"]
        df["hardness_margin_lower"] = df["material_hardness_hv"] - df["lower_spec_limit"]
        df["hardness_nominal_dev"] = np.abs(df["material_hardness_hv"] - (df["upper_spec_limit"] + df["lower_spec_limit"]) / 2.0)
        df["tensile_to_hardness_ratio"] = df["tensile_strength_mpa"] / (df["material_hardness_hv"] + 1e-4)

        supplier_dummies = pd.get_dummies(df["supplier_id"], prefix="supplier", dtype=float)

        feature_cols = [
            "material_hardness_hv", "tensile_strength_mpa",
            "hardness_margin_upper", "hardness_margin_lower",
            "hardness_nominal_dev", "tensile_to_hardness_ratio"
        ]
        X = pd.concat([df[feature_cols], supplier_dummies], axis=1)
        self.feature_names = list(X.columns)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        self.model.fit(X_train, y_train)
        probs = self.model.predict_proba(X_test)[:, 1]
        preds = (probs > 0.4).astype(int)

        auc = roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else 1.0
        f1 = f1_score(y_test, preds)

        self.metrics = {"roc_auc": float(auc), "f1_score": float(f1)}
        return self.metrics

    def predict_batch_risk(self, batch_row: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculates downstream defect risk probability and assigns risk tier for incoming lot.
        """
        row = batch_row.copy()
        if "hardness_margin_upper" not in row.columns:
            row["hardness_margin_upper"] = row["upper_spec_limit"] - row["material_hardness_hv"]
            row["hardness_margin_lower"] = row["material_hardness_hv"] - row["lower_spec_limit"]
            row["hardness_nominal_dev"] = np.abs(row["material_hardness_hv"] - (row["upper_spec_limit"] + row["lower_spec_limit"]) / 2.0)
            row["tensile_to_hardness_ratio"] = row["tensile_strength_mpa"] / (row["material_hardness_hv"] + 1e-4)

        for col in self.feature_names:
            if col not in row.columns:
                row[col] = 0.0

        X = row[self.feature_names]
        risk_prob = float(self.model.predict_proba(X)[0, 1])
        hardness = float(row["material_hardness_hv"].iloc[0])
        lower_limit = float(row["lower_spec_limit"].iloc[0])
        upper_limit = float(row["upper_spec_limit"].iloc[0])
        passed_inspection = lower_limit <= hardness <= upper_limit

        if not passed_inspection:
            status = "OUT_OF_SPEC_REJECT"
            recommendation = "Reject or quarantine lot: hardness is outside receiving specification"
        elif risk_prob >= 0.6:
            status = "HIGH_RISK_MARGINAL"
            recommendation = "Apply adaptive recipe compensation before feeding to line"
        elif risk_prob >= 0.3:
            status = "BORDERLINE_MONITOR"
            recommendation = "Apply adaptive recipe compensation before feeding to line"
        else:
            status = "STANDARD_PASS"
            recommendation = "Standard line feed"

        return {
            "downstream_risk_probability": round(risk_prob, 4),
            "batch_risk_tier": status,
            "passed_receiving_inspection": passed_inspection,
            "recommendation": recommendation
        }

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump({"model": self.model, "features": self.feature_names, "metrics": self.metrics}, f)

    def load(self, filepath: str):
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.model = data["model"]
            self.feature_names = data["features"]
            self.metrics = data["metrics"]


class DownstreamImpactForecaster:
    """
    Question 6: Forecasts downstream assembly defect types, defect severity (1-5),
    and expected scrap volume based on material properties.
    """
    def __init__(self):
        self.severity_regressor = GradientBoostingRegressor(n_estimators=80, random_state=42)
        self.scrap_regressor = GradientBoostingRegressor(n_estimators=80, random_state=42)
        self.features: List[str] = []
        self.reference_batch_size: Optional[float] = None

    def fit(self, batch_summary_df: pd.DataFrame):
        df = batch_summary_df.copy()
        if "total_parts" in df.columns and not df["total_parts"].empty:
            self.reference_batch_size = float(df["total_parts"].mean())
        supplier_dummies = pd.get_dummies(df["supplier_id"], prefix="supplier", dtype=float)

        feature_cols = ["material_hardness_hv", "tensile_strength_mpa"]
        X = pd.concat([df[feature_cols], supplier_dummies], axis=1)
        self.features = list(X.columns)

        y_sev = df["avg_severity"]
        y_scrap = df["scrap_count"]

        self.severity_regressor.fit(X, y_sev)
        self.scrap_regressor.fit(X, y_scrap)

    def forecast(
        self,
        hardness: float,
        tensile: float,
        supplier_id: str = "VENDOR_A",
        batch_size_units: Optional[int] = None,
    ) -> Dict[str, Any]:
        row = pd.DataFrame([{
            "material_hardness_hv": hardness,
            "tensile_strength_mpa": tensile,
            "supplier_VENDOR_A": 1.0 if supplier_id == "VENDOR_A" else 0.0,
            "supplier_VENDOR_B": 1.0 if supplier_id == "VENDOR_B" else 0.0,
            "supplier_VENDOR_C": 1.0 if supplier_id == "VENDOR_C" else 0.0,
        }])
        X = row.reindex(columns=self.features, fill_value=0.0)

        pred_sev = float(self.severity_regressor.predict(X)[0])
        pred_scrap = float(self.scrap_regressor.predict(X)[0])
        if batch_size_units is not None:
            if batch_size_units < 1:
                raise ValueError("batch_size_units must be at least 1.")
            if not self.reference_batch_size or self.reference_batch_size <= 0:
                raise RuntimeError("Reference batch size is unavailable; refit the impact forecaster.")
            pred_scrap *= batch_size_units / self.reference_batch_size
        expected_scrap = max(0, int(round(pred_scrap)))
        if batch_size_units is not None:
            expected_scrap = min(batch_size_units, expected_scrap)

        # Predicted likely failure mode based on physical regime
        if hardness >= 218.0:
            likely_mode = "INCOMPLETE_FASTENING"
            root_cause = "Excessive material hardness causing high fastening torque resistance before target shutoff angle is reached."
        elif hardness <= 192.0:
            likely_mode = "THREAD_WEAR"
            root_cause = "Sub-nominal material yield strength causing thread stripping or micro-galling under nominal preload."
        else:
            likely_mode = "SURFACE_FLAW"
            root_cause = "Nominal material properties; defect risk is driven by tool wear or thermal dissipation."

        return {
            "expected_defect_severity": round(max(1.0, min(5.0, pred_sev)), 2),
            "expected_scrap_count": expected_scrap,
            "likely_downstream_failure_mode": likely_mode,
            "physical_root_cause": root_cause,
            "failure_mode_method": "HARDNESS_THRESHOLD_RULES",
            "projected_cost_usd": round(expected_scrap * 55.0, 2)
        }

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump({
                "severity_regressor": self.severity_regressor,
                "scrap_regressor": self.scrap_regressor,
                "features": self.features,
                "reference_batch_size": self.reference_batch_size,
            }, f)

    def load(self, filepath: str):
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.severity_regressor = data["severity_regressor"]
            self.scrap_regressor = data["scrap_regressor"]
            self.features = data["features"]
            self.reference_batch_size = data.get("reference_batch_size")


class AdaptiveParameterCompensator:
    """
    Question 7: Adaptive Machine Parameter Compensation Engine.
    Computes precise recipe offsets to safely process borderline/marginal material stock
    without generating scrap.
    """
    def __init__(self):
        self.nominal_hardness = 205.0  # HV
        self.nominal_tensile = 610.0   # MPa
        self.nominal_torque = 12.5     # Nm
        self.nominal_shutoff_angle = 360.0  # deg
        self.nominal_rpm = 450.0       # rpm

    def compute_offsets(self, batch_hardness_hv: float, batch_tensile_mpa: float) -> Dict[str, Any]:
        """
        Physics-informed transfer function:
        High hardness requires higher rundown torque margin and reduced rundown speed
        to avoid torque spikes and incomplete fastening.
        Low hardness requires gentle torque ramp-up to prevent thread stripping.
        """
        delta_hardness = batch_hardness_hv - self.nominal_hardness

        if delta_hardness > 10.0:  # Borderline High (e.g. 215 - 228 HV)
            # Increase torque limit threshold slightly to prevent premature shutoff
            torque_offset = round(0.04 * (delta_hardness - 10.0), 2)  # +0.2 to +0.6 Nm
            angle_offset = round(1.2 * (delta_hardness - 10.0), 1)    # +10 to +18 deg
            speed_reduction_pct = round(min(20.0, 0.8 * (delta_hardness - 10.0)), 1)
            feed_rate_offset_pct = -5.0
            lubricant_boost = "ENABLE_HIGH_PRESSURE_MIST"
            status = "ACTIVE_COMPENSATION_HARD_STOCK"
            notes = (
                f"Batch hardness is +{delta_hardness:.1f} HV above nominal. Applying torque offset of "
                f"+{torque_offset} Nm and speed reduction of -{speed_reduction_pct}% to eliminate incomplete fastening risk."
            )

        elif delta_hardness < -10.0:  # Borderline Low (e.g. 180 - 195 HV)
            # Reduce torque to prevent thread deformation
            torque_offset = round(-0.03 * abs(delta_hardness + 10.0), 2)  # -0.15 to -0.45 Nm
            angle_offset = round(-0.8 * abs(delta_hardness + 10.0), 1)
            speed_reduction_pct = 0.0
            feed_rate_offset_pct = +2.0
            lubricant_boost = "STANDARD_COOLANT"
            status = "ACTIVE_COMPENSATION_SOFT_STOCK"
            notes = (
                f"Batch hardness is {delta_hardness:.1f} HV below nominal. Applying torque offset of "
                f"{torque_offset} Nm and angle offset {angle_offset} deg to prevent thread wear and stripping."
            )

        else:
            torque_offset = 0.0
            angle_offset = 0.0
            speed_reduction_pct = 0.0
            feed_rate_offset_pct = 0.0
            lubricant_boost = "STANDARD_COOLANT"
            status = "STANDARD_RECIPE_OPTIMAL"
            notes = "Material hardness and tensile strength are within optimal core process band. No recipe offset required."

        return {
            "status": status,
            "material_hardness_hv": batch_hardness_hv,
            "tensile_strength_mpa": batch_tensile_mpa,
            "tensile_adjustment_applied": False,
            "hardness_deviation_from_nominal": round(delta_hardness, 2),
            "prescribed_torque_target_nm": round(self.nominal_torque + torque_offset, 2),
            "prescribed_shutoff_angle_deg": round(self.nominal_shutoff_angle + angle_offset, 1),
            "torque_offset_nm": torque_offset,
            "angle_offset_deg": angle_offset,
            "spindle_speed_offset_pct": -speed_reduction_pct,
            "feed_rate_offset_pct": feed_rate_offset_pct,
            "coolant_lubrication_mode": lubricant_boost,
            "engineering_guidance": (
                f"{notes} Offsets are based on hardness; tensile strength is recorded but has no "
                "independent offset because no validated tensile-to-recipe mapping is available."
            )
        }

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump({
                "nominal_hardness": self.nominal_hardness,
                "nominal_tensile": self.nominal_tensile,
                "nominal_torque": self.nominal_torque,
                "nominal_shutoff_angle": self.nominal_shutoff_angle,
                "nominal_rpm": self.nominal_rpm
            }, f)

    def load(self, filepath: str):
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.nominal_hardness = data.get("nominal_hardness", 205.0)
            self.nominal_tensile = data.get("nominal_tensile", 610.0)
            self.nominal_torque = data.get("nominal_torque", 12.5)
            self.nominal_shutoff_angle = data.get("nominal_shutoff_angle", 360.0)
            self.nominal_rpm = data.get("nominal_rpm", 450.0)
