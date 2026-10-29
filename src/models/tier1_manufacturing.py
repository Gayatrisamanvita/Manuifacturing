"""
Tier 1: Manufacturing Issue Level ML Models
Addresses:
1. Defect Count & High-Risk Station Prediction
2. Process Drift & Root Parameter Attribution (SHAP)
3. Immediate Adjustment / Preventive Action Engine
4. Failure Mode Multi-Class Classifier & Top 5 Physical Contributors
"""

import os
import pickle
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    classification_report, accuracy_score, f1_score
)
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import shap


class ShiftDefectPredictor:
    """
    Question 1: Shift-level Defect Count Regressor and High-Risk Station Ranking.
    """
    def __init__(self, model_path: Optional[str] = None):
        self.model = lgb.LGBMRegressor(
            n_estimators=150,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            objective="poisson",  # defect count is naturally Poisson distributed
            verbose=-1
        )
        self.feature_names: List[str] = []
        self.metrics: Dict[str, float] = {}
        self.model_path = model_path

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        self.feature_names = list(X.columns)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
        )

        preds = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        self.metrics = {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}
        return self.metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Align features
        X_aligned = X.reindex(columns=self.feature_names, fill_value=0.0)
        return self.model.predict(X_aligned)

    def rank_station_risk(self, current_shift_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Ranks stations ST-10, ST-20, ST-30, ST-40 by predicted defect risk for the shift.
        """
        predictions = self.predict(current_shift_data)
        stations = current_shift_data["station_id"].values if "station_id" in current_shift_data.columns else [f"ST-{i}" for i in range(len(predictions))]

        results = []
        for st, pred in zip(stations, predictions):
            risk_level = "HIGH" if pred > 60 else ("MEDIUM" if pred > 30 else "LOW")
            results.append({
                "station_id": st,
                "predicted_defect_count": round(float(pred), 1),
                "risk_level": risk_level
            })

        # Sort descending by predicted defects
        results.sort(key=lambda x: x["predicted_defect_count"], reverse=True)
        for rank, item in enumerate(results, start=1):
            item["station_rank"] = rank
        return results

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


class FailureModeClassifier:
    """
    Question 4 & 2: Multi-Class Failure Mode Classifier + Top 5 Physical Contributors via SHAP.
    """
    def __init__(self, model_path: Optional[str] = None):
        self.model = lgb.LGBMClassifier(
            n_estimators=120,
            learning_rate=0.08,
            num_leaves=31,
            random_state=42,
            class_weight="balanced",
            verbose=-1
        )
        self.label_encoder = LabelEncoder()
        self.feature_names: List[str] = []
        self.explainer: Optional[shap.TreeExplainer] = None
        self.metrics: Dict[str, Any] = {}
        self.model_path = model_path

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        self.feature_names = list(X.columns)
        y_encoded = self.label_encoder.fit_transform(y)

        unique, counts = np.unique(y_encoded, return_counts=True)
        strat = y_encoded if np.min(counts) >= 2 else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=strat
        )

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
        )

        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        macro_f1 = f1_score(y_test, preds, average="macro")

        # TreeExplainer for fast SHAP attribution
        self.explainer = shap.TreeExplainer(self.model)

        self.metrics = {
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "classes": list(self.label_encoder.classes_)
        }
        return self.metrics

    def predict_with_contributors(self, X_sample: pd.DataFrame, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Predicts failure mode and extracts top-5 physical contributors using SHAP values.
        """
        X_aligned = X_sample.reindex(columns=self.feature_names, fill_value=0.0)
        probs = self.model.predict_proba(X_aligned)
        pred_indices = np.argmax(probs, axis=1)
        pred_labels = self.label_encoder.inverse_transform(pred_indices)

        # Compute SHAP values
        if self.explainer is None:
            self.explainer = shap.TreeExplainer(self.model)

        shap_values = self.explainer.shap_values(X_aligned)

        results = []
        for i in range(len(X_aligned)):
            label = pred_labels[i]
            cls_idx = pred_indices[i]
            prob = float(probs[i][cls_idx])

            # SHAP for predicted class
            if isinstance(shap_values, list):
                # multi-class list of shap matrices
                row_shap = shap_values[cls_idx][i]
            else:
                # 3D array (n_samples, n_features, n_classes)
                if len(shap_values.shape) == 3:
                    row_shap = shap_values[i, :, cls_idx]
                else:
                    row_shap = shap_values[i]

            # Top contributors by absolute SHAP impact
            abs_shap = np.abs(row_shap)
            top_indices = np.argsort(abs_shap)[::-1][:top_k]

            contributors = []
            for idx in top_indices:
                feature_name = self.feature_names[idx]
                feat_val = float(X_aligned.iloc[i, idx])
                shap_val = float(row_shap[idx])
                contributors.append({
                    "feature": feature_name,
                    "value": round(feat_val, 2),
                    "shap_impact": round(shap_val, 4),
                    "direction": (
                        "PROTECTIVE" if shap_val > 0 else "RISK_INFLATING"
                    ) if label in {"NONE", "NORMAL"} else (
                        "RISK_INFLATING" if shap_val > 0 else "PROTECTIVE"
                    )
                })

            results.append({
                "predicted_failure_mode": label,
                "confidence": round(prob, 4),
                "top_physical_contributors": contributors
            })

        return results

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump({
                "model": self.model,
                "label_encoder": self.label_encoder,
                "features": self.feature_names,
                "metrics": self.metrics
            }, f)

    def load(self, filepath: str):
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.model = data["model"]
            self.label_encoder = data["label_encoder"]
            self.feature_names = data["features"]
            self.metrics = data["metrics"]
            self.explainer = shap.TreeExplainer(self.model)


class PrescriptiveActionEngine:
    """
    Question 3: Immediate Adjustment / Preventive Action Engine.
    Maps detected root causes, failure modes, and current sensor drifts to prescriptive
    machine recipe offsets and maintenance actions.
    """
    def __init__(self):
        # Recommended standard operating nominals
        self.nominals = {
            "peak_torque_nm": 12.5,
            "shutoff_angle_deg": 360.0,
            "spindle_temp_c": 55.0,
            "system_pressure_psi": 115.0,
            "vibration_rms_g": 1.2,
            "tool_wear_index": 40.0
        }

    def recommend_action(
        self,
        predicted_failure_mode: str,
        current_telemetry: Dict[str, float],
        top_contributors: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Synthesizes operational recommendations and exact machine recipe offsets.
        """
        temp = current_telemetry.get("spindle_temp_c", 55.0)
        torque = current_telemetry.get("peak_torque_nm", 12.5)
        angle = current_telemetry.get("shutoff_angle_deg", 360.0)
        vibration = current_telemetry.get("vibration_rms_g", 1.2)
        tool_wear = current_telemetry.get("tool_wear_index", 30.0)
        pressure = current_telemetry.get("system_pressure_psi", 115.0)

        # Baseline decision rule mapped to industrial Action Codes
        if tool_wear >= 85.0 or vibration > 3.2:
            action_code = "STOP_LINE_REPLACE_TOOL"
            urgency = "CRITICAL - IMMEDIATE HALT"
            maintenance_task = f"Replace cutting/fastening bit immediately. Tool wear index is {tool_wear:.1f}% (critical threshold 85%) and vibration is {vibration:.2f}g (limit 3.0g)."
            recipe_offsets = {"tool_replacement": "REQUIRED", "reset_tool_wear_counter": True}

        elif predicted_failure_mode == "INCOMPLETE_FASTENING" or torque > 13.1 or angle < 300.0:
            action_code = "RECALIBRATE_TOOL"
            urgency = "HIGH - BEFORE NEXT BATCH"
            delta_torque = round(self.nominals["peak_torque_nm"] - torque, 2)
            delta_angle = round(self.nominals["shutoff_angle_deg"] - angle, 1)
            maintenance_task = "Recalibrate torque shutoff sensor and inspect fastener thread engagement depth."
            recipe_offsets = {
                "torque_offset_nm": delta_torque,
                "shutoff_angle_offset_deg": delta_angle,
                "rundown_speed_reduction_pct": 10.0 if torque > 13.2 else 5.0
            }

        elif predicted_failure_mode == "SURFACE_FLAW" or temp > 65.0:
            action_code = "ADJUST_SPEED_PRESSURE"
            urgency = "MEDIUM - ACTIVE INTERVENTION"
            delta_press = round(self.nominals["system_pressure_psi"] - pressure, 1)
            maintenance_task = "Increase coolant manifold flow rate and throttle spindle speed to avoid surface thermal scoring."
            recipe_offsets = {
                "spindle_rpm_offset_pct": -8.0,
                "coolant_pressure_offset_psi": max(5.0, delta_press),
                "feed_rate_reduction_pct": 5.0
            }

        elif predicted_failure_mode == "HYDRAULIC_PRESSURE_DROP" or pressure < 95.0:
            action_code = "ADJUST_SPEED_PRESSURE"
            urgency = "CRITICAL - PRESSURE LOSS"
            maintenance_task = f"Inspect hydraulic line filters and pump seal. Current pressure {pressure:.1f} psi is below 95 psi minimum."
            recipe_offsets = {
                "hydraulic_pump_boost_psi": round(115.0 - pressure, 1),
                "line_speed_throttle_upm": -10.0
            }

        elif predicted_failure_mode == "NONE" and tool_wear < 65.0 and vibration < 2.2:
            action_code = "CONTINUE_MONITORING"
            urgency = "NORMAL - OPERATIONAL"
            maintenance_task = "Process is operating well within 6-sigma tolerance boundaries. Continue standard monitoring."
            recipe_offsets = {"no_offset_needed": True}

        else:
            action_code = "RECALIBRATE_TOOL"
            urgency = "MODERATE - DRIFT DETECTED"
            maintenance_task = "Minor parameter drift detected. Schedule station recalibration at the end of current cycle."
            recipe_offsets = {"torque_offset_nm": -0.2, "shutoff_angle_offset_deg": +5.0}

        return {
            "action_code": action_code,
            "urgency": urgency,
            "recommended_task": maintenance_task,
            "prescriptive_recipe_offsets": recipe_offsets,
            "root_driver": top_contributors[0]["feature"] if top_contributors else "UNKNOWN"
        }

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump({"nominals": self.nominals}, f)

    def load(self, filepath: str):
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.nominals = data.get("nominals", self.nominals)
