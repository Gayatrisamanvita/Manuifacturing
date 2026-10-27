"""
Feature Engineering Module for Manufacturing Intelligence Suite.
Includes physical domain features, drift statistics (Z-scores, rolling EWMA, CUSUM),
and interaction terms across sensor, tool, torque, and material domains.
"""

from typing import List, Tuple, Dict
import pandas as pd
import numpy as np


class FeatureEngineering:
    def __init__(self):
        # Numerical feature lists
        self.sensor_features = [
            "spindle_temp_c", "system_pressure_psi", "vibration_rms_g",
            "tool_wear_index", "hours_since_service",
            "peak_torque_nm", "shutoff_angle_deg"
        ]
        self.material_features = [
            "material_hardness_hv", "tensile_strength_mpa"
        ]

    def add_physical_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derives physics-informed and interaction features.
        """
        data = df.copy()

        # 1. Torque Mechanical Energy Proxy (Joules approx: Torque * Angle in radians)
        if "peak_torque_nm" in data.columns and "shutoff_angle_deg" in data.columns:
            angle_rad = data["shutoff_angle_deg"] * (np.pi / 180.0)
            data["torque_energy_proxy"] = data["peak_torque_nm"] * angle_rad

        # 2. Tool Stress Factor (Tool Wear Index combined with Vibration)
        if "tool_wear_index" in data.columns and "vibration_rms_g" in data.columns:
            data["tool_stress_factor"] = data["tool_wear_index"] * data["vibration_rms_g"]

        # 3. Thermal-Mechanical Dissipation Index
        if "spindle_temp_c" in data.columns and "vibration_rms_g" in data.columns:
            data["thermal_vibration_index"] = data["spindle_temp_c"] * data["vibration_rms_g"]

        # 4. Hydraulic-Thermal Stability Ratio
        if "system_pressure_psi" in data.columns and "spindle_temp_c" in data.columns:
            data["hydraulic_thermal_ratio"] = data["system_pressure_psi"] / (data["spindle_temp_c"] + 1e-3)

        # 5. Material-Tool Resistance Interaction
        if "material_hardness_hv" in data.columns and "tool_wear_index" in data.columns:
            data["material_tool_resistance"] = data["material_hardness_hv"] * (data["tool_wear_index"] + 1.0)

        # 6. Torque shutoff deviation from target (standard target is ~12.5 Nm, 360 deg)
        if "peak_torque_nm" in data.columns:
            data["torque_delta_nominal"] = data["peak_torque_nm"] - 12.5
        if "shutoff_angle_deg" in data.columns:
            data["angle_delta_nominal"] = data["shutoff_angle_deg"] - 360.0

        return data

    def compute_cusum_drift(self, series: pd.Series, target: float = None, slack: float = 0.5) -> np.ndarray:
        """
        Computes two-sided Cumulative Sum (CUSUM) for detecting subtle process drift.
        """
        vals = series.values
        if target is None:
            target = np.nanmean(vals)
        sigma = np.nanstd(vals) if np.nanstd(vals) > 0 else 1.0
        k = slack * sigma

        pos_cusum = np.zeros(len(vals), dtype=np.float32)
        neg_cusum = np.zeros(len(vals), dtype=np.float32)

        for i in range(1, len(vals)):
            diff = vals[i] - target
            pos_cusum[i] = max(0.0, pos_cusum[i - 1] + diff - k)
            neg_cusum[i] = max(0.0, neg_cusum[i - 1] - diff - k)

        return np.maximum(pos_cusum, neg_cusum)

    def add_rolling_drift_features(self, df: pd.DataFrame, window: int = 50) -> pd.DataFrame:
        """
        Computes rolling statistics and Z-scores per station to identify dynamic drifts.
        """
        data = df.copy()

        cols_to_drift = ["spindle_temp_c", "vibration_rms_g", "peak_torque_nm", "tool_wear_index"]
        cols_present = [c for c in cols_to_drift if c in data.columns]

        # Group by station if station_id is available
        if "station_id" in data.columns:
            for col in cols_present:
                grouped = data.groupby("station_id")[col]
                r_mean = grouped.transform(lambda s: s.rolling(window, min_periods=5).mean())
                r_std = grouped.transform(lambda s: s.rolling(window, min_periods=5).std()).replace(0, 1e-4)
                data[f"{col}_rolling_mean"] = r_mean
                data[f"{col}_zscore"] = (data[col] - r_mean) / r_std
        else:
            for col in cols_present:
                r_mean = data[col].rolling(window, min_periods=5).mean()
                r_std = data[col].rolling(window, min_periods=5).std().replace(0, 1e-4)
                data[f"{col}_rolling_mean"] = r_mean
                data[f"{col}_zscore"] = (data[col] - r_mean) / r_std

        # Fill NaNs from rolling warmup
        data = data.bfill().ffill()
        return data

    def prepare_shift_features(self, shift_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Prepares shift-level features for Defect Count regression & Station ranking.
        """
        df = shift_df.copy()

        # One-hot encode station and shift
        station_dummies = pd.get_dummies(df["station_id"], prefix="station", dtype=float)
        shift_dummies = pd.get_dummies(df["shift_id"], prefix="shift", dtype=float)

        feature_cols = [
            "total_units", "avg_line_speed", "std_line_speed", "operator_tenure_yrs",
            "mean_spindle_temp", "max_spindle_temp", "mean_vibration", "max_vibration",
            "mean_pressure", "mean_tool_wear", "max_tool_wear",
            "mean_peak_torque", "std_peak_torque", "mean_shutoff_angle"
        ]
        # Include available engineered columns
        X = pd.concat([df[feature_cols], station_dummies, shift_dummies], axis=1)
        y = df["defect_count"]
        return X, y, list(X.columns)

    def prepare_cycle_features(self, cycle_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.Series], List[str]]:
        """
        Prepares high-resolution cycle features for Failure Mode, Defect Flag, and Action classification.
        """
        df = self.add_physical_domain_features(cycle_df)

        # Torque status encoding
        df["torque_status_nok"] = (df["torque_status"] != "OK").astype(float)
        df["torque_status_high"] = (df["torque_status"] == "NOK_HIGH").astype(float)
        df["torque_status_low"] = (df["torque_status"] == "NOK_LOW").astype(float)

        # Station one-hot
        station_dummies = pd.get_dummies(df["station_id"], prefix="st", dtype=float)

        feature_cols = [
            "peak_torque_nm", "shutoff_angle_deg",
            "spindle_temp_c", "system_pressure_psi", "vibration_rms_g",
            "tool_wear_index", "hours_since_service",
            "torque_energy_proxy", "tool_stress_factor", "thermal_vibration_index",
            "hydraulic_thermal_ratio", "material_tool_resistance",
            "torque_delta_nominal", "angle_delta_nominal",
            "torque_status_nok", "torque_status_high", "torque_status_low"
        ]

        # If material columns are present
        if "material_hardness_hv" in df.columns:
            feature_cols.extend(["material_hardness_hv", "tensile_strength_mpa"])

        X = pd.concat([df[feature_cols], station_dummies], axis=1)

        targets = {
            "defect_flag": df["defect_flag"],
            "defect_severity": df["defect_severity"],
            "failure_mode": df["failure_mode"],
            "action_code": df["action_code"],
            "vendor_action_code": df["vendor_action_code"],
        }

        return X, targets, list(X.columns)
