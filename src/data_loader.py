"""
Manufacturing Intelligence Suite - Unified Data Loader
Handles ingestion, schema validation, and relational joining across all 9 data tiers.
"""

import os
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np


class DataLoader:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.files = {
            "shift_logs": os.path.join(data_dir, "production_shift_logs.csv"),
            "machine_sensors": os.path.join(data_dir, "machine_sensor_readings.csv"),
            "tool_health": os.path.join(data_dir, "tool_machine_health.csv"),
            "torque_cycles": os.path.join(data_dir, "torque_cycle_logs.csv"),
            "part_inspections": os.path.join(data_dir, "part_inspection_defect_logs.csv"),
            "material_batches": os.path.join(data_dir, "material_batch_master.csv"),
            "supplier_history": os.path.join(data_dir, "supplier_quality_history.csv"),
            "warranty_claims": os.path.join(data_dir, "warranty_field_claims.csv"),
            "failure_modes": os.path.join(data_dir, "failure_mode_reference.csv"),
            "supplier_products": os.path.join(data_dir, "supplier_products_catalog.csv"),
        }

    def load_raw(self, key: str, nrows: Optional[int] = None) -> pd.DataFrame:
        if key not in self.files:
            raise KeyError(f"Dataset key '{key}' not found. Available: {list(self.files.keys())}")
        return pd.read_csv(self.files[key], nrows=nrows)

    def load_all_raw(self, nrows: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        return {k: self.load_raw(k, nrows=nrows) for k in self.files}

    def get_unified_cycle_data(self, sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Merges cycle-level datasets: torque, sensors, tool health, part inspections,
        and material batch characteristics into a unified operational dataset.
        """
        # Load core tables
        df_torq = pd.read_csv(self.files["torque_cycles"], nrows=sample_size)
        df_sens = pd.read_csv(self.files["machine_sensors"], nrows=sample_size)
        df_tool = pd.read_csv(self.files["tool_health"], nrows=sample_size)
        df_insp = pd.read_csv(self.files["part_inspections"], nrows=sample_size)
        df_mat = pd.read_csv(self.files["material_batches"])

        # Cycle / Inspection table alignment
        # The primary continuous tables share index / timestamp / station ordering
        unified = pd.DataFrame({
            "fastener_cycle_id": df_torq["fastener_cycle_id"],
            "part_serial_no": df_insp["part_serial_no"],
            "event_timestamp": pd.to_datetime(df_torq["event_timestamp"]),
            "station_id": df_torq["station_id"],
            # Torque features
            "peak_torque_nm": df_torq["peak_torque_nm"].astype(np.float32),
            "shutoff_angle_deg": df_torq["shutoff_angle_deg"].astype(np.float32),
            "torque_status": df_torq["torque_status"],
            # Machine Sensor features
            "machine_state": df_sens["machine_state"],
            "spindle_temp_c": df_sens["spindle_temp_c"].astype(np.float32),
            "system_pressure_psi": df_sens["system_pressure_psi"].astype(np.float32),
            "vibration_rms_g": df_sens["vibration_rms_g"].astype(np.float32),
            # Tool Health features
            "tool_id": df_tool["tool_id"],
            "tool_wear_index": df_tool["tool_wear_index"].astype(np.float32),
            "hours_since_service": df_tool["hours_since_service"].astype(np.float32),
            # Inspection Targets & Material Join Key
            "material_batch_id": df_insp["material_batch_id"],
            "defect_flag": df_insp["defect_flag"].astype(np.int8),
            "defect_severity": df_insp["defect_severity"].astype(np.int8),
            "failure_mode": df_insp["failure_mode"],
            "action_code": df_insp["action_code"],
            "vendor_action_code": df_insp["vendor_action_code"],
        })

        # Enrich with Material Master specs
        mat_cols = [
            "material_batch_id", "supplier_id", "material_hardness_hv",
            "tensile_strength_mpa", "lower_spec_limit", "upper_spec_limit",
            "material_risk_status"
        ]
        unified = unified.merge(df_mat[mat_cols], on="material_batch_id", how="left")

        return unified

    def get_shift_aggregated_data(self) -> pd.DataFrame:
        """
        Aggregates shift-level logs with summary statistics from sensor telemetry,
        torque execution, and material batches processed during each shift.
        """
        df_shift = pd.read_csv(self.files["shift_logs"])
        df_shift["event_timestamp"] = pd.to_datetime(df_shift["event_timestamp"])
        df_shift["date"] = df_shift["event_timestamp"].dt.date.astype(str)

        # Base shift-station aggregates
        shift_summary = df_shift.groupby(["date", "shift_id", "station_id"]).agg(
            total_units=("units_produced", "sum"),
            defect_count=("defect_count", "sum"),
            avg_line_speed=("line_speed_upm", "mean"),
            std_line_speed=("line_speed_upm", "std"),
            operator_tenure_yrs=("operator_tenure_yrs", "mean"),
        ).reset_index()

        # Fill std na
        shift_summary["std_line_speed"] = shift_summary["std_line_speed"].fillna(0.0)

        # Merge in sensor and tool telemetry summaries per shift-station
        df_sens = pd.read_csv(self.files["machine_sensors"])
        df_tool = pd.read_csv(self.files["tool_health"])
        df_torq = pd.read_csv(self.files["torque_cycles"])

        df_telemetry = pd.DataFrame({
            "event_timestamp": pd.to_datetime(df_sens["event_timestamp"]),
            "station_id": df_sens["station_id"],
            "spindle_temp_c": df_sens["spindle_temp_c"],
            "system_pressure_psi": df_sens["system_pressure_psi"],
            "vibration_rms_g": df_sens["vibration_rms_g"],
            "tool_wear_index": df_tool["tool_wear_index"],
            "peak_torque_nm": df_torq["peak_torque_nm"],
            "shutoff_angle_deg": df_torq["shutoff_angle_deg"],
        })
        df_telemetry["date"] = df_telemetry["event_timestamp"].dt.date.astype(str)
        # Infer shift_id from timestamp hour
        hour = df_telemetry["event_timestamp"].dt.hour
        conditions = [
            (hour >= 6) & (hour < 14),
            (hour >= 14) & (hour < 22),
        ]
        choices = ["SHIFT_1", "SHIFT_2"]
        df_telemetry["shift_id"] = np.select(conditions, choices, default="SHIFT_3")

        telem_agg = df_telemetry.groupby(["date", "shift_id", "station_id"]).agg(
            mean_spindle_temp=("spindle_temp_c", "mean"),
            max_spindle_temp=("spindle_temp_c", "max"),
            mean_vibration=("vibration_rms_g", "mean"),
            max_vibration=("vibration_rms_g", "max"),
            mean_pressure=("system_pressure_psi", "mean"),
            mean_tool_wear=("tool_wear_index", "mean"),
            max_tool_wear=("tool_wear_index", "max"),
            mean_peak_torque=("peak_torque_nm", "mean"),
            std_peak_torque=("peak_torque_nm", "std"),
            mean_shutoff_angle=("shutoff_angle_deg", "mean"),
        ).reset_index()

        telem_agg["std_peak_torque"] = telem_agg["std_peak_torque"].fillna(0.0)

        merged_shift = shift_summary.merge(telem_agg, on=["date", "shift_id", "station_id"], how="left")
        merged_shift["defect_rate"] = merged_shift["defect_count"] / merged_shift["total_units"].replace(0, 1)

        return merged_shift

    def get_material_batch_summary(self) -> pd.DataFrame:
        """
        Summarizes material batches with downstream production outcomes and warranty claims.
        """
        df_mat = pd.read_csv(self.files["material_batches"])
        df_insp = pd.read_csv(self.files["part_inspections"])
        df_warr = pd.read_csv(self.files["warranty_claims"])

        # Aggregate downstream inspection outcomes per batch
        insp_agg = df_insp.groupby("material_batch_id").agg(
            total_parts=("part_serial_no", "count"),
            defect_parts=("defect_flag", "sum"),
            avg_severity=("defect_severity", "mean"),
            capa_triggers=("vendor_action_code", lambda x: (x != "NO_ACTION").sum())
        ).reset_index()
        insp_agg["batch_defect_rate"] = insp_agg["defect_parts"] / insp_agg["total_parts"]

        # Aggregate warranty claims per batch
        warr_agg = df_warr.groupby("material_batch_id").agg(
            claim_count=("claim_id", "count"),
            total_claim_cost_usd=("claim_cost_usd", "sum"),
            avg_claim_cost_usd=("claim_cost_usd", "mean")
        ).reset_index()

        batch_master = df_mat.merge(insp_agg, on="material_batch_id", how="left")
        batch_master = batch_master.merge(warr_agg, on="material_batch_id", how="left")
        batch_master["claim_count"] = batch_master["claim_count"].fillna(0).astype(int)
        batch_master["total_claim_cost_usd"] = batch_master["total_claim_cost_usd"].fillna(0.0)
        batch_master["avg_claim_cost_usd"] = batch_master["avg_claim_cost_usd"].fillna(0.0)

        # Distance to nominal & spec margins
        batch_master["nominal_hardness"] = (batch_master["lower_spec_limit"] + batch_master["upper_spec_limit"]) / 2.0
        batch_master["hardness_margin_high"] = batch_master["upper_spec_limit"] - batch_master["material_hardness_hv"]
        batch_master["hardness_margin_low"] = batch_master["material_hardness_hv"] - batch_master["lower_spec_limit"]

        return batch_master

    def get_warranty_traceability(self) -> pd.DataFrame:
        """
        Full genealogical chain linking Warranty Claims -> Part -> Batch -> Supplier.
        """
        df_warr = pd.read_csv(self.files["warranty_claims"])
        df_insp = pd.read_csv(self.files["part_inspections"])
        df_mat = pd.read_csv(self.files["material_batches"])
        df_torq = pd.read_csv(self.files["torque_cycles"])

        trace = df_warr.merge(
            df_insp[["part_serial_no", "fastener_cycle_id", "defect_flag", "defect_severity", "failure_mode", "action_code"]],
            on="part_serial_no", how="left"
        )
        trace = trace.merge(
            df_torq[["fastener_cycle_id", "station_id", "peak_torque_nm", "shutoff_angle_deg", "torque_status"]],
            on="fastener_cycle_id", how="left"
        )
        trace = trace.merge(
            df_mat[["material_batch_id", "supplier_id", "material_hardness_hv", "tensile_strength_mpa", "material_risk_status", "chargeback_amount_usd"]],
            on="material_batch_id", how="left"
        )
        return trace
