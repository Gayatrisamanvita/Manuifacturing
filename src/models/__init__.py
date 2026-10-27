"""
Machine Learning and Analytical Models for Manufacturing, Materials, and Suppliers.
Provides direct access to all Tier 1, Tier 2, and Tier 3 models, as well as
convenience loaders for serialized pickle models.
"""

import os
from typing import Dict, Any, Optional

from src.models.tier1_manufacturing import (
    ShiftDefectPredictor,
    FailureModeClassifier,
    PrescriptiveActionEngine
)
from src.models.tier2_materials import (
    MarginalBatchRiskClassifier,
    DownstreamImpactForecaster,
    AdaptiveParameterCompensator
)
from src.models.tier3_supplier import (
    SupplierQualityTrendAnalyzer,
    WarrantyTraceabilityEngine,
    ToleranceLeakageAnalyzer,
    SCARGenerator
)

__all__ = [
    "ShiftDefectPredictor",
    "FailureModeClassifier",
    "PrescriptiveActionEngine",
    "MarginalBatchRiskClassifier",
    "DownstreamImpactForecaster",
    "AdaptiveParameterCompensator",
    "SupplierQualityTrendAnalyzer",
    "WarrantyTraceabilityEngine",
    "ToleranceLeakageAnalyzer",
    "SCARGenerator",
    "load_all_models",
]


def load_all_models(models_dir: str = "models", data_dir: str = "data") -> Dict[str, Any]:
    """
    Convenience loader: loads all serialized models and engines from pickle files
    and dataset references.
    """
    models: Dict[str, Any] = {}

    # Tier 1: Shift Defect Predictor
    shift_pred = ShiftDefectPredictor()
    shift_pkl = os.path.join(models_dir, "shift_defect_predictor.pkl")
    if os.path.exists(shift_pkl):
        shift_pred.load(shift_pkl)
        models["shift_defect_predictor"] = shift_pred

    # Tier 1: Failure Mode Classifier
    fm_clf = FailureModeClassifier()
    fm_pkl = os.path.join(models_dir, "failure_mode_classifier.pkl")
    if os.path.exists(fm_pkl):
        fm_clf.load(fm_pkl)
        models["failure_mode_classifier"] = fm_clf

    # Tier 1: Prescriptive Action Engine
    prescriptive = PrescriptiveActionEngine()
    presc_pkl = os.path.join(models_dir, "prescriptive_action_engine.pkl")
    if os.path.exists(presc_pkl):
        prescriptive.load(presc_pkl)
    models["prescriptive_action_engine"] = prescriptive

    # Tier 2: Marginal Batch Classifier
    batch_clf = MarginalBatchRiskClassifier()
    batch_pkl = os.path.join(models_dir, "marginal_batch_classifier.pkl")
    if os.path.exists(batch_pkl):
        batch_clf.load(batch_pkl)
        models["marginal_batch_classifier"] = batch_clf

    # Tier 2: Downstream Impact Forecaster
    impact_forecaster = DownstreamImpactForecaster()
    impact_pkl = os.path.join(models_dir, "downstream_impact_forecaster.pkl")
    if os.path.exists(impact_pkl):
        impact_forecaster.load(impact_pkl)
        models["downstream_impact_forecaster"] = impact_forecaster

    # Tier 2: Adaptive Parameter Compensator
    compensator = AdaptiveParameterCompensator()
    comp_pkl = os.path.join(models_dir, "adaptive_parameter_compensator.pkl")
    if os.path.exists(comp_pkl):
        compensator.load(comp_pkl)
    models["adaptive_parameter_compensator"] = compensator

    # Tier 3: Supplier Analytics Engines (if data directory exists)
    if os.path.exists(data_dir):
        models["trend_analyzer"] = SupplierQualityTrendAnalyzer(data_dir=data_dir)
        models["trace_engine"] = WarrantyTraceabilityEngine(data_dir=data_dir)
        models["gap_analyzer"] = ToleranceLeakageAnalyzer(data_dir=data_dir)
        models["scar_generator"] = SCARGenerator(data_dir=data_dir)

    return models
