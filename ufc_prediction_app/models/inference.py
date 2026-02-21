"""
Model Inference Module.

Provides the PredictionEngine class for generating fight predictions
using trained ML models. Supports loading from MLFlow Model Registry
or local files.
"""

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    CURRENT_MODELS_DIR,
    MODEL_VERSIONS_DIR,
    MODEL_VERSION,
    MLFLOW_MODEL_NAME_WINNER,
    MLFLOW_MODEL_NAME_METHOD,
    MLFLOW_MODEL_NAME_ROUND,
    get_confidence_level,
    get_confidence_color,
)
from services.mlflow_service import get_mlflow_service

logger = logging.getLogger(__name__)


@dataclass
class PredictionOutput:
    """Structured output for fight predictions."""
    predicted_winner_id: int
    predicted_winner_name: str
    winner_confidence: float
    confidence_level: str
    confidence_color: str

    method_ko_prob: float
    method_sub_prob: float
    method_dec_prob: float
    predicted_method: str

    predicted_round: float

    feature_importance: Dict[str, float]
    top_factors: List[str]
    shap_values: Optional[Dict[str, float]]

    model_version: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class PredictionEngine:
    """
    Engine for generating fight predictions using trained models.

    Handles model loading (from MLFlow or local files), feature transformation,
    and prediction generation.
    """

    METHOD_LABELS = ["KO/TKO", "Submission", "Decision"]

    def __init__(self, models_dir: Optional[Path] = None, use_mlflow: bool = True):
        """
        Initialize the prediction engine.

        Args:
            models_dir: Directory containing trained models (fallback)
            use_mlflow: Whether to try loading from MLFlow first
        """
        self.models_dir = models_dir or CURRENT_MODELS_DIR
        self.model_version = MODEL_VERSION
        self.use_mlflow = use_mlflow
        self.mlflow_service = get_mlflow_service()
        self._loaded_from_mlflow = False

        # Model components
        self.winner_model = None
        self.method_model = None
        self.round_model = None
        self.scaler = None
        self.feature_names: List[str] = []

        # SHAP explainer (optional)
        self.shap_explainer = None

        self._load_models()

    def _load_models(self):
        """Load all trained models from MLFlow or local disk."""
        # Try MLFlow first if enabled
        if self.use_mlflow and self.mlflow_service.is_available():
            mlflow_loaded = self._load_models_from_mlflow()
            if mlflow_loaded:
                self._loaded_from_mlflow = True
                logger.info("Models loaded from MLFlow")
                self._load_shap_explainer()
                return

        # Fallback to local files
        self._load_models_from_disk()

    def _load_models_from_mlflow(self) -> bool:
        """
        Try to load models from MLFlow Model Registry.

        Returns:
            True if all models loaded successfully
        """
        try:
            # Load winner model
            self.winner_model = self.mlflow_service.load_model(
                MLFLOW_MODEL_NAME_WINNER, stage="Production"
            )
            if self.winner_model:
                logger.info("Loaded winner model from MLFlow")

            # Load method model
            self.method_model = self.mlflow_service.load_model(
                MLFLOW_MODEL_NAME_METHOD, stage="Production"
            )
            if self.method_model:
                logger.info("Loaded method model from MLFlow")

            # Load round model
            self.round_model = self.mlflow_service.load_model(
                MLFLOW_MODEL_NAME_ROUND, stage="Production"
            )
            if self.round_model:
                logger.info("Loaded round model from MLFlow")

            # Get model version from MLFlow
            winner_info = self.mlflow_service.get_latest_model_info(
                MLFLOW_MODEL_NAME_WINNER
            )
            if winner_info:
                self.model_version = f"v{winner_info.get('version', '1')}"

            # Still need to load scaler and feature names from local
            # (these are logged as artifacts but simpler to load locally)
            self._load_auxiliary_files()

            return self.winner_model is not None

        except Exception as e:
            logger.warning(f"Failed to load models from MLFlow: {e}")
            return False

    def _load_auxiliary_files(self):
        """Load scaler and feature names from local files."""
        try:
            import joblib

            # Load scaler
            scaler_path = self.models_dir / "scaler.pkl"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                logger.info("Loaded feature scaler")

            # Load feature names
            feature_names_path = self.models_dir / "feature_names.json"
            if feature_names_path.exists():
                with open(feature_names_path, "r") as f:
                    self.feature_names = json.load(f)
                logger.info(f"Loaded {len(self.feature_names)} feature names")

        except Exception as e:
            logger.warning(f"Error loading auxiliary files: {e}")

    def _load_models_from_disk(self):
        """Load all trained models from local disk."""
        try:
            import joblib

            # Load winner model
            winner_path = self.models_dir / "winner_model.pkl"
            if winner_path.exists():
                self.winner_model = joblib.load(winner_path)
                logger.info("Loaded winner prediction model from disk")

            # Load method model
            method_path = self.models_dir / "method_model.pkl"
            if method_path.exists():
                self.method_model = joblib.load(method_path)
                logger.info("Loaded method prediction model from disk")

            # Load round model
            round_path = self.models_dir / "round_model.pkl"
            if round_path.exists():
                self.round_model = joblib.load(round_path)
                logger.info("Loaded round prediction model from disk")

            # Load scaler and feature names
            self._load_auxiliary_files()

            # Try to load SHAP explainer
            self._load_shap_explainer()

        except Exception as e:
            logger.error(f"Error loading models from disk: {e}")

    def _load_shap_explainer(self):
        """Load or create SHAP explainer for model explanations."""
        try:
            import shap

            explainer_path = self.models_dir / "shap_explainer.pkl"
            if explainer_path.exists():
                import joblib
                self.shap_explainer = joblib.load(explainer_path)
                logger.info("Loaded SHAP explainer")
            elif self.winner_model is not None:
                # Create new explainer
                self.shap_explainer = shap.TreeExplainer(self.winner_model)
                logger.info("Created new SHAP explainer")

        except ImportError:
            logger.warning("SHAP not available - explanations will be limited")
        except Exception as e:
            logger.warning(f"Could not create SHAP explainer: {e}")

    def is_ready(self) -> bool:
        """Check if the engine is ready for predictions."""
        return self.winner_model is not None

    def predict(
        self,
        features: np.ndarray,
        fighter_a_id: int,
        fighter_b_id: int,
        fighter_a_name: str,
        fighter_b_name: str,
    ) -> PredictionOutput:
        """
        Generate a prediction from features.

        Args:
            features: Feature array
            fighter_a_id: ID of fighter A (red corner)
            fighter_b_id: ID of fighter B (blue corner)
            fighter_a_name: Name of fighter A
            fighter_b_name: Name of fighter B

        Returns:
            PredictionOutput with all prediction details
        """
        if not self.is_ready():
            raise RuntimeError("Prediction engine not ready - models not loaded")

        # Scale features
        if self.scaler is not None:
            features_scaled = self.scaler.transform([features])[0]
        else:
            features_scaled = features

        # Winner prediction
        winner_proba = self.winner_model.predict_proba([features_scaled])[0]
        fighter_a_prob = winner_proba[1]  # Probability of fighter A winning

        if fighter_a_prob >= 0.5:
            predicted_winner_id = fighter_a_id
            predicted_winner_name = fighter_a_name
            confidence = fighter_a_prob
        else:
            predicted_winner_id = fighter_b_id
            predicted_winner_name = fighter_b_name
            confidence = 1 - fighter_a_prob

        # Method prediction
        if self.method_model is not None:
            method_proba = self.method_model.predict_proba([features_scaled])[0]
        else:
            method_proba = np.array([0.35, 0.20, 0.45])

        predicted_method = self.METHOD_LABELS[np.argmax(method_proba)]

        # Round prediction
        if self.round_model is not None:
            predicted_round = float(self.round_model.predict([features_scaled])[0])
        else:
            predicted_round = 2.5

        # Feature importance
        feature_importance = self._calculate_feature_importance(features_scaled)

        # Top factors
        top_factors = self._generate_top_factors(
            feature_importance,
            fighter_a_name,
            fighter_b_name,
            features
        )

        # SHAP values
        shap_values = self._calculate_shap_values(features_scaled)

        return PredictionOutput(
            predicted_winner_id=predicted_winner_id,
            predicted_winner_name=predicted_winner_name,
            winner_confidence=float(confidence),
            confidence_level=get_confidence_level(confidence),
            confidence_color=get_confidence_color(confidence),
            method_ko_prob=float(method_proba[0]),
            method_sub_prob=float(method_proba[1]),
            method_dec_prob=float(method_proba[2]),
            predicted_method=predicted_method,
            predicted_round=predicted_round,
            feature_importance=feature_importance,
            top_factors=top_factors,
            shap_values=shap_values,
            model_version=self.model_version,
        )

    def _calculate_feature_importance(
        self,
        features: np.ndarray
    ) -> Dict[str, float]:
        """Calculate feature importance for the prediction."""
        importance = {}

        if hasattr(self.winner_model, "feature_importances_"):
            importances = self.winner_model.feature_importances_
        elif hasattr(self.winner_model, "coef_"):
            importances = np.abs(self.winner_model.coef_[0])
        else:
            importances = np.ones(len(features)) / len(features)

        names = self.feature_names or [f"feature_{i}" for i in range(len(features))]

        for name, imp in zip(names, importances):
            importance[name] = float(imp)

        # Sort by importance and return top 10
        sorted_importance = dict(
            sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
        )

        return sorted_importance

    def _calculate_shap_values(
        self,
        features: np.ndarray
    ) -> Optional[Dict[str, float]]:
        """Calculate SHAP values for the prediction."""
        if self.shap_explainer is None:
            return None

        try:
            shap_values = self.shap_explainer.shap_values([features])

            # For binary classification, get values for positive class
            if isinstance(shap_values, list):
                values = shap_values[1][0]
            else:
                values = shap_values[0]

            names = self.feature_names or [f"feature_{i}" for i in range(len(features))]

            shap_dict = {}
            for name, val in zip(names, values):
                shap_dict[name] = float(val)

            # Sort by absolute value
            return dict(
                sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
            )

        except Exception as e:
            logger.warning(f"SHAP calculation failed: {e}")
            return None

    def _generate_top_factors(
        self,
        feature_importance: Dict[str, float],
        fighter_a_name: str,
        fighter_b_name: str,
        features: np.ndarray
    ) -> List[str]:
        """Generate human-readable top factors."""
        factors = []

        feature_names = self.feature_names or []
        feature_dict = {name: features[i] for i, name in enumerate(feature_names)}

        for feature, importance in list(feature_importance.items())[:5]:
            if importance < 0.01:
                continue

            value = feature_dict.get(feature, 0)

            if "height_diff" in feature:
                if value > 0:
                    factors.append(f"{fighter_a_name} has a height advantage")
                else:
                    factors.append(f"{fighter_b_name} has a height advantage")

            elif "reach_diff" in feature:
                if value > 0:
                    factors.append(f"{fighter_a_name} has longer reach")
                else:
                    factors.append(f"{fighter_b_name} has longer reach")

            elif "sig_str_acc" in feature:
                factors.append("Striking accuracy is a significant factor")

            elif "sig_str_def" in feature:
                factors.append("Defensive striking ability matters")

            elif "td_acc" in feature or "td_avg" in feature:
                factors.append("Takedown ability is relevant to this matchup")

            elif "td_def" in feature:
                factors.append("Takedown defense is a key differentiator")

            elif "win_rate" in feature:
                factors.append("Historical win rates influence the prediction")

            elif "experience" in feature:
                factors.append("Experience level is a factor")

            elif "finish_rate" in feature:
                factors.append("Finishing ability is important")

        return factors[:5]

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        info = {
            "version": self.model_version,
            "winner_model_loaded": self.winner_model is not None,
            "method_model_loaded": self.method_model is not None,
            "round_model_loaded": self.round_model is not None,
            "scaler_loaded": self.scaler is not None,
            "shap_available": self.shap_explainer is not None,
            "feature_count": len(self.feature_names),
            "models_dir": str(self.models_dir),
            "loaded_from_mlflow": self._loaded_from_mlflow,
        }

        # Add MLFlow model info if available
        if self._loaded_from_mlflow and self.mlflow_service.is_available():
            info["mlflow_models"] = self.mlflow_service.get_all_model_info()

        return info


def get_model_registry() -> Dict[str, Any]:
    """
    Load and return the model registry.

    Returns:
        Model registry dict or empty dict if not found
    """
    registry_path = MODEL_VERSIONS_DIR / "model_registry.json"

    if registry_path.exists():
        try:
            with open(registry_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load model registry: {e}")

    return {}


def get_current_model_version() -> str:
    """Get the current active model version."""
    registry = get_model_registry()
    return registry.get("current_version", MODEL_VERSION)
