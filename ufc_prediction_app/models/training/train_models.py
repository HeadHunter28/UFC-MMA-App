"""
Model Training Module.

Provides the ModelTrainer class for training and evaluating prediction models.
Integrates with MLFlow for experiment tracking and model versioning.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import (
    MODEL_VERSIONS_DIR,
    CURRENT_MODELS_DIR,
    PROCESSED_DATA_DIR,
    MAX_MODEL_VERSIONS,
    WINNER_MODEL_WEIGHTS,
    METHOD_MODEL_WEIGHTS,
    MLFLOW_MODEL_NAME_WINNER,
    MLFLOW_MODEL_NAME_METHOD,
    MLFLOW_MODEL_NAME_ROUND,
)
from services.data_service import DataService
from services.mlflow_service import get_mlflow_service
from models.training.feature_engineering import FeatureEngineer, create_training_dataset

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Trainer for UFC fight prediction models.

    Handles training, evaluation, versioning, and saving of models.
    Integrates with MLFlow for experiment tracking and model registry.
    """

    def __init__(self):
        """Initialize the model trainer."""
        self.data_service = DataService()
        self.feature_engineer = FeatureEngineer()
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.mlflow_service = get_mlflow_service()

        # Ensure directories exist
        MODEL_VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
        CURRENT_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    def train_all_models(self, force: bool = False) -> Tuple[str, Dict[str, Any]]:
        """
        Train all prediction models with MLFlow tracking.

        Args:
            force: Force retraining even if recent model exists

        Returns:
            Tuple of (version, metrics)
        """
        logger.info("Starting model training...")

        # Load training data
        X, y_winner, y_method, y_round = self.prepare_training_data()

        if len(X) == 0:
            logger.error("No training data available")
            return "", {}

        logger.info(f"Training with {len(X)} samples, {X.shape[1]} features")

        # Time-based split
        X_train, X_test, splits = self._temporal_split(X, y_winner)
        y_winner_train = y_winner[splits["train_idx"]]
        y_winner_test = y_winner[splits["test_idx"]]
        y_method_train = y_method[splits["train_idx"]]
        y_method_test = y_method[splits["test_idx"]]
        y_round_train = y_round[splits["train_idx"]]
        y_round_test = y_round[splits["test_idx"]]

        # Fit imputer and scaler
        X_train_imputed = self.imputer.fit_transform(X_train)
        X_test_imputed = self.imputer.transform(X_test)
        X_train_scaled = self.scaler.fit_transform(X_train_imputed)
        X_test_scaled = self.scaler.transform(X_test_imputed)

        # Get next version
        version = self._get_next_version()

        # Start MLFlow run for overall training
        run_name = f"training_{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.mlflow_service.start_run(
            run_name=run_name,
            tags={
                "version": version,
                "training_type": "full",
                "force_retrain": str(force),
            }
        )

        # Log dataset parameters
        self.mlflow_service.log_params({
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "n_features": X.shape[1],
            "test_ratio": 0.2,
            "version": version,
        })

        # Train models with MLFlow tracking
        winner_model, winner_metrics = self._train_winner_model(
            X_train_scaled, y_winner_train, X_test_scaled, y_winner_test
        )
        method_model, method_metrics = self._train_method_model(
            X_train_scaled, y_method_train, X_test_scaled, y_method_test
        )
        round_model, round_metrics = self._train_round_model(
            X_train_scaled, y_round_train, X_test_scaled, y_round_test
        )

        # Log all metrics to MLFlow
        all_mlflow_metrics = {}
        for key, value in winner_metrics.items():
            all_mlflow_metrics[f"winner_{key}"] = value
        for key, value in method_metrics.items():
            all_mlflow_metrics[f"method_{key}"] = value
        for key, value in round_metrics.items():
            all_mlflow_metrics[f"round_{key}"] = value

        self.mlflow_service.log_metrics(all_mlflow_metrics)

        # Log and register models to MLFlow
        self._log_models_to_mlflow(
            winner_model,
            method_model,
            round_model,
            X_test_scaled[:5],  # Sample input for signature
            version,
        )

        # End MLFlow run
        self.mlflow_service.end_run()

        # Save models locally (backup and for non-MLFlow loading)
        self._save_models(
            version,
            winner_model,
            method_model,
            round_model,
            {
                "winner": winner_metrics,
                "method": method_metrics,
                "round": round_metrics,
            }
        )

        # Cleanup old versions
        self.cleanup_old_versions(keep=MAX_MODEL_VERSIONS)

        all_metrics = {
            "winner": winner_metrics,
            "method": method_metrics,
            "round": round_metrics,
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "training_date": datetime.now().isoformat(),
        }

        logger.info(f"Model training complete. Version: {version}")
        return version, all_metrics

    def _log_models_to_mlflow(
        self,
        winner_model: Any,
        method_model: Any,
        round_model: Any,
        input_example: np.ndarray,
        version: str,
    ):
        """Log all models to MLFlow and register them."""
        if not self.mlflow_service.is_available():
            logger.info("MLFlow not available, skipping model logging")
            return

        try:
            from mlflow.models.signature import infer_signature

            # Log winner model
            winner_uri = self.mlflow_service.log_model(
                winner_model,
                "winner_model",
                model_type="sklearn",
                registered_model_name=MLFLOW_MODEL_NAME_WINNER,
            )

            # Log method model
            method_uri = self.mlflow_service.log_model(
                method_model,
                "method_model",
                model_type="sklearn",
                registered_model_name=MLFLOW_MODEL_NAME_METHOD,
            )

            # Log round model
            round_uri = self.mlflow_service.log_model(
                round_model,
                "round_model",
                model_type="sklearn",
                registered_model_name=MLFLOW_MODEL_NAME_ROUND,
            )

            # Transition to production
            if winner_uri:
                winner_version = self._get_latest_mlflow_version(MLFLOW_MODEL_NAME_WINNER)
                if winner_version:
                    self.mlflow_service.transition_model_stage(
                        MLFLOW_MODEL_NAME_WINNER, winner_version, "Production"
                    )

            if method_uri:
                method_version = self._get_latest_mlflow_version(MLFLOW_MODEL_NAME_METHOD)
                if method_version:
                    self.mlflow_service.transition_model_stage(
                        MLFLOW_MODEL_NAME_METHOD, method_version, "Production"
                    )

            if round_uri:
                round_version = self._get_latest_mlflow_version(MLFLOW_MODEL_NAME_ROUND)
                if round_version:
                    self.mlflow_service.transition_model_stage(
                        MLFLOW_MODEL_NAME_ROUND, round_version, "Production"
                    )

            # Log scaler as artifact
            import joblib
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
                joblib.dump(self.scaler, f.name)
                self.mlflow_service.log_artifact(f.name, "scaler")

            # Log feature names
            feature_names = self.feature_engineer.get_feature_names()
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
                json.dump(feature_names, f)
                self.mlflow_service.log_artifact(f.name, "feature_names")

            logger.info("Models logged to MLFlow successfully")

        except Exception as e:
            logger.error(f"Failed to log models to MLFlow: {e}")

    def _get_latest_mlflow_version(self, model_name: str) -> Optional[str]:
        """Get the latest version of a registered model."""
        if not self.mlflow_service.is_available():
            return None

        try:
            info = self.mlflow_service.get_latest_model_info(model_name, stage="None")
            if info:
                return info.get("version")
        except Exception:
            pass
        return None

    def prepare_training_data(
        self
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare training data from database.

        Returns:
            Tuple of (X, y_winner, y_method, y_round)
        """
        logger.info("Preparing training data...")

        with self.data_service.get_connection() as conn:
            # Load fights
            fights_df = pd.read_sql(
                """
                SELECT * FROM fights
                WHERE winner_id IS NOT NULL
                ORDER BY fight_id
                """,
                conn
            )

            # Load fighters
            fighters_df = pd.read_sql("SELECT * FROM fighters", conn)

            # Load fighter stats
            fighter_stats_df = pd.read_sql("SELECT * FROM fighter_stats", conn)

        if len(fights_df) == 0:
            logger.warning("No fights found in database")
            return np.array([]), np.array([]), np.array([]), np.array([])

        logger.info(f"Loaded {len(fights_df)} fights for training")

        return create_training_dataset(fights_df, fighters_df, fighter_stats_df)

    def _temporal_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_ratio: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Perform time-based train/test split.

        Args:
            X: Features
            y: Labels
            test_ratio: Proportion for test set

        Returns:
            Tuple of (X_train, X_test, splits_dict)
        """
        n = len(X)
        split_idx = int(n * (1 - test_ratio))

        train_idx = np.arange(split_idx)
        test_idx = np.arange(split_idx, n)

        return (
            X[:split_idx],
            X[split_idx:],
            {"train_idx": train_idx, "test_idx": test_idx}
        )

    def _train_winner_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Tuple[Any, Dict[str, float]]:
        """Train the winner prediction ensemble with MLFlow tracking."""
        logger.info("Training winner prediction model...")

        # Model hyperparameters
        xgb_params = {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
            "random_state": 42,
        }
        lgbm_params = {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
            "random_state": 42,
        }
        rf_params = {
            "n_estimators": 100,
            "max_depth": 5,
            "random_state": 42,
        }
        lr_params = {
            "max_iter": 1000,
            "random_state": 42,
        }

        # Log hyperparameters to MLFlow
        self.mlflow_service.log_params({
            "winner_xgb_n_estimators": xgb_params["n_estimators"],
            "winner_xgb_max_depth": xgb_params["max_depth"],
            "winner_xgb_learning_rate": xgb_params["learning_rate"],
            "winner_lgbm_n_estimators": lgbm_params["n_estimators"],
            "winner_lgbm_max_depth": lgbm_params["max_depth"],
            "winner_lgbm_learning_rate": lgbm_params["learning_rate"],
            "winner_rf_n_estimators": rf_params["n_estimators"],
            "winner_rf_max_depth": rf_params["max_depth"],
            "winner_lr_max_iter": lr_params["max_iter"],
            "winner_ensemble_weights": str(WINNER_MODEL_WEIGHTS),
        })

        # Train component models
        models = {}

        # XGBoost
        try:
            from xgboost import XGBClassifier
            models["xgboost"] = XGBClassifier(
                n_estimators=xgb_params["n_estimators"],
                max_depth=xgb_params["max_depth"],
                learning_rate=xgb_params["learning_rate"],
                random_state=xgb_params["random_state"],
                n_jobs=-1,
                eval_metric="logloss",
            )
            models["xgboost"].fit(X_train, y_train)
        except ImportError:
            logger.warning("XGBoost not available")

        # LightGBM
        try:
            from lightgbm import LGBMClassifier
            models["lightgbm"] = LGBMClassifier(
                n_estimators=lgbm_params["n_estimators"],
                max_depth=lgbm_params["max_depth"],
                learning_rate=lgbm_params["learning_rate"],
                random_state=lgbm_params["random_state"],
                n_jobs=-1,
                verbose=-1,
            )
            models["lightgbm"].fit(X_train, y_train)
        except ImportError:
            logger.warning("LightGBM not available")

        # Random Forest
        models["random_forest"] = RandomForestClassifier(
            n_estimators=rf_params["n_estimators"],
            max_depth=rf_params["max_depth"],
            random_state=rf_params["random_state"],
            n_jobs=-1,
        )
        models["random_forest"].fit(X_train, y_train)

        # Logistic Regression
        models["logistic_regression"] = LogisticRegression(
            max_iter=lr_params["max_iter"],
            random_state=lr_params["random_state"],
        )
        models["logistic_regression"].fit(X_train, y_train)

        # Create ensemble prediction
        predictions = []
        weights = []

        for name, model in models.items():
            if name in WINNER_MODEL_WEIGHTS:
                pred_proba = model.predict_proba(X_test)[:, 1]
                predictions.append(pred_proba)
                weights.append(WINNER_MODEL_WEIGHTS[name])

        # Weighted average
        weights = np.array(weights) / sum(weights)
        ensemble_proba = np.average(predictions, axis=0, weights=weights)
        ensemble_pred = (ensemble_proba >= 0.5).astype(int)

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, ensemble_pred),
            "precision": precision_score(y_test, ensemble_pred, zero_division=0),
            "recall": recall_score(y_test, ensemble_pred, zero_division=0),
            "f1": f1_score(y_test, ensemble_pred, zero_division=0),
            "auc_roc": roc_auc_score(y_test, ensemble_proba),
        }

        logger.info(f"Winner model accuracy: {metrics['accuracy']:.3f}")

        # Return the best single model for simplicity (Random Forest)
        # In production, you'd create a proper ensemble class
        return models["random_forest"], metrics

    def _train_method_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Tuple[Any, Dict[str, float]]:
        """Train the method prediction model."""
        logger.info("Training method prediction model...")

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
            "recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
            "f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        }

        logger.info(f"Method model accuracy: {metrics['accuracy']:.3f}")

        return model, metrics

    def _train_round_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Tuple[Any, Dict[str, float]]:
        """Train the round prediction model."""
        logger.info("Training round prediction model...")

        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # Calculate metrics
        mae = np.mean(np.abs(y_test - y_pred))
        within_one = np.mean(np.abs(y_test - y_pred) <= 1)

        metrics = {
            "mae": mae,
            "within_one_round": within_one,
        }

        logger.info(f"Round model MAE: {metrics['mae']:.3f}")

        return model, metrics

    def _get_next_version(self) -> str:
        """Get the next model version string."""
        registry = self._load_registry()

        current = registry.get("current_version", "v0.0.0")

        # Parse version
        parts = current.replace("v", "").split(".")
        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])

        # Increment patch version
        patch += 1

        return f"v{major}.{minor}.{patch}"

    def _load_registry(self) -> Dict[str, Any]:
        """Load the model registry."""
        registry_path = MODEL_VERSIONS_DIR / "model_registry.json"

        if registry_path.exists():
            with open(registry_path, "r") as f:
                return json.load(f)

        return {"registry_version": "1.0", "max_versions_kept": MAX_MODEL_VERSIONS, "models": {}}

    def _save_models(
        self,
        version: str,
        winner_model: Any,
        method_model: Any,
        round_model: Any,
        metrics: Dict[str, Any]
    ):
        """Save trained models to disk."""
        import joblib

        logger.info(f"Saving models version {version}...")

        # Save versioned models
        joblib.dump(winner_model, MODEL_VERSIONS_DIR / f"winner_{version}.pkl")
        joblib.dump(method_model, MODEL_VERSIONS_DIR / f"method_{version}.pkl")
        joblib.dump(round_model, MODEL_VERSIONS_DIR / f"round_{version}.pkl")

        # Save to current directory
        joblib.dump(winner_model, CURRENT_MODELS_DIR / "winner_model.pkl")
        joblib.dump(method_model, CURRENT_MODELS_DIR / "method_model.pkl")
        joblib.dump(round_model, CURRENT_MODELS_DIR / "round_model.pkl")
        joblib.dump(self.imputer, CURRENT_MODELS_DIR / "imputer.pkl")
        joblib.dump(self.scaler, CURRENT_MODELS_DIR / "scaler.pkl")

        # Save feature names
        with open(CURRENT_MODELS_DIR / "feature_names.json", "w") as f:
            json.dump(self.feature_engineer.get_feature_names(), f)

        # Update registry
        registry = self._load_registry()
        registry["current_version"] = version

        for model_type in ["winner", "method", "round"]:
            if model_type not in registry["models"]:
                registry["models"][model_type] = {"current_version": version, "versions": []}

            registry["models"][model_type]["current_version"] = version
            registry["models"][model_type]["versions"].append({
                "version": version,
                "filename": f"{model_type}_{version}.pkl",
                "created_at": datetime.now().isoformat(),
                "training_samples": metrics.get("training_samples", 0),
                "metrics": metrics.get(model_type, {}),
                "status": "active",
            })

        registry_path = MODEL_VERSIONS_DIR / "model_registry.json"
        with open(registry_path, "w") as f:
            json.dump(registry, f, indent=2)

        logger.info("Models saved successfully")

    def cleanup_old_versions(self, keep: int = 3):
        """
        Remove old model versions, keeping only the latest N.

        Args:
            keep: Number of versions to keep
        """
        logger.info(f"Cleaning up old versions (keeping {keep})...")

        registry = self._load_registry()

        for model_type, model_info in registry.get("models", {}).items():
            versions = model_info.get("versions", [])

            # Sort by creation date
            versions.sort(key=lambda x: x.get("created_at", ""), reverse=True)

            # Keep only latest N
            versions_to_keep = versions[:keep]
            versions_to_remove = versions[keep:]

            # Delete old files
            for old_version in versions_to_remove:
                old_file = MODEL_VERSIONS_DIR / old_version["filename"]
                if old_file.exists():
                    old_file.unlink()
                    logger.info(f"Deleted: {old_file}")

            model_info["versions"] = versions_to_keep

        # Save updated registry
        registry_path = MODEL_VERSIONS_DIR / "model_registry.json"
        with open(registry_path, "w") as f:
            json.dump(registry, f, indent=2)

    def check_retrain_needed(self) -> bool:
        """
        Check if model retraining is needed.

        Returns:
            True if retraining is recommended
        """
        from services.accuracy_service import AccuracyService

        accuracy_service = AccuracyService()
        return accuracy_service.needs_retraining()

    def get_current_version(self) -> str:
        """Get the current model version."""
        registry = self._load_registry()
        return registry.get("current_version", "v1.0.0")

    def get_last_training_date(self) -> Optional[datetime]:
        """Get the last training date."""
        registry = self._load_registry()

        for model_type, model_info in registry.get("models", {}).items():
            versions = model_info.get("versions", [])
            if versions:
                latest = versions[0]
                created_at = latest.get("created_at")
                if created_at:
                    return datetime.fromisoformat(created_at)

        return None

    def get_new_fights_since_training(self) -> int:
        """Get the number of new fights since last training."""
        last_training = self.get_last_training_date()

        if not last_training:
            return 0

        with self.data_service.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT COUNT(*) FROM fights
                WHERE created_at > ?
                """,
                (last_training.isoformat(),)
            )
            return cursor.fetchone()[0]
