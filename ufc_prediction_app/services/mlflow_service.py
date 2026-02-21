"""
MLFlow Service Module.

Provides centralized MLFlow integration for experiment tracking,
model versioning, and registry operations.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_MODEL_NAME_WINNER,
    MLFLOW_MODEL_NAME_METHOD,
    MLFLOW_MODEL_NAME_ROUND,
    MLFLOW_ENABLED,
    CURRENT_MODELS_DIR,
)

logger = logging.getLogger(__name__)

# Lazy import MLFlow to avoid import errors if not installed
_mlflow = None
_MlflowClient = None


def _get_mlflow():
    """Lazy load MLFlow module."""
    global _mlflow, _MlflowClient
    if _mlflow is None:
        try:
            import mlflow
            from mlflow.tracking import MlflowClient
            _mlflow = mlflow
            _MlflowClient = MlflowClient
        except ImportError:
            logger.warning("MLFlow not installed. MLFlow features will be disabled.")
            return None, None
    return _mlflow, _MlflowClient


class MLFlowService:
    """
    Service for MLFlow experiment tracking and model registry.

    Handles all MLFlow operations including:
    - Experiment management
    - Run tracking (parameters, metrics, artifacts)
    - Model registration and versioning
    - Model loading for inference
    """

    def __init__(self):
        """Initialize the MLFlow service."""
        self.enabled = MLFLOW_ENABLED
        self.tracking_uri = MLFLOW_TRACKING_URI
        self.experiment_name = MLFLOW_EXPERIMENT_NAME
        self._initialized = False
        self._client = None

        if self.enabled:
            self._initialize()

    def _initialize(self):
        """Initialize MLFlow tracking."""
        mlflow, MlflowClient = _get_mlflow()

        if mlflow is None:
            self.enabled = False
            logger.warning("MLFlow not available. Falling back to local model storage.")
            return

        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            self._client = MlflowClient(self.tracking_uri)
            self._initialized = True
            logger.info(f"MLFlow initialized with tracking URI: {self.tracking_uri}")
        except Exception as e:
            logger.error(f"Failed to initialize MLFlow: {e}")
            self.enabled = False

    def is_available(self) -> bool:
        """Check if MLFlow is available and enabled."""
        return self.enabled and self._initialized

    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        nested: bool = False,
    ) -> Optional[Any]:
        """
        Start a new MLFlow run.

        Args:
            run_name: Optional name for the run
            tags: Optional tags for the run
            nested: Whether this is a nested run

        Returns:
            MLFlow run object or None
        """
        if not self.is_available():
            return None

        mlflow, _ = _get_mlflow()

        try:
            run = mlflow.start_run(run_name=run_name, nested=nested)

            if tags:
                for key, value in tags.items():
                    mlflow.set_tag(key, value)

            return run
        except Exception as e:
            logger.error(f"Failed to start MLFlow run: {e}")
            return None

    def end_run(self, status: str = "FINISHED"):
        """End the current MLFlow run."""
        if not self.is_available():
            return

        mlflow, _ = _get_mlflow()

        try:
            mlflow.end_run(status=status)
        except Exception as e:
            logger.error(f"Failed to end MLFlow run: {e}")

    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters to the current run.

        Args:
            params: Dictionary of parameters
        """
        if not self.is_available():
            return

        mlflow, _ = _get_mlflow()

        try:
            # MLFlow has a limit on param value length, truncate if needed
            for key, value in params.items():
                str_value = str(value)
                if len(str_value) > 500:
                    str_value = str_value[:497] + "..."
                mlflow.log_param(key, str_value)
        except Exception as e:
            logger.error(f"Failed to log parameters: {e}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics to the current run.

        Args:
            metrics: Dictionary of metrics
            step: Optional step number
        """
        if not self.is_available():
            return

        mlflow, _ = _get_mlflow()

        try:
            mlflow.log_metrics(metrics, step=step)
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Log an artifact file.

        Args:
            local_path: Path to the local file
            artifact_path: Optional path within the artifact directory
        """
        if not self.is_available():
            return

        mlflow, _ = _get_mlflow()

        try:
            mlflow.log_artifact(local_path, artifact_path)
        except Exception as e:
            logger.error(f"Failed to log artifact: {e}")

    def log_model(
        self,
        model: Any,
        model_name: str,
        model_type: str = "sklearn",
        signature: Optional[Any] = None,
        input_example: Optional[Any] = None,
        registered_model_name: Optional[str] = None,
    ) -> Optional[str]:
        """
        Log a model to MLFlow.

        Args:
            model: The model object
            model_name: Name for the model artifact
            model_type: Type of model (sklearn, xgboost, lightgbm)
            signature: Model signature
            input_example: Example input for the model
            registered_model_name: Name to register in model registry

        Returns:
            Model URI or None
        """
        if not self.is_available():
            return None

        mlflow, _ = _get_mlflow()

        try:
            if model_type == "sklearn":
                model_info = mlflow.sklearn.log_model(
                    model,
                    model_name,
                    signature=signature,
                    input_example=input_example,
                    registered_model_name=registered_model_name,
                )
            elif model_type == "xgboost":
                model_info = mlflow.xgboost.log_model(
                    model,
                    model_name,
                    signature=signature,
                    input_example=input_example,
                    registered_model_name=registered_model_name,
                )
            elif model_type == "lightgbm":
                model_info = mlflow.lightgbm.log_model(
                    model,
                    model_name,
                    signature=signature,
                    input_example=input_example,
                    registered_model_name=registered_model_name,
                )
            else:
                # Default to sklearn
                model_info = mlflow.sklearn.log_model(
                    model,
                    model_name,
                    signature=signature,
                    input_example=input_example,
                    registered_model_name=registered_model_name,
                )

            return model_info.model_uri
        except Exception as e:
            logger.error(f"Failed to log model: {e}")
            return None

    def register_model(
        self,
        model_uri: str,
        name: str,
        tags: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        """
        Register a model in the MLFlow Model Registry.

        Args:
            model_uri: URI of the logged model
            name: Name for the registered model
            tags: Optional tags for the model version

        Returns:
            Model version or None
        """
        if not self.is_available():
            return None

        mlflow, _ = _get_mlflow()

        try:
            result = mlflow.register_model(model_uri, name)

            if tags and self._client:
                for key, value in tags.items():
                    self._client.set_model_version_tag(
                        name, result.version, key, value
                    )

            return result.version
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            return None

    def transition_model_stage(
        self,
        model_name: str,
        version: str,
        stage: str = "Production",
        archive_existing: bool = True,
    ):
        """
        Transition a model version to a new stage.

        Args:
            model_name: Name of the registered model
            version: Version to transition
            stage: Target stage (Staging, Production, Archived)
            archive_existing: Whether to archive existing models in the stage
        """
        if not self.is_available() or not self._client:
            return

        try:
            self._client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage,
                archive_existing_versions=archive_existing,
            )
            logger.info(f"Transitioned {model_name} v{version} to {stage}")
        except Exception as e:
            logger.error(f"Failed to transition model stage: {e}")

    def load_model(
        self,
        model_name: str,
        stage: str = "Production",
        version: Optional[str] = None,
    ) -> Optional[Any]:
        """
        Load a model from the MLFlow Model Registry.

        Args:
            model_name: Name of the registered model
            stage: Stage to load from (if version not specified)
            version: Specific version to load

        Returns:
            Loaded model or None
        """
        if not self.is_available():
            return None

        mlflow, _ = _get_mlflow()

        try:
            if version:
                model_uri = f"models:/{model_name}/{version}"
            else:
                model_uri = f"models:/{model_name}/{stage}"

            # Try loading as sklearn first, then other formats
            try:
                return mlflow.sklearn.load_model(model_uri)
            except Exception:
                pass

            try:
                return mlflow.xgboost.load_model(model_uri)
            except Exception:
                pass

            try:
                return mlflow.lightgbm.load_model(model_uri)
            except Exception:
                pass

            # Generic pyfunc load as fallback
            return mlflow.pyfunc.load_model(model_uri)
        except Exception as e:
            logger.warning(f"Failed to load model from MLFlow: {e}")
            return None

    def get_latest_model_info(
        self,
        model_name: str,
        stage: str = "Production",
    ) -> Optional[Dict[str, Any]]:
        """
        Get information about the latest model version.

        Args:
            model_name: Name of the registered model
            stage: Stage to query

        Returns:
            Dictionary with model information or None
        """
        if not self.is_available() or not self._client:
            return None

        try:
            versions = self._client.get_latest_versions(model_name, stages=[stage])

            if not versions:
                # Try getting any version
                versions = self._client.search_model_versions(f"name='{model_name}'")
                if versions:
                    versions = sorted(versions, key=lambda x: int(x.version), reverse=True)

            if not versions:
                return None

            latest = versions[0]

            # Get run info for metrics
            run_info = None
            if latest.run_id:
                run = self._client.get_run(latest.run_id)
                run_info = {
                    "params": run.data.params,
                    "metrics": run.data.metrics,
                    "tags": run.data.tags,
                    "start_time": datetime.fromtimestamp(run.info.start_time / 1000),
                    "end_time": datetime.fromtimestamp(run.info.end_time / 1000) if run.info.end_time else None,
                }

            return {
                "name": latest.name,
                "version": latest.version,
                "stage": latest.current_stage,
                "status": latest.status,
                "created_at": datetime.fromtimestamp(latest.creation_timestamp / 1000),
                "updated_at": datetime.fromtimestamp(latest.last_updated_timestamp / 1000),
                "run_id": latest.run_id,
                "source": latest.source,
                "run_info": run_info,
            }
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return None

    def get_all_model_info(self) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Get information about all registered models.

        Returns:
            Dictionary with model information for winner, method, and round models
        """
        return {
            "winner": self.get_latest_model_info(MLFLOW_MODEL_NAME_WINNER),
            "method": self.get_latest_model_info(MLFLOW_MODEL_NAME_METHOD),
            "round": self.get_latest_model_info(MLFLOW_MODEL_NAME_ROUND),
        }

    def get_experiment_runs(
        self,
        max_results: int = 10,
        order_by: str = "start_time DESC",
    ) -> List[Dict[str, Any]]:
        """
        Get recent experiment runs.

        Args:
            max_results: Maximum number of runs to return
            order_by: Sort order

        Returns:
            List of run information dictionaries
        """
        if not self.is_available() or not self._client:
            return []

        mlflow, _ = _get_mlflow()

        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if not experiment:
                return []

            runs = self._client.search_runs(
                experiment_ids=[experiment.experiment_id],
                max_results=max_results,
                order_by=[order_by],
            )

            return [
                {
                    "run_id": run.info.run_id,
                    "run_name": run.info.run_name,
                    "status": run.info.status,
                    "start_time": datetime.fromtimestamp(run.info.start_time / 1000),
                    "end_time": datetime.fromtimestamp(run.info.end_time / 1000) if run.info.end_time else None,
                    "metrics": run.data.metrics,
                    "params": run.data.params,
                    "tags": run.data.tags,
                }
                for run in runs
            ]
        except Exception as e:
            logger.error(f"Failed to get experiment runs: {e}")
            return []

    def get_status(self) -> Dict[str, Any]:
        """
        Get MLFlow service status.

        Returns:
            Status dictionary
        """
        return {
            "enabled": self.enabled,
            "initialized": self._initialized,
            "tracking_uri": self.tracking_uri,
            "experiment_name": self.experiment_name,
            "models": {
                "winner": MLFLOW_MODEL_NAME_WINNER,
                "method": MLFLOW_MODEL_NAME_METHOD,
                "round": MLFLOW_MODEL_NAME_ROUND,
            },
        }


# Singleton instance
_mlflow_service = None


def get_mlflow_service() -> MLFlowService:
    """Get the singleton MLFlow service instance."""
    global _mlflow_service
    if _mlflow_service is None:
        _mlflow_service = MLFlowService()
    return _mlflow_service
