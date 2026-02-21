"""
Prediction Service Module.

Orchestrates ML predictions using trained models and feature engineering.
Includes proper form/momentum calculation and stats snapshot for ground truth collection.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    MODEL_VERSION,
    MIN_CONFIDENCE_THRESHOLD,
    get_confidence_level,
    CURRENT_MODELS_DIR,
)
from services.data_service import DataService

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Container for prediction results."""
    predicted_winner_id: int
    winner_confidence: float
    method_ko_prob: float
    method_sub_prob: float
    method_dec_prob: float
    predicted_method: str
    predicted_round: float
    confidence_level: str
    feature_importance: Dict[str, float]
    top_factors: List[str]
    model_version: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "predicted_winner_id": self.predicted_winner_id,
            "winner_confidence": self.winner_confidence,
            "method_ko_prob": self.method_ko_prob,
            "method_sub_prob": self.method_sub_prob,
            "method_dec_prob": self.method_dec_prob,
            "predicted_method": self.predicted_method,
            "predicted_round": self.predicted_round,
            "confidence_level": self.confidence_level,
            "feature_importance": json.dumps(self.feature_importance),
            "top_factors": json.dumps(self.top_factors),
            "model_version": self.model_version,
        }


class PredictionService:
    """
    Service for generating fight predictions.

    Handles loading models, feature engineering, and prediction generation.
    Includes proper form/momentum calculation and stats snapshot for ground truth.
    """

    def __init__(self):
        """Initialize the prediction service."""
        self.data_service = DataService()
        self.model_version = MODEL_VERSION
        self.models_loaded = False

        # Model placeholders
        self.winner_model = None
        self.method_model = None
        self.round_model = None
        self.scaler = None
        self.feature_names = []

        # Cache for form data to save with stats snapshot
        self._last_form_data = None

        # Try to load models
        self._load_models()

    def _load_models(self):
        """Load trained models from disk."""
        try:
            import joblib

            winner_path = CURRENT_MODELS_DIR / "winner_model.pkl"
            method_path = CURRENT_MODELS_DIR / "method_model.pkl"
            round_path = CURRENT_MODELS_DIR / "round_model.pkl"
            scaler_path = CURRENT_MODELS_DIR / "scaler.pkl"

            if winner_path.exists():
                self.winner_model = joblib.load(winner_path)
                logger.info("Loaded winner model")

            if method_path.exists():
                self.method_model = joblib.load(method_path)
                logger.info("Loaded method model")

            if round_path.exists():
                self.round_model = joblib.load(round_path)
                logger.info("Loaded round model")

            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                logger.info("Loaded scaler")

            # Load feature names
            feature_names_path = CURRENT_MODELS_DIR / "feature_names.json"
            if feature_names_path.exists():
                with open(feature_names_path, "r") as f:
                    self.feature_names = json.load(f)

            self.models_loaded = all([
                self.winner_model is not None,
                self.method_model is not None,
            ])

            if self.models_loaded:
                logger.info("All models loaded successfully")
            else:
                logger.warning("Some models not found - predictions may be limited")

        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            self.models_loaded = False

    def is_available(self) -> bool:
        """Check if prediction service is available."""
        return self.models_loaded

    def predict(
        self,
        fighter_a_id: int,
        fighter_b_id: int,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[PredictionResult]:
        """
        Generate prediction for a matchup.

        Args:
            fighter_a_id: ID of fighter A (red corner)
            fighter_b_id: ID of fighter B (blue corner)
            context: Optional context (weight_class, is_title_fight, etc.)

        Returns:
            PredictionResult or None if prediction fails
        """
        if not self.is_available():
            logger.warning("Prediction service not available - models not loaded")
            return self._generate_heuristic_prediction(fighter_a_id, fighter_b_id, context)

        try:
            # Get fighter data
            fighter_a = self.data_service.get_fighter_by_id(fighter_a_id)
            fighter_b = self.data_service.get_fighter_by_id(fighter_b_id)

            if not fighter_a or not fighter_b:
                logger.error("Could not find one or both fighters")
                return None

            # Create features
            features = self._create_features(fighter_a, fighter_b, context or {})

            # Scale features
            if self.scaler:
                features_scaled = self.scaler.transform([features])[0]
            else:
                features_scaled = features

            # Predict winner
            winner_proba = self.winner_model.predict_proba([features_scaled])[0]
            fighter_a_prob = winner_proba[1]  # Probability of fighter A winning

            if fighter_a_prob >= 0.5:
                predicted_winner_id = fighter_a_id
                confidence = fighter_a_prob
            else:
                predicted_winner_id = fighter_b_id
                confidence = 1 - fighter_a_prob

            # Predict method
            method_proba = self.method_model.predict_proba([features_scaled])[0]
            method_labels = ["KO/TKO", "Submission", "Decision"]
            predicted_method = method_labels[np.argmax(method_proba)]

            # Predict round
            if self.round_model:
                predicted_round = float(self.round_model.predict([features_scaled])[0])
            else:
                predicted_round = 2.5  # Default to mid-fight

            # Calculate feature importance
            feature_importance = self._get_feature_importance(features_scaled)
            top_factors = self._get_top_factors(feature_importance, fighter_a, fighter_b)

            return PredictionResult(
                predicted_winner_id=predicted_winner_id,
                winner_confidence=float(confidence),
                method_ko_prob=float(method_proba[0]),
                method_sub_prob=float(method_proba[1]),
                method_dec_prob=float(method_proba[2]),
                predicted_method=predicted_method,
                predicted_round=predicted_round,
                confidence_level=get_confidence_level(confidence),
                feature_importance=feature_importance,
                top_factors=top_factors,
                model_version=self.model_version,
            )

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return None

    def _create_features(
        self,
        fighter_a: Dict[str, Any],
        fighter_b: Dict[str, Any],
        context: Dict[str, Any]
    ) -> np.ndarray:
        """
        Create feature vector for a matchup.

        Features MUST be in the exact order defined in config.ALL_FEATURES
        to match the scaler's expected input.

        Args:
            fighter_a: Fighter A data
            fighter_b: Fighter B data
            context: Context data

        Returns:
            Feature array
        """
        # Calculate form features from fight history
        form_a = self._calculate_fighter_form(fighter_a.get("fighter_id"))
        form_b = self._calculate_fighter_form(fighter_b.get("fighter_id"))

        # Store form data for stats snapshot
        self._last_form_data = {"red": form_a, "blue": form_b}

        # Record info for ratio features
        a_wins = fighter_a.get("wins") or 1
        a_losses = fighter_a.get("losses") or 0
        b_wins = fighter_b.get("wins") or 1
        b_losses = fighter_b.get("losses") or 0

        a_win_rate = a_wins / max(a_wins + a_losses, 1)
        b_win_rate = b_wins / max(b_wins + b_losses, 1)

        # Build features in EXACT order matching config.ALL_FEATURES:
        # DIFFERENTIAL_FEATURES + RATIO_FEATURES + FORM_FEATURES + CONTEXTUAL_FEATURES
        features = [
            # === DIFFERENTIAL_FEATURES (11) ===
            # height_diff
            (fighter_a.get("height_cm") or 175) - (fighter_b.get("height_cm") or 175),
            # reach_diff
            (fighter_a.get("reach_cm") or 180) - (fighter_b.get("reach_cm") or 180),
            # age_diff
            0,  # Default age diff
            # sig_str_acc_diff
            (fighter_a.get("sig_strike_accuracy") or 0.45) - (fighter_b.get("sig_strike_accuracy") or 0.45),
            # sig_str_def_diff
            (fighter_a.get("sig_strike_defense") or 0.55) - (fighter_b.get("sig_strike_defense") or 0.55),
            # td_acc_diff
            (fighter_a.get("takedown_accuracy") or 0.40) - (fighter_b.get("takedown_accuracy") or 0.40),
            # td_def_diff
            (fighter_a.get("takedown_defense") or 0.60) - (fighter_b.get("takedown_defense") or 0.60),
            # str_landed_pm_diff
            (fighter_a.get("sig_strikes_landed_per_min") or 3.5) - (fighter_b.get("sig_strikes_landed_per_min") or 3.5),
            # str_absorbed_pm_diff
            (fighter_a.get("sig_strikes_absorbed_per_min") or 3.0) - (fighter_b.get("sig_strikes_absorbed_per_min") or 3.0),
            # td_avg_diff
            (fighter_a.get("takedowns_avg_per_15min") or 1.5) - (fighter_b.get("takedowns_avg_per_15min") or 1.5),
            # sub_avg_diff
            (fighter_a.get("submissions_avg_per_15min") or 0.5) - (fighter_b.get("submissions_avg_per_15min") or 0.5),

            # === RATIO_FEATURES (6) ===
            # win_rate_ratio
            a_win_rate / max(b_win_rate, 0.01),
            # finish_rate_ratio
            (fighter_a.get("finish_rate") or 0.5) / max(fighter_b.get("finish_rate") or 0.5, 0.01),
            # ko_rate_ratio
            (fighter_a.get("ko_rate") or 0.3) / max(fighter_b.get("ko_rate") or 0.3, 0.01),
            # sub_rate_ratio
            (fighter_a.get("submission_rate") or 0.2) / max(fighter_b.get("submission_rate") or 0.2, 0.01),
            # experience_ratio
            (a_wins + a_losses) / max(b_wins + b_losses, 1),
            # ufc_experience_ratio
            (a_wins + a_losses) / max(b_wins + b_losses, 1),

            # === FORM_FEATURES (10) ===
            # win_streak_a
            form_a["win_streak"],
            # win_streak_b
            form_b["win_streak"],
            # lose_streak_a
            form_a["loss_streak"],
            # lose_streak_b
            form_b["loss_streak"],
            # recent_form_a
            form_a["recent_form_score"],
            # recent_form_b
            form_b["recent_form_score"],
            # days_since_fight_a
            form_a["days_since_last_fight"],
            # days_since_fight_b
            form_b["days_since_last_fight"],
            # momentum_a
            form_a["momentum"],
            # momentum_b
            form_b["momentum"],

            # === CONTEXTUAL_FEATURES (4) ===
            # weight_class_encoded
            0,  # Simplified encoding
            # is_title_fight
            int(context.get("is_title_fight", False)),
            # is_main_event
            int(context.get("is_main_event", False)),
            # rounds_scheduled
            context.get("rounds_scheduled", 3),
        ]

        return np.array(features)

    def _calculate_fighter_form(
        self,
        fighter_id: int,
        num_recent_fights: int = 5
    ) -> Dict[str, Any]:
        """
        Calculate form/momentum features from fight history.

        Args:
            fighter_id: Fighter's database ID
            num_recent_fights: Number of recent fights to consider

        Returns:
            Dict with form metrics
        """
        default_form = {
            "win_streak": 0,
            "loss_streak": 0,
            "recent_form_score": 0.5,
            "days_since_last_fight": 365,
            "momentum": 0.0,
            "recent_wins": 0,
            "recent_losses": 0,
        }

        try:
            # Get recent fight history
            fight_history = self.data_service.get_fighter_fight_history(
                fighter_id, limit=num_recent_fights
            )

            if not fight_history:
                return default_form

            # Calculate win/loss streak
            win_streak = 0
            loss_streak = 0

            for fight in fight_history:
                result = fight.get("result", "")
                if result == "Win":
                    if loss_streak == 0:
                        win_streak += 1
                    else:
                        break
                elif result == "Loss":
                    if win_streak == 0:
                        loss_streak += 1
                    else:
                        break
                else:
                    # Draw or NC - break streak calculation
                    break

            # Calculate recent form score (weighted recent results)
            weights = [1.0, 0.8, 0.6, 0.4, 0.2]  # More weight to recent fights
            total_weight = 0
            weighted_score = 0

            for i, fight in enumerate(fight_history[:5]):
                weight = weights[i] if i < len(weights) else 0.2
                total_weight += weight

                result = fight.get("result", "")
                if result == "Win":
                    weighted_score += weight * 1.0
                elif result == "Loss":
                    weighted_score += weight * 0.0
                else:
                    weighted_score += weight * 0.5  # Draw/NC

            recent_form_score = weighted_score / total_weight if total_weight > 0 else 0.5

            # Calculate days since last fight
            days_since_last = 365
            if fight_history and fight_history[0].get("event_date"):
                try:
                    last_fight_date = datetime.fromisoformat(
                        str(fight_history[0]["event_date"])
                    )
                    days_since_last = (datetime.now() - last_fight_date).days
                except (ValueError, TypeError):
                    pass

            # Calculate momentum (combining streak and form)
            momentum = (win_streak - loss_streak) * 0.3 + (recent_form_score - 0.5) * 2

            # Count recent wins/losses
            recent_wins = sum(1 for f in fight_history[:5] if f.get("result") == "Win")
            recent_losses = sum(1 for f in fight_history[:5] if f.get("result") == "Loss")

            return {
                "win_streak": win_streak,
                "loss_streak": loss_streak,
                "recent_form_score": recent_form_score,
                "days_since_last_fight": days_since_last,
                "momentum": momentum,
                "recent_wins": recent_wins,
                "recent_losses": recent_losses,
            }

        except Exception as e:
            logger.warning(f"Error calculating fighter form: {e}")
            return default_form

    def _get_feature_importance(self, features: np.ndarray) -> Dict[str, float]:
        """Get feature importance for the prediction."""
        importance = {}

        if hasattr(self.winner_model, "feature_importances_"):
            importances = self.winner_model.feature_importances_
        elif hasattr(self.winner_model, "coef_"):
            importances = np.abs(self.winner_model.coef_[0])
        else:
            # Default equal importance
            importances = np.ones(len(features)) / len(features)

        feature_names = self.feature_names or [f"feature_{i}" for i in range(len(features))]

        for name, imp in zip(feature_names, importances):
            importance[name] = float(imp)

        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10])

    def _get_top_factors(
        self,
        feature_importance: Dict[str, float],
        fighter_a: Dict[str, Any],
        fighter_b: Dict[str, Any]
    ) -> List[str]:
        """Generate human-readable top factors for the prediction."""
        factors = []

        for feature, importance in list(feature_importance.items())[:5]:
            if importance < 0.01:
                continue

            if "height_diff" in feature:
                height_a = fighter_a.get("height_cm") or 175
                height_b = fighter_b.get("height_cm") or 175
                if height_a > height_b:
                    factors.append(f"{fighter_a['name']} has a height advantage ({height_a - height_b:.0f} cm)")
                else:
                    factors.append(f"{fighter_b['name']} has a height advantage ({height_b - height_a:.0f} cm)")

            elif "reach_diff" in feature:
                reach_a = fighter_a.get("reach_cm") or 180
                reach_b = fighter_b.get("reach_cm") or 180
                if reach_a > reach_b:
                    factors.append(f"{fighter_a['name']} has longer reach ({reach_a - reach_b:.0f} cm)")
                else:
                    factors.append(f"{fighter_b['name']} has longer reach ({reach_b - reach_a:.0f} cm)")

            elif "sig_str" in feature:
                factors.append("Significant strike differential is a key factor")

            elif "td" in feature:
                factors.append("Takedown ability is a key factor in this matchup")

            elif "win_rate" in feature:
                factors.append("Historical win rate difference matters")

            elif "experience" in feature:
                factors.append("Experience level is a factor")

        return factors[:5]

    def _generate_heuristic_prediction(
        self,
        fighter_a_id: int,
        fighter_b_id: int,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[PredictionResult]:
        """
        Generate a simple heuristic-based prediction when models aren't available.

        Args:
            fighter_a_id: ID of fighter A
            fighter_b_id: ID of fighter B
            context: Optional context

        Returns:
            PredictionResult based on simple heuristics
        """
        fighter_a = self.data_service.get_fighter_by_id(fighter_a_id)
        fighter_b = self.data_service.get_fighter_by_id(fighter_b_id)

        if not fighter_a or not fighter_b:
            return None

        # Simple win rate comparison
        a_wins = fighter_a.get("wins") or 1
        a_losses = fighter_a.get("losses") or 0
        b_wins = fighter_b.get("wins") or 1
        b_losses = fighter_b.get("losses") or 0

        a_win_rate = a_wins / max(a_wins + a_losses, 1)
        b_win_rate = b_wins / max(b_wins + b_losses, 1)

        if a_win_rate >= b_win_rate:
            predicted_winner_id = fighter_a_id
            confidence = 0.5 + (a_win_rate - b_win_rate) * 0.3
        else:
            predicted_winner_id = fighter_b_id
            confidence = 0.5 + (b_win_rate - a_win_rate) * 0.3

        confidence = min(max(confidence, 0.5), 0.75)

        return PredictionResult(
            predicted_winner_id=predicted_winner_id,
            winner_confidence=confidence,
            method_ko_prob=0.35,
            method_sub_prob=0.20,
            method_dec_prob=0.45,
            predicted_method="Decision",
            predicted_round=2.5,
            confidence_level=get_confidence_level(confidence),
            feature_importance={"win_rate": 1.0},
            top_factors=["Based on historical win rates (models not loaded)"],
            model_version="heuristic",
        )

    def predict_upcoming_events(self, save_snapshots: bool = True) -> List[Dict[str, Any]]:
        """
        Generate predictions for all upcoming fights.

        Args:
            save_snapshots: Whether to save stats snapshots for ground truth

        Returns:
            List of prediction results
        """
        predictions = []
        upcoming_fights = self.data_service.get_upcoming_fights()

        for fight in upcoming_fights:
            # Get fighter data for snapshot
            fighter_red = self.data_service.get_fighter_by_id(fight["fighter_red_id"])
            fighter_blue = self.data_service.get_fighter_by_id(fight["fighter_blue_id"])

            result = self.predict(
                fight["fighter_red_id"],
                fight["fighter_blue_id"],
                context={
                    "is_title_fight": fight.get("is_title_fight", False),
                    "is_main_event": fight.get("is_main_event", False),
                    "weight_class": fight.get("weight_class"),
                }
            )

            if result:
                pred_dict = result.to_dict()
                pred_dict["upcoming_id"] = fight["upcoming_id"]
                pred_dict["fighter_red_id"] = fight["fighter_red_id"]
                pred_dict["fighter_blue_id"] = fight["fighter_blue_id"]
                pred_dict["event_date"] = fight.get("event_date")
                pred_dict["is_valid_timing"] = True  # New predictions are valid

                # Save prediction to database
                prediction_id = self.data_service.save_prediction(pred_dict)

                # Save stats snapshots for ground truth collection
                if save_snapshots and prediction_id:
                    self._save_stats_snapshot(
                        prediction_id,
                        fighter_red,
                        fighter_blue
                    )

                pred_dict["prediction_id"] = prediction_id
                predictions.append(pred_dict)

        logger.info(f"Generated {len(predictions)} predictions for upcoming fights")
        return predictions

    def _save_stats_snapshot(
        self,
        prediction_id: int,
        fighter_red: Dict[str, Any],
        fighter_blue: Dict[str, Any]
    ) -> None:
        """
        Save fighter stats snapshot at prediction time for ground truth collection.

        Args:
            prediction_id: ID of the saved prediction
            fighter_red: Red corner fighter data
            fighter_blue: Blue corner fighter data
        """
        try:
            # Get form data from cache (calculated during predict)
            form_data = self._last_form_data or {"red": {}, "blue": {}}

            for corner, fighter in [("red", fighter_red), ("blue", fighter_blue)]:
                if not fighter:
                    continue

                form = form_data.get(corner, {})

                snapshot_data = {
                    "fighter_id": fighter.get("fighter_id"),
                    "wins": fighter.get("wins"),
                    "losses": fighter.get("losses"),
                    "draws": fighter.get("draws"),
                    "sig_strikes_landed_per_min": fighter.get("sig_strikes_landed_per_min"),
                    "sig_strikes_absorbed_per_min": fighter.get("sig_strikes_absorbed_per_min"),
                    "sig_strike_accuracy": fighter.get("sig_strike_accuracy"),
                    "sig_strike_defense": fighter.get("sig_strike_defense"),
                    "takedowns_avg_per_15min": fighter.get("takedowns_avg_per_15min"),
                    "takedown_accuracy": fighter.get("takedown_accuracy"),
                    "takedown_defense": fighter.get("takedown_defense"),
                    "submissions_avg_per_15min": fighter.get("submissions_avg_per_15min"),
                    "finish_rate": fighter.get("finish_rate"),
                    "ko_rate": fighter.get("ko_rate"),
                    "submission_rate": fighter.get("submission_rate"),
                    "decision_rate": fighter.get("decision_rate"),
                    "win_streak": form.get("win_streak", 0),
                    "loss_streak": form.get("loss_streak", 0),
                    "recent_form_score": form.get("recent_form_score"),
                    "days_since_last_fight": form.get("days_since_last_fight"),
                }

                self.data_service.save_prediction_stats_snapshot(
                    prediction_id, snapshot_data, corner
                )

            logger.debug(f"Saved stats snapshot for prediction {prediction_id}")

        except Exception as e:
            logger.warning(f"Failed to save stats snapshot: {e}")

    def get_confidence_level(self, confidence: float) -> str:
        """Get confidence level string."""
        return get_confidence_level(confidence)
