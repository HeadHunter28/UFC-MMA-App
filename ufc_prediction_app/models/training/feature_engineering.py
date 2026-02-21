"""
Feature Engineering Module.

Provides feature extraction and transformation for fight prediction models.
"""

import logging
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import (
    DIFFERENTIAL_FEATURES,
    RATIO_FEATURES,
    FORM_FEATURES,
    CONTEXTUAL_FEATURES,
    ALL_FEATURES,
    WEIGHT_CLASSES,
)

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering pipeline for UFC fight predictions.

    Transforms raw fighter and fight data into ML-ready features.
    """

    # Feature ordering for consistency
    FEATURE_ORDER = ALL_FEATURES

    def __init__(self):
        """Initialize the feature engineer."""
        self.feature_names = self.FEATURE_ORDER.copy()

    def create_fight_features(
        self,
        fighter_a_stats: Dict[str, Any],
        fighter_b_stats: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Create feature vector for a matchup.

        Args:
            fighter_a_stats: Statistics for fighter A (red corner)
            fighter_b_stats: Statistics for fighter B (blue corner)
            context: Fight context (weight_class, is_title_fight, etc.)

        Returns:
            Feature array
        """
        context = context or {}
        features = {}

        # Differential features
        features.update(self._create_differential_features(fighter_a_stats, fighter_b_stats))

        # Ratio features
        features.update(self._create_ratio_features(fighter_a_stats, fighter_b_stats))

        # Form features
        features.update(self._create_form_features(fighter_a_stats, fighter_b_stats))

        # Contextual features
        features.update(self._create_contextual_features(context))

        # Ensure correct ordering
        return np.array([features.get(f, 0) for f in self.FEATURE_ORDER])

    def _create_differential_features(
        self,
        a_stats: Dict[str, Any],
        b_stats: Dict[str, Any]
    ) -> Dict[str, float]:
        """Create differential (A - B) features."""
        features = {}

        # Physical differentials
        features["height_diff"] = (
            self._get_stat(a_stats, "height_cm", 175) -
            self._get_stat(b_stats, "height_cm", 175)
        )
        features["reach_diff"] = (
            self._get_stat(a_stats, "reach_cm", 180) -
            self._get_stat(b_stats, "reach_cm", 180)
        )

        # Age differential
        a_age = self._calculate_age(a_stats.get("dob"))
        b_age = self._calculate_age(b_stats.get("dob"))
        features["age_diff"] = a_age - b_age

        # Striking differentials
        features["sig_str_acc_diff"] = (
            self._get_stat(a_stats, "sig_strike_accuracy", 0.45) -
            self._get_stat(b_stats, "sig_strike_accuracy", 0.45)
        )
        features["sig_str_def_diff"] = (
            self._get_stat(a_stats, "sig_strike_defense", 0.55) -
            self._get_stat(b_stats, "sig_strike_defense", 0.55)
        )
        features["str_landed_pm_diff"] = (
            self._get_stat(a_stats, "sig_strikes_landed_per_min", 3.5) -
            self._get_stat(b_stats, "sig_strikes_landed_per_min", 3.5)
        )
        features["str_absorbed_pm_diff"] = (
            self._get_stat(a_stats, "sig_strikes_absorbed_per_min", 3.0) -
            self._get_stat(b_stats, "sig_strikes_absorbed_per_min", 3.0)
        )

        # Grappling differentials
        features["td_acc_diff"] = (
            self._get_stat(a_stats, "takedown_accuracy", 0.40) -
            self._get_stat(b_stats, "takedown_accuracy", 0.40)
        )
        features["td_def_diff"] = (
            self._get_stat(a_stats, "takedown_defense", 0.60) -
            self._get_stat(b_stats, "takedown_defense", 0.60)
        )
        features["td_avg_diff"] = (
            self._get_stat(a_stats, "takedowns_avg_per_15min", 1.5) -
            self._get_stat(b_stats, "takedowns_avg_per_15min", 1.5)
        )
        features["sub_avg_diff"] = (
            self._get_stat(a_stats, "submissions_avg_per_15min", 0.5) -
            self._get_stat(b_stats, "submissions_avg_per_15min", 0.5)
        )

        return features

    def _create_ratio_features(
        self,
        a_stats: Dict[str, Any],
        b_stats: Dict[str, Any]
    ) -> Dict[str, float]:
        """Create ratio (A / B) features."""
        features = {}

        # Win rate ratio
        a_wins = self._get_stat(a_stats, "wins", 1)
        a_losses = self._get_stat(a_stats, "losses", 0)
        b_wins = self._get_stat(b_stats, "wins", 1)
        b_losses = self._get_stat(b_stats, "losses", 0)

        a_win_rate = a_wins / max(a_wins + a_losses, 1)
        b_win_rate = b_wins / max(b_wins + b_losses, 1)

        features["win_rate_ratio"] = a_win_rate / max(b_win_rate, 0.01)

        # Experience ratio
        a_total = a_wins + a_losses
        b_total = b_wins + b_losses
        features["experience_ratio"] = a_total / max(b_total, 1)

        # UFC experience (placeholder - would need UFC-specific fight count)
        features["ufc_experience_ratio"] = features["experience_ratio"]

        # Finish rates
        features["finish_rate_ratio"] = (
            self._get_stat(a_stats, "finish_rate", 0.5) /
            max(self._get_stat(b_stats, "finish_rate", 0.5), 0.01)
        )
        features["ko_rate_ratio"] = (
            self._get_stat(a_stats, "ko_rate", 0.3) /
            max(self._get_stat(b_stats, "ko_rate", 0.3), 0.01)
        )
        features["sub_rate_ratio"] = (
            self._get_stat(a_stats, "submission_rate", 0.2) /
            max(self._get_stat(b_stats, "submission_rate", 0.2), 0.01)
        )

        return features

    def _create_form_features(
        self,
        a_stats: Dict[str, Any],
        b_stats: Dict[str, Any]
    ) -> Dict[str, float]:
        """Create form-related features."""
        features = {}

        # Win/lose streaks
        features["win_streak_a"] = self._get_stat(a_stats, "current_win_streak", 0)
        features["win_streak_b"] = self._get_stat(b_stats, "current_win_streak", 0)
        features["lose_streak_a"] = self._get_stat(a_stats, "current_lose_streak", 0)
        features["lose_streak_b"] = self._get_stat(b_stats, "current_lose_streak", 0)

        # Recent form (weighted last 5 fights - would need fight history)
        features["recent_form_a"] = self._get_stat(a_stats, "recent_form", 0.5)
        features["recent_form_b"] = self._get_stat(b_stats, "recent_form", 0.5)

        # Days since last fight
        features["days_since_fight_a"] = self._get_stat(a_stats, "days_since_fight", 180)
        features["days_since_fight_b"] = self._get_stat(b_stats, "days_since_fight", 180)

        # Momentum (win streak - lose streak)
        features["momentum_a"] = features["win_streak_a"] - features["lose_streak_a"]
        features["momentum_b"] = features["win_streak_b"] - features["lose_streak_b"]

        return features

    def _create_contextual_features(
        self,
        context: Dict[str, Any]
    ) -> Dict[str, float]:
        """Create fight context features."""
        features = {}

        # Weight class encoding
        weight_class = context.get("weight_class", "Welterweight")
        features["weight_class_encoded"] = self._encode_weight_class(weight_class)

        # Fight importance
        features["is_title_fight"] = float(context.get("is_title_fight", False))
        features["is_main_event"] = float(context.get("is_main_event", False))

        # Rounds scheduled
        if context.get("is_title_fight") or context.get("is_main_event"):
            features["rounds_scheduled"] = context.get("rounds_scheduled", 5)
        else:
            features["rounds_scheduled"] = context.get("rounds_scheduled", 3)

        return features

    def _get_stat(
        self,
        stats: Dict[str, Any],
        key: str,
        default: float
    ) -> float:
        """Get a stat value with default."""
        value = stats.get(key)
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    def _calculate_age(self, dob: Optional[str]) -> float:
        """Calculate age from date of birth."""
        if not dob:
            return 30.0  # Default age

        try:
            if isinstance(dob, str):
                birth_date = datetime.strptime(dob, "%Y-%m-%d").date()
            else:
                birth_date = dob

            today = date.today()
            age = today.year - birth_date.year
            if (today.month, today.day) < (birth_date.month, birth_date.day):
                age -= 1
            return float(age)
        except (ValueError, TypeError):
            return 30.0

    def _encode_weight_class(self, weight_class: str) -> float:
        """Encode weight class as ordinal value."""
        # Order from lightest to heaviest
        order = [
            "Strawweight",
            "Flyweight",
            "Bantamweight",
            "Featherweight",
            "Lightweight",
            "Welterweight",
            "Middleweight",
            "Light Heavyweight",
            "Heavyweight",
        ]

        # Handle women's divisions
        if weight_class.startswith("Women's"):
            base_class = weight_class.replace("Women's ", "")
            if base_class in order:
                return order.index(base_class) / len(order)

        if weight_class in order:
            return order.index(weight_class) / len(order)

        return 0.5  # Default to middle

    def get_feature_names(self) -> List[str]:
        """Get list of feature names in order."""
        return self.feature_names.copy()


def create_fight_features(
    fighter_a_stats: Dict[str, Any],
    fighter_b_stats: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Convenience function to create fight features.

    Args:
        fighter_a_stats: Statistics for fighter A
        fighter_b_stats: Statistics for fighter B
        context: Fight context

    Returns:
        Feature array
    """
    engineer = FeatureEngineer()
    return engineer.create_fight_features(fighter_a_stats, fighter_b_stats, context)


def create_training_dataset(
    fights_df: pd.DataFrame,
    fighters_df: pd.DataFrame,
    fighter_stats_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create training dataset from fight history.

    Args:
        fights_df: DataFrame of historical fights
        fighters_df: DataFrame of fighter info
        fighter_stats_df: DataFrame of fighter statistics

    Returns:
        Tuple of (X, y_winner, y_method, y_round)
    """
    engineer = FeatureEngineer()

    X_list = []
    y_winner = []
    y_method = []
    y_round = []

    for _, fight in fights_df.iterrows():
        try:
            # Get fighter stats at time of fight (or current as approximation)
            a_stats = fighter_stats_df[
                fighter_stats_df["fighter_id"] == fight["fighter_red_id"]
            ].iloc[0].to_dict() if len(fighter_stats_df[
                fighter_stats_df["fighter_id"] == fight["fighter_red_id"]
            ]) > 0 else {}

            b_stats = fighter_stats_df[
                fighter_stats_df["fighter_id"] == fight["fighter_blue_id"]
            ].iloc[0].to_dict() if len(fighter_stats_df[
                fighter_stats_df["fighter_id"] == fight["fighter_blue_id"]
            ]) > 0 else {}

            # Add record info from fighters table
            a_fighter = fighters_df[
                fighters_df["fighter_id"] == fight["fighter_red_id"]
            ].iloc[0].to_dict() if len(fighters_df[
                fighters_df["fighter_id"] == fight["fighter_red_id"]
            ]) > 0 else {}

            b_fighter = fighters_df[
                fighters_df["fighter_id"] == fight["fighter_blue_id"]
            ].iloc[0].to_dict() if len(fighters_df[
                fighters_df["fighter_id"] == fight["fighter_blue_id"]
            ]) > 0 else {}

            a_stats.update(a_fighter)
            b_stats.update(b_fighter)

            context = {
                "weight_class": fight.get("weight_class", "Welterweight"),
                "is_title_fight": fight.get("is_title_fight", False),
                "is_main_event": fight.get("is_main_event", False),
            }

            features = engineer.create_fight_features(a_stats, b_stats, context)
            X_list.append(features)

            # Labels
            winner = 1 if fight["winner_id"] == fight["fighter_red_id"] else 0
            y_winner.append(winner)

            # Method encoding
            method = (fight.get("method") or "").upper()
            if "KO" in method or "TKO" in method:
                y_method.append(0)
            elif "SUB" in method:
                y_method.append(1)
            else:
                y_method.append(2)

            # Round (ensure no NaN)
            fight_round = fight.get("round")
            if fight_round is None or (isinstance(fight_round, float) and np.isnan(fight_round)):
                fight_round = 3
            y_round.append(int(fight_round))

        except Exception as e:
            logger.warning(f"Failed to process fight {fight.get('fight_id')}: {e}")
            continue

    return (
        np.array(X_list),
        np.array(y_winner),
        np.array(y_method),
        np.array(y_round),
    )
