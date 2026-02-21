#!/usr/bin/env python3
"""
Feature Importance Analysis for UFC Fight Prediction.

This script analyzes which features are most important for predicting fight outcomes
using SHAP values, permutation importance, and built-in feature importance.

Usage:
    python scripts/analyze_feature_importance.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import SHAP (optional but recommended)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not installed. Run: pip install shap")

from services.data_service import DataService
from config import ALL_FEATURES


def load_training_data(data_service: DataService) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load historical fight data for analysis.

    Returns:
        Tuple of (features_df, labels_df)
    """
    logger.info("Loading historical fight data...")

    # Get all fights
    with data_service.get_connection() as conn:
        fights_query = """
            SELECT
                f.fight_id,
                f.fighter_red_id,
                f.fighter_blue_id,
                f.winner_id,
                f.method,
                f.round,
                f.event_id,
                f.weight_class,
                e.date as event_date
            FROM fights f
            JOIN events e ON f.event_id = e.event_id
            WHERE f.winner_id IS NOT NULL
            ORDER BY e.date DESC
        """
        fights_df = pd.read_sql_query(fights_query, conn)

        # Get fighter stats
        fighters_query = """
            SELECT
                fighter_id, name, wins, losses, draws,
                height_cm, reach_cm, dob
            FROM fighters
        """
        fighters_df = pd.read_sql_query(fighters_query, conn)

        # Get fighter stats
        stats_query = """
            SELECT * FROM fighter_stats
        """
        stats_df = pd.read_sql_query(stats_query, conn)

    logger.info(f"Loaded {len(fights_df)} fights")

    # Merge fighter data
    fighters_dict = fighters_df.set_index('fighter_id').to_dict('index')
    stats_dict = stats_df.set_index('fighter_id').to_dict('index') if len(stats_df) > 0 else {}

    return fights_df, fighters_dict, stats_dict


def create_features(
    fights_df: pd.DataFrame,
    fighters_dict: Dict,
    stats_dict: Dict
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Create feature vectors from fight data.

    Returns:
        Tuple of (X, y, feature_names)
    """
    logger.info("Creating feature vectors...")

    X_list = []
    y_list = []

    for _, fight in fights_df.iterrows():
        try:
            red_id = fight['fighter_red_id']
            blue_id = fight['fighter_blue_id']
            winner_id = fight['winner_id']

            # Get fighter data
            red = fighters_dict.get(red_id, {})
            blue = fighters_dict.get(blue_id, {})
            red_stats = stats_dict.get(red_id, {})
            blue_stats = stats_dict.get(blue_id, {})

            # Skip if no data
            if not red or not blue:
                continue

            # Calculate features
            features = calculate_fight_features(red, blue, red_stats, blue_stats)

            if features is not None:
                X_list.append(features)
                # 1 if red corner wins, 0 if blue corner wins
                y_list.append(1 if winner_id == red_id else 0)

        except Exception as e:
            logger.debug(f"Skipping fight {fight.get('fight_id')}: {e}")
            continue

    X = np.array(X_list)
    y = np.array(y_list)

    logger.info(f"Created {len(X)} feature vectors")

    return X, y, ALL_FEATURES


def calculate_fight_features(
    red: Dict,
    blue: Dict,
    red_stats: Dict,
    blue_stats: Dict
) -> np.ndarray:
    """Calculate features for a single fight."""

    def get_stat(stats: Dict, key: str, default: float) -> float:
        val = stats.get(key)
        if val is None:
            return default
        try:
            return float(val)
        except:
            return default

    # Win rates
    red_wins = red.get('wins', 1)
    red_losses = red.get('losses', 0)
    blue_wins = blue.get('wins', 1)
    blue_losses = blue.get('losses', 0)

    red_win_rate = red_wins / max(red_wins + red_losses, 1)
    blue_win_rate = blue_wins / max(blue_wins + blue_losses, 1)

    features = [
        # DIFFERENTIAL FEATURES (11)
        (red.get('height_cm') or 175) - (blue.get('height_cm') or 175),  # height_diff
        (red.get('reach_cm') or 180) - (blue.get('reach_cm') or 180),    # reach_diff
        0,  # age_diff (would need DOB calculation)
        get_stat(red_stats, 'sig_strike_accuracy', 0.45) - get_stat(blue_stats, 'sig_strike_accuracy', 0.45),
        get_stat(red_stats, 'sig_strike_defense', 0.55) - get_stat(blue_stats, 'sig_strike_defense', 0.55),
        get_stat(red_stats, 'takedown_accuracy', 0.40) - get_stat(blue_stats, 'takedown_accuracy', 0.40),
        get_stat(red_stats, 'takedown_defense', 0.60) - get_stat(blue_stats, 'takedown_defense', 0.60),
        get_stat(red_stats, 'sig_strikes_landed_per_min', 3.5) - get_stat(blue_stats, 'sig_strikes_landed_per_min', 3.5),
        get_stat(red_stats, 'sig_strikes_absorbed_per_min', 3.0) - get_stat(blue_stats, 'sig_strikes_absorbed_per_min', 3.0),
        get_stat(red_stats, 'takedowns_avg_per_15min', 1.5) - get_stat(blue_stats, 'takedowns_avg_per_15min', 1.5),
        get_stat(red_stats, 'submissions_avg_per_15min', 0.5) - get_stat(blue_stats, 'submissions_avg_per_15min', 0.5),

        # RATIO FEATURES (6)
        red_win_rate / max(blue_win_rate, 0.01),  # win_rate_ratio
        get_stat(red_stats, 'finish_rate', 0.5) / max(get_stat(blue_stats, 'finish_rate', 0.5), 0.01),
        get_stat(red_stats, 'ko_rate', 0.3) / max(get_stat(blue_stats, 'ko_rate', 0.3), 0.01),
        get_stat(red_stats, 'submission_rate', 0.2) / max(get_stat(blue_stats, 'submission_rate', 0.2), 0.01),
        (red_wins + red_losses) / max(blue_wins + blue_losses, 1),  # experience_ratio
        (red_wins + red_losses) / max(blue_wins + blue_losses, 1),  # ufc_experience_ratio

        # FORM FEATURES (10)
        0,  # win_streak_a (would need fight history)
        0,  # win_streak_b
        0,  # lose_streak_a
        0,  # lose_streak_b
        0.5,  # recent_form_a
        0.5,  # recent_form_b
        180,  # days_since_fight_a
        180,  # days_since_fight_b
        0,  # momentum_a
        0,  # momentum_b

        # CONTEXTUAL FEATURES (4)
        0,  # weight_class_encoded
        0,  # is_title_fight
        0,  # is_main_event
        3,  # rounds_scheduled
    ]

    return np.array(features)


def analyze_with_shap(
    model,
    X: np.ndarray,
    feature_names: List[str],
    output_dir: Path
):
    """Analyze feature importance using SHAP values."""
    if not SHAP_AVAILABLE:
        logger.warning("SHAP not available, skipping SHAP analysis")
        return

    logger.info("Computing SHAP values...")

    # Create explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / 'shap_summary.png', dpi=150)
    plt.close()

    # Bar plot of mean absolute SHAP values
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(output_dir / 'shap_importance_bar.png', dpi=150)
    plt.close()

    # Calculate mean absolute SHAP values
    mean_shap = np.abs(shap_values).mean(axis=0)
    shap_importance = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_shap
    }).sort_values('mean_abs_shap', ascending=False)

    logger.info("\nTop 15 Features by SHAP Importance:")
    print(shap_importance.head(15).to_string(index=False))

    shap_importance.to_csv(output_dir / 'shap_importance.csv', index=False)

    return shap_importance


def analyze_permutation_importance(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    output_dir: Path
) -> pd.DataFrame:
    """Analyze feature importance using permutation importance."""
    logger.info("Computing permutation importance...")

    result = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=-1)

    perm_importance = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std
    }).sort_values('importance_mean', ascending=False)

    logger.info("\nTop 15 Features by Permutation Importance:")
    print(perm_importance.head(15).to_string(index=False))

    # Plot
    plt.figure(figsize=(12, 8))
    top_features = perm_importance.head(20)
    plt.barh(range(len(top_features)), top_features['importance_mean'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Permutation Importance')
    plt.title('Top 20 Features by Permutation Importance')
    plt.tight_layout()
    plt.savefig(output_dir / 'permutation_importance.png', dpi=150)
    plt.close()

    perm_importance.to_csv(output_dir / 'permutation_importance.csv', index=False)

    return perm_importance


def analyze_builtin_importance(
    model,
    feature_names: List[str],
    output_dir: Path,
    model_name: str
) -> pd.DataFrame:
    """Analyze feature importance using model's built-in importance."""
    logger.info(f"Computing {model_name} built-in importance...")

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        logger.warning(f"Model {model_name} does not have feature_importances_")
        return None

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    logger.info(f"\nTop 15 Features by {model_name} Importance:")
    print(importance_df.head(15).to_string(index=False))

    # Plot
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(20)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title(f'Top 20 Features - {model_name}')
    plt.tight_layout()
    plt.savefig(output_dir / f'{model_name.lower().replace(" ", "_")}_importance.png', dpi=150)
    plt.close()

    importance_df.to_csv(output_dir / f'{model_name.lower().replace(" ", "_")}_importance.csv', index=False)

    return importance_df


def main():
    """Run feature importance analysis."""
    print("=" * 60)
    print("UFC Fight Prediction - Feature Importance Analysis")
    print("=" * 60)

    # Create output directory
    output_dir = Path(__file__).parent.parent / 'analysis_output'
    output_dir.mkdir(exist_ok=True)

    # Load data
    data_service = DataService()
    fights_df, fighters_dict, stats_dict = load_training_data(data_service)

    # Create features
    X, y, feature_names = create_features(fights_df, fighters_dict, stats_dict)

    if len(X) < 100:
        logger.error(f"Not enough data for analysis ({len(X)} samples). Need at least 100.")
        return

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    logger.info(f"\nTraining set: {len(X_train)}, Test set: {len(X_test)}")

    # Train models
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            eval_metric='logloss'
        )
    }

    all_importances = {}

    for name, model in models.items():
        logger.info(f"\n{'='*40}")
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)

        # Evaluate
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        logger.info(f"Train accuracy: {train_acc:.3f}, Test accuracy: {test_acc:.3f}")

        # Built-in importance
        importance_df = analyze_builtin_importance(model, feature_names, output_dir, name)
        if importance_df is not None:
            all_importances[name] = importance_df

    # SHAP analysis on best model (XGBoost usually best for SHAP)
    logger.info("\n" + "="*40)
    if SHAP_AVAILABLE:
        shap_importance = analyze_with_shap(
            models['XGBoost'], X_test, feature_names, output_dir
        )
        if shap_importance is not None:
            all_importances['SHAP'] = shap_importance

    # Permutation importance
    logger.info("\n" + "="*40)
    perm_importance = analyze_permutation_importance(
        models['XGBoost'], X_test, y_test, feature_names, output_dir
    )
    all_importances['Permutation'] = perm_importance

    # Consensus ranking
    logger.info("\n" + "="*40)
    logger.info("CONSENSUS FEATURE RANKING")
    logger.info("="*40)

    # Combine rankings
    ranking_df = pd.DataFrame({'feature': feature_names})

    for method_name, imp_df in all_importances.items():
        # Create rank (1 = most important)
        imp_col = [c for c in imp_df.columns if c != 'feature'][0]
        imp_df = imp_df.copy()
        imp_df['rank'] = range(1, len(imp_df) + 1)

        ranking_df = ranking_df.merge(
            imp_df[['feature', 'rank']].rename(columns={'rank': f'{method_name}_rank'}),
            on='feature',
            how='left'
        )

    # Calculate average rank
    rank_cols = [c for c in ranking_df.columns if c.endswith('_rank')]
    ranking_df['avg_rank'] = ranking_df[rank_cols].mean(axis=1)
    ranking_df = ranking_df.sort_values('avg_rank')

    print("\nTop 15 Features by Consensus Ranking:")
    print(ranking_df[['feature', 'avg_rank'] + rank_cols].head(15).to_string(index=False))

    # Save consensus ranking
    ranking_df.to_csv(output_dir / 'consensus_ranking.csv', index=False)

    print(f"\n{'='*60}")
    print(f"Analysis complete! Results saved to: {output_dir}")
    print(f"{'='*60}")

    # Key insights
    print("\n" + "="*40)
    print("KEY INSIGHTS")
    print("="*40)
    print("\nMost important feature categories:")

    # Group features by category
    categories = {
        'Differential': ['_diff', 'height', 'reach', 'age'],
        'Ratio': ['_ratio'],
        'Form': ['streak', 'form', 'momentum', 'days_since'],
        'Contextual': ['weight_class', 'title', 'main_event', 'rounds']
    }

    for category, patterns in categories.items():
        cat_features = ranking_df[
            ranking_df['feature'].str.contains('|'.join(patterns), case=False, na=False)
        ]
        if len(cat_features) > 0:
            avg = cat_features['avg_rank'].mean()
            best = cat_features.iloc[0]['feature'] if len(cat_features) > 0 else 'N/A'
            print(f"\n{category}:")
            print(f"  Average rank: {avg:.1f}")
            print(f"  Best feature: {best}")


if __name__ == "__main__":
    main()
