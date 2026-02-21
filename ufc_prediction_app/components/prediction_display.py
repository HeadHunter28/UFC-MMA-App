"""
Prediction Display Component.

Displays prediction results with confidence meters and explanations.
"""

import streamlit as st
from typing import Any, Dict, List, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import COLORS, CONFIDENCE_COLORS, get_confidence_level, MIN_CONFIDENCE_THRESHOLD


def render_prediction_result(
    prediction: Dict[str, Any],
    fighter_a: Dict[str, Any],
    fighter_b: Dict[str, Any],
    show_details: bool = True
):
    """
    Render a complete prediction result.

    Args:
        prediction: Prediction data
        fighter_a: Fighter A data
        fighter_b: Fighter B data
        show_details: Whether to show detailed breakdown
    """
    winner_id = prediction.get("predicted_winner_id")
    confidence = prediction.get("winner_confidence", 0.5)

    # Determine winner name
    if winner_id == fighter_a.get("fighter_id"):
        winner_name = fighter_a.get("name", "Fighter A")
        loser_name = fighter_b.get("name", "Fighter B")
    else:
        winner_name = fighter_b.get("name", "Fighter B")
        loser_name = fighter_a.get("name", "Fighter A")

    # Main prediction header
    st.markdown(
        f"""
        <div style="
            text-align: center;
            padding: 20px;
            background: {COLORS['card_bg']};
            border-radius: 10px;
            border-left: 4px solid {get_confidence_color(confidence)};
        ">
            <h2 style="color: {COLORS['text_primary']}; margin-bottom: 5px;">
                🏆 {winner_name}
            </h2>
            <p style="color: {COLORS['text_secondary']}; margin: 0;">
                Predicted to defeat {loser_name}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("")

    # Confidence meter
    render_confidence_meter(confidence)

    # Low confidence warning
    if confidence < MIN_CONFIDENCE_THRESHOLD:
        st.warning(
            f"""
            **Low Confidence Prediction**

            This matchup is difficult to predict with high certainty.
            Our model shows only {confidence:.0%} confidence, suggesting
            this fight could go either way.
            """
        )

    st.markdown("---")

    # Method and round predictions
    col1, col2, col3 = st.columns(3)

    with col1:
        predicted_method = prediction.get("predicted_method", "Decision")
        st.metric("Predicted Method", predicted_method)

    with col2:
        predicted_round = prediction.get("predicted_round", 2.5)
        st.metric("Predicted Round", f"Round {predicted_round:.1f}")

    with col3:
        model_version = prediction.get("model_version", "v1.0.0")
        st.metric("Model Version", model_version)

    # Method probabilities
    st.markdown("#### Win Method Probabilities")

    method_data = [
        ("KO/TKO", prediction.get("method_ko_prob", 0.33)),
        ("Submission", prediction.get("method_sub_prob", 0.33)),
        ("Decision", prediction.get("method_dec_prob", 0.34)),
    ]

    for method, prob in method_data:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.progress(prob, text=method)
        with col2:
            st.write(f"{prob:.0%}")

    # Detailed breakdown
    if show_details:
        render_prediction_details(prediction, fighter_a, fighter_b)


def render_confidence_meter(confidence: float):
    """
    Render a confidence meter visualization.

    Args:
        confidence: Confidence value (0.0 - 1.0)
    """
    level = get_confidence_level(confidence)
    color = CONFIDENCE_COLORS[level]

    level_labels = {
        "high": "High Confidence",
        "medium": "Medium Confidence",
        "low": "Low Confidence",
    }

    level_icons = {
        "high": "🟢",
        "medium": "🟡",
        "low": "🟠",
    }

    st.markdown(
        f"""
        <div style="
            text-align: center;
            padding: 15px;
            background: {COLORS['card_bg']};
            border-radius: 8px;
        ">
            <h3 style="margin: 0; color: {color};">
                {level_icons[level]} {confidence:.0%} {level_labels[level]}
            </h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Progress bar
    st.progress(confidence)


def render_prediction_details(
    prediction: Dict[str, Any],
    fighter_a: Dict[str, Any],
    fighter_b: Dict[str, Any]
):
    """
    Render detailed prediction breakdown.

    Args:
        prediction: Prediction data
        fighter_a: Fighter A data
        fighter_b: Fighter B data
    """
    with st.expander("📊 Prediction Details", expanded=False):
        # Top factors
        st.markdown("#### Top Contributing Factors")

        top_factors = prediction.get("top_factors", [])
        if isinstance(top_factors, str):
            import json
            try:
                top_factors = json.loads(top_factors)
            except:
                top_factors = [top_factors]

        if top_factors:
            for i, factor in enumerate(top_factors[:5], 1):
                st.markdown(f"{i}. {factor}")
        else:
            st.write("No specific factors available")

        # Feature importance
        st.markdown("#### Feature Importance")

        feature_importance = prediction.get("feature_importance", {})
        if isinstance(feature_importance, str):
            import json
            try:
                feature_importance = json.loads(feature_importance)
            except:
                feature_importance = {}

        if feature_importance:
            # Sort by importance
            sorted_features = sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]

            for feature, importance in sorted_features:
                feature_label = feature.replace("_", " ").title()
                st.progress(min(importance, 1.0), text=f"{feature_label}: {importance:.3f}")
        else:
            st.write("Feature importance not available")


def render_fight_card_prediction(
    prediction: Dict[str, Any],
    fighter_a: Dict[str, Any],
    fighter_b: Dict[str, Any],
    compact: bool = False
):
    """
    Render prediction in fight card format.

    Args:
        prediction: Prediction data
        fighter_a: Fighter A data
        fighter_b: Fighter B data
        compact: Use compact layout
    """
    winner_id = prediction.get("predicted_winner_id")
    confidence = prediction.get("winner_confidence", 0.5)
    level = get_confidence_level(confidence)
    color = CONFIDENCE_COLORS[level]

    # Determine winner
    if winner_id == fighter_a.get("fighter_id"):
        winner_name = fighter_a.get("name", "Fighter A")
        winner_record = f"{fighter_a.get('wins', 0)}-{fighter_a.get('losses', 0)}"
        loser_name = fighter_b.get("name", "Fighter B")
        loser_record = f"{fighter_b.get('wins', 0)}-{fighter_b.get('losses', 0)}"
    else:
        winner_name = fighter_b.get("name", "Fighter B")
        winner_record = f"{fighter_b.get('wins', 0)}-{fighter_b.get('losses', 0)}"
        loser_name = fighter_a.get("name", "Fighter A")
        loser_record = f"{fighter_a.get('wins', 0)}-{fighter_a.get('losses', 0)}"

    if compact:
        st.markdown(
            f"""
            <div style="
                padding: 10px;
                background: {COLORS['card_bg']};
                border-radius: 5px;
                border-left: 3px solid {color};
            ">
                <strong>{fighter_a.get('name')} vs {fighter_b.get('name')}</strong><br>
                <span style="color: {color};">
                    Pick: {winner_name} ({confidence:.0%})
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 3])

        with col1:
            is_winner_a = winner_id == fighter_a.get("fighter_id")
            name_style = f"color: {COLORS['success']}; font-weight: bold;" if is_winner_a else ""
            st.markdown(f"<span style='{name_style}'>{fighter_a.get('name')}</span>", unsafe_allow_html=True)
            st.caption(f"{fighter_a.get('wins', 0)}-{fighter_a.get('losses', 0)}")

        with col2:
            st.markdown("vs")

        with col3:
            st.markdown(f"<span style='color: {color};'>{confidence:.0%}</span>", unsafe_allow_html=True)

        with col4:
            st.caption(prediction.get("predicted_method", "DEC")[:3])

        with col5:
            is_winner_b = winner_id == fighter_b.get("fighter_id")
            name_style = f"color: {COLORS['success']}; font-weight: bold;" if is_winner_b else ""
            st.markdown(f"<span style='{name_style}'>{fighter_b.get('name')}</span>", unsafe_allow_html=True)
            st.caption(f"{fighter_b.get('wins', 0)}-{fighter_b.get('losses', 0)}")


def get_confidence_color(confidence: float) -> str:
    """Get color for confidence level."""
    level = get_confidence_level(confidence)
    return CONFIDENCE_COLORS[level]
