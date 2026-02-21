"""
Fighter Card Component.

Displays fighter profile information in card format.
"""

import streamlit as st
from typing import Any, Dict, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import COLORS


def get_fighter_image_url(fighter_name: str, image_url: Optional[str] = None) -> str:
    """
    Get fighter image URL.

    Args:
        fighter_name: Fighter's name
        image_url: Optional existing image URL

    Returns:
        Image URL or placeholder
    """
    if image_url:
        return image_url

    # Generate UFC.com URL
    slug = fighter_name.lower().replace(" ", "-").replace("'", "")
    return f"https://www.ufc.com/athlete/{slug}"


def render_fighter_card(
    fighter: Dict[str, Any],
    show_actions: bool = True,
    show_stats: bool = True,
    key_prefix: str = ""
):
    """
    Render a full fighter profile card.

    Args:
        fighter: Fighter data dictionary
        show_actions: Whether to show action buttons
        show_stats: Whether to show detailed stats
        key_prefix: Prefix for widget keys
    """
    col1, col2 = st.columns([1, 3])

    with col1:
        # Fighter image placeholder
        st.markdown(
            f"""
            <div style="
                width: 150px;
                height: 150px;
                background: {COLORS['card_bg']};
                border-radius: 8px;
                display: flex;
                align-items: center;
                justify-content: center;
                border: 2px solid {COLORS['primary']};
            ">
                <span style="font-size: 48px;">👤</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        # Name and nickname
        st.subheader(fighter.get("name", "Unknown Fighter"))
        if fighter.get("nickname"):
            st.caption(f'"{fighter["nickname"]}"')

        # Record
        wins = fighter.get("wins", 0)
        losses = fighter.get("losses", 0)
        draws = fighter.get("draws", 0)
        record = f"{wins}-{losses}-{draws}"

        st.metric("Record", record)

        # Physical details
        details = []
        if fighter.get("height_cm"):
            height_ft = fighter["height_cm"] / 30.48
            feet = int(height_ft)
            inches = int((height_ft - feet) * 12)
            details.append(f"Height: {feet}'{inches}\" ({fighter['height_cm']:.0f}cm)")
        if fighter.get("reach_cm"):
            details.append(f"Reach: {fighter['reach_cm']:.0f}cm")
        if fighter.get("stance"):
            details.append(f"Stance: {fighter['stance']}")
        if fighter.get("nationality"):
            details.append(f"Country: {fighter['nationality']}")

        if details:
            st.text(" | ".join(details))

    # Stats section
    if show_stats:
        st.markdown("---")
        render_fighter_stats(fighter)

    # Action buttons
    if show_actions:
        st.markdown("---")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button(
                "⚔️ Compare",
                key=f"{key_prefix}compare_{fighter.get('fighter_id', 0)}",
                width='stretch'
            ):
                st.session_state.compare_fighter = fighter.get("fighter_id")
                st.switch_page("pages/2_⚔️_Fighter_Comparison.py")

        with col2:
            if st.button(
                "🎯 Predict",
                key=f"{key_prefix}predict_{fighter.get('fighter_id', 0)}",
                width='stretch'
            ):
                st.session_state.predict_fighter = fighter.get("fighter_id")
                st.switch_page("pages/3_🎯_Predictions.py")

        with col3:
            if st.button(
                "📋 Full Profile",
                key=f"{key_prefix}profile_{fighter.get('fighter_id', 0)}",
                width='stretch'
            ):
                st.session_state.selected_fighter = fighter.get("fighter_id")


def render_fighter_mini_card(
    fighter: Dict[str, Any],
    show_record: bool = True,
    key_prefix: str = ""
):
    """
    Render a compact fighter card.

    Args:
        fighter: Fighter data dictionary
        show_record: Whether to show record
        key_prefix: Prefix for widget keys
    """
    with st.container():
        col1, col2 = st.columns([1, 4])

        with col1:
            st.markdown(
                f"""
                <div style="
                    width: 60px;
                    height: 60px;
                    background: {COLORS['card_bg']};
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    border: 2px solid {COLORS['primary']};
                ">
                    <span style="font-size: 24px;">👤</span>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:
            st.markdown(f"**{fighter.get('name', 'Unknown')}**")
            if show_record:
                wins = fighter.get("wins", 0)
                losses = fighter.get("losses", 0)
                draws = fighter.get("draws", 0)
                st.caption(f"Record: {wins}-{losses}-{draws}")


def render_fighter_stats(fighter: Dict[str, Any]):
    """
    Render fighter statistics section.

    Args:
        fighter: Fighter data with stats
    """
    st.markdown("#### Career Statistics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Striking**")
        stats_data = [
            ("Sig. Strikes/Min", fighter.get("sig_strikes_landed_per_min")),
            ("Strike Accuracy", fighter.get("sig_strike_accuracy")),
            ("Strikes Absorbed/Min", fighter.get("sig_strikes_absorbed_per_min")),
            ("Strike Defense", fighter.get("sig_strike_defense")),
        ]

        for label, value in stats_data:
            if value is not None:
                if "Accuracy" in label or "Defense" in label:
                    display = f"{value * 100:.0f}%"
                else:
                    display = f"{value:.2f}"
                st.text(f"{label}: {display}")

    with col2:
        st.markdown("**Grappling**")
        stats_data = [
            ("Takedowns/15min", fighter.get("takedowns_avg_per_15min")),
            ("Takedown Accuracy", fighter.get("takedown_accuracy")),
            ("Takedown Defense", fighter.get("takedown_defense")),
            ("Submissions/15min", fighter.get("submissions_avg_per_15min")),
        ]

        for label, value in stats_data:
            if value is not None:
                if "Accuracy" in label or "Defense" in label:
                    display = f"{value * 100:.0f}%"
                else:
                    display = f"{value:.2f}"
                st.text(f"{label}: {display}")


def render_fighter_comparison_header(
    fighter_a: Dict[str, Any],
    fighter_b: Dict[str, Any]
):
    """
    Render side-by-side fighter headers for comparison.

    Args:
        fighter_a: Fighter A data
        fighter_b: Fighter B data
    """
    col1, col2, col3 = st.columns([2, 1, 2])

    with col1:
        st.markdown(
            f"""
            <div style="text-align: center;">
                <div style="
                    width: 100px;
                    height: 100px;
                    background: {COLORS['card_bg']};
                    border-radius: 50%;
                    display: inline-flex;
                    align-items: center;
                    justify-content: center;
                    border: 3px solid {COLORS['primary']};
                    margin-bottom: 10px;
                ">
                    <span style="font-size: 40px;">👤</span>
                </div>
                <h3 style="margin: 0;">{fighter_a.get('name', 'Fighter A')}</h3>
                <p style="color: {COLORS['text_secondary']};">
                    {fighter_a.get('wins', 0)}-{fighter_a.get('losses', 0)}-{fighter_a.get('draws', 0)}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div style="text-align: center; padding-top: 40px;">
                <h2 style="color: {COLORS['primary']};">VS</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div style="text-align: center;">
                <div style="
                    width: 100px;
                    height: 100px;
                    background: {COLORS['card_bg']};
                    border-radius: 50%;
                    display: inline-flex;
                    align-items: center;
                    justify-content: center;
                    border: 3px solid {COLORS['info']};
                    margin-bottom: 10px;
                ">
                    <span style="font-size: 40px;">👤</span>
                </div>
                <h3 style="margin: 0;">{fighter_b.get('name', 'Fighter B')}</h3>
                <p style="color: {COLORS['text_secondary']};">
                    {fighter_b.get('wins', 0)}-{fighter_b.get('losses', 0)}-{fighter_b.get('draws', 0)}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
