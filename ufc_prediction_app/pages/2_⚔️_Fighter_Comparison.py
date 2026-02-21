"""
Fighter Comparison Page.

Head-to-head statistical comparison of two fighters.
"""

import streamlit as st

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import COLORS
from services.data_service import DataService
from services.llm_service import LLMService
from components.fighter_card import render_fighter_comparison_header
from components.comparison_charts import (
    render_comparison_charts,
    render_edge_summary,
)


st.set_page_config(
    page_title="Compare Fighters - UFC Prediction App",
    page_icon="⚔️",
    layout="wide",
)

st.title("⚔️ Fighter Comparison")
st.markdown("Compare two fighters head-to-head with detailed statistics.")

# Initialize services
data_service = DataService()
llm_service = LLMService()

# Fighter selection
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Fighter A (Red Corner)")
    fighter_a_search = st.text_input(
        "Search Fighter A",
        placeholder="Enter fighter name...",
        key="fighter_a_search",
    )

    fighter_a = None
    if fighter_a_search:
        results_a = data_service.search_fighters(fighter_a_search, limit=5)
        if results_a:
            selected_a = st.selectbox(
                "Select Fighter A",
                options=results_a,
                format_func=lambda x: f"{x['name']} ({x['wins']}-{x['losses']})",
                key="select_a",
            )
            fighter_a = selected_a
        else:
            st.warning("No fighters found")

    # Check for pre-selected fighter from other pages
    if "compare_fighter" in st.session_state and st.session_state.compare_fighter:
        preset = data_service.get_fighter_by_id(st.session_state.compare_fighter)
        if preset:
            st.info(f"Pre-selected: {preset['name']}")
            fighter_a = preset
            st.session_state.compare_fighter = None

with col2:
    st.markdown("### Fighter B (Blue Corner)")
    fighter_b_search = st.text_input(
        "Search Fighter B",
        placeholder="Enter fighter name...",
        key="fighter_b_search",
    )

    fighter_b = None
    if fighter_b_search:
        results_b = data_service.search_fighters(fighter_b_search, limit=5)
        if results_b:
            selected_b = st.selectbox(
                "Select Fighter B",
                options=results_b,
                format_func=lambda x: f"{x['name']} ({x['wins']}-{x['losses']})",
                key="select_b",
            )
            fighter_b = selected_b
        else:
            st.warning("No fighters found")

# Comparison display
if fighter_a and fighter_b:
    st.markdown("---")

    # Header with photos
    render_fighter_comparison_header(fighter_a, fighter_b)

    st.markdown("---")

    # Comparison charts
    render_comparison_charts(fighter_a, fighter_b)

    st.markdown("---")

    # Edge summary
    render_edge_summary(fighter_a, fighter_b)

    st.markdown("---")

    # AI Comparison
    st.markdown("### AI Comparison Analysis")

    if llm_service.is_available():
        with st.spinner("Generating comparison analysis..."):
            analysis = llm_service.generate_comparison_summary(fighter_a, fighter_b)
            if analysis:
                st.markdown(analysis)
            else:
                st.info("Could not generate analysis at this time.")
    else:
        st.info("AI analysis is currently unavailable.")

    st.markdown("---")

    # Action buttons
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🎯 Get Prediction", width='stretch'):
            st.session_state.predict_fighter_a = fighter_a.get("fighter_id")
            st.session_state.predict_fighter_b = fighter_b.get("fighter_id")
            st.switch_page("pages/3_🎯_Predictions.py")

    with col2:
        if st.button("🔄 Swap Fighters", width='stretch'):
            # Swap the search queries
            temp_a = fighter_a_search
            temp_b = fighter_b_search
            st.session_state.fighter_a_search = temp_b
            st.session_state.fighter_b_search = temp_a
            st.rerun()

elif fighter_a or fighter_b:
    st.info("Select both fighters to see comparison.")
else:
    st.markdown("---")
    st.markdown("### How to use")
    st.markdown(
        """
        1. Search for Fighter A in the left column
        2. Search for Fighter B in the right column
        3. View the head-to-head comparison
        4. Get a prediction for the matchup
        """
    )

    # Quick comparison suggestions
    st.markdown("---")
    st.markdown("### Popular Matchups")
    st.info("Search for fighters above to compare them.")
