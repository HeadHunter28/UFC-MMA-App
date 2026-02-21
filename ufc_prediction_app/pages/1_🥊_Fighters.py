"""
Fighters Page.

Search and explore UFC fighter profiles with detailed statistics.
"""

import streamlit as st

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import COLORS, WEIGHT_CLASSES
from services.data_service import DataService
from services.llm_service import LLMService
from components.fighter_card import render_fighter_card, render_fighter_stats
from components.stats_charts import render_win_method_chart


st.set_page_config(
    page_title="Fighters - UFC Prediction App",
    page_icon="🥊",
    layout="wide",
)

st.title("🥊 UFC Fighters")
st.markdown("Search and explore fighter profiles, statistics, and fight history.")

# Initialize services
data_service = DataService()
llm_service = LLMService()

# Search and filters
col1, col2, col3 = st.columns([3, 1, 1])

with col1:
    search_query = st.text_input(
        "Search fighters",
        placeholder="Enter fighter name...",
        key="fighter_search",
    )

with col2:
    active_filter = st.selectbox(
        "Status",
        options=["All", "Active", "Inactive"],
        key="active_filter",
    )

with col3:
    nationality_filter = st.text_input(
        "Nationality",
        placeholder="e.g., USA",
        key="nationality_filter",
    )

# Search results
if search_query:
    st.markdown("---")
    st.markdown("### Search Results")

    # Build filters
    filters = {}
    if active_filter == "Active":
        filters["is_active"] = True
    elif active_filter == "Inactive":
        filters["is_active"] = False

    # Search fighters
    fighters = data_service.search_fighters(
        query=search_query,
        limit=20,
        is_active=filters.get("is_active"),
    )

    if fighters:
        st.write(f"Found {len(fighters)} fighters matching '{search_query}'")

        # Display results in a grid
        for i in range(0, len(fighters), 2):
            cols = st.columns(2)

            for j, col in enumerate(cols):
                idx = i + j
                if idx < len(fighters):
                    fighter = fighters[idx]
                    with col:
                        with st.container():
                            st.markdown(
                                f"""
                                <div style="
                                    background: {COLORS['card_bg']};
                                    padding: 15px;
                                    border-radius: 8px;
                                    margin-bottom: 10px;
                                ">
                                    <h4 style="margin: 0;">{fighter.get('name', 'Unknown')}</h4>
                                    <p style="color: {COLORS['text_secondary']}; margin: 5px 0;">
                                        Record: {fighter.get('wins', 0)}-{fighter.get('losses', 0)}-{fighter.get('draws', 0)}
                                    </p>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                            if st.button(
                                "View Profile",
                                key=f"view_{fighter.get('fighter_id', idx)}",
                            ):
                                st.session_state.selected_fighter = fighter.get("fighter_id")
                                st.rerun()
    else:
        st.info(f"No fighters found matching '{search_query}'")

# Selected fighter profile
if "selected_fighter" in st.session_state and st.session_state.selected_fighter:
    st.markdown("---")
    st.markdown("### Fighter Profile")

    fighter = data_service.get_fighter_by_id(st.session_state.selected_fighter)

    if fighter:
        # Clear selection button
        if st.button("← Back to Search"):
            st.session_state.selected_fighter = None
            st.rerun()

        # Main profile
        render_fighter_card(fighter, show_actions=True, show_stats=True)

        # Fight history
        st.markdown("---")
        st.markdown("### Fight History")

        history = data_service.get_fighter_fight_history(
            st.session_state.selected_fighter,
            limit=10,
        )

        if history:
            for fight in history:
                result_color = (
                    COLORS["success"] if fight.get("result") == "Win"
                    else COLORS["danger"] if fight.get("result") == "Loss"
                    else COLORS["warning"]
                )

                col1, col2, col3, col4 = st.columns([1, 3, 2, 2])

                with col1:
                    st.markdown(
                        f"<span style='color: {result_color}; font-weight: bold;'>{fight.get('result', '?')}</span>",
                        unsafe_allow_html=True,
                    )

                with col2:
                    st.write(f"vs {fight.get('opponent_name', 'Unknown')}")

                with col3:
                    st.caption(f"{fight.get('method', 'N/A')} R{fight.get('round', '?')}")

                with col4:
                    st.caption(fight.get('event_name', '')[:30])
        else:
            st.info("No fight history available.")

        # AI Analysis
        st.markdown("---")
        st.markdown("### AI Analysis")

        if llm_service.is_available():
            with st.spinner("Generating analysis..."):
                stats = data_service.get_fighter_stats(st.session_state.selected_fighter)
                analysis = llm_service.generate_fighter_analysis(
                    fighter,
                    stats=stats,
                    history=history,
                )

                if analysis:
                    st.markdown(analysis)
                else:
                    st.info("Could not generate analysis at this time.")
        else:
            st.info("AI analysis is currently unavailable.")

# Browse all fighters
else:
    if not search_query:
        st.markdown("---")
        st.markdown("### Browse Fighters")
        st.info("Enter a fighter name above to search, or browse recent additions below.")

        # Show some fighters
        try:
            all_fighters = data_service.get_all_fighters(
                filters={"is_active": True},
                limit=12,
            )

            if all_fighters:
                for i in range(0, len(all_fighters), 3):
                    cols = st.columns(3)
                    for j, col in enumerate(cols):
                        idx = i + j
                        if idx < len(all_fighters):
                            fighter = all_fighters[idx]
                            with col:
                                st.markdown(
                                    f"""
                                    <div style="
                                        background: {COLORS['card_bg']};
                                        padding: 15px;
                                        border-radius: 8px;
                                        margin-bottom: 10px;
                                    ">
                                        <h4 style="margin: 0;">{fighter.get('name', 'Unknown')}</h4>
                                        <p style="color: {COLORS['text_secondary']}; margin: 5px 0;">
                                            {fighter.get('wins', 0)}-{fighter.get('losses', 0)}-{fighter.get('draws', 0)}
                                        </p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                                if st.button("View", key=f"browse_{fighter.get('fighter_id', idx)}"):
                                    st.session_state.selected_fighter = fighter.get("fighter_id")
                                    st.rerun()
        except Exception as e:
            st.info("No fighters in database. Run the data initialization script first.")
