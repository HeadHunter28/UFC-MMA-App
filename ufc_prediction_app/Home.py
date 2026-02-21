"""
UFC Fighter Analysis & Prediction App.

Main entry point for the Streamlit application.
"""

import streamlit as st

from config import COLORS, ensure_directories
from services.data_service import DataService
from services.accuracy_service import AccuracyService
from utils.helpers import get_countdown, format_date

# Ensure directories exist
ensure_directories()

# Page configuration
st.set_page_config(
    page_title="UFC Prediction App",
    page_icon="🥊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for UFC theme
st.markdown(
    f"""
    <style>
        .stApp {{
            background-color: {COLORS['background']};
        }}
        .main-header {{
            text-align: center;
            color: {COLORS['primary']};
            font-size: 48px;
            font-weight: bold;
            margin-bottom: 10px;
        }}
        .sub-header {{
            text-align: center;
            color: {COLORS['text_secondary']};
            font-size: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: {COLORS['card_bg']};
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            border-left: 4px solid {COLORS['primary']};
        }}
        .stat-value {{
            font-size: 36px;
            font-weight: bold;
            color: {COLORS['text_primary']};
        }}
        .stat-label {{
            font-size: 16px;
            color: {COLORS['text_secondary']};
        }}
        .feature-card {{
            background: {COLORS['card_bg']};
            padding: 25px;
            border-radius: 10px;
            text-align: center;
            cursor: pointer;
            transition: transform 0.2s;
        }}
        .feature-card:hover {{
            transform: translateY(-5px);
        }}
        .feature-icon {{
            font-size: 48px;
            margin-bottom: 15px;
        }}
        .feature-title {{
            font-size: 20px;
            font-weight: bold;
            color: {COLORS['text_primary']};
        }}
        .feature-desc {{
            font-size: 16px;
            color: {COLORS['text_secondary']};
        }}
        .event-card {{
            background: {COLORS['card_bg']};
            padding: 20px;
            border-radius: 10px;
            border-top: 4px solid {COLORS['primary']};
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


def main():
    """Main application entry point."""
    # Header
    st.markdown('<div class="main-header">🥊 UFC Prediction App</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">AI-Powered Fight Analysis & Predictions</div>',
        unsafe_allow_html=True,
    )

    # Initialize services
    data_service = DataService()
    accuracy_service = AccuracyService()

    # Navigation cards
    st.markdown("### Quick Navigation")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="feature-card">
                <div class="feature-icon">🥊</div>
                <div class="feature-title">Fighters</div>
                <div class="feature-desc">Search and explore UFC fighter profiles</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Go to Fighters", key="nav_fighters", width='stretch'):
            st.switch_page("pages/1_🥊_Fighters.py")

    with col2:
        st.markdown(
            f"""
            <div class="feature-card">
                <div class="feature-icon">⚔️</div>
                <div class="feature-title">Compare Fighters</div>
                <div class="feature-desc">Head-to-head statistical comparison</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Compare Fighters", key="nav_compare", width='stretch'):
            st.switch_page("pages/2_⚔️_Fighter_Comparison.py")

    with col3:
        st.markdown(
            f"""
            <div class="feature-card">
                <div class="feature-icon">🎯</div>
                <div class="feature-title">Predictions</div>
                <div class="feature-desc">Get AI predictions for any matchup</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Make Predictions", key="nav_predict", width='stretch'):
            st.switch_page("pages/3_🎯_Predictions.py")

    col4, col5, col6 = st.columns(3)

    with col4:
        st.markdown(
            f"""
            <div class="feature-card">
                <div class="feature-icon">📅</div>
                <div class="feature-title">Upcoming Fights</div>
                <div class="feature-desc">View predictions for upcoming events</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("View Upcoming", key="nav_upcoming", width='stretch'):
            st.switch_page("pages/4_📅_Upcoming_Fights.py")

    with col5:
        st.markdown(
            f"""
            <div class="feature-card">
                <div class="feature-icon">📊</div>
                <div class="feature-title">Analysis & Insights</div>
                <div class="feature-desc">Statistics, ML patterns & records</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Explore Analysis", key="nav_trends", width='stretch'):
            st.switch_page("pages/4_📊_Analysis_Insights.py")

    with col6:
        st.markdown(
            f"""
            <div class="feature-card">
                <div class="feature-icon">❓</div>
                <div class="feature-title">How It Works</div>
                <div class="feature-desc">Learn about our methodology</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Learn More", key="nav_how", width='stretch'):
            st.switch_page("pages/6_❓_How_It_Works.py")

    col7, col8, col9 = st.columns(3)

    with col7:
        st.markdown(
            f"""
            <div class="feature-card">
                <div class="feature-icon">🎮</div>
                <div class="feature-title">Fight Simulator</div>
                <div class="feature-desc">Simulate hypothetical matchups</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Simulate Fights", key="nav_simulator", width='stretch'):
            st.switch_page("pages/7_🎮_Fight_Simulator.py")

    with col8:
        st.markdown(
            f"""
            <div class="feature-card">
                <div class="feature-icon">📰</div>
                <div class="feature-title">Latest News</div>
                <div class="feature-desc">MMA headlines & updates</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("View News", key="nav_news", width='stretch'):
            st.switch_page("pages/5_📰_Latest_News.py")

    st.markdown("---")

    # Database stats
    st.markdown("### Database Statistics")

    try:
        stats = data_service.get_database_stats()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-value">{stats.get('fighters', 0):,}</div>
                    <div class="stat-label">Total Fighters</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-value">{stats.get('fights', 0):,}</div>
                    <div class="stat-label">Historical Fights</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-value">{stats.get('events', 0):,}</div>
                    <div class="stat-label">UFC Events</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col4:
            accuracy = accuracy_service.get_rolling_accuracy(100)
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-value">{accuracy:.0%}</div>
                    <div class="stat-label">Model Accuracy (Last 100)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    except Exception as e:
        st.info("Database not initialized. Run `python scripts/init_database.py` to set up.")

    st.markdown("---")

    # Next event
    st.markdown("### Next UFC Event")

    try:
        upcoming_events = data_service.get_upcoming_events(limit=1)

        if upcoming_events:
            event = upcoming_events[0]
            countdown = get_countdown(event.get("date"))

            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(
                    f"""
                    <div class="event-card">
                        <h3 style="color: {COLORS['text_primary']}; margin: 0;">
                            {event.get('name', 'Upcoming Event')}
                        </h3>
                        <p style="color: {COLORS['text_secondary']}; margin: 5px 0;">
                            📍 {event.get('location', 'TBA')}
                        </p>
                        <p style="color: {COLORS['text_secondary']}; margin: 5px 0;">
                            📅 {format_date(event.get('date'))}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col2:
                st.markdown(
                    f"""
                    <div style="
                        background: {COLORS['primary']};
                        padding: 20px;
                        border-radius: 10px;
                        text-align: center;
                    ">
                        <div style="font-size: 16px; color: white;">COUNTDOWN</div>
                        <div style="font-size: 28px; font-weight: bold; color: white;">
                            {countdown}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if st.button("View Event Predictions", key="view_next_event"):
                st.switch_page("pages/4_📅_Upcoming_Fights.py")
        else:
            st.info("No upcoming events scheduled.")

    except Exception as e:
        st.info("No upcoming events data available.")

    st.markdown("---")

    # Recent accuracy
    st.markdown("### Recent Prediction Performance")

    try:
        accuracy_summary = accuracy_service.get_accuracy_summary()

        col1, col2, col3 = st.columns(3)

        with col1:
            rolling_30 = accuracy_summary.get("rolling_30", 0)
            st.metric(
                "Last 30 Predictions",
                f"{rolling_30:.0%}",
                delta=None,
            )

        with col2:
            rolling_100 = accuracy_summary.get("rolling_100", 0)
            st.metric(
                "Last 100 Predictions",
                f"{rolling_100:.0%}",
                delta=None,
            )

        with col3:
            by_confidence = accuracy_summary.get("by_confidence", {})
            high_acc = by_confidence.get("high", {}).get("accuracy", 0)
            st.metric(
                "High Confidence Accuracy",
                f"{high_acc:.0%}",
                delta=None,
            )

    except Exception as e:
        st.info("No prediction accuracy data available yet.")

    # Footer
    st.markdown("---")
    st.markdown(
        f"""
        <div style="text-align: center; color: {COLORS['text_muted']}; font-size: 14px;">
            <p>UFC Prediction App v1.0.0 | Data from UFCStats.com</p>
            <p>For entertainment purposes only. Please gamble responsibly.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
