"""
Latest News Page.

UFC news feed from major MMA news sources.
"""

import streamlit as st

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import COLORS
from services.news_service import UFCNewsService


st.set_page_config(
    page_title="Latest News - UFC Prediction App",
    page_icon="📰",
    layout="wide",
)

st.title("📰 Latest UFC News")
st.markdown("*Stay updated with the latest headlines from the MMA world*")

# Initialize service
news_service = UFCNewsService()


@st.cache_data(ttl=3600)  # Cache for 1 hour in Streamlit
def get_cached_news(limit=16):
    """Fetch and cache UFC news."""
    return news_service.get_latest_news(limit=limit)


st.markdown("---")

# Get news limit from session state (set in How It Works page)
news_limit = st.session_state.get('news_article_count', 12)

# News controls
col1, col2 = st.columns([3, 1])

with col1:
    cache_status = news_service.get_cache_status()
    if cache_status['exists'] and cache_status['timestamp']:
        st.caption(f"Last updated: {cache_status['age_hours']} hours ago | Showing {news_limit} articles")
    else:
        st.caption("Fetching fresh news...")

with col2:
    if st.button("Refresh News", key="refresh_news", use_container_width=True):
        news_service.get_latest_news(limit=news_limit, force_refresh=True)
        st.cache_data.clear()
        st.rerun()

st.markdown("---")

try:
    news_items = get_cached_news(limit=news_limit)

    if news_items:
        # Display news in a grid layout
        cols = st.columns(2)

        for idx, item in enumerate(news_items[:news_limit]):
            with cols[idx % 2]:
                with st.container():
                    st.markdown(f"""
                    <div style="background: {COLORS['card_bg']}; padding: 18px; border-radius: 12px; margin-bottom: 15px; border-left: 4px solid {COLORS['primary']};">
                        <a href="{item['link']}" target="_blank" style="text-decoration: none;">
                            <h4 style="color: {COLORS['text_primary']}; margin: 0 0 10px 0; font-size: 16px; line-height: 1.4;">{item['title']}</h4>
                        </a>
                        <p style="color: {COLORS['text_secondary']}; font-size: 13px; margin: 0 0 12px 0; line-height: 1.5;">{item.get('description', '')[:200]}{'...' if len(item.get('description', '')) > 200 else ''}</p>
                        <div style="display: flex; justify-content: space-between; align-items: center; padding-top: 10px; border-top: 1px solid {COLORS['text_muted']}30;">
                            <span style="color: {COLORS['primary']}; font-size: 12px; font-weight: 500;">{item['source']}</span>
                            <span style="color: {COLORS['text_muted']}; font-size: 12px;">{item['published_display']}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("---")

        # News sources info
        st.markdown("### News Sources")
        st.markdown(f"""
        <div style="background: {COLORS['card_bg']}; padding: 20px; border-radius: 10px;">
            <p style="color: {COLORS['text_secondary']}; margin: 0; line-height: 1.8;">
                News is aggregated from trusted MMA sources including:
            </p>
            <ul style="color: {COLORS['text_primary']}; margin: 10px 0;">
                <li><b>MMAJunkie</b> - Comprehensive MMA news coverage</li>
                <li><b>MMA Fighting</b> - In-depth analysis and breaking news</li>
                <li><b>ESPN MMA</b> - Sports network MMA coverage</li>
            </ul>
            <p style="color: {COLORS['text_muted']}; margin: 0; font-size: 13px;">
                News is refreshed automatically every few hours. Click "Refresh News" for the latest updates.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Unable to fetch news at this time. Please try again later.")

except Exception as e:
    st.warning("News feed temporarily unavailable.")
    st.caption(f"Error: {str(e)}")


# Footer
st.markdown("---")
st.markdown(
    f"""
    <div style="text-align: center; color: {COLORS['text_muted']}; font-size: 14px;">
        <p>News aggregated from multiple sources | External links open in new tab</p>
    </div>
    """,
    unsafe_allow_html=True,
)
