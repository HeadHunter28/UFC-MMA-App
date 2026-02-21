"""
Upcoming Fights & Predictions Page.

View all upcoming UFC events with fight cards, AI predictions, and betting odds comparison.
"""

import logging
import streamlit as st
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

from config import COLORS
from services.data_service import DataService
from services.prediction_service import PredictionService
from services.simulation_service import FightSimulationService
from services.llm_service import LLMService
from services.betting_odds_service import BettingOddsService, BettingOdds
from utils.helpers import format_date, get_countdown


st.set_page_config(
    page_title="Upcoming Fights - UFC Prediction App",
    page_icon="📅",
    layout="wide",
)

st.title("📅 Upcoming Fights & Predictions")
st.markdown("AI-powered predictions for all upcoming UFC events with betting odds comparison.")

# Initialize services
data_service = DataService()
prediction_service = PredictionService()
simulation_service = FightSimulationService()
llm_service = LLMService()
betting_service = BettingOddsService()


# =============================================================================
# RENDERING FUNCTIONS
# =============================================================================

def render_event_card(event: dict, is_next: bool = False):
    """Render an event card with countdown and fight count."""
    countdown = get_countdown(event.get("date"))

    # Get fights for this event
    fights = data_service.get_upcoming_fights(event_id=event.get("event_id"))
    fight_count = len(fights) if fights else 0

    # Determine card style
    if is_next:
        border_color = COLORS['primary']
        bg_gradient = f"linear-gradient(135deg, {COLORS['primary']}20, {COLORS['card_bg']})"
        label = "NEXT EVENT"
    else:
        border_color = COLORS['info']
        bg_gradient = COLORS['card_bg']
        label = ""

    label_html = ""
    if label:
        label_html = f'<div style="color: {COLORS["primary"]}; font-size: 12px; font-weight: bold; margin-bottom: 10px;">{label}</div>'

    badge_color = COLORS['primary'] if is_next else COLORS['info']

    event_html = f'''
    <div style="background: {bg_gradient}; border: 2px solid {border_color}; border-radius: 12px; padding: 20px; margin-bottom: 20px;">
        {label_html}
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <div style="font-size: 22px; font-weight: bold; color: {COLORS["text_primary"]};">{event.get("name", "UFC Event")}</div>
                <div style="color: {COLORS["text_secondary"]}; margin-top: 5px;">📍 {event.get("location", "TBA")} | 📅 {format_date(event.get("date"))}</div>
            </div>
            <div style="text-align: right;">
                <div style="background: {badge_color}; color: white; padding: 8px 15px; border-radius: 20px; font-weight: bold; display: inline-block;">⏱️ {countdown}</div>
                <div style="color: {COLORS["text_muted"]}; margin-top: 8px; font-size: 14px;">{fight_count} fights announced</div>
            </div>
        </div>
    </div>
    '''
    st.markdown(event_html, unsafe_allow_html=True)

    return fights


def render_fighter_card(fighter: dict, corner: str = "red"):
    """Render a detailed fighter card with stats."""
    corner_color = COLORS['primary'] if corner == "red" else COLORS['info']
    corner_label = "RED CORNER" if corner == "red" else "BLUE CORNER"

    stats = data_service.get_fighter_stats(fighter.get("fighter_id")) or {}

    # Calculate age
    age = "N/A"
    dob = fighter.get("dob")
    if dob:
        try:
            from datetime import datetime, date
            if isinstance(dob, str):
                dob_date = datetime.strptime(dob, "%Y-%m-%d").date()
            else:
                dob_date = dob
            today = date.today()
            age = today.year - dob_date.year - ((today.month, today.day) < (dob_date.month, dob_date.day))
        except:
            pass

    # Format physical stats
    height = fighter.get("height_cm")
    height_display = f"{height:.0f}cm" if height else "N/A"
    reach = fighter.get("reach_cm")
    reach_display = f"{reach:.0f}cm" if reach else "N/A"

    fighter_name = fighter.get('name', 'TBA')
    record = f"{fighter.get('wins', 0)}-{fighter.get('losses', 0)}-{fighter.get('draws', 0)}"

    card_html = f'''
    <div style="background: {COLORS["card_bg"]}; border-left: 4px solid {corner_color}; padding: 15px; border-radius: 8px; margin-bottom: 10px;">
        <div style="font-size: 11px; color: {corner_color}; font-weight: 600;">{corner_label}</div>
        <div style="font-size: 20px; font-weight: bold; color: {COLORS["text_primary"]}; margin: 5px 0;">{fighter_name}</div>
        <div style="font-size: 24px; color: {corner_color}; font-weight: bold;">{record}</div>
    </div>
    '''
    st.markdown(card_html, unsafe_allow_html=True)

    # Physical stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Height", height_display)
    with col2:
        st.metric("Reach", reach_display)
    with col3:
        st.metric("Age", age if age != "N/A" else "N/A")

    # Fighting stats
    col1, col2 = st.columns(2)
    with col1:
        str_acc = stats.get("sig_strike_accuracy")
        st.metric("Strike Acc.", f"{str_acc:.0%}" if str_acc else "N/A")
        td_acc = stats.get("takedown_accuracy")
        st.metric("TD Acc.", f"{td_acc:.0%}" if td_acc else "N/A")
    with col2:
        str_def = stats.get("sig_strike_defense")
        st.metric("Strike Def.", f"{str_def:.0%}" if str_def else "N/A")
        td_def = stats.get("takedown_defense")
        st.metric("TD Def.", f"{td_def:.0%}" if td_def else "N/A")

    # Finish rates
    ko_rate = stats.get("ko_rate")
    sub_rate = stats.get("submission_rate")
    if ko_rate or sub_rate:
        ko_display = f"{ko_rate:.0%}" if ko_rate else "0%"
        sub_display = f"{sub_rate:.0%}" if sub_rate else "0%"
        finish_html = f'''
        <div style="margin-top: 10px; padding: 10px; background: {COLORS["background"]}; border-radius: 6px;">
            <span style="color: {COLORS["danger"]};">KO Rate: {ko_display}</span> |
            <span style="color: {COLORS["info"]};">Sub Rate: {sub_display}</span>
        </div>
        '''
        st.markdown(finish_html, unsafe_allow_html=True)


def render_fight_analysis(fighter_a: dict, fighter_b: dict):
    """Render detailed analysis for both fighters."""
    analysis_a = simulation_service.analyze_fighter(fighter_a.get("fighter_id"))
    analysis_b = simulation_service.analyze_fighter(fighter_b.get("fighter_id"))

    col1, col2 = st.columns(2)

    with col1:
        if analysis_a:
            st.markdown(f"#### {fighter_a.get('name')} Analysis")
            st.markdown(f"**Style:** {analysis_a.style_description}")

            trend_emoji = {"improving": "📈", "declining": "📉", "consistent": "➡️", "unknown": "❓"}
            st.markdown(f"**Form:** {trend_emoji.get(analysis_a.recent_trend, '➡️')} {analysis_a.recent_trend.capitalize()}")

            if analysis_a.strengths:
                st.markdown("**Strengths:**")
                for s in analysis_a.strengths[:3]:
                    st.markdown(f"- ✅ {s}")

            if analysis_a.keys_to_victory:
                st.markdown("**Keys to Victory:**")
                for k in analysis_a.keys_to_victory[:2]:
                    st.markdown(f"- 🔑 {k}")

    with col2:
        if analysis_b:
            st.markdown(f"#### {fighter_b.get('name')} Analysis")
            st.markdown(f"**Style:** {analysis_b.style_description}")

            trend_emoji = {"improving": "📈", "declining": "📉", "consistent": "➡️", "unknown": "❓"}
            st.markdown(f"**Form:** {trend_emoji.get(analysis_b.recent_trend, '➡️')} {analysis_b.recent_trend.capitalize()}")

            if analysis_b.strengths:
                st.markdown("**Strengths:**")
                for s in analysis_b.strengths[:3]:
                    st.markdown(f"- ✅ {s}")

            if analysis_b.keys_to_victory:
                st.markdown("**Keys to Victory:**")
                for k in analysis_b.keys_to_victory[:2]:
                    st.markdown(f"- 🔑 {k}")


def render_fight_row(fight: dict, prediction: dict = None):
    """Render a single fight row with prediction (compact view)."""
    fighter_a = data_service.get_fighter_by_id(fight.get("fighter_red_id"))
    fighter_b = data_service.get_fighter_by_id(fight.get("fighter_blue_id"))

    if not fighter_a or not fighter_b:
        return

    # Determine winner if prediction exists
    if prediction:
        winner_id = prediction.get("predicted_winner_id")
        confidence = prediction.get("winner_confidence", 0.5)
        method = prediction.get("predicted_method", "Decision")

        a_is_winner = winner_id == fighter_a.get("fighter_id")
        a_style = f"color: {COLORS['success']}; font-weight: bold;" if a_is_winner else ""
        b_style = f"color: {COLORS['success']}; font-weight: bold;" if not a_is_winner else ""

        winner_name = fighter_a.get('name') if a_is_winner else fighter_b.get('name')
        pick_text = f'<span style="color: {COLORS["success"]}; font-size: 12px;">Pick: {winner_name} by {method} ({confidence:.0%})</span>'
    else:
        a_style = ""
        b_style = ""
        pick_text = f'<span style="color: {COLORS["text_muted"]}; font-size: 12px;">Prediction pending</span>'

    # Fight label
    title_tag = ""
    if fight.get("is_title_fight"):
        title_tag = f'<span style="background: {COLORS["warning"]}; color: black; padding: 2px 8px; border-radius: 4px; font-size: 11px; margin-left: 10px;">TITLE</span>'
    elif fight.get("is_main_event"):
        title_tag = f'<span style="background: {COLORS["primary"]}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; margin-left: 10px;">MAIN</span>'

    fighter_a_name = fighter_a.get('name', 'TBA')
    fighter_a_record = f"({fighter_a.get('wins', 0)}-{fighter_a.get('losses', 0)})"
    fighter_b_name = fighter_b.get('name', 'TBA')
    fighter_b_record = f"({fighter_b.get('wins', 0)}-{fighter_b.get('losses', 0)})"
    weight_class = fight.get('weight_class', 'TBA')

    fight_html = f'''
    <div style="background: {COLORS["card_bg"]}; padding: 15px; border-radius: 8px; margin: 8px 0; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
        <div style="flex: 1;">
            <div style="font-size: 16px;">
                <span style="{a_style}">{fighter_a_name} {fighter_a_record}</span>
                <span style="color: {COLORS["text_muted"]};"> vs </span>
                <span style="{b_style}">{fighter_b_name} {fighter_b_record}</span>
                {title_tag}
            </div>
            <div style="margin-top: 5px;">
                <span style="color: {COLORS["text_secondary"]}; font-size: 13px;">{weight_class}</span>
            </div>
        </div>
        <div style="text-align: right; min-width: 200px;">{pick_text}</div>
    </div>
    '''
    st.markdown(fight_html, unsafe_allow_html=True)


def render_detailed_prediction(
    fight: dict,
    prediction: dict,
    betting_odds: BettingOdds = None,
    is_main_event: bool = False,
    show_fighter_cards: bool = True,
    show_analysis: bool = True,
    show_ai_preview: bool = True
):
    """
    Render a detailed prediction with betting odds, fighter cards, and analysis.
    """
    fighter_a = data_service.get_fighter_by_id(fight.get("fighter_red_id"))
    fighter_b = data_service.get_fighter_by_id(fight.get("fighter_blue_id"))

    if not fighter_a or not fighter_b:
        return

    # Header styling
    if is_main_event:
        header_bg = f"linear-gradient(135deg, {COLORS['primary']}30, {COLORS['secondary']}30)"
        border_color = COLORS['primary']
        header_label = "MAIN EVENT"
    elif fight.get("is_title_fight"):
        header_bg = f"linear-gradient(135deg, {COLORS['warning']}20, {COLORS['card_bg']})"
        border_color = COLORS['warning']
        header_label = "TITLE FIGHT"
    else:
        header_bg = COLORS['card_bg']
        border_color = COLORS['info']
        header_label = ""

    # Fighter info
    fighter_a_name = fighter_a.get('name', 'TBA')
    fighter_a_record = f"{fighter_a.get('wins', 0)}-{fighter_a.get('losses', 0)}-{fighter_a.get('draws', 0)}"
    fighter_b_name = fighter_b.get('name', 'TBA')
    fighter_b_record = f"{fighter_b.get('wins', 0)}-{fighter_b.get('losses', 0)}-{fighter_b.get('draws', 0)}"
    weight_class = fight.get('weight_class', 'TBA')

    # Build header
    label_html = f'<div style="color: {border_color}; font-size: 12px; font-weight: bold; margin-bottom: 10px;">{header_label}</div>' if header_label else ""

    fight_header = f'''
    <div style="background: {header_bg}; border: 2px solid {border_color}; border-radius: 12px; padding: 20px; margin-bottom: 15px;">
        {label_html}
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div style="flex: 1; text-align: center;">
                <div style="font-size: 11px; color: {COLORS["primary"]};">RED CORNER</div>
                <div style="font-size: 22px; font-weight: bold; color: {COLORS["text_primary"]}; margin: 5px 0;">{fighter_a_name}</div>
                <div style="font-size: 16px; color: {COLORS["text_secondary"]};">{fighter_a_record}</div>
            </div>
            <div style="padding: 0 20px;">
                <div style="font-size: 24px; color: {COLORS["text_muted"]};">VS</div>
                <div style="font-size: 12px; color: {COLORS["text_secondary"]}; margin-top: 5px;">{weight_class}</div>
            </div>
            <div style="flex: 1; text-align: center;">
                <div style="font-size: 11px; color: {COLORS["info"]};">BLUE CORNER</div>
                <div style="font-size: 22px; font-weight: bold; color: {COLORS["text_primary"]}; margin: 5px 0;">{fighter_b_name}</div>
                <div style="font-size: 16px; color: {COLORS["text_secondary"]};">{fighter_b_record}</div>
            </div>
        </div>
    </div>
    '''
    st.markdown(fight_header, unsafe_allow_html=True)

    # Fighter detailed cards (optional)
    if show_fighter_cards:
        col1, col2 = st.columns(2)
        with col1:
            render_fighter_card(fighter_a, corner="red")
        with col2:
            render_fighter_card(fighter_b, corner="blue")
        st.markdown("---")

    # Predictions comparison - AI vs Vegas
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"#### 🤖 AI Prediction")

        if prediction:
            winner_id = prediction.get("predicted_winner_id")
            confidence = prediction.get("winner_confidence", 0.5)
            method = prediction.get("predicted_method", "Decision")
            ko_prob = prediction.get("method_ko_prob", 0.33)
            sub_prob = prediction.get("method_sub_prob", 0.33)
            dec_prob = prediction.get("method_dec_prob", 0.34)

            winner = fighter_a if winner_id == fighter_a.get("fighter_id") else fighter_b
            loser = fighter_b if winner_id == fighter_a.get("fighter_id") else fighter_a
            winner_color = COLORS['primary'] if winner_id == fighter_a.get("fighter_id") else COLORS['info']

            winner_name = winner.get('name', 'TBA')
            loser_name = loser.get('name', 'TBA')

            pred_html = f'''
            <div style="background: {winner_color}20; border: 2px solid {winner_color}; border-radius: 10px; padding: 20px; text-align: center;">
                <div style="font-size: 12px; color: {COLORS["text_muted"]};">PREDICTED WINNER</div>
                <div style="font-size: 24px; font-weight: bold; color: {winner_color}; margin: 8px 0;">{winner_name}</div>
                <div style="font-size: 14px; color: {COLORS["text_secondary"]};">defeats {loser_name}</div>
                <div style="margin-top: 15px;">
                    <span style="background: {winner_color}; color: white; padding: 8px 20px; border-radius: 20px; font-weight: bold; font-size: 18px;">{confidence:.0%}</span>
                </div>
                <div style="margin-top: 15px; font-size: 13px; color: {COLORS["text_secondary"]};">via <strong>{method}</strong></div>
            </div>
            '''
            st.markdown(pred_html, unsafe_allow_html=True)

            # Method breakdown
            st.markdown("**Method Probabilities:**")
            method_col1, method_col2, method_col3 = st.columns(3)
            with method_col1:
                st.metric("KO/TKO", f"{ko_prob:.0%}")
            with method_col2:
                st.metric("Submission", f"{sub_prob:.0%}")
            with method_col3:
                st.metric("Decision", f"{dec_prob:.0%}")
        else:
            st.info("Generating prediction...")

    with col2:
        st.markdown(f"#### 📊 Vegas Odds")

        if betting_odds:
            odds_a = betting_odds.fighter_a_odds
            odds_b = betting_odds.fighter_b_odds
            prob_a = betting_odds.fighter_a_implied_prob
            prob_b = betting_odds.fighter_b_implied_prob
            favorite = betting_odds.favorite

            # Determine favorite display
            if favorite == "fighter_a":
                fav_color = COLORS['primary']
                fav_name = fighter_a_name
            elif favorite == "fighter_b":
                fav_color = COLORS['info']
                fav_name = fighter_b_name
            else:
                fav_color = COLORS['warning']
                fav_name = "Pick 'Em"

            # Format odds display
            odds_a_display = f"+{odds_a}" if odds_a and odds_a > 0 else str(odds_a) if odds_a else "N/A"
            odds_b_display = f"+{odds_b}" if odds_b and odds_b > 0 else str(odds_b) if odds_b else "N/A"

            odds_a_color = COLORS["primary"] if odds_a and odds_a < 0 else COLORS["text_secondary"]
            odds_b_color = COLORS["info"] if odds_b and odds_b < 0 else COLORS["text_secondary"]

            odds_html = f'''
            <div style="background: {COLORS["card_bg"]}; border: 2px solid {COLORS["text_muted"]}; border-radius: 10px; padding: 20px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 15px;">
                    <div style="text-align: center; flex: 1;">
                        <div style="font-size: 12px; color: {COLORS["text_muted"]};">{fighter_a_name}</div>
                        <div style="font-size: 24px; font-weight: bold; color: {odds_a_color};">{odds_a_display}</div>
                        <div style="font-size: 13px; color: {COLORS["text_secondary"]};">{prob_a:.0%} implied</div>
                    </div>
                    <div style="text-align: center; flex: 1;">
                        <div style="font-size: 12px; color: {COLORS["text_muted"]};">{fighter_b_name}</div>
                        <div style="font-size: 24px; font-weight: bold; color: {odds_b_color};">{odds_b_display}</div>
                        <div style="font-size: 13px; color: {COLORS["text_secondary"]};">{prob_b:.0%} implied</div>
                    </div>
                </div>
                <div style="text-align: center; padding-top: 10px; border-top: 1px solid {COLORS["text_muted"]}30;">
                    <div style="font-size: 12px; color: {COLORS["text_muted"]};">VEGAS FAVORITE</div>
                    <div style="font-size: 18px; font-weight: bold; color: {fav_color}; margin-top: 5px;">{fav_name}</div>
                </div>
            </div>
            '''
            st.markdown(odds_html, unsafe_allow_html=True)
            st.caption(f"Source: {betting_odds.source}")
        else:
            no_odds_html = f'''
            <div style="background: {COLORS["card_bg"]}; border: 2px dashed {COLORS["text_muted"]}; border-radius: 10px; padding: 30px; text-align: center;">
                <div style="font-size: 32px; margin-bottom: 10px;">📈</div>
                <div style="color: {COLORS["text_muted"]}; font-size: 14px;">Betting odds not yet available</div>
                <div style="color: {COLORS["text_secondary"]}; font-size: 12px; margin-top: 10px;">Odds typically open 1-2 weeks before the event</div>
            </div>
            '''
            st.markdown(no_odds_html, unsafe_allow_html=True)

    # Agreement/Disagreement indicator
    if prediction and betting_odds and betting_odds.favorite != "pick_em":
        our_pick = prediction.get("predicted_winner_id")
        vegas_pick_id = fighter_a.get("fighter_id") if betting_odds.favorite == "fighter_a" else fighter_b.get("fighter_id")

        if our_pick == vegas_pick_id:
            agreement_html = f'''
            <div style="background: {COLORS["success"]}20; border: 1px solid {COLORS["success"]}; border-radius: 8px; padding: 12px; margin-top: 15px; text-align: center;">
                <span style="color: {COLORS["success"]}; font-weight: bold;">✅ AGREEMENT</span>
                <span style="color: {COLORS["text_secondary"]}; margin-left: 10px;">Our AI and Vegas both favor the same fighter</span>
            </div>
            '''
        else:
            agreement_html = f'''
            <div style="background: {COLORS["warning"]}20; border: 1px solid {COLORS["warning"]}; border-radius: 8px; padding: 12px; margin-top: 15px; text-align: center;">
                <span style="color: {COLORS["warning"]}; font-weight: bold;">⚡ CONTRARIAN PICK</span>
                <span style="color: {COLORS["text_secondary"]}; margin-left: 10px;">Our AI disagrees with Vegas - potential value!</span>
            </div>
            '''
        st.markdown(agreement_html, unsafe_allow_html=True)

    # Fighter analysis section
    if show_analysis:
        st.markdown("---")
        st.markdown("### Fighter Analysis")
        render_fight_analysis(fighter_a, fighter_b)

    # AI narrative preview
    if show_ai_preview and llm_service.is_available() and prediction:
        st.markdown("---")
        st.markdown("### AI Fight Preview")

        with st.spinner("Generating analysis..."):
            winner_name = (
                fighter_a.get('name')
                if prediction.get('predicted_winner_id') == fighter_a.get('fighter_id')
                else fighter_b.get('name')
            )
            loser_name = (
                fighter_b.get('name')
                if prediction.get('predicted_winner_id') == fighter_a.get('fighter_id')
                else fighter_a.get('name')
            )

            prompt = f"""Write a brief fight preview for {fighter_a.get('name')} vs {fighter_b.get('name')}.

            {fighter_a.get('name')} record: {fighter_a.get('wins', 0)}-{fighter_a.get('losses', 0)}
            {fighter_b.get('name')} record: {fighter_b.get('wins', 0)}-{fighter_b.get('losses', 0)}

            Prediction: {winner_name} to defeat {loser_name} via {prediction.get('predicted_method', 'Decision')} ({prediction.get('winner_confidence', 0.5):.0%} confidence).

            Keep it to 3-4 sentences about key matchup factors and what to watch for."""

            analysis = llm_service._generate(prompt)
            if analysis:
                st.markdown(analysis)

    st.markdown("---")


# =============================================================================
# MAIN PAGE CONTENT
# =============================================================================

# Get all upcoming events
try:
    upcoming_events = data_service.get_upcoming_events(limit=10)
except Exception as e:
    upcoming_events = []
    st.error(f"Error fetching events: {e}")

if upcoming_events:
    st.markdown(f"**{len(upcoming_events)} upcoming events**")
    st.markdown("---")

    # Render each event
    for idx, event in enumerate(upcoming_events):
        is_next = (idx == 0)

        # Render event card
        fights = render_event_card(event, is_next=is_next)

        if fights:
            # Group by card position
            main_card = [f for f in fights if f.get("card_position") == "main_card"]
            prelims = [f for f in fights if f.get("card_position") == "prelims"]
            early_prelims = [f for f in fights if f.get("card_position") == "early_prelims"]

            # If no positions set, assume order
            if not main_card and not prelims and not early_prelims:
                if len(fights) <= 5:
                    main_card = fights
                else:
                    main_card = fights[:5]
                    prelims = fights[5:]

            # Main Card - Enhanced display for the next event
            if main_card:
                if is_next:
                    # Detailed predictions view for the next event's main card
                    st.markdown("### Main Card Predictions")
                    st.markdown("*AI predictions vs Vegas betting odds*")

                    # Try to get betting odds for the event
                    event_name = event.get("name", "")

                    for fight_idx, fight in enumerate(main_card):
                        is_main_event = (fight_idx == 0 or fight.get("is_main_event"))

                        # Get or generate prediction
                        prediction = data_service.get_prediction(upcoming_id=fight.get("upcoming_id"))

                        if not prediction:
                            # Auto-generate prediction
                            pred_result = prediction_service.predict(
                                fight.get("fighter_red_id"),
                                fight.get("fighter_blue_id"),
                                context={
                                    "weight_class": fight.get("weight_class"),
                                    "is_title_fight": fight.get("is_title_fight"),
                                    "is_main_event": fight.get("is_main_event"),
                                },
                            )
                            if pred_result:
                                prediction = pred_result.to_dict()

                        # Get betting odds
                        fighter_a = data_service.get_fighter_by_id(fight.get("fighter_red_id"))
                        fighter_b = data_service.get_fighter_by_id(fight.get("fighter_blue_id"))

                        betting_odds = None
                        if fighter_a and fighter_b:
                            try:
                                betting_odds = betting_service.get_fight_odds(
                                    fighter_a.get("name", ""),
                                    fighter_b.get("name", ""),
                                    event_name
                                )
                            except Exception as e:
                                logger.warning(f"Could not fetch betting odds: {e}")

                        # Render detailed prediction
                        # Main event gets full treatment, others get slightly condensed
                        render_detailed_prediction(
                            fight,
                            prediction,
                            betting_odds,
                            is_main_event,
                            show_fighter_cards=(fight_idx < 2),  # Only first 2 fights get full cards
                            show_analysis=(fight_idx == 0),  # Only main event gets analysis
                            show_ai_preview=(fight_idx == 0)  # Only main event gets AI preview
                        )

                    # Summary table
                    st.markdown("### Predictions Summary")
                    summary_data = []
                    for fight in main_card:
                        fighter_a = data_service.get_fighter_by_id(fight.get("fighter_red_id"))
                        fighter_b = data_service.get_fighter_by_id(fight.get("fighter_blue_id"))
                        prediction = data_service.get_prediction(upcoming_id=fight.get("upcoming_id"))

                        if prediction and fighter_a and fighter_b:
                            winner_id = prediction.get("predicted_winner_id")
                            winner_name = fighter_a.get('name') if winner_id == fighter_a.get('fighter_id') else fighter_b.get('name')

                            summary_data.append({
                                "Fight": f"{fighter_a.get('name')} vs {fighter_b.get('name')}",
                                "Pick": winner_name,
                                "Method": prediction.get("predicted_method", "Decision"),
                                "Confidence": f"{prediction.get('winner_confidence', 0.5):.0%}",
                            })

                    if summary_data:
                        df = pd.DataFrame(summary_data)
                        st.dataframe(df, use_container_width=True, hide_index=True)

                else:
                    # Compact view for future events
                    with st.expander(f"Main Card ({len(main_card)} fights)", expanded=False):
                        for fight in main_card:
                            prediction = data_service.get_prediction(upcoming_id=fight.get("upcoming_id"))
                            render_fight_row(fight, prediction)

            # Prelims
            if prelims:
                with st.expander(f"Prelims ({len(prelims)} fights)", expanded=False):
                    for fight in prelims:
                        prediction = data_service.get_prediction(upcoming_id=fight.get("upcoming_id"))
                        render_fight_row(fight, prediction)

            # Early Prelims
            if early_prelims:
                with st.expander(f"Early Prelims ({len(early_prelims)} fights)", expanded=False):
                    for fight in early_prelims:
                        prediction = data_service.get_prediction(upcoming_id=fight.get("upcoming_id"))
                        render_fight_row(fight, prediction)

        else:
            no_card_html = f'''
            <div style="padding: 20px; text-align: center; color: {COLORS["text_muted"]}; background: {COLORS["card_bg"]}; border-radius: 8px; margin-bottom: 20px;">
                Fight card not yet announced
            </div>
            '''
            st.markdown(no_card_html, unsafe_allow_html=True)

        st.markdown("---")

    # Export section
    st.markdown("## 📥 Export Predictions")

    if st.button("Generate CSV Export", type="primary"):
        export_data = []

        for event in upcoming_events:
            fights = data_service.get_upcoming_fights(event_id=event.get("event_id"))

            if fights:
                for fight in fights:
                    fighter_a = data_service.get_fighter_by_id(fight.get("fighter_red_id"))
                    fighter_b = data_service.get_fighter_by_id(fight.get("fighter_blue_id"))
                    prediction = data_service.get_prediction(upcoming_id=fight.get("upcoming_id"))

                    if fighter_a and fighter_b:
                        winner_name = "N/A"
                        if prediction:
                            winner_id = prediction.get("predicted_winner_id")
                            winner_name = fighter_a.get('name') if winner_id == fighter_a.get('fighter_id') else fighter_b.get('name')

                        export_data.append({
                            "Event": event.get("name"),
                            "Date": event.get("date"),
                            "Fighter A": fighter_a.get("name"),
                            "Fighter A Record": f"{fighter_a.get('wins', 0)}-{fighter_a.get('losses', 0)}",
                            "Fighter B": fighter_b.get("name"),
                            "Fighter B Record": f"{fighter_b.get('wins', 0)}-{fighter_b.get('losses', 0)}",
                            "Weight Class": fight.get("weight_class"),
                            "Card Position": fight.get("card_position", "main_card"),
                            "Predicted Winner": winner_name,
                            "Method": prediction.get("predicted_method", "N/A") if prediction else "N/A",
                            "Confidence": f"{prediction.get('winner_confidence', 0):.0%}" if prediction else "N/A",
                        })

        if export_data:
            df = pd.DataFrame(export_data)
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                csv,
                file_name="ufc_predictions_all_events.csv",
                mime="text/csv",
            )
        else:
            st.warning("No data to export")

else:
    st.markdown("---")
    st.warning(
        """
        **No upcoming events found in the database.**

        To populate upcoming events, run:
        ```
        python scripts/manual_update.py
        ```

        This will scrape the latest UFC event data from UFCStats.com.
        """
    )

    # Show what data we have
    st.markdown("### Database Status")

    try:
        with data_service.get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM events WHERE is_completed = 0")
            pending_events = cursor.fetchone()[0]

            cursor = conn.execute("SELECT COUNT(*) FROM events")
            total_events = cursor.fetchone()[0]

            cursor = conn.execute("SELECT COUNT(*) FROM fighters")
            total_fighters = cursor.fetchone()[0]

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Pending Events", pending_events)
            with col2:
                st.metric("Total Events", total_events)
            with col3:
                st.metric("Total Fighters", total_fighters)
    except:
        st.info("Database not initialized")

# Footer
st.markdown("---")
ai_available = 'Yes' if prediction_service.is_available() else 'No (using heuristics)'
footer_html = f'''
<div style="text-align: center; color: {COLORS["text_muted"]}; font-size: 12px;">
    <p>Model Version: {prediction_service.model_version} | AI Available: {ai_available}</p>
    <p>Predictions are for entertainment purposes only</p>
</div>
'''
st.markdown(footer_html, unsafe_allow_html=True)
