"""
Fight Simulator Page.

Simulate hypothetical fights between any two active UFC fighters.
Enhanced with detailed fighter profiles, physical attributes, and sanity checks.
"""

import streamlit as st
import time

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    COLORS,
    SIMULATION_DEFAULT_ROUNDS,
    SIMULATION_TITLE_ROUNDS,
    FIGHTER_ACTIVITY_CUTOFF_YEARS,
)
from services.simulation_service import (
    FightSimulationService,
    SimulationResult,
    FighterProfile,
    FighterTendencies,
    SanityCheck,
)


st.set_page_config(
    page_title="Fight Simulator - UFC Prediction App",
    page_icon="🎮",
    layout="wide",
)

st.title("🎮 Fight Simulator")
st.markdown("Simulate hypothetical fights between any two active UFC fighters.")

# Activity cutoff notice
cutoff_date = FightSimulationService().get_activity_cutoff_date()
st.info(
    f"**Note:** Only active fighters are available for simulation. "
    f"A fighter is considered active if they have competed since **{cutoff_date.strftime('%B %Y')}** "
    f"({FIGHTER_ACTIVITY_CUTOFF_YEARS} year cutoff)."
)

# Initialize services
simulation_service = FightSimulationService()


def render_fighter_profile_card(
    profile: FighterProfile,
    corner: str = "red"
) -> None:
    """Render an enhanced fighter profile card with all UFC stats."""
    corner_color = COLORS['primary'] if corner == "red" else COLORS['info']
    corner_emoji = "🔴" if corner == "red" else "🔵"

    # Build nickname HTML separately to avoid nested f-string issues
    nickname_html = ""
    if profile.nickname:
        nickname_html = f'<div style="color: {COLORS["text_secondary"]}; font-style: italic; font-size: 14px;">"{profile.nickname}"</div>'

    # Header with name and record - inline to avoid rendering issues
    corner_label = f"{corner.upper()} CORNER"
    header_html = f'<div style="background: linear-gradient(135deg, {corner_color}20, {COLORS["card_bg"]}); padding: 20px; border-radius: 12px; border-left: 4px solid {corner_color}; margin-bottom: 15px;"><div style="display: flex; justify-content: space-between; align-items: center;"><div><span style="font-size: 14px; color: {COLORS["text_muted"]};">{corner_emoji} {corner_label}</span><h3 style="margin: 5px 0; color: {COLORS["text_primary"]};">{profile.name}</h3>{nickname_html}</div><div style="text-align: right;"><div style="font-size: 28px; font-weight: bold; color: {corner_color};">{profile.record}</div><div style="color: {COLORS["text_muted"]}; font-size: 14px;">W-L-D</div></div></div></div>'
    st.markdown(header_html, unsafe_allow_html=True)

    # Physical attributes row 1
    col1, col2, col3 = st.columns(3)

    with col1:
        height_html = f'<div style="text-align: center; padding: 10px; background: {COLORS["card_bg"]}; border-radius: 8px;"><div style="color: {COLORS["text_muted"]}; font-size: 14px;">HEIGHT</div><div style="color: {COLORS["text_primary"]}; font-size: 16px; font-weight: bold;">{profile.height_display}</div></div>'
        st.markdown(height_html, unsafe_allow_html=True)

    with col2:
        reach_html = f'<div style="text-align: center; padding: 10px; background: {COLORS["card_bg"]}; border-radius: 8px;"><div style="color: {COLORS["text_muted"]}; font-size: 14px;">REACH</div><div style="color: {COLORS["text_primary"]}; font-size: 16px; font-weight: bold;">{profile.reach_display}</div></div>'
        st.markdown(reach_html, unsafe_allow_html=True)

    with col3:
        age_display = f"{profile.age} yrs" if profile.age else "N/A"
        age_html = f'<div style="text-align: center; padding: 10px; background: {COLORS["card_bg"]}; border-radius: 8px;"><div style="color: {COLORS["text_muted"]}; font-size: 14px;">AGE</div><div style="color: {COLORS["text_primary"]}; font-size: 16px; font-weight: bold;">{age_display}</div></div>'
        st.markdown(age_html, unsafe_allow_html=True)

    # Physical attributes row 2: Stance, Division, Nationality
    col1, col2, col3 = st.columns(3)

    with col1:
        stance = profile.stance or "N/A"
        stance_html = f'<div style="text-align: center; padding: 10px; background: {COLORS["card_bg"]}; border-radius: 8px; margin-top: 8px;"><div style="color: {COLORS["text_muted"]}; font-size: 14px;">STANCE</div><div style="color: {COLORS["text_primary"]}; font-size: 14px; font-weight: bold;">{stance}</div></div>'
        st.markdown(stance_html, unsafe_allow_html=True)

    with col2:
        division = profile.weight_class or "Unknown"
        div_display = division[:15] + ('...' if len(division) > 15 else '')
        division_html = f'<div style="text-align: center; padding: 10px; background: {COLORS["card_bg"]}; border-radius: 8px; margin-top: 8px;"><div style="color: {COLORS["text_muted"]}; font-size: 14px;">DIVISION</div><div style="color: {COLORS["text_primary"]}; font-size: 14px; font-weight: bold;">{div_display}</div></div>'
        st.markdown(division_html, unsafe_allow_html=True)

    with col3:
        nationality = profile.nationality or "N/A"
        nat_display = nationality[:12] + ('...' if len(nationality) > 12 else '')
        nationality_html = f'<div style="text-align: center; padding: 10px; background: {COLORS["card_bg"]}; border-radius: 8px; margin-top: 8px;"><div style="color: {COLORS["text_muted"]}; font-size: 14px;">NATIONALITY</div><div style="color: {COLORS["text_primary"]}; font-size: 14px; font-weight: bold;">{nat_display}</div></div>'
        st.markdown(nationality_html, unsafe_allow_html=True)

    # ============ UFC STRIKING STATS ============
    st.markdown(f'<div style="margin-top: 15px; padding: 8px 12px; background: {corner_color}30; border-radius: 8px 8px 0 0; border-bottom: 2px solid {corner_color};"><strong style="color: {COLORS["text_primary"]};">🥊 STRIKING</strong></div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        slpm = f"{profile.sig_strikes_landed_per_min:.2f}" if profile.sig_strikes_landed_per_min else "N/A"
        slpm_html = f'<div style="text-align: center; padding: 12px; background: {COLORS["card_bg"]}; border-radius: 0 0 0 8px;"><div style="color: {COLORS["text_muted"]}; font-size: 14px;">SLpM</div><div style="color: {COLORS["text_primary"]}; font-size: 18px; font-weight: bold;">{slpm}</div><div style="color: {COLORS["text_secondary"]}; font-size: 12px;">Strikes Landed/Min</div></div>'
        st.markdown(slpm_html, unsafe_allow_html=True)

    with col2:
        str_acc = f"{profile.sig_strike_accuracy:.0%}" if profile.sig_strike_accuracy else "N/A"
        str_acc_html = f'<div style="text-align: center; padding: 12px; background: {COLORS["card_bg"]};"><div style="color: {COLORS["text_muted"]}; font-size: 14px;">Str. Acc.</div><div style="color: {COLORS["text_primary"]}; font-size: 18px; font-weight: bold;">{str_acc}</div><div style="color: {COLORS["text_secondary"]}; font-size: 12px;">Striking Accuracy</div></div>'
        st.markdown(str_acc_html, unsafe_allow_html=True)

    with col3:
        sapm = f"{profile.sig_strikes_absorbed_per_min:.2f}" if profile.sig_strikes_absorbed_per_min else "N/A"
        sapm_html = f'<div style="text-align: center; padding: 12px; background: {COLORS["card_bg"]};"><div style="color: {COLORS["text_muted"]}; font-size: 14px;">SApM</div><div style="color: {COLORS["text_primary"]}; font-size: 18px; font-weight: bold;">{sapm}</div><div style="color: {COLORS["text_secondary"]}; font-size: 12px;">Strikes Absorbed/Min</div></div>'
        st.markdown(sapm_html, unsafe_allow_html=True)

    with col4:
        str_def = f"{profile.sig_strike_defense:.0%}" if profile.sig_strike_defense else "N/A"
        str_def_html = f'<div style="text-align: center; padding: 12px; background: {COLORS["card_bg"]}; border-radius: 0 0 8px 0;"><div style="color: {COLORS["text_muted"]}; font-size: 14px;">Str. Def.</div><div style="color: {COLORS["text_primary"]}; font-size: 18px; font-weight: bold;">{str_def}</div><div style="color: {COLORS["text_secondary"]}; font-size: 12px;">Striking Defense</div></div>'
        st.markdown(str_def_html, unsafe_allow_html=True)

    # ============ UFC GRAPPLING STATS ============
    st.markdown(f'<div style="margin-top: 10px; padding: 8px 12px; background: {corner_color}30; border-radius: 8px 8px 0 0; border-bottom: 2px solid {corner_color};"><strong style="color: {COLORS["text_primary"]};">🤼 GRAPPLING</strong></div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        td_avg = f"{profile.takedowns_per_15min:.2f}" if profile.takedowns_per_15min else "N/A"
        td_avg_html = f'<div style="text-align: center; padding: 12px; background: {COLORS["card_bg"]}; border-radius: 0 0 0 8px;"><div style="color: {COLORS["text_muted"]}; font-size: 14px;">TD Avg.</div><div style="color: {COLORS["text_primary"]}; font-size: 18px; font-weight: bold;">{td_avg}</div><div style="color: {COLORS["text_secondary"]}; font-size: 12px;">Takedowns/15min</div></div>'
        st.markdown(td_avg_html, unsafe_allow_html=True)

    with col2:
        td_acc = f"{profile.takedown_accuracy:.0%}" if profile.takedown_accuracy else "N/A"
        td_acc_html = f'<div style="text-align: center; padding: 12px; background: {COLORS["card_bg"]};"><div style="color: {COLORS["text_muted"]}; font-size: 14px;">TD Acc.</div><div style="color: {COLORS["text_primary"]}; font-size: 18px; font-weight: bold;">{td_acc}</div><div style="color: {COLORS["text_secondary"]}; font-size: 12px;">Takedown Accuracy</div></div>'
        st.markdown(td_acc_html, unsafe_allow_html=True)

    with col3:
        td_def = f"{profile.takedown_defense:.0%}" if profile.takedown_defense else "N/A"
        td_def_html = f'<div style="text-align: center; padding: 12px; background: {COLORS["card_bg"]};"><div style="color: {COLORS["text_muted"]}; font-size: 14px;">TD Def.</div><div style="color: {COLORS["text_primary"]}; font-size: 18px; font-weight: bold;">{td_def}</div><div style="color: {COLORS["text_secondary"]}; font-size: 12px;">Takedown Defense</div></div>'
        st.markdown(td_def_html, unsafe_allow_html=True)

    with col4:
        sub_avg = f"{profile.submissions_per_15min:.1f}" if profile.submissions_per_15min else "N/A"
        sub_avg_html = f'<div style="text-align: center; padding: 12px; background: {COLORS["card_bg"]}; border-radius: 0 0 8px 0;"><div style="color: {COLORS["text_muted"]}; font-size: 14px;">Sub. Avg.</div><div style="color: {COLORS["text_primary"]}; font-size: 18px; font-weight: bold;">{sub_avg}</div><div style="color: {COLORS["text_secondary"]}; font-size: 12px;">Submissions/15min</div></div>'
        st.markdown(sub_avg_html, unsafe_allow_html=True)

    # ============ WIN METHODS & FINISH RATES ============
    st.markdown(f'<div style="margin-top: 10px; padding: 8px 12px; background: {corner_color}30; border-radius: 8px 8px 0 0; border-bottom: 2px solid {corner_color};"><strong style="color: {COLORS["text_primary"]};">🏆 WIN BREAKDOWN</strong></div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        ko_rate = f"{profile.ko_rate:.0%}" if profile.ko_rate else "N/A"
        ko_wins_display = f"({profile.ko_wins} KOs)" if profile.ko_wins else ""
        ko_html = f'<div style="text-align: center; padding: 12px; background: {COLORS["card_bg"]}; border-radius: 0 0 0 8px;"><div style="color: {COLORS["danger"]}; font-size: 14px;">KO/TKO</div><div style="color: {COLORS["text_primary"]}; font-size: 18px; font-weight: bold;">{ko_rate}</div><div style="color: {COLORS["text_secondary"]}; font-size: 12px;">{ko_wins_display}</div></div>'
        st.markdown(ko_html, unsafe_allow_html=True)

    with col2:
        sub_rate = f"{profile.submission_rate:.0%}" if profile.submission_rate else "N/A"
        sub_wins_display = f"({profile.sub_wins} Subs)" if profile.sub_wins else ""
        sub_html = f'<div style="text-align: center; padding: 12px; background: {COLORS["card_bg"]};"><div style="color: {COLORS["info"]}; font-size: 14px;">Submission</div><div style="color: {COLORS["text_primary"]}; font-size: 18px; font-weight: bold;">{sub_rate}</div><div style="color: {COLORS["text_secondary"]}; font-size: 12px;">{sub_wins_display}</div></div>'
        st.markdown(sub_html, unsafe_allow_html=True)

    with col3:
        dec_rate = f"{profile.decision_rate:.0%}" if profile.decision_rate else "N/A"
        dec_wins_display = f"({profile.dec_wins} Dec)" if profile.dec_wins else ""
        dec_html = f'<div style="text-align: center; padding: 12px; background: {COLORS["card_bg"]};"><div style="color: {COLORS["warning"]}; font-size: 14px;">Decision</div><div style="color: {COLORS["text_primary"]}; font-size: 18px; font-weight: bold;">{dec_rate}</div><div style="color: {COLORS["text_secondary"]}; font-size: 12px;">{dec_wins_display}</div></div>'
        st.markdown(dec_html, unsafe_allow_html=True)

    with col4:
        finish_rate = f"{profile.finish_rate:.0%}" if profile.finish_rate else "N/A"
        finish_html = f'<div style="text-align: center; padding: 12px; background: {COLORS["card_bg"]}; border-radius: 0 0 8px 0;"><div style="color: {COLORS["success"]}; font-size: 14px;">Finish Rate</div><div style="color: {COLORS["text_primary"]}; font-size: 18px; font-weight: bold;">{finish_rate}</div><div style="color: {COLORS["text_secondary"]}; font-size: 12px;">Total Finishes</div></div>'
        st.markdown(finish_html, unsafe_allow_html=True)

    # ============ ACTIVITY & EXPERIENCE ============
    st.markdown(f'<div style="margin-top: 10px; padding: 8px 12px; background: {corner_color}30; border-radius: 8px 8px 0 0; border-bottom: 2px solid {corner_color};"><strong style="color: {COLORS["text_primary"]};">📊 EXPERIENCE</strong></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        total_fights = profile.total_fights
        total_html = f'<div style="text-align: center; padding: 12px; background: {COLORS["card_bg"]}; border-radius: 0 0 0 8px;"><div style="color: {COLORS["text_muted"]}; font-size: 14px;">Total Pro Fights</div><div style="color: {COLORS["text_primary"]}; font-size: 18px; font-weight: bold;">{total_fights}</div></div>'
        st.markdown(total_html, unsafe_allow_html=True)

    with col2:
        ufc_fights = profile.ufc_fights
        ufc_html = f'<div style="text-align: center; padding: 12px; background: {COLORS["card_bg"]};"><div style="color: {COLORS["text_muted"]}; font-size: 14px;">UFC Fights</div><div style="color: {COLORS["text_primary"]}; font-size: 18px; font-weight: bold;">{ufc_fights}</div></div>'
        st.markdown(ufc_html, unsafe_allow_html=True)

    with col3:
        last_fight = profile.last_fight_date or "N/A"
        if last_fight != "N/A":
            try:
                from datetime import datetime
                if isinstance(last_fight, str):
                    last_date = datetime.strptime(last_fight, "%Y-%m-%d")
                    last_fight = last_date.strftime("%b %Y")
            except:
                pass
        last_html = f'<div style="text-align: center; padding: 12px; background: {COLORS["card_bg"]}; border-radius: 0 0 8px 0;"><div style="color: {COLORS["text_muted"]}; font-size: 14px;">Last Fight</div><div style="color: {COLORS["text_primary"]}; font-size: 14px; font-weight: bold;">{last_fight}</div></div>'
        st.markdown(last_html, unsafe_allow_html=True)

    # ============ RECENT FORM ============
    if profile.recent_results:
        st.markdown(f'<div style="margin-top: 10px; padding: 8px 12px; background: {corner_color}30; border-radius: 8px 8px 0 0; border-bottom: 2px solid {corner_color};"><strong style="color: {COLORS["text_primary"]};">📈 RECENT FORM</strong></div>', unsafe_allow_html=True)

        form_html = f'<div style="padding: 12px; background: {COLORS["card_bg"]}; border-radius: 0 0 8px 8px;">'
        for result in profile.recent_results[:5]:
            if result == "Win":
                form_html += f'<span style="background: {COLORS["success"]}; color: white; padding: 5px 12px; border-radius: 4px; margin-right: 6px; font-size: 14px; font-weight: bold;">W</span>'
            elif result == "Loss":
                form_html += f'<span style="background: {COLORS["danger"]}; color: white; padding: 5px 12px; border-radius: 4px; margin-right: 6px; font-size: 14px; font-weight: bold;">L</span>'
            else:
                form_html += f'<span style="background: {COLORS["warning"]}; color: black; padding: 5px 12px; border-radius: 4px; margin-right: 6px; font-size: 14px; font-weight: bold;">D</span>'

        # Add streak indicator
        if profile.win_streak >= 2:
            form_html += f'<span style="margin-left: 15px; color: {COLORS["success"]}; font-weight: bold;">🔥 {profile.win_streak}-fight win streak</span>'
        elif profile.loss_streak >= 2:
            form_html += f'<span style="margin-left: 15px; color: {COLORS["danger"]};">📉 {profile.loss_streak}-fight losing streak</span>'

        form_html += '</div>'
        st.markdown(form_html, unsafe_allow_html=True)


def render_recent_activity(
    fighter_id: int,
    corner: str = "red"
) -> None:
    """Render detailed recent activity for a fighter."""
    from datetime import datetime
    corner_color = COLORS['primary'] if corner == "red" else COLORS['info']

    # Get recent fight history
    history = simulation_service.data_service.get_fighter_fight_history(fighter_id, limit=5)

    if not history:
        return

    st.markdown(f'<div style="margin-top: 10px; padding: 8px 12px; background: {corner_color}30; border-radius: 8px 8px 0 0; border-bottom: 2px solid {corner_color};"><strong style="color: {COLORS["text_primary"]};">🕐 RECENT ACTIVITY</strong></div>', unsafe_allow_html=True)

    activity_html = f'<div style="background: {COLORS["card_bg"]}; padding: 12px; border-radius: 0 0 8px 8px;">'

    for i, fight in enumerate(history[:3]):
        result = fight.get("result", "")
        opponent = fight.get("opponent_name", "Unknown")
        method = fight.get("method", "")
        round_num = fight.get("round", "")
        event_date = fight.get("event_date", "")

        # Format date
        date_display = ""
        if event_date:
            try:
                if isinstance(event_date, str):
                    fight_date = datetime.strptime(event_date, "%Y-%m-%d")
                    date_display = fight_date.strftime("%b %d, %Y")
            except:
                date_display = str(event_date)

        # Result styling
        if result == "Win":
            result_color = COLORS["success"]
            result_icon = "✅"
        elif result == "Loss":
            result_color = COLORS["danger"]
            result_icon = "❌"
        else:
            result_color = COLORS["warning"]
            result_icon = "➖"

        # Method abbreviation
        method_short = method
        if "KO" in method.upper() or "TKO" in method.upper():
            method_short = "KO/TKO"
        elif "SUB" in method.upper():
            method_short = "SUB"
        elif "DEC" in method.upper() or "UNANIMOUS" in method.upper() or "SPLIT" in method.upper():
            method_short = "DEC"

        round_text = f"R{round_num}" if round_num else ""

        activity_html += f'<div style="display: flex; justify-content: space-between; align-items: center; padding: 8px; margin-bottom: 6px; background: {COLORS["background"]}; border-radius: 6px; border-left: 3px solid {result_color};"><div><span style="font-weight: bold; color: {result_color};">{result_icon} {result}</span><span style="color: {COLORS["text_primary"]}; margin-left: 8px;">vs {opponent}</span></div><div style="text-align: right;"><span style="color: {COLORS["text_secondary"]}; font-size: 14px;">{method_short} {round_text}</span><span style="color: {COLORS["text_muted"]}; font-size: 14px; margin-left: 8px;">{date_display}</span></div></div>'

    activity_html += '</div>'
    st.markdown(activity_html, unsafe_allow_html=True)


def render_fighter_tendencies(
    tendencies: FighterTendencies,
    corner: str = "red"
) -> None:
    """Render fighter tendencies and style analysis."""
    corner_color = COLORS['primary'] if corner == "red" else COLORS['info']

    st.markdown(f'<div style="margin-top: 10px; padding: 8px 12px; background: {corner_color}30; border-radius: 8px 8px 0 0; border-bottom: 2px solid {corner_color};"><strong style="color: {COLORS["text_primary"]};">🎯 FIGHTING STYLE & TENDENCIES</strong></div>', unsafe_allow_html=True)

    # Primary style badge
    style_colors = {
        "Striker": COLORS["danger"],
        "Wrestler": COLORS["info"],
        "Grappler": COLORS["warning"],
        "Balanced": COLORS["success"],
    }
    style_color = style_colors.get(tendencies.primary_style, COLORS["text_secondary"])

    # Style breakdown bar
    striking_pct = tendencies.style_breakdown.get("striking", 0) * 100
    wrestling_pct = tendencies.style_breakdown.get("wrestling", 0) * 100
    submission_pct = tendencies.style_breakdown.get("submissions", 0) * 100

    style_html = f'<div style="background: {COLORS["card_bg"]}; padding: 15px; border-radius: 0 0 8px 8px;"><div style="text-align: center; margin-bottom: 12px;"><span style="background: {style_color}; color: white; padding: 6px 16px; border-radius: 20px; font-weight: bold; font-size: 14px;">{tendencies.primary_style.upper()}</span></div><div style="margin-bottom: 15px;"><div style="font-size: 14px; color: {COLORS["text_muted"]}; margin-bottom: 5px;">STYLE BREAKDOWN</div><div style="display: flex; height: 20px; border-radius: 10px; overflow: hidden; background: {COLORS["background"]};"><div style="width: {striking_pct}%; background: {COLORS["danger"]}; display: flex; align-items: center; justify-content: center;"><span style="font-size: 14px; color: white; font-weight: bold;">{striking_pct:.0f}%</span></div><div style="width: {wrestling_pct}%; background: {COLORS["info"]}; display: flex; align-items: center; justify-content: center;"><span style="font-size: 14px; color: white; font-weight: bold;">{wrestling_pct:.0f}%</span></div><div style="width: {submission_pct}%; background: {COLORS["warning"]}; display: flex; align-items: center; justify-content: center;"><span style="font-size: 14px; color: white; font-weight: bold;">{submission_pct:.0f}%</span></div></div><div style="display: flex; justify-content: space-between; margin-top: 4px; font-size: 14px;"><span style="color: {COLORS["danger"]};">Striking</span><span style="color: {COLORS["info"]};">Wrestling</span><span style="color: {COLORS["warning"]};">Submissions</span></div></div>'

    # Collect tendency tags
    offensive_tags = []
    if tendencies.volume_striker:
        offensive_tags.append(("Volume Striker", COLORS["danger"]))
    if tendencies.power_puncher:
        offensive_tags.append(("Power Puncher", COLORS["danger"]))
    if tendencies.counter_striker:
        offensive_tags.append(("Counter Striker", COLORS["warning"]))
    if tendencies.pressure_fighter:
        offensive_tags.append(("Pressure Fighter", COLORS["primary"]))
    if tendencies.active_wrestler:
        offensive_tags.append(("Active Wrestler", COLORS["info"]))
    if tendencies.ground_and_pound:
        offensive_tags.append(("Ground & Pound", COLORS["info"]))
    if tendencies.submission_hunter:
        offensive_tags.append(("Submission Hunter", COLORS["warning"]))
    if tendencies.top_control:
        offensive_tags.append(("Top Control", COLORS["info"]))

    finishing_tags = []
    if tendencies.early_finisher:
        finishing_tags.append(("Early Finisher", COLORS["danger"]))
    if tendencies.late_finisher:
        finishing_tags.append(("Late Finisher", COLORS["warning"]))
    if tendencies.goes_to_decision:
        finishing_tags.append(("Decision Fighter", COLORS["text_secondary"]))
    if tendencies.prefers_finish:
        finishing_tags.append(("Seeks Finish", COLORS["success"]))

    defensive_tags = []
    if tendencies.hard_to_finish:
        defensive_tags.append(("Durable", COLORS["success"]))
    if tendencies.chin_issues:
        defensive_tags.append(("Chin Concerns", COLORS["danger"]))
    if tendencies.grappling_vulnerable:
        defensive_tags.append(("Grappling Vulnerable", COLORS["warning"]))
    if tendencies.performs_in_later_rounds:
        defensive_tags.append(("Good Cardio", COLORS["success"]))
    if tendencies.fades_in_later_rounds:
        defensive_tags.append(("Cardio Concerns", COLORS["warning"]))

    # Render offensive tendencies
    if offensive_tags:
        style_html += f'<div style="margin-bottom: 10px;"><div style="font-size: 14px; color: {COLORS["text_muted"]}; margin-bottom: 5px;">OFFENSIVE</div>'
        for tag, color in offensive_tags[:4]:
            style_html += f'<span style="background: {color}20; color: {color}; padding: 3px 8px; border-radius: 12px; font-size: 14px; margin-right: 4px; margin-bottom: 4px; display: inline-block;">{tag}</span>'
        style_html += '</div>'

    # Render finishing tendencies
    if finishing_tags:
        style_html += f'<div style="margin-bottom: 10px;"><div style="font-size: 14px; color: {COLORS["text_muted"]}; margin-bottom: 5px;">FINISHING</div>'
        for tag, color in finishing_tags[:3]:
            style_html += f'<span style="background: {color}20; color: {color}; padding: 3px 8px; border-radius: 12px; font-size: 14px; margin-right: 4px; margin-bottom: 4px; display: inline-block;">{tag}</span>'
        style_html += '</div>'

    # Render defensive profile
    if defensive_tags:
        style_html += f'<div style="margin-bottom: 10px;"><div style="font-size: 14px; color: {COLORS["text_muted"]}; margin-bottom: 5px;">DEFENSE & CARDIO</div>'
        for tag, color in defensive_tags[:3]:
            style_html += f'<span style="background: {color}20; color: {color}; padding: 3px 8px; border-radius: 12px; font-size: 14px; margin-right: 4px; margin-bottom: 4px; display: inline-block;">{tag}</span>'
        style_html += '</div>'

    # Momentum indicator
    if tendencies.improving:
        momentum_html = f'<div style="margin-top: 10px; padding: 8px; background: {COLORS["success"]}20; border-radius: 6px; text-align: center;"><span style="color: {COLORS["success"]}; font-weight: bold;">📈 ON THE RISE</span><span style="color: {COLORS["text_secondary"]}; font-size: 14px; margin-left: 8px;">Recent form trending up</span></div>'
    elif tendencies.declining:
        momentum_html = f'<div style="margin-top: 10px; padding: 8px; background: {COLORS["danger"]}20; border-radius: 6px; text-align: center;"><span style="color: {COLORS["danger"]}; font-weight: bold;">📉 STRUGGLING</span><span style="color: {COLORS["text_secondary"]}; font-size: 14px; margin-left: 8px;">Recent form trending down</span></div>'
    else:
        momentum_html = ""

    style_html += momentum_html + '</div>'
    st.markdown(style_html, unsafe_allow_html=True)


def render_sanity_checks(checks: list) -> None:
    """Render sanity check warnings/info."""
    if not checks:
        return

    st.markdown("### ⚠️ Matchup Notes")

    for check in checks:
        if check.severity == "critical":
            icon = "🚨"
            bg_color = f"{COLORS['danger']}20"
            border_color = COLORS['danger']
        elif check.severity == "warning":
            icon = "⚠️"
            bg_color = f"{COLORS['warning']}20"
            border_color = COLORS['warning']
        else:
            icon = "ℹ️"
            bg_color = f"{COLORS['info']}20"
            border_color = COLORS['info']

        details_html = ""
        if check.details:
            details_html = f'<div style="color: {COLORS["text_secondary"]}; font-size: 14px; margin-top: 5px;">{check.details}</div>'

        check_html = f'<div style="background: {bg_color}; border-left: 3px solid {border_color}; padding: 12px 15px; border-radius: 0 8px 8px 0; margin: 8px 0;"><div style="font-weight: bold; color: {COLORS["text_primary"]};">{icon} {check.message}</div>{details_html}</div>'
        st.markdown(check_html, unsafe_allow_html=True)


def render_round_breakdown(rounds, fighter_a_name: str, fighter_b_name: str):
    """Render round-by-round breakdown."""
    st.markdown("#### Round-by-Round Breakdown")

    for round_result in rounds:
        with st.expander(
            f"Round {round_result.round_number}" +
            (f" - {round_result.finish} @ {round_result.finish_time}" if round_result.finish else ""),
            expanded=(round_result.finish is not None)
        ):
            col1, col2, col3 = st.columns([2, 1, 2])

            with col1:
                st.markdown(f"**🔴 {fighter_a_name}**")
                st.metric("Strikes", round_result.fighter_a_strikes)
                st.metric("Takedowns", round_result.fighter_a_takedowns)
                if round_result.fighter_a_control_time > 0:
                    mins = round_result.fighter_a_control_time // 60
                    secs = round_result.fighter_a_control_time % 60
                    st.metric("Control Time", f"{mins}:{secs:02d}")

            with col2:
                if round_result.winner == "fighter_a":
                    win_html = f'<div style="text-align: center; padding: 20px;"><span style="font-size: 24px;">◀️</span><br><span style="color: {COLORS["success"]}; font-weight: bold;">WIN</span></div>'
                    st.markdown(win_html, unsafe_allow_html=True)
                elif round_result.winner == "fighter_b":
                    win_html = f'<div style="text-align: center; padding: 20px;"><span style="font-size: 24px;">▶️</span><br><span style="color: {COLORS["success"]}; font-weight: bold;">WIN</span></div>'
                    st.markdown(win_html, unsafe_allow_html=True)
                else:
                    even_html = f'<div style="text-align: center; padding: 20px;"><span style="font-size: 24px;">⚖️</span><br><span style="color: {COLORS["warning"]};">EVEN</span></div>'
                    st.markdown(even_html, unsafe_allow_html=True)

            with col3:
                st.markdown(f"**🔵 {fighter_b_name}**")
                st.metric("Strikes", round_result.fighter_b_strikes)
                st.metric("Takedowns", round_result.fighter_b_takedowns)
                if round_result.fighter_b_control_time > 0:
                    mins = round_result.fighter_b_control_time // 60
                    secs = round_result.fighter_b_control_time % 60
                    st.metric("Control Time", f"{mins}:{secs:02d}")

            if round_result.significant_moments:
                st.markdown("**Key Moments:**")
                for moment in round_result.significant_moments:
                    st.markdown(f"- {moment}")

            if round_result.finish:
                finish_html = f'<div style="background: {COLORS["primary"]}; padding: 15px; border-radius: 8px; text-align: center; margin-top: 10px;"><span style="font-size: 20px; font-weight: bold; color: white;">🏆 {round_result.finish.upper()} @ {round_result.finish_time}</span></div>'
                st.markdown(finish_html, unsafe_allow_html=True)


def render_simulation_result(
    sim: SimulationResult,
    fighter_a_name: str,
    fighter_b_name: str,
    profile_a: FighterProfile,
    profile_b: FighterProfile,
    is_primary: bool = True
):
    """Render a simulation result with sanity checks."""
    if is_primary:
        st.markdown("### 🏆 Most Realistic Simulation")
    else:
        st.markdown("### 📊 Alternative Simulation")

    # Result header
    finish_text = f"via {sim.method}"
    if sim.finish_round:
        finish_text += f" (R{sim.finish_round} @ {sim.finish_time})"

    # Winner banner
    winner_color = COLORS['primary'] if sim.winner_name == fighter_a_name else COLORS['info']

    winner_html = f'<div style="background: linear-gradient(135deg, {winner_color}, {COLORS["secondary"]}); padding: 25px; border-radius: 12px; text-align: center; margin-bottom: 20px;"><div style="font-size: 14px; color: #ccc; margin-bottom: 5px;">WINNER</div><div style="font-size: 32px; font-weight: bold; color: white; margin-bottom: 10px;">{sim.winner_name}</div><div style="font-size: 18px; color: #eee;">defeats {sim.loser_name} {finish_text}</div><div style="margin-top: 15px;"><span style="background: rgba(255,255,255,0.2); padding: 8px 16px; border-radius: 20px; color: white; font-weight: bold;">{sim.confidence:.0%} Confidence</span><span style="background: rgba(255,255,255,0.2); padding: 8px 16px; border-radius: 20px; color: white; margin-left: 10px;">Realism: {sim.realism_score:.0%}</span></div></div>'
    st.markdown(winner_html, unsafe_allow_html=True)

    # Fight stats comparison
    st.markdown("#### Fight Statistics")

    col1, col2, col3 = st.columns([2, 1, 2])

    with col1:
        st.markdown(f"**🔴 {fighter_a_name}**")
        a_stats = sim.fighter_a_stats
        st.metric("Total Strikes", a_stats.get("total_strikes", 0))
        st.metric("Total Takedowns", a_stats.get("total_takedowns", 0))
        ctrl_time = a_stats.get("total_control_time", 0)
        st.metric("Control Time", f"{ctrl_time // 60}:{ctrl_time % 60:02d}")
        st.metric("Rounds Won", a_stats.get("rounds_won", 0))

    with col2:
        vs_html = f'<div style="text-align: center; padding-top: 50px;"><span style="font-size: 24px;">⚔️</span></div>'
        st.markdown(vs_html, unsafe_allow_html=True)

    with col3:
        st.markdown(f"**🔵 {fighter_b_name}**")
        b_stats = sim.fighter_b_stats
        st.metric("Total Strikes", b_stats.get("total_strikes", 0))
        st.metric("Total Takedowns", b_stats.get("total_takedowns", 0))
        ctrl_time = b_stats.get("total_control_time", 0)
        st.metric("Control Time", f"{ctrl_time // 60}:{ctrl_time % 60:02d}")
        st.metric("Rounds Won", b_stats.get("rounds_won", 0))

    # Key factors
    st.markdown("#### Key Factors")
    for factor in sim.key_factors:
        st.markdown(f"- 🔑 {factor}")

    # Round breakdown for primary simulation
    if is_primary:
        st.markdown("---")
        render_round_breakdown(sim.rounds, fighter_a_name, fighter_b_name)

    # Render matchup notes after the simulation result (only for primary)
    if is_primary and sim.sanity_checks:
        st.markdown("---")
        render_sanity_checks(sim.sanity_checks)

    # Model info
    st.caption(f"Simulation Model: {sim.model_name.capitalize()}")


def render_simulation_summary(sim: SimulationResult, fighter_a_name: str, fighter_b_name: str):
    """Render a condensed simulation summary."""
    finish_text = f"{sim.method}"
    if sim.finish_round:
        finish_text += f" R{sim.finish_round}"

    summary_html = f'<div style="background: {COLORS["card_bg"]}; padding: 15px; border-radius: 8px; border-left: 4px solid {COLORS["info"]};"><div style="display: flex; justify-content: space-between; align-items: center;"><div><strong>{sim.winner_name}</strong> wins via <strong>{finish_text}</strong></div><div><span style="color: {COLORS["text_secondary"]};">{sim.confidence:.0%} conf. | {sim.realism_score:.0%} realism</span></div></div><div style="margin-top: 10px; color: {COLORS["text_secondary"]}; font-size: 14px;">Model: {sim.model_name.capitalize()}</div></div>'
    st.markdown(summary_html, unsafe_allow_html=True)


# Fighter Selection
st.markdown("---")
st.markdown("### 🥊 Select Fighters")

col1, col2 = st.columns(2)

fighter_a_selection = None
fighter_b_selection = None
profile_a = None
profile_b = None

tendencies_a = None
tendencies_b = None

with col1:
    st.markdown("#### 🔴 Fighter A (Red Corner)")
    fighter_a_search = st.text_input(
        "Search Fighter A",
        placeholder="Type fighter name...",
        key="fighter_a_search"
    )

    if fighter_a_search:
        results_a = simulation_service.search_active_fighters(fighter_a_search, limit=5)
        if results_a:
            fighter_a_options = {f["name"]: f for f in results_a}
            selected_name_a = st.selectbox(
                "Select Fighter A",
                options=list(fighter_a_options.keys()),
                key="fighter_a_select"
            )
            fighter_a_selection = fighter_a_options[selected_name_a]

            # Get full profile
            profile_a = simulation_service.get_fighter_profile(fighter_a_selection.get("fighter_id"))

            if profile_a:
                render_fighter_profile_card(profile_a, corner="red")

                # Display recent activity
                render_recent_activity(fighter_a_selection.get("fighter_id"), corner="red")

                # Get and display tendencies
                tendencies_a = simulation_service.get_fighter_tendencies(fighter_a_selection.get("fighter_id"))
                if tendencies_a:
                    render_fighter_tendencies(tendencies_a, corner="red")
        else:
            st.warning("No active fighters found. Try a different search.")

with col2:
    st.markdown("#### 🔵 Fighter B (Blue Corner)")
    fighter_b_search = st.text_input(
        "Search Fighter B",
        placeholder="Type fighter name...",
        key="fighter_b_search"
    )

    if fighter_b_search:
        results_b = simulation_service.search_active_fighters(fighter_b_search, limit=5)
        if results_b:
            fighter_b_options = {f["name"]: f for f in results_b}
            selected_name_b = st.selectbox(
                "Select Fighter B",
                options=list(fighter_b_options.keys()),
                key="fighter_b_select"
            )
            fighter_b_selection = fighter_b_options[selected_name_b]

            # Get full profile
            profile_b = simulation_service.get_fighter_profile(fighter_b_selection.get("fighter_id"))

            if profile_b:
                render_fighter_profile_card(profile_b, corner="blue")

                # Display recent activity
                render_recent_activity(fighter_b_selection.get("fighter_id"), corner="blue")

                # Get and display tendencies
                tendencies_b = simulation_service.get_fighter_tendencies(fighter_b_selection.get("fighter_id"))
                if tendencies_b:
                    render_fighter_tendencies(tendencies_b, corner="blue")
        else:
            st.warning("No active fighters found. Try a different search.")

# Physical Comparison (if both fighters selected)
if profile_a and profile_b:
    st.markdown("---")
    st.markdown("### 📏 Physical Comparison")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        height_cmp_html = f'<div style="text-align: center;"><div style="color: {COLORS["text_muted"]}; font-size: 14px;">HEIGHT</div><div style="color: {COLORS["primary"]}; font-weight: bold;">{profile_a.height_display}</div><div style="color: {COLORS["text_muted"]};">vs</div><div style="color: {COLORS["info"]}; font-weight: bold;">{profile_b.height_display}</div></div>'
        st.markdown(height_cmp_html, unsafe_allow_html=True)

    with col2:
        reach_cmp_html = f'<div style="text-align: center;"><div style="color: {COLORS["text_muted"]}; font-size: 14px;">REACH</div><div style="color: {COLORS["primary"]}; font-weight: bold;">{profile_a.reach_display}</div><div style="color: {COLORS["text_muted"]};">vs</div><div style="color: {COLORS["info"]}; font-weight: bold;">{profile_b.reach_display}</div></div>'
        st.markdown(reach_cmp_html, unsafe_allow_html=True)

    with col3:
        age_a = f"{profile_a.age} yrs" if profile_a.age else "N/A"
        age_b = f"{profile_b.age} yrs" if profile_b.age else "N/A"
        age_cmp_html = f'<div style="text-align: center;"><div style="color: {COLORS["text_muted"]}; font-size: 14px;">AGE</div><div style="color: {COLORS["primary"]}; font-weight: bold;">{age_a}</div><div style="color: {COLORS["text_muted"]};">vs</div><div style="color: {COLORS["info"]}; font-weight: bold;">{age_b}</div></div>'
        st.markdown(age_cmp_html, unsafe_allow_html=True)

    with col4:
        record_cmp_html = f'<div style="text-align: center;"><div style="color: {COLORS["text_muted"]}; font-size: 14px;">RECORD</div><div style="color: {COLORS["primary"]}; font-weight: bold;">{profile_a.record}</div><div style="color: {COLORS["text_muted"]};">vs</div><div style="color: {COLORS["info"]}; font-weight: bold;">{profile_b.record}</div></div>'
        st.markdown(record_cmp_html, unsafe_allow_html=True)

    with col5:
        stance_a = profile_a.stance or "N/A"
        stance_b = profile_b.stance or "N/A"
        stance_cmp_html = f'<div style="text-align: center;"><div style="color: {COLORS["text_muted"]}; font-size: 14px;">STANCE</div><div style="color: {COLORS["primary"]}; font-weight: bold;">{stance_a}</div><div style="color: {COLORS["text_muted"]};">vs</div><div style="color: {COLORS["info"]}; font-weight: bold;">{stance_b}</div></div>'
        st.markdown(stance_cmp_html, unsafe_allow_html=True)

# Fight Parameters
st.markdown("---")
st.markdown("### ⚙️ Fight Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    num_rounds = st.radio(
        "Number of Rounds",
        options=[3, 5],
        index=0,
        horizontal=True,
        help="3 rounds for regular fights, 5 rounds for main events and title fights"
    )

with col2:
    format_text = 'Championship Format' if num_rounds == 5 else 'Standard Format'
    scheduled_html = f'<div style="background: {COLORS["card_bg"]}; padding: 15px; border-radius: 8px; text-align: center;"><div style="color: {COLORS["text_secondary"]}; font-size: 14px;">SCHEDULED</div><div style="font-size: 24px; font-weight: bold; color: {COLORS["text_primary"]};">{num_rounds} x 5:00</div><div style="color: {COLORS["text_secondary"]}; font-size: 14px;">{format_text}</div></div>'
    st.markdown(scheduled_html, unsafe_allow_html=True)

with col3:
    factors_html = f'<div style="background: {COLORS["card_bg"]}; padding: 15px; border-radius: 8px; text-align: center;"><div style="color: {COLORS["text_secondary"]}; font-size: 14px;">FACTORS ANALYZED</div><div style="font-size: 24px; font-weight: bold; color: {COLORS["text_primary"]};">20+</div><div style="color: {COLORS["text_secondary"]}; font-size: 14px;">Click below for details</div></div>'
    st.markdown(factors_html, unsafe_allow_html=True)

# Detailed factors expander
with st.expander("View All Factors Analyzed in Simulation"):
    st.markdown(f'<h4 style="color: {COLORS["primary"]}; margin-top: 0;">Physical Attributes</h4>', unsafe_allow_html=True)
    st.markdown("""
- **Height** - Reach advantage affects striking range
- **Reach** - Longer reach can control distance
- **Age** - Prime years vs potential decline
- **Stance** - Orthodox vs Southpaw matchups
- **Weight Class** - Division-specific tendencies
    """)

    st.markdown(f'<h4 style="color: {COLORS["danger"]};">Striking Statistics</h4>', unsafe_allow_html=True)
    st.markdown("""
- **Sig. Strikes Landed/Min (SLpM)** - Offensive output
- **Striking Accuracy %** - Precision of strikes
- **Sig. Strikes Absorbed/Min (SApM)** - Damage taken
- **Striking Defense %** - Ability to avoid strikes
- **KO/TKO Rate** - Knockout finishing ability
    """)

    st.markdown(f'<h4 style="color: {COLORS["info"]};">Grappling Statistics</h4>', unsafe_allow_html=True)
    st.markdown("""
- **Takedowns/15min** - Takedown frequency
- **Takedown Accuracy %** - Takedown success rate
- **Takedown Defense %** - Ability to stay on feet
- **Submissions/15min** - Submission attempt rate
- **Submission Rate** - Submission finishing ability
    """)

    st.markdown(f'<h4 style="color: {COLORS["success"]};">Experience & Record</h4>', unsafe_allow_html=True)
    st.markdown("""
- **Win-Loss Record** - Overall career success
- **Total Pro Fights** - Overall experience
- **UFC Fights** - Elite-level experience
- **Finish Rate** - Ability to end fights early
- **Decision Rate** - Going the distance ability
    """)

    st.markdown(f'<h4 style="color: {COLORS["warning"]};">Form & Momentum</h4>', unsafe_allow_html=True)
    st.markdown("""
- **Recent Results** - Last 5 fight outcomes
- **Win/Loss Streak** - Current momentum
- **Last Fight Date** - Ring rust consideration
- **Fighter Trajectory** - Improving or declining
    """)

    st.markdown(f'<h4 style="color: {COLORS["text_secondary"]};">Style Analysis</h4>', unsafe_allow_html=True)
    st.markdown("""
- **Primary Style** - Striker, Wrestler, Grappler, or Balanced
- **Offensive Tendencies** - Volume striker, Power puncher, Pressure fighter, etc.
- **Finishing Tendencies** - Early finisher, Late finisher, Decision fighter
- **Defensive Profile** - Durability, Chin issues, Cardio concerns
    """)

# Simulate Button
st.markdown("---")

can_simulate = (
    fighter_a_selection is not None and
    fighter_b_selection is not None and
    fighter_a_selection.get("fighter_id") != fighter_b_selection.get("fighter_id")
)

if not can_simulate:
    if fighter_a_selection and fighter_b_selection:
        if fighter_a_selection.get("fighter_id") == fighter_b_selection.get("fighter_id"):
            st.warning("Please select two different fighters.")
    else:
        st.info("Select two fighters to simulate a fight.")

simulate_button = st.button(
    "🎮 SIMULATE FIGHT",
    disabled=not can_simulate,
    width='stretch',
    type="primary"
)

if simulate_button and can_simulate:
    fighter_a_id = fighter_a_selection.get("fighter_id")
    fighter_b_id = fighter_b_selection.get("fighter_id")
    fighter_a_name = fighter_a_selection.get("name")
    fighter_b_name = fighter_b_selection.get("name")

    # Simulation animation (directly below button)

    progress_placeholder = st.empty()
    status_placeholder = st.empty()

    # Show simulating animation
    with progress_placeholder.container():
        anim_html = f'<div style="background: linear-gradient(135deg, {COLORS["primary"]}, {COLORS["secondary"]}); padding: 40px; border-radius: 15px; text-align: center;"><div style="font-size: 48px; margin-bottom: 20px;">🥊</div><div style="font-size: 24px; font-weight: bold; color: white;">{fighter_a_name} vs {fighter_b_name}</div><div style="margin-top: 20px; color: #ccc;">Analyzing physical attributes, age, form, and statistics...</div></div>'
        st.markdown(anim_html, unsafe_allow_html=True)

    # Progress bar
    progress_bar = st.progress(0)

    simulation_stages = [
        "Analyzing fighter records...",
        "Comparing physical attributes...",
        "Evaluating age and experience...",
        "Checking recent form and momentum...",
        "Running statistical model...",
        "Running momentum model...",
        "Running stylistic model...",
        "Running ensemble model...",
        "Running sanity checks...",
        "Calculating realism scores...",
        "Selecting most realistic outcome..."
    ]

    for i, stage in enumerate(simulation_stages):
        status_placeholder.markdown(f"*{stage}*")
        progress_bar.progress((i + 1) / len(simulation_stages))
        time.sleep(0.25)

    # Run actual simulation
    simulations = simulation_service.simulate_fight(
        fighter_a_id,
        fighter_b_id,
        num_rounds=num_rounds
    )

    # Clear animation
    progress_placeholder.empty()
    status_placeholder.empty()
    progress_bar.empty()

    if simulations:
        most_realistic, second_realistic = simulation_service.get_most_realistic_simulation(simulations)

        if most_realistic:
            render_simulation_result(
                most_realistic,
                fighter_a_name,
                fighter_b_name,
                profile_a,
                profile_b,
                is_primary=True
            )

            st.markdown("---")

            # Second most realistic
            if second_realistic:
                st.markdown("### 🔮 Alternative Outcome")
                st.markdown("*If the fight had gone differently...*")
                render_simulation_summary(second_realistic, fighter_a_name, fighter_b_name)

            # All simulations summary
            st.markdown("---")
            st.markdown("### 📋 All Simulation Results")

            for i, sim in enumerate(simulations):
                with st.expander(
                    f"{sim.model_name.capitalize()} Model - {sim.winner_name} wins ({sim.realism_score:.0%} realism)"
                ):
                    render_simulation_summary(sim, fighter_a_name, fighter_b_name)
    else:
        st.error("Simulation failed. Please ensure both fighters have sufficient data.")

# Fighter Analysis Section
if fighter_a_selection or fighter_b_selection:
    st.markdown("---")
    st.markdown("### 📊 Fighter Analysis")

    col1, col2 = st.columns(2)

    if fighter_a_selection:
        with col1:
            analysis_a = simulation_service.analyze_fighter(fighter_a_selection.get("fighter_id"))
            if analysis_a:
                st.markdown(f"#### 🔴 {analysis_a.name}")

                st.markdown("**Strengths**")
                if analysis_a.strengths:
                    for s in analysis_a.strengths:
                        st.markdown(f"- ✅ {s}")
                else:
                    st.markdown("- No significant strengths identified")

                st.markdown("**Weaknesses**")
                if analysis_a.weaknesses:
                    for w in analysis_a.weaknesses:
                        st.markdown(f"- ⚠️ {w}")
                else:
                    st.markdown("- No significant weaknesses identified")

                st.markdown(f"**Style:** {analysis_a.style_description}")

                trend_emoji = {
                    "improving": "📈",
                    "declining": "📉",
                    "consistent": "➡️",
                    "unknown": "❓"
                }
                st.markdown(f"**Recent Form:** {trend_emoji.get(analysis_a.recent_trend, '➡️')} {analysis_a.recent_trend.capitalize()}")

    if fighter_b_selection:
        with col2:
            analysis_b = simulation_service.analyze_fighter(fighter_b_selection.get("fighter_id"))
            if analysis_b:
                st.markdown(f"#### 🔵 {analysis_b.name}")

                st.markdown("**Strengths**")
                if analysis_b.strengths:
                    for s in analysis_b.strengths:
                        st.markdown(f"- ✅ {s}")
                else:
                    st.markdown("- No significant strengths identified")

                st.markdown("**Weaknesses**")
                if analysis_b.weaknesses:
                    for w in analysis_b.weaknesses:
                        st.markdown(f"- ⚠️ {w}")
                else:
                    st.markdown("- No significant weaknesses identified")

                st.markdown(f"**Style:** {analysis_b.style_description}")

                trend_emoji = {
                    "improving": "📈",
                    "declining": "📉",
                    "consistent": "➡️",
                    "unknown": "❓"
                }
                st.markdown(f"**Recent Form:** {trend_emoji.get(analysis_b.recent_trend, '➡️')} {analysis_b.recent_trend.capitalize()}")

# Footer
st.markdown("---")
footer_html = f'<div style="text-align: center; color: {COLORS["text_muted"]}; font-size: 14px;">Simulations consider: Record, Physical Attributes, Age, Weight Class, Striking & Wrestling Stats, Recent Form | For entertainment purposes only</div>'
st.markdown(footer_html, unsafe_allow_html=True)
