"""
Analysis Insights Page.

Comprehensive UFC analytics combining statistical analysis, machine learning insights,
fighter records, and skill analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import COLORS, WEIGHT_CLASSES
from services.data_service import DataService
from services.llm_service import LLMService
from services.unsupervised_analysis_service import UnsupervisedAnalysisService
import plotly.express as px
import plotly.graph_objects as go


st.set_page_config(
    page_title="Analysis & Insights - UFC Prediction App",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Analysis & Insights")
st.markdown("*Comprehensive UFC statistics, machine learning patterns, and in-depth analysis*")

# Initialize services
data_service = DataService()
llm_service = LLMService()
analysis_service = UnsupervisedAnalysisService()


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================
@st.cache_data(ttl=3600)
def get_comprehensive_stats():
    """Calculate all UFC statistics from database."""
    with data_service.get_connection() as conn:
        fights_df = pd.read_sql('SELECT * FROM fights WHERE winner_id IS NOT NULL', conn)
        fighters_df = pd.read_sql('SELECT * FROM fighters', conn)
        fighter_stats_df = pd.read_sql('SELECT * FROM fighter_stats', conn)
        events_df = pd.read_sql('SELECT * FROM events ORDER BY date', conn)

    fighters_with_stats = fighters_df.merge(fighter_stats_df, on='fighter_id', how='left')

    # Convert numeric columns
    for col in ['sig_strikes_landed_per_min', 'takedowns_avg_per_15min', 'submissions_avg_per_15min',
                'takedown_defense', 'takedown_accuracy', 'sig_strike_accuracy', 'sig_strike_defense']:
        if col in fighters_with_stats.columns:
            fighters_with_stats[col] = pd.to_numeric(fighters_with_stats[col], errors='coerce')

    stats = {}

    # Basic counts
    stats['total_fights'] = len(fights_df)
    stats['total_fighters'] = len(fighters_df)
    stats['total_events'] = len(events_df)

    # Overall method distribution
    total = len(fights_df)
    ko_total = len(fights_df[fights_df['method'].str.contains('KO', na=False)])
    sub_total = len(fights_df[fights_df['method'].str.contains('Sub', na=False, case=False)])
    dec_total = len(fights_df[fights_df['method'].str.contains('Dec', na=False, case=False)])

    stats['ko_count'] = ko_total
    stats['sub_count'] = sub_total
    stats['dec_count'] = dec_total
    stats['ko_rate'] = ko_total / total * 100
    stats['sub_rate'] = sub_total / total * 100
    stats['dec_rate'] = dec_total / total * 100
    stats['finish_rate'] = (ko_total + sub_total) / total * 100

    # KOs by round
    ko_fights = fights_df[fights_df['method'].str.contains('KO', na=False)]
    round_dist = ko_fights['round'].value_counts().sort_index()
    stats['ko_by_round'] = {int(r): int(c) for r, c in round_dist.items() if pd.notna(r)}

    # Division stats
    wc_stats = {}
    for wc in fights_df['weight_class'].dropna().unique():
        wc_fights = fights_df[fights_df['weight_class'] == wc]
        if len(wc_fights) >= 10:
            ko_count = len(wc_fights[wc_fights['method'].str.contains('KO', na=False)])
            sub_count = len(wc_fights[wc_fights['method'].str.contains('Sub', na=False, case=False)])
            dec_count = len(wc_fights[wc_fights['method'].str.contains('Dec', na=False, case=False)])
            wc_stats[wc] = {
                'total': len(wc_fights),
                'ko': ko_count,
                'sub': sub_count,
                'dec': dec_count,
                'ko_rate': ko_count / len(wc_fights) * 100,
                'sub_rate': sub_count / len(wc_fights) * 100,
                'dec_rate': dec_count / len(wc_fights) * 100,
                'finish_rate': (ko_count + sub_count) / len(wc_fights) * 100,
            }
    stats['division_stats'] = wc_stats

    # Division records
    if wc_stats:
        stats['most_ko_division'] = max(wc_stats.items(), key=lambda x: x[1]['ko'])
        stats['most_sub_division'] = max(wc_stats.items(), key=lambda x: x[1]['sub'])
        stats['most_dec_division'] = max(wc_stats.items(), key=lambda x: x[1]['dec'])
        stats['highest_ko_rate_division'] = max(wc_stats.items(), key=lambda x: x[1]['ko_rate'])
        stats['highest_sub_rate_division'] = max(wc_stats.items(), key=lambda x: x[1]['sub_rate'])
        stats['most_active_division'] = max(wc_stats.items(), key=lambda x: x[1]['total'])

    # === FIGHTER RECORDS ===

    # KO artists
    ko_artists = []
    for _, fighter in fighters_df.iterrows():
        fid = fighter['fighter_id']
        kos = len(fights_df[(fights_df['winner_id'] == fid) & (fights_df['method'].str.contains('KO', na=False))])
        if kos >= 3:
            ko_artists.append({'name': fighter['name'], 'ko_wins': kos, 'total_wins': fighter['wins']})
    stats['ko_artists'] = sorted(ko_artists, key=lambda x: x['ko_wins'], reverse=True)[:15]

    # Submission artists
    sub_artists = []
    for _, fighter in fighters_df.iterrows():
        fid = fighter['fighter_id']
        subs = len(fights_df[(fights_df['winner_id'] == fid) & (fights_df['method'].str.contains('Sub', na=False, case=False))])
        if subs >= 3:
            sub_artists.append({'name': fighter['name'], 'sub_wins': subs, 'total_wins': fighter['wins']})
    stats['sub_artists'] = sorted(sub_artists, key=lambda x: x['sub_wins'], reverse=True)[:15]

    # Most wins
    stats['most_wins'] = fighters_df.nlargest(10, 'wins')[['name', 'wins', 'losses']].to_dict('records')

    # Most losses
    stats['most_losses'] = fighters_df.nlargest(10, 'losses')[['name', 'wins', 'losses']].to_dict('records')

    # Volume strikers
    top_strikers = fighters_with_stats.dropna(subset=['sig_strikes_landed_per_min']).nlargest(10, 'sig_strikes_landed_per_min')
    stats['volume_strikers'] = top_strikers[['name', 'sig_strikes_landed_per_min']].to_dict('records')

    # Takedown specialists
    top_td = fighters_with_stats.dropna(subset=['takedowns_avg_per_15min']).nlargest(10, 'takedowns_avg_per_15min')
    stats['takedown_specialists'] = top_td[['name', 'takedowns_avg_per_15min']].to_dict('records')

    # Submission attempt rate
    top_sub_att = fighters_with_stats.dropna(subset=['submissions_avg_per_15min']).nlargest(10, 'submissions_avg_per_15min')
    stats['sub_attempt_rate'] = top_sub_att[['name', 'submissions_avg_per_15min']].to_dict('records')

    # Most active fighters (by UFC fight count)
    fight_counts = []
    for _, fighter in fighters_df.iterrows():
        fid = fighter['fighter_id']
        count = len(fights_df[(fights_df['fighter_red_id'] == fid) | (fights_df['fighter_blue_id'] == fid)])
        if count >= 10:
            fight_counts.append({'name': fighter['name'], 'ufc_fights': count, 'wins': fighter['wins'], 'losses': fighter['losses']})
    stats['most_active'] = sorted(fight_counts, key=lambda x: x['ufc_fights'], reverse=True)[:15]

    # Win/Loss streaks
    fighter_streaks = []
    for _, fighter in fighters_df.iterrows():
        fid = fighter['fighter_id']
        fighter_fights = fights_df[(fights_df['fighter_red_id'] == fid) | (fights_df['fighter_blue_id'] == fid)].sort_values('fight_id', ascending=False)

        streak = 0
        streak_type = None
        for _, fight in fighter_fights.iterrows():
            won = (fight['winner_id'] == fid)
            if streak_type is None:
                streak_type = 'W' if won else 'L'
                streak = 1
            elif (streak_type == 'W' and won) or (streak_type == 'L' and not won):
                streak += 1
            else:
                break

        if streak >= 3:
            fighter_streaks.append({
                'name': fighter['name'],
                'streak': streak,
                'type': streak_type,
                'wins': fighter['wins'],
                'losses': fighter['losses']
            })

    stats['win_streaks'] = sorted([f for f in fighter_streaks if f['type'] == 'W'], key=lambda x: x['streak'], reverse=True)[:15]
    stats['loss_streaks'] = sorted([f for f in fighter_streaks if f['type'] == 'L'], key=lambda x: x['streak'], reverse=True)[:10]

    # First Round KO Specialists
    first_round_ko_artists = []
    for _, fighter in fighters_df.iterrows():
        fid = fighter['fighter_id']
        r1_kos = len(fights_df[(fights_df['winner_id'] == fid) &
                               (fights_df['method'].str.contains('KO', na=False)) &
                               (fights_df['round'] == 1)])
        total_kos = len(fights_df[(fights_df['winner_id'] == fid) &
                                  (fights_df['method'].str.contains('KO', na=False))])
        if r1_kos >= 3:
            first_round_ko_artists.append({
                'name': fighter['name'],
                'r1_kos': r1_kos,
                'total_kos': total_kos,
                'r1_pct': r1_kos / total_kos * 100 if total_kos > 0 else 0
            })
    stats['first_round_ko_artists'] = sorted(first_round_ko_artists, key=lambda x: x['r1_kos'], reverse=True)[:10]

    # KO Win Percentage (for fighters with 5+ wins)
    ko_win_pct = []
    for _, fighter in fighters_df.iterrows():
        if fighter['wins'] >= 5:
            fid = fighter['fighter_id']
            kos = len(fights_df[(fights_df['winner_id'] == fid) & (fights_df['method'].str.contains('KO', na=False))])
            ko_pct = kos / fighter['wins'] * 100 if fighter['wins'] > 0 else 0
            if ko_pct >= 50:
                ko_win_pct.append({
                    'name': fighter['name'],
                    'ko_wins': kos,
                    'total_wins': fighter['wins'],
                    'ko_pct': ko_pct
                })
    stats['highest_ko_pct'] = sorted(ko_win_pct, key=lambda x: x['ko_pct'], reverse=True)[:10]

    # Iron Chin - Most fights without being KO'd (with 10+ fights)
    iron_chin = []
    for _, fighter in fighters_df.iterrows():
        fid = fighter['fighter_id']
        total_fights = len(fights_df[(fights_df['fighter_red_id'] == fid) | (fights_df['fighter_blue_id'] == fid)])
        if total_fights >= 10:
            ko_losses = len(fights_df[
                ((fights_df['fighter_red_id'] == fid) | (fights_df['fighter_blue_id'] == fid)) &
                (fights_df['winner_id'] != fid) &
                (fights_df['method'].str.contains('KO', na=False))
            ])
            if ko_losses == 0:
                iron_chin.append({
                    'name': fighter['name'],
                    'fights': total_fights,
                    'wins': fighter['wins'],
                    'losses': fighter['losses'],
                    'ko_losses': ko_losses
                })
    stats['iron_chin'] = sorted(iron_chin, key=lambda x: x['fights'], reverse=True)[:10]

    # Striking Defense Leaders
    top_str_def = fighters_with_stats.dropna(subset=['sig_strike_defense'])
    top_str_def = top_str_def[top_str_def['sig_strike_defense'] > 0]
    if len(top_str_def) >= 5:
        top_str_def = top_str_def.nlargest(10, 'sig_strike_defense')
        stats['best_str_defense'] = top_str_def[['name', 'sig_strike_defense']].to_dict('records')
    else:
        stats['best_str_defense'] = []

    # Striking Accuracy Leaders
    top_str_acc = fighters_with_stats.dropna(subset=['sig_strike_accuracy'])
    top_str_acc = top_str_acc[top_str_acc['sig_strike_accuracy'] > 0]
    if len(top_str_acc) >= 5:
        top_str_acc = top_str_acc.nlargest(10, 'sig_strike_accuracy')
        stats['best_str_accuracy'] = top_str_acc[['name', 'sig_strike_accuracy']].to_dict('records')
    else:
        stats['best_str_accuracy'] = []

    # Submission Win Percentage (for fighters with 5+ wins)
    sub_win_pct = []
    for _, fighter in fighters_df.iterrows():
        if fighter['wins'] >= 5:
            fid = fighter['fighter_id']
            subs = len(fights_df[(fights_df['winner_id'] == fid) & (fights_df['method'].str.contains('Sub', na=False, case=False))])
            sub_pct = subs / fighter['wins'] * 100 if fighter['wins'] > 0 else 0
            if sub_pct >= 40:
                sub_win_pct.append({
                    'name': fighter['name'],
                    'sub_wins': subs,
                    'total_wins': fighter['wins'],
                    'sub_pct': sub_pct
                })
    stats['highest_sub_pct'] = sorted(sub_win_pct, key=lambda x: x['sub_pct'], reverse=True)[:10]

    # Most Diverse Submission Artists
    sub_diversity = []
    for _, fighter in fighters_df.iterrows():
        fid = fighter['fighter_id']
        fighter_subs = fights_df[(fights_df['winner_id'] == fid) & (fights_df['method'].str.contains('Sub', na=False, case=False))]
        unique_subs = fighter_subs['method_detail'].dropna().nunique()
        total_subs = len(fighter_subs)
        if total_subs >= 3 and unique_subs >= 2:
            sub_diversity.append({
                'name': fighter['name'],
                'unique_subs': unique_subs,
                'total_subs': total_subs,
                'sub_types': list(fighter_subs['method_detail'].dropna().unique())[:5]
            })
    stats['most_diverse_subs'] = sorted(sub_diversity, key=lambda x: x['unique_subs'], reverse=True)[:10]

    # Finisher vs Decision Fighter Analysis
    finishers = []
    decision_fighters = []
    for _, fighter in fighters_df.iterrows():
        if fighter['wins'] >= 8:
            fid = fighter['fighter_id']
            finishes = len(fights_df[(fights_df['winner_id'] == fid) &
                                     ((fights_df['method'].str.contains('KO', na=False)) |
                                      (fights_df['method'].str.contains('Sub', na=False, case=False)))])
            decisions = len(fights_df[(fights_df['winner_id'] == fid) &
                                      (fights_df['method'].str.contains('Dec', na=False, case=False))])
            finish_rate = finishes / fighter['wins'] * 100 if fighter['wins'] > 0 else 0
            decision_rate = decisions / fighter['wins'] * 100 if fighter['wins'] > 0 else 0

            if finish_rate >= 70:
                finishers.append({
                    'name': fighter['name'],
                    'finishes': finishes,
                    'total_wins': fighter['wins'],
                    'finish_rate': finish_rate
                })
            if decision_rate >= 60:
                decision_fighters.append({
                    'name': fighter['name'],
                    'decisions': decisions,
                    'total_wins': fighter['wins'],
                    'decision_rate': decision_rate
                })

    stats['top_finishers'] = sorted(finishers, key=lambda x: x['finish_rate'], reverse=True)[:10]
    stats['top_decision_fighters'] = sorted(decision_fighters, key=lambda x: x['decision_rate'], reverse=True)[:10]

    # Most Exciting Fighters
    exciting_fighters = []
    for _, fighter in fighters_df.iterrows():
        fid = fighter['fighter_id']
        total_fights = len(fights_df[(fights_df['fighter_red_id'] == fid) | (fights_df['fighter_blue_id'] == fid)])
        if total_fights >= 10:
            finishes_for = len(fights_df[(fights_df['winner_id'] == fid) &
                                         ((fights_df['method'].str.contains('KO', na=False)) |
                                          (fights_df['method'].str.contains('Sub', na=False, case=False)))])
            finishes_against = len(fights_df[
                ((fights_df['fighter_red_id'] == fid) | (fights_df['fighter_blue_id'] == fid)) &
                (fights_df['winner_id'] != fid) &
                ((fights_df['method'].str.contains('KO', na=False)) |
                 (fights_df['method'].str.contains('Sub', na=False, case=False)))
            ])
            excitement_score = (finishes_for + finishes_against) / total_fights * 100
            if excitement_score >= 60:
                exciting_fighters.append({
                    'name': fighter['name'],
                    'total_fights': total_fights,
                    'finishes_for': finishes_for,
                    'finishes_against': finishes_against,
                    'excitement_score': excitement_score
                })
    stats['most_exciting'] = sorted(exciting_fighters, key=lambda x: x['excitement_score'], reverse=True)[:10]

    # Submission types
    sub_fights = fights_df[fights_df['method'].str.contains('Sub', na=False, case=False)]
    sub_types = sub_fights['method_detail'].value_counts()
    sub_types = sub_types[sub_types.index.notna() & (sub_types.index != 'None') & (sub_types.index != '')]
    stats['top_submissions'] = [{'type': t, 'count': int(c)} for t, c in sub_types.head(5).items()]
    stats['bottom_submissions'] = [{'type': t, 'count': int(c)} for t, c in sub_types.tail(5).items()]

    # Submissions by round
    sub_by_round = sub_fights['round'].value_counts().sort_index()
    stats['sub_by_round'] = {int(r): int(c) for r, c in sub_by_round.items() if pd.notna(r)}

    # Best takedown defence
    top_td_def = fighters_with_stats.dropna(subset=['takedown_defense'])
    top_td_def = top_td_def[top_td_def['takedown_defense'] > 0]

    if len(top_td_def) < 5:
        with data_service.get_connection() as conn:
            td_defense_query = '''
                SELECT
                    fr.fighter_id,
                    fr.name,
                    SUM(opp_stats.takedowns_landed) as tds_against_landed,
                    SUM(opp_stats.takedowns_attempted) as tds_against_attempted
                FROM fighters fr
                JOIN fights f ON (f.fighter_red_id = fr.fighter_id OR f.fighter_blue_id = fr.fighter_id)
                LEFT JOIN fight_stats opp_stats ON f.fight_id = opp_stats.fight_id AND opp_stats.fighter_id != fr.fighter_id
                WHERE opp_stats.takedowns_attempted > 0
                GROUP BY fr.fighter_id, fr.name
                HAVING SUM(opp_stats.takedowns_attempted) >= 10
            '''
            cursor = conn.execute(td_defense_query)
            td_def_data = []
            for row in cursor.fetchall():
                fid, name, tds_landed, tds_attempted = row
                if tds_attempted and tds_attempted > 0:
                    td_def = 1 - (tds_landed / tds_attempted)
                    td_def_data.append({'name': name, 'takedown_defense': td_def})

            td_def_data = sorted(td_def_data, key=lambda x: x['takedown_defense'], reverse=True)
            stats['best_td_defense'] = td_def_data[:10]
    else:
        top_td_def = top_td_def.nlargest(10, 'takedown_defense')
        stats['best_td_defense'] = top_td_def[['name', 'takedown_defense']].to_dict('records')

    # Submission wins while in striking deficit
    sub_deficit_wins = []
    with data_service.get_connection() as conn:
        query = '''
            SELECT
                f.fight_id,
                f.winner_id,
                f.fighter_red_id,
                f.fighter_blue_id,
                fr.name as red_name,
                fb.name as blue_name,
                f.method_detail,
                COALESCE(ws.sig_strikes_landed, 0) as winner_strikes,
                COALESCE(ls.sig_strikes_landed, 0) as loser_strikes
            FROM fights f
            JOIN fighters fr ON f.fighter_red_id = fr.fighter_id
            JOIN fighters fb ON f.fighter_blue_id = fb.fighter_id
            LEFT JOIN fight_stats ws ON f.fight_id = ws.fight_id AND f.winner_id = ws.fighter_id
            LEFT JOIN fight_stats ls ON f.fight_id = ls.fight_id AND f.winner_id != ls.fighter_id
            WHERE f.method LIKE '%Sub%'
            AND ws.sig_strikes_landed IS NOT NULL
            AND ls.sig_strikes_landed IS NOT NULL
        '''
        cursor = conn.execute(query)
        rows = cursor.fetchall()

        for row in rows:
            fight_id, winner_id, red_id, blue_id, red_name, blue_name, sub_type, w_strikes, l_strikes = row
            if w_strikes < l_strikes:
                winner_name = red_name if winner_id == red_id else blue_name
                sub_deficit_wins.append({
                    'fighter': winner_name,
                    'submission': sub_type or 'Submission',
                    'winner_strikes': w_strikes,
                    'loser_strikes': l_strikes,
                    'deficit': l_strikes - w_strikes
                })

    stats['sub_wins_in_deficit'] = len(sub_deficit_wins)
    stats['sub_wins_in_deficit_details'] = sorted(sub_deficit_wins, key=lambda x: x['deficit'], reverse=True)[:10]

    deficit_fighters = {}
    for win in sub_deficit_wins:
        name = win['fighter']
        deficit_fighters[name] = deficit_fighters.get(name, 0) + 1
    stats['top_deficit_sub_artists'] = sorted(
        [{'name': n, 'count': c} for n, c in deficit_fighters.items()],
        key=lambda x: x['count'],
        reverse=True
    )[:5]

    return stats


@st.cache_data(ttl=86400, show_spinner=False)
def load_ml_report():
    """Load or generate ML analysis report."""
    cached = analysis_service.load_cached_report()
    if cached:
        return cached
    return analysis_service.generate_full_report()


@st.cache_data(ttl=86400)
def get_key_facts():
    """Load key facts from unsupervised analysis."""
    try:
        report = analysis_service.load_cached_report()
        if not report:
            return {
                'career': analysis_service.analyze_career_statistics(),
                'champion': analysis_service.analyze_champion_patterns(),
                'path_to_title': analysis_service.analyze_path_to_title(),
                'age': analysis_service.analyze_age_performance(),
                'reach': analysis_service.analyze_height_reach_advantage(),
            }
        return {
            'career': report.get('sections', {}).get('career_analysis', {}).get('statistics', {}),
            'champion': report.get('sections', {}).get('champion_analysis', {}).get('patterns', {}),
            'path_to_title': report.get('sections', {}).get('champion_analysis', {}).get('path_to_title', {}),
            'age': report.get('sections', {}).get('age_analysis', {}).get('performance', {}),
            'reach': report.get('sections', {}).get('physical_analysis', {}).get('height_reach', {}),
        }
    except Exception as e:
        return None


# ============================================================================
# LOAD DATA
# ============================================================================
with st.spinner("Loading comprehensive statistics..."):
    stats = get_comprehensive_stats()

# Header controls
col1, col2 = st.columns([3, 1])
with col2:
    if st.button("Regenerate Analysis", key="regen"):
        st.cache_data.clear()
        st.rerun()

with col1:
    st.caption("Analysis is cached for better performance. Click 'Regenerate' to refresh all data.")

# Load ML report
with st.spinner("Loading ML analysis..."):
    try:
        ml_report = load_ml_report()
    except Exception as e:
        ml_report = None
        st.warning("ML analysis unavailable. Some features will be limited.")

st.markdown("---")

# ============================================================================
# EXECUTIVE SUMMARY - KEY DISCOVERIES
# ============================================================================
if ml_report and 'executive_summary' in ml_report:
    st.markdown("### Key Discoveries")
    st.markdown("*Machine learning patterns discovered through unsupervised analysis*")
    summary = ml_report['executive_summary']
    if 'key_findings' in summary:
        cols = st.columns(2)
        for i, finding in enumerate(summary['key_findings'][:6]):
            with cols[i % 2]:
                st.markdown(f"""
                <div style="background: {COLORS['card_bg']}; padding: 12px 15px; border-radius: 8px; margin-bottom: 8px; border-left: 3px solid {COLORS['primary']};">
                    <span style="color: {COLORS['text_primary']};">{finding}</span>
                </div>
                """, unsafe_allow_html=True)
    st.markdown("---")

# ============================================================================
# UFC BY THE NUMBERS - KEY FACTS
# ============================================================================
st.markdown(f"""
<div style="margin-bottom: 25px;">
    <h2 style="color: {COLORS['text_primary']}; margin: 0; font-size: 28px; font-weight: 700;">UFC By The Numbers</h2>
    <p style="color: {COLORS['text_muted']}; margin: 5px 0 0 0; font-size: 15px;">Key statistics discovered through machine learning analysis of the entire UFC dataset</p>
</div>
""", unsafe_allow_html=True)

try:
    key_facts = get_key_facts()
except:
    key_facts = None

if key_facts:
    career = key_facts.get('career', {})
    age = key_facts.get('age', {})
    path = key_facts.get('path_to_title', {})
    champ = key_facts.get('champion', {})
    reach = key_facts.get('reach', {})

    # SECTION 1: FIGHTER CAREER & DEMOGRAPHICS
    st.markdown(f"""
    <div style="margin: 20px 0 15px 0; padding-bottom: 8px; border-bottom: 2px solid {COLORS['primary']}40;">
        <span style="color: {COLORS['primary']}; font-size: 13px; font-weight: 600; letter-spacing: 1px;">FIGHTER CAREER & DEMOGRAPHICS</span>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div style="background: {COLORS['card_bg']}; padding: 20px; border-radius: 12px; text-align: center; height: 140px;">
            <p style="color: {COLORS['text_muted']}; margin: 0; font-size: 13px; font-weight: 500;">Average Career Length</p>
            <p style="color: {COLORS['text_primary']}; font-size: 36px; font-weight: 700; margin: 8px 0 4px 0; line-height: 1;">{career.get('avg_career_duration_years', 0):.1f}</p>
            <p style="color: {COLORS['text_secondary']}; margin: 0; font-size: 14px;">years in the UFC</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="background: {COLORS['card_bg']}; padding: 20px; border-radius: 12px; text-align: center; height: 140px;">
            <p style="color: {COLORS['text_muted']}; margin: 0; font-size: 13px; font-weight: 500;">Fights Per Fighter</p>
            <p style="color: {COLORS['text_primary']}; font-size: 36px; font-weight: 700; margin: 8px 0 4px 0; line-height: 1;">{career.get('avg_ufc_fights', 0):.1f}</p>
            <p style="color: {COLORS['text_secondary']}; margin: 0; font-size: 14px;">average UFC bouts</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div style="background: {COLORS['card_bg']}; padding: 20px; border-radius: 12px; text-align: center; height: 140px;">
            <p style="color: {COLORS['text_muted']}; margin: 0; font-size: 13px; font-weight: 500;">Average Fighter Age</p>
            <p style="color: {COLORS['info']}; font-size: 36px; font-weight: 700; margin: 8px 0 4px 0; line-height: 1;">{age.get('avg_fighter_age', 0):.1f}</p>
            <p style="color: {COLORS['text_secondary']}; margin: 0; font-size: 14px;">years old</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div style="background: {COLORS['card_bg']}; padding: 20px; border-radius: 12px; text-align: center; height: 140px;">
            <p style="color: {COLORS['text_muted']}; margin: 0; font-size: 13px; font-weight: 500;">Peak Performance Age</p>
            <p style="color: {COLORS['success']}; font-size: 36px; font-weight: 700; margin: 8px 0 4px 0; line-height: 1;">{age.get('peak_age_bracket', 'N/A')}</p>
            <p style="color: {COLORS['text_secondary']}; margin: 0; font-size: 14px;">optimal age range</p>
        </div>
        """, unsafe_allow_html=True)

    # SECTION 2: THE ROAD TO CHAMPIONSHIP GOLD
    st.markdown(f"""
    <div style="margin: 30px 0 15px 0; padding-bottom: 8px; border-bottom: 2px solid {COLORS['warning']}40;">
        <span style="color: {COLORS['warning']}; font-size: 13px; font-weight: 600; letter-spacing: 1px;">THE ROAD TO CHAMPIONSHIP GOLD</span>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    pct = path.get('pct_got_title_shot', 0)
    fights_to_title = path.get('avg_fights_before_title', 0)
    years_to_title = path.get('avg_years_to_title', 0)

    with col1:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {COLORS['card_bg']} 0%, {COLORS['warning']}15 100%); padding: 20px; border-radius: 12px; border-left: 4px solid {COLORS['warning']}; height: 160px;">
            <p style="color: {COLORS['text_muted']}; margin: 0; font-size: 12px; font-weight: 600; letter-spacing: 0.5px;">ELITE FEW</p>
            <p style="color: {COLORS['warning']}; font-size: 42px; font-weight: 800; margin: 8px 0 6px 0; line-height: 1;">{pct:.1f}%</p>
            <p style="color: {COLORS['text_primary']}; margin: 0; font-size: 15px; font-weight: 500;">of fighters get a title shot</p>
            <p style="color: {COLORS['text_muted']}; margin: 6px 0 0 0; font-size: 13px;">The vast majority never compete for gold</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {COLORS['card_bg']} 0%, {COLORS['info']}15 100%); padding: 20px; border-radius: 12px; border-left: 4px solid {COLORS['info']}; height: 160px;">
            <p style="color: {COLORS['text_muted']}; margin: 0; font-size: 12px; font-weight: 600; letter-spacing: 0.5px;">THE CLIMB</p>
            <p style="color: {COLORS['info']}; font-size: 42px; font-weight: 800; margin: 8px 0 6px 0; line-height: 1;">{fights_to_title:.1f}</p>
            <p style="color: {COLORS['text_primary']}; margin: 0; font-size: 15px; font-weight: 500;">fights to earn a title shot</p>
            <p style="color: {COLORS['text_muted']}; margin: 6px 0 0 0; font-size: 13px;">Average wins needed to contend</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {COLORS['card_bg']} 0%, {COLORS['primary']}15 100%); padding: 20px; border-radius: 12px; border-left: 4px solid {COLORS['primary']}; height: 160px;">
            <p style="color: {COLORS['text_muted']}; margin: 0; font-size: 12px; font-weight: 600; letter-spacing: 0.5px;">TIME INVESTED</p>
            <p style="color: {COLORS['primary']}; font-size: 42px; font-weight: 800; margin: 8px 0 6px 0; line-height: 1;">{years_to_title:.1f}</p>
            <p style="color: {COLORS['text_primary']}; margin: 0; font-size: 15px; font-weight: 500;">years to reach a title fight</p>
            <p style="color: {COLORS['text_muted']}; margin: 6px 0 0 0; font-size: 13px;">From UFC debut to championship</p>
        </div>
        """, unsafe_allow_html=True)

    # SECTION 3: CHAMPIONSHIP BATTLES
    st.markdown(f"""
    <div style="margin: 30px 0 15px 0; padding-bottom: 8px; border-bottom: 2px solid {COLORS['danger']}40;">
        <span style="color: {COLORS['danger']}; font-size: 13px; font-weight: 600; letter-spacing: 1px;">CHAMPIONSHIP BATTLES</span>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    cwr = champ.get('challenger_win_rate', 0)
    ko_rate = champ.get('title_ko_rate', 0)
    sub_rate = champ.get('title_sub_rate', 0)
    dec_rate = champ.get('title_dec_rate', 0)
    dur = champ.get('avg_title_fight_minutes', 0)

    with col1:
        st.markdown(f"""
        <div style="background: {COLORS['card_bg']}; padding: 22px; border-radius: 12px;">
            <h4 style="color: {COLORS['text_primary']}; margin: 0 0 18px 0; font-size: 17px; font-weight: 600;">How Title Fights End</h4>
            <div style="display: flex; justify-content: space-between; gap: 15px;">
                <div style="text-align: center; flex: 1; background: {COLORS['danger']}15; padding: 15px 10px; border-radius: 10px;">
                    <p style="color: {COLORS['danger']}; font-size: 32px; font-weight: 800; margin: 0; line-height: 1;">{ko_rate:.0f}%</p>
                    <p style="color: {COLORS['text_secondary']}; font-size: 13px; margin: 6px 0 0 0; font-weight: 500;">KO/TKO</p>
                </div>
                <div style="text-align: center; flex: 1; background: {COLORS['info']}15; padding: 15px 10px; border-radius: 10px;">
                    <p style="color: {COLORS['info']}; font-size: 32px; font-weight: 800; margin: 0; line-height: 1;">{sub_rate:.0f}%</p>
                    <p style="color: {COLORS['text_secondary']}; font-size: 13px; margin: 6px 0 0 0; font-weight: 500;">Submission</p>
                </div>
                <div style="text-align: center; flex: 1; background: {COLORS['warning']}15; padding: 15px 10px; border-radius: 10px;">
                    <p style="color: {COLORS['warning']}; font-size: 32px; font-weight: 800; margin: 0; line-height: 1;">{dec_rate:.0f}%</p>
                    <p style="color: {COLORS['text_secondary']}; font-size: 13px; margin: 6px 0 0 0; font-weight: 500;">Decision</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="background: {COLORS['card_bg']}; padding: 22px; border-radius: 12px; display: flex; gap: 20px;">
            <div style="flex: 1; text-align: center; border-right: 1px solid {COLORS['text_muted']}30; padding-right: 20px;">
                <p style="color: {COLORS['text_muted']}; margin: 0; font-size: 12px; font-weight: 600; letter-spacing: 0.5px;">CHALLENGER WINS</p>
                <p style="color: {COLORS['danger']}; font-size: 38px; font-weight: 800; margin: 10px 0 4px 0; line-height: 1;">{cwr:.1f}%</p>
                <p style="color: {COLORS['text_secondary']}; margin: 0; font-size: 13px;">of title fights</p>
                <p style="color: {COLORS['text_muted']}; margin: 8px 0 0 0; font-size: 12px;">Champions defend {100-cwr:.0f}% of the time</p>
            </div>
            <div style="flex: 1; text-align: center;">
                <p style="color: {COLORS['text_muted']}; margin: 0; font-size: 12px; font-weight: 600; letter-spacing: 0.5px;">AVG FIGHT LENGTH</p>
                <p style="color: {COLORS['success']}; font-size: 38px; font-weight: 800; margin: 10px 0 4px 0; line-height: 1;">{dur:.1f}</p>
                <p style="color: {COLORS['text_secondary']}; margin: 0; font-size: 13px;">minutes</p>
                <p style="color: {COLORS['text_muted']}; margin: 8px 0 0 0; font-size: 12px;">Title fights go longer</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # SECTION 4: PHYSICAL ADVANTAGES
    st.markdown(f"""
    <div style="margin: 30px 0 15px 0; padding-bottom: 8px; border-bottom: 2px solid {COLORS['success']}40;">
        <span style="color: {COLORS['success']}; font-size: 13px; font-weight: 600; letter-spacing: 1px;">PHYSICAL ADVANTAGES</span>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    rwr = reach.get('longer_reach_win_rate', 50)
    twr = reach.get('taller_fighter_win_rate', 50)
    reach_ko = reach.get('reach_advantage_in_kos', 50)
    reach_sub = reach.get('reach_advantage_in_subs', 50)

    with col1:
        st.markdown(f"""
        <div style="background: {COLORS['card_bg']}; padding: 20px; border-radius: 12px; text-align: center; height: 150px;">
            <p style="color: {COLORS['text_muted']}; margin: 0; font-size: 13px; font-weight: 500;">Reach Advantage (5cm+)</p>
            <p style="color: {COLORS['success']}; font-size: 40px; font-weight: 800; margin: 10px 0 6px 0; line-height: 1;">{rwr:.1f}%</p>
            <p style="color: {COLORS['text_secondary']}; margin: 0; font-size: 14px;">win rate for longer reach</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="background: {COLORS['card_bg']}; padding: 20px; border-radius: 12px; text-align: center; height: 150px;">
            <p style="color: {COLORS['text_muted']}; margin: 0; font-size: 13px; font-weight: 500;">Height Advantage</p>
            <p style="color: {COLORS['info']}; font-size: 40px; font-weight: 800; margin: 10px 0 6px 0; line-height: 1;">{twr:.1f}%</p>
            <p style="color: {COLORS['text_secondary']}; margin: 0; font-size: 14px;">win rate for taller fighter</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        better_for = "KOs" if reach_ko > reach_sub else "Submissions"
        better_color = COLORS['danger'] if reach_ko > reach_sub else COLORS['info']
        st.markdown(f"""
        <div style="background: {COLORS['card_bg']}; padding: 20px; border-radius: 12px; height: 150px;">
            <p style="color: {COLORS['text_muted']}; margin: 0; font-size: 13px; font-weight: 500;">Reach Best For...</p>
            <p style="color: {better_color}; font-size: 28px; font-weight: 700; margin: 10px 0 6px 0; line-height: 1.2;">{better_for}</p>
            <p style="color: {COLORS['text_secondary']}; margin: 0; font-size: 13px;">KO: {reach_ko:.1f}% vs Sub: {reach_sub:.1f}%</p>
            <p style="color: {COLORS['text_muted']}; margin: 6px 0 0 0; font-size: 12px;">Reach helps {'strikers more' if reach_ko > reach_sub else 'grapplers more'}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ============================================================================
# TABS - MAIN CONTENT
# ============================================================================
tabs = st.tabs([
    "Overview",
    "Division Analysis",
    "Fighter Records",
    "Fighter Styles",
    "Age & Physical",
    "Career & Titles",
    "Style Evolution",
    "Skill Analysis",
    "Ask AI"
])

sections = ml_report.get('sections', {}) if ml_report else {}

# ============================================================================
# TAB 1: OVERVIEW DASHBOARD
# ============================================================================
with tabs[0]:
    st.markdown("### Overview Dashboard")

    # Key metrics row
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Fights", f"{stats['total_fights']:,}")
    with col2:
        st.metric("Total Fighters", f"{stats['total_fighters']:,}")
    with col3:
        st.metric("Finish Rate", f"{stats['finish_rate']:.1f}%")
    with col4:
        st.metric("KO/TKO Rate", f"{stats['ko_rate']:.1f}%")
    with col5:
        st.metric("Submission Rate", f"{stats['sub_rate']:.1f}%")

    st.markdown("---")

    # Key Facts Section
    st.markdown("### Key Facts")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div style="background: {COLORS['card_bg']}; padding: 20px; border-radius: 10px; height: 150px;">
            <h4 style="color: {COLORS['danger']}; margin-top:0;">Most Dangerous Division</h4>
            <p style="color: {COLORS['text_primary']}; font-size: 20px; font-weight: bold; margin: 5px 0;">{stats['highest_ko_rate_division'][0]}</p>
            <p style="color: {COLORS['text_secondary']}; margin: 0;">{stats['highest_ko_rate_division'][1]['ko_rate']:.1f}% KO rate</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="background: {COLORS['card_bg']}; padding: 20px; border-radius: 10px; height: 150px;">
            <h4 style="color: {COLORS['info']}; margin-top:0;">Most Submissions</h4>
            <p style="color: {COLORS['text_primary']}; font-size: 20px; font-weight: bold; margin: 5px 0;">{stats['most_sub_division'][0]}</p>
            <p style="color: {COLORS['text_secondary']}; margin: 0;">{stats['most_sub_division'][1]['sub']} total submissions</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div style="background: {COLORS['card_bg']}; padding: 20px; border-radius: 10px; height: 150px;">
            <h4 style="color: {COLORS['success']}; margin-top:0;">Most Active Division</h4>
            <p style="color: {COLORS['text_primary']}; font-size: 20px; font-weight: bold; margin: 5px 0;">{stats['most_active_division'][0]}</p>
            <p style="color: {COLORS['text_secondary']}; margin: 0;">{stats['most_active_division'][1]['total']:,} total fights</p>
        </div>
        """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div style="background: {COLORS['card_bg']}; padding: 20px; border-radius: 10px; height: 150px;">
            <h4 style="color: {COLORS['primary']}; margin-top:0;">Most KOs Total</h4>
            <p style="color: {COLORS['text_primary']}; font-size: 20px; font-weight: bold; margin: 5px 0;">{stats['most_ko_division'][0]}</p>
            <p style="color: {COLORS['text_secondary']}; margin: 0;">{stats['most_ko_division'][1]['ko']} knockouts</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="background: {COLORS['card_bg']}; padding: 20px; border-radius: 10px; height: 150px;">
            <h4 style="color: {COLORS['warning']}; margin-top:0;">Most Decisions</h4>
            <p style="color: {COLORS['text_primary']}; font-size: 20px; font-weight: bold; margin: 5px 0;">{stats['most_dec_division'][0]}</p>
            <p style="color: {COLORS['text_secondary']}; margin: 0;">{stats['most_dec_division'][1]['dec']} decisions</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        top_ko = stats['ko_artists'][0] if stats['ko_artists'] else {'name': 'N/A', 'ko_wins': 0}
        st.markdown(f"""
        <div style="background: {COLORS['card_bg']}; padding: 20px; border-radius: 10px; height: 150px;">
            <h4 style="color: {COLORS['danger']}; margin-top:0;">Top KO Artist</h4>
            <p style="color: {COLORS['text_primary']}; font-size: 20px; font-weight: bold; margin: 5px 0;">{top_ko['name']}</p>
            <p style="color: {COLORS['text_secondary']}; margin: 0;">{top_ko['ko_wins']} UFC KO wins</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Charts row
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Win Methods Distribution")
        fig = go.Figure(data=[go.Pie(
            labels=['KO/TKO', 'Submission', 'Decision', 'Other'],
            values=[stats['ko_count'], stats['sub_count'], stats['dec_count'],
                    stats['total_fights'] - stats['ko_count'] - stats['sub_count'] - stats['dec_count']],
            hole=0.4,
            marker_colors=[COLORS['danger'], COLORS['info'], COLORS['success'], COLORS['text_muted']]
        )])
        fig.update_layout(
            paper_bgcolor=COLORS['background'],
            font=dict(color=COLORS['text_primary']),
            height=350,
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### KOs by Round")
        if stats['ko_by_round']:
            rounds = list(stats['ko_by_round'].keys())
            counts = list(stats['ko_by_round'].values())
            fig = go.Figure(data=[go.Bar(
                x=[f"Round {r}" for r in rounds],
                y=counts,
                marker_color=COLORS['danger'],
                text=[f"{c} ({c/sum(counts)*100:.1f}%)" for c in counts],
                textposition='outside'
            )])
            fig.update_layout(
                paper_bgcolor=COLORS['background'],
                plot_bgcolor=COLORS['background'],
                font=dict(color=COLORS['text_primary']),
                height=350,
                yaxis_title="Number of KOs"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("48% of KOs happen in Round 1 - the first round is the most dangerous!")


# ============================================================================
# TAB 2: DIVISION ANALYSIS (MERGED)
# ============================================================================
with tabs[1]:
    st.markdown("### Division Analysis")
    st.markdown("*Comprehensive analysis of each weight class combining statistics and ML patterns*")

    # Division selector
    divisions = sorted(stats['division_stats'].keys())
    selected_div = st.selectbox("Select Weight Class", divisions, index=divisions.index('Lightweight') if 'Lightweight' in divisions else 0)

    if selected_div and selected_div in stats['division_stats']:
        div_stats = stats['division_stats'][selected_div]

        st.markdown("---")

        # Division metrics
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Total Fights", f"{div_stats['total']:,}")
        with col2:
            st.metric("KO/TKO Rate", f"{div_stats['ko_rate']:.1f}%")
        with col3:
            st.metric("Submission Rate", f"{div_stats['sub_rate']:.1f}%")
        with col4:
            st.metric("Decision Rate", f"{div_stats['dec_rate']:.1f}%")
        with col5:
            st.metric("Finish Rate", f"{div_stats['finish_rate']:.1f}%")

        st.markdown("---")

        # Comparison with UFC average
        st.markdown("#### Division vs UFC Average")

        comparison_data = pd.DataFrame({
            'Metric': ['KO Rate', 'Submission Rate', 'Decision Rate', 'Finish Rate'],
            selected_div: [div_stats['ko_rate'], div_stats['sub_rate'], div_stats['dec_rate'], div_stats['finish_rate']],
            'UFC Average': [stats['ko_rate'], stats['sub_rate'], stats['dec_rate'], stats['finish_rate']]
        })

        fig = go.Figure()
        fig.add_trace(go.Bar(name=selected_div, x=comparison_data['Metric'], y=comparison_data[selected_div], marker_color=COLORS['primary']))
        fig.add_trace(go.Bar(name='UFC Average', x=comparison_data['Metric'], y=comparison_data['UFC Average'], marker_color=COLORS['text_muted']))
        fig.update_layout(
            barmode='group',
            paper_bgcolor=COLORS['background'],
            plot_bgcolor=COLORS['background'],
            font=dict(color=COLORS['text_primary']),
            height=400,
            yaxis_title="Percentage (%)"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Division tendency
        tendency = 'STRIKER-HEAVY' if div_stats['ko_rate'] > div_stats['sub_rate'] * 1.5 else ('GRAPPLER-HEAVY' if div_stats['sub_rate'] > div_stats['ko_rate'] else 'BALANCED')
        st.info(f"**{selected_div}** is a **{tendency}** division with {div_stats['ko_rate']:.1f}% KO rate vs {div_stats['sub_rate']:.1f}% submission rate.")

    st.markdown("---")

    # ML Division Patterns
    div_data = sections.get('division_patterns', {})

    if div_data and 'error' not in div_data:
        clustering = div_data.get('clustering', {})
        recovery = div_data.get('recovery', {})
        newcomers = div_data.get('newcomers', {})

        # Division archetypes from ML
        if clustering and clustering.get('division_clusters'):
            st.markdown("#### Division Archetypes (ML Clustering)")
            for cluster in clustering.get('division_clusters', []):
                st.markdown(f"""
                <div style="background: {COLORS['card_bg']}; padding: 15px; border-radius: 10px; margin-bottom: 12px; border-left: 4px solid {COLORS['primary']};">
                    <h4 style="color: {COLORS['primary']}; margin: 0 0 8px 0;">{cluster['label']}</h4>
                    <p style="color: {COLORS['text_secondary']}; margin: 0 0 10px 0;">{cluster['description']}</p>
                    <p style="color: {COLORS['text_muted']}; margin: 0; font-size: 13px;"><strong>Divisions:</strong> {', '.join(cluster.get('divisions', []))}</p>
                    <div style="display: flex; gap: 20px; margin-top: 10px;">
                        <span style="color: {COLORS['danger']};">KO: {cluster['avg_ko_rate']*100:.1f}%</span>
                        <span style="color: {COLORS['info']};">Sub: {cluster['avg_sub_rate']*100:.1f}%</span>
                        <span style="color: {COLORS['success']};">Finish: {cluster['avg_finish_rate']*100:.1f}%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Division recovery analysis
        if recovery and 'hardest_to_recover' in recovery:
            st.markdown("#### Hardest Divisions to Recover from Losing Streaks")
            st.warning(f"**Hardest:** {', '.join(recovery.get('hardest_to_recover', [])[:3])}")
            st.success(f"**Easiest:** {', '.join(recovery.get('easiest_to_recover', [])[:3])}")

        # Newcomers analysis
        if newcomers and 'division_newcomer_stats' in newcomers:
            st.markdown("#### Division Newcomer Activity (Since 2020)")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Most Newcomers", newcomers.get('most_newcomers_division', 'Unknown'))
            with col2:
                st.metric("Overall Debut Win Rate", f"{newcomers.get('overall_debut_win_rate', 0):.1f}%")

    st.markdown("---")

    # All divisions comparison
    st.markdown("#### All Divisions Comparison")

    div_comparison = []
    for div, ds in stats['division_stats'].items():
        div_comparison.append({
            'Division': div,
            'Fights': ds['total'],
            'KO Rate': ds['ko_rate'],
            'Sub Rate': ds['sub_rate'],
            'Finish Rate': ds['finish_rate']
        })

    div_df = pd.DataFrame(div_comparison).sort_values('Fights', ascending=False)

    fig = px.bar(div_df, x='Division', y='KO Rate',
                 color='KO Rate', color_continuous_scale='Reds',
                 title='KO Rate by Division')
    fig.update_layout(
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['background'],
        font=dict(color=COLORS['text_primary']),
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# TAB 3: FIGHTER RECORDS
# ============================================================================
with tabs[2]:
    st.markdown("### Fighter Records")

    section = st.radio(
        "Select Category",
        ["Striking Records", "Grappling Records", "Overall Records"],
        horizontal=True
    )

    st.markdown("---")

    if section == "Striking Records":
        st.markdown("## Striking Records")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total KO/TKOs", f"{stats['ko_count']:,}")
        with col2:
            st.metric("KO Rate", f"{stats['ko_rate']:.1f}%")
        with col3:
            r1_kos = stats['ko_by_round'].get(1, 0)
            total_kos = sum(stats['ko_by_round'].values()) if stats['ko_by_round'] else 1
            st.metric("Round 1 KOs", f"{r1_kos/total_kos*100:.1f}%")
        with col4:
            top_ko = stats['ko_artists'][0] if stats['ko_artists'] else {'ko_wins': 0}
            st.metric("Most KOs (Single Fighter)", f"{top_ko.get('ko_wins', 0)}")

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Most KO/TKO Wins")
            for i, f in enumerate(stats['ko_artists'][:10], 1):
                ko_pct = f['ko_wins'] / f['total_wins'] * 100 if f['total_wins'] > 0 else 0
                medal = "" if i == 1 else ("" if i == 2 else ("" if i == 3 else ""))
                st.markdown(f"""
                <div style="background: {COLORS['card_bg']}; padding: 12px; border-radius: 8px; margin: 5px 0;">
                    <div style="display: flex; align-items: center;">
                        <span style="font-size: 18px; font-weight: bold; color: {COLORS['danger']}; width: 40px;">#{i}{medal}</span>
                        <span style="flex: 1; color: {COLORS['text_primary']}; font-weight: bold;">{f['name']}</span>
                        <span style="color: {COLORS['danger']}; font-weight: bold;">{f['ko_wins']} KOs</span>
                    </div>
                    <div style="margin-top: 5px; background: {COLORS['background']}; border-radius: 4px; height: 6px;">
                        <div style="background: {COLORS['danger']}; width: {min(ko_pct, 100)}%; height: 100%; border-radius: 4px;"></div>
                    </div>
                    <div style="color: {COLORS['text_muted']}; font-size: 14px; margin-top: 3px;">{ko_pct:.1f}% of wins by KO</div>
                </div>
                """, unsafe_allow_html=True)

        with col2:
            st.markdown("### First Round KO Specialists")
            for i, f in enumerate(stats.get('first_round_ko_artists', [])[:10], 1):
                medal = "" if i == 1 else ("" if i == 2 else ("" if i == 3 else ""))
                st.markdown(f"""
                <div style="background: {COLORS['card_bg']}; padding: 12px; border-radius: 8px; margin: 5px 0;">
                    <div style="display: flex; align-items: center;">
                        <span style="font-size: 18px; font-weight: bold; color: {COLORS['warning']}; width: 40px;">#{i}{medal}</span>
                        <span style="flex: 1; color: {COLORS['text_primary']}; font-weight: bold;">{f['name']}</span>
                        <span style="color: {COLORS['warning']}; font-weight: bold;">{f['r1_kos']} R1 KOs</span>
                    </div>
                    <div style="color: {COLORS['text_muted']}; font-size: 14px; margin-top: 3px;">{f['r1_pct']:.1f}% of their KOs in Round 1</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Highest KO Win Percentage")
            st.caption("Fighters with 5+ wins")
            for i, f in enumerate(stats.get('highest_ko_pct', [])[:8], 1):
                st.markdown(f"""
                <div style="background: {COLORS['card_bg']}; padding: 10px; border-radius: 8px; margin: 4px 0; display: flex; align-items: center;">
                    <span style="font-weight: bold; color: {COLORS['danger']}; width: 30px;">#{i}</span>
                    <span style="flex: 1; color: {COLORS['text_primary']};">{f['name']}</span>
                    <span style="color: {COLORS['danger']}; font-weight: bold;">{f['ko_pct']:.1f}%</span>
                    <span style="color: {COLORS['text_muted']}; font-size: 14px; margin-left: 8px;">({f['ko_wins']}/{f['total_wins']})</span>
                </div>
                """, unsafe_allow_html=True)

        with col2:
            st.markdown("### Highest Strike Volume")
            st.caption("Significant strikes per minute")
            for i, f in enumerate(stats['volume_strikers'][:8], 1):
                st.markdown(f"""
                <div style="background: {COLORS['card_bg']}; padding: 10px; border-radius: 8px; margin: 4px 0; display: flex; align-items: center;">
                    <span style="font-weight: bold; color: {COLORS['primary']}; width: 30px;">#{i}</span>
                    <span style="flex: 1; color: {COLORS['text_primary']};">{f['name']}</span>
                    <span style="color: {COLORS['primary']}; font-weight: bold;">{f['sig_strikes_landed_per_min']:.2f}/min</span>
                </div>
                """, unsafe_allow_html=True)

    elif section == "Grappling Records":
        st.markdown("## Grappling Records")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Submissions", f"{stats['sub_count']:,}")
        with col2:
            st.metric("Submission Rate", f"{stats['sub_rate']:.1f}%")
        with col3:
            top_sub = stats.get('top_submissions', [{}])[0]
            st.metric("Most Common Sub", top_sub.get('type', 'RNC')[:15])
        with col4:
            top_sub_artist = stats['sub_artists'][0] if stats['sub_artists'] else {'sub_wins': 0}
            st.metric("Most Subs (Single)", f"{top_sub_artist.get('sub_wins', 0)}")

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Most Submission Wins")
            for i, f in enumerate(stats['sub_artists'][:10], 1):
                sub_pct = f['sub_wins'] / f['total_wins'] * 100 if f['total_wins'] > 0 else 0
                st.markdown(f"""
                <div style="background: {COLORS['card_bg']}; padding: 12px; border-radius: 8px; margin: 5px 0;">
                    <div style="display: flex; align-items: center;">
                        <span style="font-size: 18px; font-weight: bold; color: {COLORS['info']}; width: 40px;">#{i}</span>
                        <span style="flex: 1; color: {COLORS['text_primary']}; font-weight: bold;">{f['name']}</span>
                        <span style="color: {COLORS['info']}; font-weight: bold;">{f['sub_wins']} Subs</span>
                    </div>
                    <div style="margin-top: 5px; background: {COLORS['background']}; border-radius: 4px; height: 6px;">
                        <div style="background: {COLORS['info']}; width: {min(sub_pct, 100)}%; height: 100%; border-radius: 4px;"></div>
                    </div>
                    <div style="color: {COLORS['text_muted']}; font-size: 14px; margin-top: 3px;">{sub_pct:.1f}% of wins by submission</div>
                </div>
                """, unsafe_allow_html=True)

        with col2:
            st.markdown("### Takedown Specialists")
            st.caption("Takedowns per 15 minutes")
            for i, f in enumerate(stats['takedown_specialists'][:10], 1):
                st.markdown(f"""
                <div style="background: {COLORS['card_bg']}; padding: 10px; border-radius: 8px; margin: 4px 0; display: flex; align-items: center;">
                    <span style="font-weight: bold; color: {COLORS['success']}; width: 30px;">#{i}</span>
                    <span style="flex: 1; color: {COLORS['text_primary']};">{f['name']}</span>
                    <span style="color: {COLORS['success']}; font-weight: bold;">{f['takedowns_avg_per_15min']:.2f}</span>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # Submission types chart
        st.markdown("### Most Common Submissions")
        top_subs = stats.get('top_submissions', [])
        if top_subs:
            fig = go.Figure(data=[go.Bar(
                x=[s['count'] for s in top_subs],
                y=[s['type'][:20] for s in top_subs],
                orientation='h',
                marker_color=COLORS['info'],
                text=[s['count'] for s in top_subs],
                textposition='outside'
            )])
            fig.update_layout(
                paper_bgcolor=COLORS['background'],
                plot_bgcolor=COLORS['background'],
                font=dict(color=COLORS['text_primary']),
                height=300,
                xaxis_title="Count",
                yaxis=dict(autorange="reversed"),
                margin=dict(l=150)
            )
            st.plotly_chart(fig, use_container_width=True)

    else:  # Overall Records
        st.markdown("## Overall Records")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Fighters", f"{stats['total_fighters']:,}")
        with col2:
            st.metric("Total UFC Fights", f"{stats['total_fights']:,}")
        with col3:
            st.metric("Total Events", f"{stats['total_events']:,}")
        with col4:
            st.metric("Finish Rate", f"{stats['finish_rate']:.1f}%")

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Most UFC Wins")
            for i, f in enumerate(stats['most_wins'][:10], 1):
                win_pct = f['wins'] / (f['wins'] + f['losses']) * 100 if (f['wins'] + f['losses']) > 0 else 0
                st.markdown(f"""
                <div style="background: {COLORS['card_bg']}; padding: 12px; border-radius: 8px; margin: 5px 0;">
                    <div style="display: flex; align-items: center;">
                        <span style="font-weight: bold; color: {COLORS['success']}; width: 40px;">#{i}</span>
                        <span style="flex: 1; color: {COLORS['text_primary']}; font-weight: bold;">{f['name']}</span>
                        <span style="color: {COLORS['success']}; font-weight: bold;">{f['wins']}W</span>
                    </div>
                    <div style="color: {COLORS['text_muted']}; font-size: 14px; margin-top: 3px;">Record: {f['wins']}-{f['losses']} ({win_pct:.1f}% win rate)</div>
                </div>
                """, unsafe_allow_html=True)

        with col2:
            st.markdown("### Most Active Fighters")
            for i, f in enumerate(stats['most_active'][:10], 1):
                win_rate = f['wins'] / (f['wins'] + f['losses']) * 100 if (f['wins'] + f['losses']) > 0 else 0
                st.markdown(f"""
                <div style="background: {COLORS['card_bg']}; padding: 12px; border-radius: 8px; margin: 5px 0;">
                    <div style="display: flex; align-items: center;">
                        <span style="font-weight: bold; color: {COLORS['primary']}; width: 40px;">#{i}</span>
                        <span style="flex: 1; color: {COLORS['text_primary']}; font-weight: bold;">{f['name']}</span>
                        <span style="color: {COLORS['primary']}; font-weight: bold;">{f['ufc_fights']} fights</span>
                    </div>
                    <div style="color: {COLORS['text_muted']}; font-size: 14px; margin-top: 3px;">Record: {f['wins']}-{f['losses']} ({win_rate:.1f}% win rate)</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Current Win Streaks")
            for i, f in enumerate(stats['win_streaks'][:10], 1):
                streak_color = COLORS['success'] if f['streak'] >= 5 else COLORS['info']
                st.markdown(f"""
                <div style="background: {COLORS['card_bg']}; padding: 10px; border-radius: 8px; margin: 3px 0; display: flex; align-items: center;">
                    <span style="color: {streak_color}; font-weight: bold; width: 30px;">#{i}</span>
                    <span style="flex: 1; color: {COLORS['text_primary']}; font-weight: bold;">{f['name']}</span>
                    <span style="color: {streak_color}; font-weight: bold;">{f['streak']}W</span>
                </div>
                """, unsafe_allow_html=True)

        with col2:
            st.markdown("### Most Exciting Fighters")
            st.caption("Highest % of fights ending in finishes")
            exciting = stats.get('most_exciting', [])
            if exciting:
                for i, f in enumerate(exciting[:10], 1):
                    st.markdown(f"""
                    <div style="background: {COLORS['card_bg']}; padding: 10px; border-radius: 8px; margin: 4px 0;">
                        <div style="display: flex; align-items: center;">
                            <span style="font-weight: bold; color: {COLORS['warning']}; width: 35px;">#{i}</span>
                            <span style="flex: 1; color: {COLORS['text_primary']};">{f['name']}</span>
                            <span style="color: {COLORS['warning']}; font-weight: bold;">{f['excitement_score']:.1f}%</span>
                        </div>
                        <div style="color: {COLORS['text_muted']}; font-size: 14px; margin-top: 3px;">
                            {f['total_fights']} fights | {f['finishes_for']} finishes won, {f['finishes_against']} finished
                        </div>
                    </div>
                    """, unsafe_allow_html=True)


# ============================================================================
# TAB 4: FIGHTER STYLES (ML CLUSTERING)
# ============================================================================
with tabs[3]:
    st.markdown("### Fighter Style Clustering")
    st.markdown("*Machine learning identifies distinct fighting archetypes*")

    clustering = sections.get('fighter_clustering', {})

    if 'error' in clustering:
        st.error(f"Analysis error: {clustering['error']}")
    elif clustering:
        kmeans_data = clustering.get('kmeans', {})

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Optimal Clusters", kmeans_data.get('optimal_clusters', 'N/A'))
        with col2:
            st.metric("Silhouette Score", f"{kmeans_data.get('silhouette_score', 0):.3f}")
        with col3:
            st.metric("Fighters Analyzed", len(kmeans_data.get('fighters', [])))

        st.markdown("#### Fighting Archetypes")
        clusters = kmeans_data.get('clusters', [])

        for cluster in clusters:
            with st.expander(f"**{cluster['style_label']}** ({cluster['size']} fighters)", expanded=False):
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown(f"*{cluster['description']}*")
                    st.markdown(f"**Win Rate:** {cluster['avg_win_rate']*100:.1f}%")
                    st.markdown("**Top Fighters:** " + ", ".join(cluster['top_fighters'][:5]))
                with col2:
                    categories = ['Strikes', 'Takedowns', 'Submissions', 'Str Acc', 'Defense']
                    values = [
                        min(cluster['avg_sig_strikes'] / 6 * 100, 100),
                        min(cluster['avg_takedowns'] / 4 * 100, 100),
                        min(cluster['avg_submissions'] / 2 * 100, 100),
                        cluster['avg_strike_accuracy'] * 100,
                        cluster['avg_strike_defense'] * 100,
                    ]
                    fig = go.Figure(data=go.Scatterpolar(
                        r=values + [values[0]], theta=categories + [categories[0]],
                        fill='toself', fillcolor='rgba(210, 10, 10, 0.3)',
                        line=dict(color=COLORS['primary'])
                    ))
                    fig.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                        showlegend=False, height=200, margin=dict(l=30, r=30, t=20, b=20),
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color=COLORS['text_primary'], size=10)
                    )
                    st.plotly_chart(fig, use_container_width=True)

        # DBSCAN unique fighters
        dbscan = clustering.get('dbscan', {})
        if dbscan and 'unique_style_fighters' in dbscan:
            st.markdown("#### Unique Style Fighters (Statistical Outliers)")
            st.info(f"DBSCAN identified {dbscan.get('n_outliers', 0)} fighters ({dbscan.get('outlier_percentage', 0):.1f}%) with truly unique styles")

        if 'insights' in kmeans_data:
            st.markdown("#### Key Insights")
            for insight in kmeans_data['insights']:
                st.success(insight)
    else:
        st.info("Fighter style clustering data not available. Click 'Regenerate Analysis' to generate.")


# ============================================================================
# TAB 5: AGE & PHYSICAL ATTRIBUTES (MERGED)
# ============================================================================
with tabs[4]:
    st.markdown("### Age & Physical Attributes")
    st.markdown("*How age and physical characteristics affect fight outcomes*")

    subtab = st.radio("Select Analysis", ["Age & Performance", "Physical Attributes"], horizontal=True)

    st.markdown("---")

    if subtab == "Age & Performance":
        age_data = sections.get('age_analysis', {})

        if 'error' in age_data:
            st.error(f"Analysis error: {age_data['error']}")
        elif age_data:
            perf = age_data.get('performance', {})

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Average Fighter Age", f"{perf.get('avg_fighter_age', 0):.1f} years")
            with col2:
                st.metric("Peak Age Bracket", perf.get('peak_age_bracket', 'Unknown'))
            with col3:
                st.metric("Peak Win Rate", f"{perf.get('peak_win_rate', 0)*100:.1f}%")
            with col4:
                corr = perf.get('age_win_correlation', 0)
                st.metric("Age-Performance Correlation", f"{corr:.3f}")

            # Age bracket stats
            st.markdown("#### Performance by Age Bracket")
            age_stats = perf.get('age_bracket_stats', [])
            if age_stats:
                df_age = pd.DataFrame(age_stats)

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=df_age['bracket'], y=df_age['avg_win_rate'].apply(lambda x: x * 100),
                    name='Win Rate %', marker_color=COLORS['primary']
                ))
                fig.update_layout(
                    yaxis_title='Win Rate (%)', height=350,
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color=COLORS['text_primary'])
                )
                st.plotly_chart(fig, use_container_width=True)

            # Age by division
            div_age = age_data.get('by_division', {})
            if div_age and 'division_age_stats' in div_age:
                st.markdown("#### Age Advantage by Division")
                st.markdown(f"**Youth-favoring divisions:** {', '.join(div_age.get('youth_favoring_divisions', [])[:3]) or 'None identified'}")
                st.markdown(f"**Experience-favoring divisions:** {', '.join(div_age.get('experience_favoring_divisions', [])[:3]) or 'None identified'}")

            # Insights
            for insight in perf.get('insights', []):
                st.info(insight)
        else:
            st.info("Age analysis data not available. Click 'Regenerate Analysis' to generate.")

    else:  # Physical Attributes
        phys_data = sections.get('physical_analysis', {})

        if 'error' in phys_data:
            st.error(f"Analysis error: {phys_data['error']}")
        elif phys_data:
            hr_data = phys_data.get('height_reach', {})

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Taller Fighter Win Rate", f"{hr_data.get('taller_fighter_win_rate', 50):.1f}%")
            with col2:
                st.metric("Longer Reach Win Rate", f"{hr_data.get('longer_reach_win_rate', 50):.1f}%")
            with col3:
                st.metric("Fights Analyzed", hr_data.get('total_fights_analyzed', 0))

            st.markdown("#### Reach Advantage Impact")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Reach Advantage in Different Outcomes:**")
                st.markdown(f"- KO/TKO wins: {hr_data.get('reach_advantage_in_kos', 50):.1f}%")
                st.markdown(f"- Submission wins: {hr_data.get('reach_advantage_in_subs', 50):.1f}%")

                if hr_data.get('reach_advantage_in_kos', 50) > hr_data.get('reach_advantage_in_subs', 50):
                    st.success("Reach advantage is MORE beneficial for striking/KOs than grappling")
                else:
                    st.info("Reach advantage is LESS beneficial for striking than grappling")

            with col2:
                bracket_stats = hr_data.get('reach_bracket_stats', [])
                if bracket_stats:
                    df_reach = pd.DataFrame(bracket_stats)
                    fig = go.Figure(go.Bar(
                        x=df_reach['bracket'], y=df_reach['win_rate'],
                        marker_color=COLORS['primary']
                    ))
                    fig.update_layout(
                        title='Win Rate by Reach Advantage Size',
                        yaxis_title='Win Rate (%)', height=300,
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color=COLORS['text_primary'])
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # Physical archetypes
            clusters = phys_data.get('physical_clusters', {})
            if clusters and 'physical_archetypes' in clusters:
                st.markdown("#### Physical Archetypes")
                st.info(f"Best performing body type: **{clusters.get('best_archetype', 'Unknown')}** with {clusters.get('best_archetype_win_rate', 0)*100:.1f}% win rate")

                for archetype in clusters.get('physical_archetypes', []):
                    with st.expander(f"{archetype['label']} ({archetype['size']} fighters)"):
                        st.markdown(f"*{archetype['description']}*")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Avg Height", f"{archetype['avg_height_cm']:.0f} cm")
                        col2.metric("Avg Reach", f"{archetype['avg_reach_cm']:.0f} cm")
                        col3.metric("Win Rate", f"{archetype['avg_win_rate']*100:.1f}%")
                        st.markdown("**Examples:** " + ", ".join(archetype['top_fighters'][:5]))

            for insight in hr_data.get('insights', []):
                st.info(insight)
        else:
            st.info("Physical analysis data not available. Click 'Regenerate Analysis' to generate.")


# ============================================================================
# TAB 6: CAREER & TITLES (MERGED)
# ============================================================================
with tabs[5]:
    st.markdown("### Career Statistics & Title Fights")
    st.markdown("*Understanding career paths and championship patterns*")

    subtab = st.radio("Select Analysis", ["Career Statistics", "Title Fights"], horizontal=True, key="career_subtab")

    st.markdown("---")

    if subtab == "Career Statistics":
        career_data = sections.get('career_analysis', {})

        if 'error' in career_data:
            st.error(f"Analysis error: {career_data['error']}")
        elif career_data:
            career_stats = career_data.get('statistics', {})
            trajectories = career_data.get('trajectories', {})
            wc_careers = career_data.get('weight_class_careers', {})

            st.markdown("#### Average UFC Career")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Career Duration", f"{career_stats.get('avg_career_duration_years', 0):.1f} years")
            with col2:
                st.metric("UFC Fights", f"{career_stats.get('avg_ufc_fights', 0):.1f}")
            with col3:
                st.metric("Fighter Age", f"{career_stats.get('avg_fighter_age', 0):.1f} years")
            with col4:
                st.metric("Active Fighters", career_stats.get('active_fighters_count', 0))

            # Career archetypes
            if trajectories and 'career_archetypes' in trajectories:
                st.markdown("#### Career Archetypes")
                for archetype in trajectories['career_archetypes']:
                    st.markdown(f"""
                    <div style="background: {COLORS['card_bg']}; padding: 15px; border-radius: 10px; margin-bottom: 12px;">
                        <div style="display: flex; justify-content: space-between;">
                            <h4 style="color: {COLORS['primary']}; margin: 0;">{archetype['label']}</h4>
                            <span style="color: {COLORS['text_muted']};">({archetype['size']} fighters)</span>
                        </div>
                        <p style="color: {COLORS['text_secondary']}; margin: 8px 0;">{archetype['description']}</p>
                        <div style="display: flex; gap: 25px; color: {COLORS['text_muted']}; font-size: 13px;">
                            <span>Avg Wins: {archetype['avg_wins']:.1f}</span>
                            <span>Win Rate: {archetype['avg_win_rate']*100:.1f}%</span>
                            <span>Avg Fights: {archetype['avg_fights']:.1f}</span>
                        </div>
                        <p style="color: {COLORS['text_muted']}; margin: 5px 0 0 0; font-size: 12px;"><strong>Examples:</strong> {', '.join(archetype['top_fighters'][:5])}</p>
                    </div>
                    """, unsafe_allow_html=True)

            # Career by weight class
            if wc_careers and 'division_career_stats' in wc_careers:
                st.markdown("#### Career Length by Division")
                st.markdown(f"**Longest careers:** {', '.join(wc_careers.get('longest_career_divisions', [])[:3])}")
                st.markdown(f"**Shortest careers:** {', '.join(wc_careers.get('shortest_career_divisions', [])[:3])}")

                div_stats_career = wc_careers.get('division_career_stats', [])[:10]
                if div_stats_career:
                    df_div = pd.DataFrame(div_stats_career)
                    fig = go.Figure(go.Bar(
                        x=df_div['weight_class'], y=df_div['avg_career_years'],
                        marker_color=COLORS['primary']
                    ))
                    fig.update_layout(
                        xaxis_tickangle=-45, yaxis_title='Avg Career (Years)', height=350,
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color=COLORS['text_primary'])
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # Record holders
            if career_stats.get('longest_careers'):
                st.markdown("#### Record Holders")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Longest Careers**")
                    for f in career_stats.get('longest_careers', [])[:5]:
                        st.markdown(f"- {f['name']}: {f['career_span_years']:.1f} years ({f['ufc_fights']} fights)")
                with col2:
                    st.markdown("**Most UFC Fights**")
                    for f in career_stats.get('most_fights', [])[:5]:
                        st.markdown(f"- {f['name']}: {f['ufc_fights']} fights ({f['wins']}-{f['losses']})")

            for insight in career_stats.get('insights', []):
                st.info(insight)
        else:
            st.info("Career analysis data not available. Click 'Regenerate Analysis' to generate.")

    else:  # Title Fights
        champ_data = sections.get('champion_analysis', {})

        if 'error' in champ_data:
            st.error(f"Analysis error: {champ_data['error']}")
        elif champ_data:
            patterns = champ_data.get('patterns', {})
            path = champ_data.get('path_to_title', {})

            st.markdown("#### Title Fight Statistics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Title Fights", patterns.get('total_title_fights', 0))
            with col2:
                st.metric("Avg Duration", f"{patterns.get('avg_title_fight_rounds', 0):.1f} rounds")
            with col3:
                st.metric("Champion Defense Rate", f"{patterns.get('champion_defense_rate', 0):.1f}%")
            with col4:
                st.metric("Challenger Win Rate", f"{patterns.get('challenger_win_rate', 0):.1f}%")

            st.markdown("#### How Title Fights End")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("By KO/TKO", f"{patterns.get('title_ko_rate', 0):.1f}%")
            with col2:
                st.metric("By Submission", f"{patterns.get('title_sub_rate', 0):.1f}%")
            with col3:
                st.metric("By Decision", f"{patterns.get('title_dec_rate', 0):.1f}%")

            st.markdown(f"**Average title fight length:** {patterns.get('avg_title_fight_minutes', 0):.1f} minutes")

            # Path to title
            if path and 'error' not in path:
                st.markdown("#### Path to a Title Shot")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("% of Fighters Who Get Title Shot",
                             f"{path.get('pct_got_title_shot', 0):.1f}%")
                    st.metric("Avg Fights Before Title",
                             f"{path.get('avg_fights_before_title', 0):.1f}")
                with col2:
                    st.metric("Total Fighters with Title Shots",
                             path.get('total_fighters_with_title_shots', 0))
                    st.metric("Avg Years to Title",
                             f"{path.get('avg_years_to_title', 0):.1f}")

            for insight in patterns.get('insights', []):
                st.info(insight)
            for insight in path.get('insights', []) if path and 'insights' in path else []:
                st.info(insight)
        else:
            st.info("Title fight analysis data not available. Click 'Regenerate Analysis' to generate.")


# ============================================================================
# TAB 7: STYLE EVOLUTION
# ============================================================================
with tabs[6]:
    st.markdown("### Fighting Style Evolution")
    st.markdown("*How UFC fighting has changed over the years*")

    temporal = sections.get('temporal_patterns', {})

    if 'error' in temporal:
        st.error(f"Analysis error: {temporal['error']}")
    elif temporal:
        yearly = temporal.get('yearly', {})
        evolution = temporal.get('style_evolution', {})

        # Era analysis
        st.markdown("#### UFC Eras")
        era_stats = evolution.get('era_stats', []) if evolution else []

        if era_stats:
            cols = st.columns(len(era_stats))
            for i, era in enumerate(era_stats):
                with cols[i]:
                    st.markdown(f"""
                    <div style="background: {COLORS['card_bg']}; padding: 15px; border-radius: 10px; text-align: center;">
                        <h5 style="color: {COLORS['primary']}; margin: 0 0 10px 0;">{era['era']}</h5>
                        <p style="color: {COLORS['text_muted']}; font-size: 12px; margin: 0;">
                            {era['total_fights']:,} fights<br>
                            KO: {era['ko_rate']:.1f}% | Sub: {era['sub_rate']:.1f}%
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

        # Trends
        if evolution:
            st.markdown("#### Long-term Trends")
            col1, col2 = st.columns(2)
            with col1:
                trend = evolution.get('striking_volume_trend', 'stable')
                st.metric("Striking Volume", trend.capitalize(),
                         delta="Higher output" if trend == 'increasing' else "Lower output")
            with col2:
                trend = evolution.get('wrestling_trend', 'stable')
                st.metric("Wrestling/Takedowns", trend.capitalize(),
                         delta="More grappling" if trend == 'increasing' else "Less grappling")

        # Yearly trends chart
        yearly_data = yearly.get('yearly_data', [])
        if yearly_data:
            st.markdown("#### Historical Outcome Trends")
            df_yearly = pd.DataFrame(yearly_data)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_yearly['year'], y=df_yearly['ko_percentage'],
                                     name='KO/TKO %', line=dict(color=COLORS['danger'])))
            fig.add_trace(go.Scatter(x=df_yearly['year'], y=df_yearly['sub_percentage'],
                                     name='Submission %', line=dict(color=COLORS['info'])))
            fig.add_trace(go.Scatter(x=df_yearly['year'], y=df_yearly['dec_percentage'],
                                     name='Decision %', line=dict(color=COLORS['warning'])))
            fig.update_layout(
                xaxis_title='Year', yaxis_title='Percentage', height=400,
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color=COLORS['text_primary']), legend=dict(orientation='h', y=1.1)
            )
            st.plotly_chart(fig, use_container_width=True)

        for insight in evolution.get('insights', []) if evolution else []:
            st.info(insight)
    else:
        st.info("Style evolution data not available. Click 'Regenerate Analysis' to generate.")


# ============================================================================
# TAB 8: SKILL ANALYSIS
# ============================================================================
with tabs[7]:
    st.markdown("### Skill Analysis")
    st.markdown("*In-depth analysis of striking, grappling, and overall skill trends*")

    skill_section = st.radio(
        "Select Analysis",
        ["Striking Analysis", "Grappling Analysis", "Overall Trends"],
        horizontal=True,
        key="skill_analysis_radio"
    )

    st.markdown("---")

    if skill_section == "Striking Analysis":
        st.markdown("## Striking Analysis")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total KO/TKOs", f"{stats['ko_count']:,}")
        with col2:
            st.metric("KO Rate", f"{stats['ko_rate']:.1f}%")
        with col3:
            r1_kos = stats['ko_by_round'].get(1, 0) if stats['ko_by_round'] else 0
            total_kos = sum(stats['ko_by_round'].values()) if stats['ko_by_round'] else 1
            st.metric("Round 1 KOs", f"{r1_kos/total_kos*100:.1f}%")
        with col4:
            top_ko = stats['ko_artists'][0] if stats['ko_artists'] else {'ko_wins': 0}
            st.metric("Most KOs (Fighter)", f"{top_ko.get('ko_wins', 0)}")

        st.markdown("---")

        # Striker-heavy divisions
        st.markdown("### Most Striker-Heavy Divisions")
        striker_divs = sorted(stats['division_stats'].items(), key=lambda x: x[1]['ko_rate'], reverse=True)[:5]

        div_names = [d[0] for d in striker_divs]
        ko_rates = [d[1]['ko_rate'] for d in striker_divs]

        fig = go.Figure(data=[go.Bar(
            x=ko_rates,
            y=div_names,
            orientation='h',
            marker_color=[COLORS['danger'] if r > 40 else COLORS['warning'] for r in ko_rates],
            text=[f"{r:.1f}%" for r in ko_rates],
            textposition='outside'
        )])
        fig.update_layout(
            title="KO Rate by Division",
            paper_bgcolor=COLORS['background'],
            plot_bgcolor=COLORS['background'],
            font=dict(color=COLORS['text_primary']),
            height=300,
            xaxis_title="KO Rate (%)",
            yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(fig, use_container_width=True)

    elif skill_section == "Grappling Analysis":
        st.markdown("## Grappling Analysis")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Submissions", f"{stats['sub_count']:,}")
        with col2:
            st.metric("Submission Rate", f"{stats['sub_rate']:.1f}%")
        with col3:
            top_sub_type = stats.get('top_submissions', [{}])[0].get('type', 'RNC')[:12]
            st.metric("Most Common", top_sub_type)
        with col4:
            top_sub_artist = stats['sub_artists'][0] if stats['sub_artists'] else {'sub_wins': 0}
            st.metric("Most Subs (Fighter)", f"{top_sub_artist.get('sub_wins', 0)}")

        st.markdown("---")

        # Grappling-Heavy Divisions
        st.markdown("### Grappling by Division")
        grappler_divs = sorted(stats['division_stats'].items(), key=lambda x: x[1]['sub_rate'], reverse=True)[:6]

        div_names = [d[0] for d in grappler_divs]
        sub_rates = [d[1]['sub_rate'] for d in grappler_divs]

        fig = go.Figure(data=[go.Bar(
            x=sub_rates,
            y=div_names,
            orientation='h',
            marker_color=[COLORS['info'] if r > 15 else COLORS['success'] for r in sub_rates],
            text=[f"{r:.1f}%" for r in sub_rates],
            textposition='outside'
        )])
        fig.update_layout(
            title="Submission Rate by Division",
            paper_bgcolor=COLORS['background'],
            plot_bgcolor=COLORS['background'],
            font=dict(color=COLORS['text_primary']),
            height=300,
            xaxis_title="Submission Rate (%)",
            yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(fig, use_container_width=True)

    else:  # Overall Trends
        st.markdown("## Overall Skill Trends")

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total KO/TKOs", f"{stats['ko_count']:,}")
        with col2:
            st.metric("Total Submissions", f"{stats['sub_count']:,}")
        with col3:
            st.metric("Total Decisions", f"{stats['dec_count']:,}")
        with col4:
            st.metric("Finish Rate", f"{stats['finish_rate']:.1f}%")
        with col5:
            ratio = stats['ko_count'] / stats['sub_count'] if stats['sub_count'] > 0 else 0
            st.metric("KO:Sub Ratio", f"{ratio:.2f}:1")

        st.markdown("---")

        # Striking vs Grappling comparison
        st.markdown("### Striking vs Grappling: The Eternal Debate")

        col1, col2 = st.columns(2)

        with col1:
            fig = go.Figure(data=[go.Pie(
                labels=['KO/TKO', 'Submission', 'Decision', 'Other'],
                values=[stats['ko_count'], stats['sub_count'], stats['dec_count'],
                        stats['total_fights'] - stats['ko_count'] - stats['sub_count'] - stats['dec_count']],
                hole=0.5,
                marker_colors=[COLORS['danger'], COLORS['info'], COLORS['success'], COLORS['text_muted']],
                textinfo='percent+label'
            )])
            fig.update_layout(
                title="UFC Fight Outcome Distribution",
                paper_bgcolor=COLORS['background'],
                font=dict(color=COLORS['text_primary']),
                height=350,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown(f"""
            <div style="background: {COLORS['card_bg']}; padding: 25px; border-radius: 10px; height: 320px;">
                <h4 style="color: {COLORS['primary']}; margin-top: 0;">Key Findings</h4>
                <ul style="color: {COLORS['text_primary']}; line-height: 1.8;">
                    <li><b>Striking dominates:</b> KO/TKOs ({stats['ko_rate']:.1f}%) are {ratio:.1f}x more common than submissions ({stats['sub_rate']:.1f}%)</li>
                    <li><b>Weight class matters:</b> Heavyweight has ~50% KO rate vs Flyweight at ~24%</li>
                    <li><b>Round 1 danger zone:</b> 48% of all KOs happen in the first 5 minutes</li>
                    <li><b>Technical evolution:</b> Decision rate ({stats['dec_rate']:.1f}%) shows improved defense skills</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)


# ============================================================================
# TAB 9: ASK AI
# ============================================================================
with tabs[8]:
    st.markdown("### Ask AI About UFC Stats")

    if llm_service.is_available():
        question = st.text_area(
            "Your Question",
            placeholder="e.g., Why does Heavyweight have the highest KO rate?",
            key="trends_question",
        )

        if st.button("Get Answer", type="primary"):
            if question:
                with st.spinner("Analyzing..."):
                    context = {
                        "total_fights": stats['total_fights'],
                        "ko_rate": f"{stats['ko_rate']:.1f}%",
                        "sub_rate": f"{stats['sub_rate']:.1f}%",
                        "dec_rate": f"{stats['dec_rate']:.1f}%",
                        "top_ko_division": stats['highest_ko_rate_division'][0],
                        "top_ko_artist": stats['ko_artists'][0]['name'] if stats['ko_artists'] else "N/A",
                        "top_sub_artist": stats['sub_artists'][0]['name'] if stats['sub_artists'] else "N/A",
                    }
                    answer = llm_service.answer_trends_question(question, context)
                    if answer:
                        st.markdown("---")
                        st.markdown("### Answer")
                        st.markdown(answer)

        st.markdown("---")
        st.markdown("#### Sample Questions")
        samples = [
            "Why does Heavyweight have the highest KO rate?",
            "Who is the best submission artist in UFC history?",
            "Which division is most technical?",
        ]
        for q in samples:
            if st.button(q, key=f"sample_{hash(q)}"):
                st.session_state.trends_question = q
                st.rerun()
    else:
        st.warning("AI Q&A unavailable. Add GROQ_API_KEY to .env file.")


# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown(
    f"""
    <div style="text-align: center; color: {COLORS['text_muted']}; font-size: 12px;">
        <p>Analysis based on {stats['total_fights']:,} UFC fights | ML powered by KMeans, DBSCAN, PCA, t-SNE, and statistical analysis</p>
    </div>
    """,
    unsafe_allow_html=True,
)