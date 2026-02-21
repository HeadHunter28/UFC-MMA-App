"""
Comparison Charts Component.

Visualizations for comparing two fighters.
"""

import streamlit as st
from typing import Any, Dict, List

import plotly.graph_objects as go
import plotly.express as px

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import COLORS


def render_comparison_charts(
    fighter_a: Dict[str, Any],
    fighter_b: Dict[str, Any]
):
    """
    Render complete comparison charts for two fighters.

    Args:
        fighter_a: Fighter A data with stats
        fighter_b: Fighter B data with stats
    """
    st.markdown("### Statistical Comparison")

    # Tabs for different comparisons
    tab1, tab2, tab3 = st.tabs(["Overview", "Striking", "Grappling"])

    with tab1:
        render_overview_comparison(fighter_a, fighter_b)

    with tab2:
        render_striking_comparison(fighter_a, fighter_b)

    with tab3:
        render_grappling_comparison(fighter_a, fighter_b)


def render_overview_comparison(
    fighter_a: Dict[str, Any],
    fighter_b: Dict[str, Any]
):
    """Render overview comparison with radar chart."""
    # Create radar chart data
    categories = [
        "Experience",
        "Striking",
        "Defense",
        "Grappling",
        "Finishing",
    ]

    # Normalize stats to 0-100 scale
    def get_normalized_stats(fighter):
        total_fights = (fighter.get("wins", 0) + fighter.get("losses", 0)) or 1
        return [
            min(total_fights / 50 * 100, 100),  # Experience (max 50 fights)
            (fighter.get("sig_strike_accuracy") or 0.45) * 100,  # Striking
            (fighter.get("sig_strike_defense") or 0.55) * 100,  # Defense
            (fighter.get("takedown_accuracy") or 0.40) * 100,  # Grappling
            (fighter.get("finish_rate") or 0.50) * 100,  # Finishing
        ]

    stats_a = get_normalized_stats(fighter_a)
    stats_b = get_normalized_stats(fighter_b)

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=stats_a,
        theta=categories,
        fill="toself",
        name=fighter_a.get("name", "Fighter A"),
        line_color=COLORS["primary"],
    ))

    fig.add_trace(go.Scatterpolar(
        r=stats_b,
        theta=categories,
        fill="toself",
        name=fighter_b.get("name", "Fighter B"),
        line_color=COLORS["info"],
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
            )
        ),
        showlegend=True,
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["background"],
        font=dict(color=COLORS["text_primary"]),
        height=400,
    )

    st.plotly_chart(fig, width='stretch')

    # Physical comparison
    render_physical_comparison(fighter_a, fighter_b)


def render_physical_comparison(
    fighter_a: Dict[str, Any],
    fighter_b: Dict[str, Any]
):
    """Render physical attributes comparison as a vertical table."""
    st.markdown("#### Physical Comparison")

    name_a = fighter_a.get("name", "Fighter A")
    name_b = fighter_b.get("name", "Fighter B")

    # Get physical stats
    height_a = fighter_a.get("height_cm") or 0
    height_b = fighter_b.get("height_cm") or 0
    reach_a = fighter_a.get("reach_cm") or 0
    reach_b = fighter_b.get("reach_cm") or 0
    age_a = fighter_a.get("age") or 0
    age_b = fighter_b.get("age") or 0
    stance_a = fighter_a.get("stance") or "N/A"
    stance_b = fighter_b.get("stance") or "N/A"
    wins_a = fighter_a.get("wins", 0)
    losses_a = fighter_a.get("losses", 0)
    wins_b = fighter_b.get("wins", 0)
    losses_b = fighter_b.get("losses", 0)

    # Helper to format with advantage indicator
    def format_val(val_a, val_b, unit="", higher_better=True):
        if val_a == val_b or val_a == 0 or val_b == 0:
            return f"{val_a}{unit}", f"{val_b}{unit}", ""
        if higher_better:
            if val_a > val_b:
                return f"**{val_a}{unit}** ✓", f"{val_b}{unit}", "red"
            else:
                return f"{val_a}{unit}", f"**{val_b}{unit}** ✓", "blue"
        else:  # Lower is better (e.g., age)
            if val_a < val_b:
                return f"**{val_a}{unit}** ✓", f"{val_b}{unit}", "red"
            else:
                return f"{val_a}{unit}", f"**{val_b}{unit}** ✓", "blue"

    # Build table data
    height_a_fmt, height_b_fmt, _ = format_val(height_a, height_b, " cm")
    reach_a_fmt, reach_b_fmt, _ = format_val(reach_a, reach_b, " cm")
    age_a_fmt, age_b_fmt, _ = format_val(age_a, age_b, " yrs", higher_better=False)
    record_a = f"{wins_a}-{losses_a}"
    record_b = f"{wins_b}-{losses_b}"

    # Create HTML table
    table_html = f"""
    <style>
        .physical-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
        }}
        .physical-table th, .physical-table td {{
            padding: 12px 15px;
            text-align: center;
            border-bottom: 1px solid {COLORS['text_muted']};
        }}
        .physical-table th {{
            background-color: {COLORS['card_bg']};
            color: {COLORS['text_primary']};
            font-weight: bold;
        }}
        .physical-table td {{
            color: {COLORS['text_primary']};
        }}
        .physical-table tr:hover {{
            background-color: {COLORS['card_bg']};
        }}
        .physical-table .label {{
            color: {COLORS['text_secondary']};
            font-weight: 500;
        }}
        .red-corner {{
            color: {COLORS['primary']} !important;
        }}
        .blue-corner {{
            color: {COLORS['info']} !important;
        }}
    </style>
    <table class="physical-table">
        <thead>
            <tr>
                <th class="red-corner">{name_a}</th>
                <th class="label">Attribute</th>
                <th class="blue-corner">{name_b}</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>{height_a_fmt}</td>
                <td class="label">Height</td>
                <td>{height_b_fmt}</td>
            </tr>
            <tr>
                <td>{reach_a_fmt}</td>
                <td class="label">Reach</td>
                <td>{reach_b_fmt}</td>
            </tr>
            <tr>
                <td>{age_a_fmt}</td>
                <td class="label">Age</td>
                <td>{age_b_fmt}</td>
            </tr>
            <tr>
                <td>{stance_a}</td>
                <td class="label">Stance</td>
                <td>{stance_b}</td>
            </tr>
            <tr>
                <td>{record_a}</td>
                <td class="label">Record</td>
                <td>{record_b}</td>
            </tr>
        </tbody>
    </table>
    """

    st.markdown(table_html, unsafe_allow_html=True)


def render_striking_comparison(
    fighter_a: Dict[str, Any],
    fighter_b: Dict[str, Any]
):
    """Render striking statistics comparison."""
    metrics = [
        ("Sig. Strikes Landed/Min", "sig_strikes_landed_per_min", 0, 10),
        ("Striking Accuracy", "sig_strike_accuracy", 0, 1),
        ("Strikes Absorbed/Min", "sig_strikes_absorbed_per_min", 0, 10),
        ("Strike Defense", "sig_strike_defense", 0, 1),
    ]

    render_stats_comparison(fighter_a, fighter_b, metrics, "Striking Stats")


def render_grappling_comparison(
    fighter_a: Dict[str, Any],
    fighter_b: Dict[str, Any]
):
    """Render grappling statistics comparison."""
    metrics = [
        ("Takedowns/15min", "takedowns_avg_per_15min", 0, 10),
        ("Takedown Accuracy", "takedown_accuracy", 0, 1),
        ("Takedown Defense", "takedown_defense", 0, 1),
        ("Submissions/15min", "submissions_avg_per_15min", 0, 5),
    ]

    render_stats_comparison(fighter_a, fighter_b, metrics, "Grappling Stats")


def render_stats_comparison(
    fighter_a: Dict[str, Any],
    fighter_b: Dict[str, Any],
    metrics: List[tuple],
    title: str = "Stats Comparison"
):
    """
    Render horizontal bar chart comparison.

    Args:
        fighter_a: Fighter A data
        fighter_b: Fighter B data
        metrics: List of (label, key, min_val, max_val)
        title: Chart title
    """
    labels = []
    values_a = []
    values_b = []

    for label, key, min_val, max_val in metrics:
        labels.append(label)

        val_a = fighter_a.get(key) or 0
        val_b = fighter_b.get(key) or 0

        # Normalize to 0-100
        range_val = max_val - min_val
        if range_val > 0:
            norm_a = (val_a - min_val) / range_val * 100
            norm_b = (val_b - min_val) / range_val * 100
        else:
            norm_a = val_a * 100
            norm_b = val_b * 100

        values_a.append(norm_a)
        values_b.append(-norm_b)  # Negative for left side

    fig = go.Figure()

    # Fighter A bars (right side)
    fig.add_trace(go.Bar(
        y=labels,
        x=values_a,
        orientation="h",
        name=fighter_a.get("name", "Fighter A"),
        marker_color=COLORS["primary"],
        text=[f"{v:.1f}" for v in values_a],
        textposition="outside",
    ))

    # Fighter B bars (left side)
    fig.add_trace(go.Bar(
        y=labels,
        x=values_b,
        orientation="h",
        name=fighter_b.get("name", "Fighter B"),
        marker_color=COLORS["info"],
        text=[f"{abs(v):.1f}" for v in values_b],
        textposition="outside",
    ))

    fig.update_layout(
        title=title,
        barmode="relative",
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["background"],
        font=dict(color=COLORS["text_primary"]),
        height=300,
        xaxis=dict(
            showticklabels=False,
            zeroline=True,
            zerolinecolor=COLORS["text_muted"],
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
    )

    st.plotly_chart(fig, width='stretch')


def render_edge_summary(
    fighter_a: Dict[str, Any],
    fighter_b: Dict[str, Any]
):
    """
    Render a summary of who has the edge in different areas.

    Args:
        fighter_a: Fighter A data
        fighter_b: Fighter B data
    """
    st.markdown("### Edge Summary")

    comparisons = [
        ("Height", fighter_a.get("height_cm") or 0, fighter_b.get("height_cm") or 0),
        ("Reach", fighter_a.get("reach_cm") or 0, fighter_b.get("reach_cm") or 0),
        ("Experience",
         (fighter_a.get("wins", 0) + fighter_a.get("losses", 0)),
         (fighter_b.get("wins", 0) + fighter_b.get("losses", 0))),
        ("Striking",
         fighter_a.get("sig_strike_accuracy") or 0,
         fighter_b.get("sig_strike_accuracy") or 0),
        ("Defense",
         fighter_a.get("sig_strike_defense") or 0,
         fighter_b.get("sig_strike_defense") or 0),
        ("Takedowns",
         fighter_a.get("takedown_accuracy") or 0,
         fighter_b.get("takedown_accuracy") or 0),
    ]

    edges_a = 0
    edges_b = 0

    for label, val_a, val_b in comparisons:
        col1, col2, col3 = st.columns([2, 3, 2])

        with col1:
            if val_a > val_b:
                st.markdown(f"✅ **{fighter_a.get('name', 'A')[:15]}**")
                edges_a += 1
            else:
                st.markdown(f"{fighter_a.get('name', 'A')[:15]}")

        with col2:
            st.markdown(f"<center>{label}</center>", unsafe_allow_html=True)

        with col3:
            if val_b > val_a:
                st.markdown(f"✅ **{fighter_b.get('name', 'B')[:15]}**")
                edges_b += 1
            else:
                st.markdown(f"{fighter_b.get('name', 'B')[:15]}")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(f"{fighter_a.get('name', 'A')} Edges", edges_a)
    with col2:
        st.metric(f"{fighter_b.get('name', 'B')} Edges", edges_b)
