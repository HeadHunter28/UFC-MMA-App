"""
Statistics Charts Component.

Various chart visualizations for UFC statistics.
"""

import streamlit as st
from typing import Any, Dict, List, Optional

import plotly.graph_objects as go
import plotly.express as px

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import COLORS


def render_win_method_chart(
    data: Dict[str, int],
    title: str = "Win Methods Distribution"
):
    """
    Render a pie chart of win methods.

    Args:
        data: Dict mapping method names to counts
        title: Chart title
    """
    if not data:
        st.write("No data available")
        return

    labels = list(data.keys())
    values = list(data.values())

    # Custom colors for methods
    method_colors = {
        "KO": COLORS["danger"],
        "KO/TKO": COLORS["danger"],
        "TKO": COLORS["danger"],
        "SUB": COLORS["info"],
        "Submission": COLORS["info"],
        "DEC": COLORS["success"],
        "Decision": COLORS["success"],
        "U-DEC": COLORS["success"],
        "S-DEC": COLORS["warning"],
    }

    colors = [method_colors.get(label, COLORS["text_muted"]) for label in labels]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker_colors=colors,
        textinfo="label+percent",
        textposition="outside",
    )])

    fig.update_layout(
        title=title,
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["background"],
        font=dict(color=COLORS["text_primary"]),
        height=350,
        showlegend=False,
    )

    st.plotly_chart(fig, width='stretch')


def render_radar_chart(
    data: Dict[str, float],
    title: str = "Fighter Profile",
    max_value: float = 100
):
    """
    Render a radar/spider chart.

    Args:
        data: Dict mapping category names to values
        title: Chart title
        max_value: Maximum value for the scale
    """
    if not data:
        st.write("No data available")
        return

    categories = list(data.keys())
    values = list(data.values())

    # Close the polygon
    categories = categories + [categories[0]]
    values = values + [values[0]]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill="toself",
        fillcolor=f"rgba({int(COLORS['primary'][1:3], 16)}, "
                  f"{int(COLORS['primary'][3:5], 16)}, "
                  f"{int(COLORS['primary'][5:7], 16)}, 0.3)",
        line_color=COLORS["primary"],
    ))

    fig.update_layout(
        title=title,
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max_value],
                gridcolor=COLORS["text_muted"],
            ),
            angularaxis=dict(
                gridcolor=COLORS["text_muted"],
            ),
            bgcolor=COLORS["card_bg"],
        ),
        paper_bgcolor=COLORS["background"],
        font=dict(color=COLORS["text_primary"]),
        height=400,
        showlegend=False,
    )

    st.plotly_chart(fig, width='stretch')


def render_accuracy_over_time(
    data: List[Dict[str, Any]],
    title: str = "Model Accuracy Over Time"
):
    """
    Render accuracy trend line chart.

    Args:
        data: List of dicts with 'date' and 'accuracy' keys
        title: Chart title
    """
    if not data:
        st.write("No accuracy data available yet")
        return

    dates = [d["date"] for d in data]
    accuracy = [d["accuracy"] * 100 for d in data]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates,
        y=accuracy,
        mode="lines+markers",
        line=dict(color=COLORS["primary"], width=2),
        marker=dict(size=6),
        fill="tozeroy",
        fillcolor=f"rgba({int(COLORS['primary'][1:3], 16)}, "
                  f"{int(COLORS['primary'][3:5], 16)}, "
                  f"{int(COLORS['primary'][5:7], 16)}, 0.2)",
    ))

    # Add threshold line
    fig.add_hline(
        y=55,
        line_dash="dash",
        line_color=COLORS["warning"],
        annotation_text="Minimum Threshold (55%)",
        annotation_position="bottom right",
    )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Accuracy (%)",
        yaxis=dict(range=[40, 80]),
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["card_bg"],
        font=dict(color=COLORS["text_primary"]),
        height=350,
    )

    st.plotly_chart(fig, width='stretch')


def render_confidence_distribution(
    data: Dict[str, Dict[str, float]],
    title: str = "Accuracy by Confidence Level"
):
    """
    Render accuracy breakdown by confidence level.

    Args:
        data: Dict with 'high', 'medium', 'low' keys containing accuracy data
        title: Chart title
    """
    if not data:
        st.write("No data available")
        return

    levels = ["High (>65%)", "Medium (55-65%)", "Low (<55%)"]
    keys = ["high", "medium", "low"]
    colors_list = [COLORS["success"], COLORS["warning"], COLORS["danger"]]

    accuracies = []
    totals = []

    for key in keys:
        bucket = data.get(key, {})
        accuracies.append(bucket.get("accuracy", 0) * 100)
        totals.append(bucket.get("total", 0))

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=levels,
        y=accuracies,
        marker_color=colors_list,
        text=[f"{a:.1f}%<br>({t} predictions)" for a, t in zip(accuracies, totals)],
        textposition="outside",
    ))

    fig.update_layout(
        title=title,
        yaxis_title="Accuracy (%)",
        yaxis=dict(range=[0, 100]),
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["card_bg"],
        font=dict(color=COLORS["text_primary"]),
        height=350,
    )

    st.plotly_chart(fig, width='stretch')


def render_country_stats(
    data: List[Dict[str, Any]],
    title: str = "Fighters by Country"
):
    """
    Render bar chart of fighters by country.

    Args:
        data: List of dicts with 'country' and 'count' keys
        title: Chart title
    """
    if not data:
        st.write("No data available")
        return

    # Sort and take top 15
    sorted_data = sorted(data, key=lambda x: x.get("count", 0), reverse=True)[:15]

    countries = [d["country"] for d in sorted_data]
    counts = [d["count"] for d in sorted_data]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=counts,
        y=countries,
        orientation="h",
        marker_color=COLORS["primary"],
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Number of Fighters",
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["card_bg"],
        font=dict(color=COLORS["text_primary"]),
        height=max(300, len(countries) * 25),
        yaxis=dict(autorange="reversed"),
    )

    st.plotly_chart(fig, width='stretch')


def render_weight_class_stats(
    data: List[Dict[str, Any]],
    title: str = "Fights by Weight Class"
):
    """
    Render horizontal bar chart of weight class distribution.

    Args:
        data: List of dicts with 'weight_class' and 'count' keys
        title: Chart title
    """
    if not data:
        st.write("No data available")
        return

    weight_classes = [d["weight_class"] for d in data]
    counts = [d["count"] for d in data]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=counts,
        y=weight_classes,
        orientation="h",
        marker_color=COLORS["info"],
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Number of Fights",
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["card_bg"],
        font=dict(color=COLORS["text_primary"]),
        height=400,
    )

    st.plotly_chart(fig, width='stretch')


def render_feature_importance_chart(
    feature_importance: Dict[str, float],
    title: str = "Feature Importance"
):
    """
    Render feature importance horizontal bar chart.

    Args:
        feature_importance: Dict mapping feature names to importance values
        title: Chart title
    """
    if not feature_importance:
        st.write("No feature importance data available")
        return

    # Sort and take top features
    sorted_features = sorted(
        feature_importance.items(),
        key=lambda x: x[1],
        reverse=True
    )[:15]

    features = [f[0].replace("_", " ").title() for f in sorted_features]
    importances = [f[1] for f in sorted_features]

    # Normalize
    max_imp = max(importances) if importances else 1
    normalized = [i / max_imp * 100 for i in importances]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=normalized,
        y=features,
        orientation="h",
        marker_color=COLORS["primary"],
        text=[f"{i:.1f}%" for i in normalized],
        textposition="outside",
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Relative Importance (%)",
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["card_bg"],
        font=dict(color=COLORS["text_primary"]),
        height=max(300, len(features) * 25),
        yaxis=dict(autorange="reversed"),
        xaxis=dict(range=[0, 120]),  # Extra space for labels
    )

    st.plotly_chart(fig, width='stretch')
