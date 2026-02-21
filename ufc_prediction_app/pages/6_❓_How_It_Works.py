"""
How It Works Page.

Documentation about data sources, methodology, and limitations.
"""

import json
import streamlit as st
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import COLORS, MODEL_VERSION, MODEL_VERSIONS_DIR, MLFLOW_ENABLED
from services.data_service import DataService
from services.llm_service import LLMService
from services.accuracy_service import AccuracyService
from services.mlflow_service import get_mlflow_service


def load_model_registry():
    """Load the model registry with training metadata."""
    registry_path = MODEL_VERSIONS_DIR / "model_registry.json"

    if registry_path.exists():
        with open(registry_path, "r") as f:
            return json.load(f)

    return None


def get_current_model_info():
    """Get information about the current best model from MLFlow or local registry."""
    mlflow_service = get_mlflow_service()

    # Try MLFlow first if available
    if mlflow_service.is_available():
        mlflow_info = get_mlflow_model_info(mlflow_service)
        if mlflow_info:
            return mlflow_info

    # Fallback to local registry
    return get_local_model_info()


def get_mlflow_model_info(mlflow_service):
    """Get model info from MLFlow."""
    try:
        all_model_info = mlflow_service.get_all_model_info()

        if not any(all_model_info.values()):
            return None

        winner_info = all_model_info.get("winner")
        method_info = all_model_info.get("method")
        round_info = all_model_info.get("round")

        model_info = {
            "version": f"v{winner_info['version']}" if winner_info else "N/A",
            "source": "mlflow",
            "models": {}
        }

        if winner_info:
            run_info = winner_info.get("run_info", {})
            model_info["models"]["winner"] = {
                "version": winner_info.get("version", "N/A"),
                "trained_at": winner_info.get("created_at", "N/A"),
                "training_samples": int(run_info.get("params", {}).get("training_samples", 0)),
                "metrics": {
                    k.replace("winner_", ""): v
                    for k, v in run_info.get("metrics", {}).items()
                    if k.startswith("winner_")
                },
                "params": {
                    k.replace("winner_", ""): v
                    for k, v in run_info.get("params", {}).items()
                    if k.startswith("winner_")
                },
                "stage": winner_info.get("stage", "None"),
            }

        if method_info:
            run_info = method_info.get("run_info", {})
            model_info["models"]["method"] = {
                "version": method_info.get("version", "N/A"),
                "trained_at": method_info.get("created_at", "N/A"),
                "training_samples": int(run_info.get("params", {}).get("training_samples", 0)),
                "metrics": {
                    k.replace("method_", ""): v
                    for k, v in run_info.get("metrics", {}).items()
                    if k.startswith("method_")
                },
                "stage": method_info.get("stage", "None"),
            }

        if round_info:
            run_info = round_info.get("run_info", {})
            model_info["models"]["round"] = {
                "version": round_info.get("version", "N/A"),
                "trained_at": round_info.get("created_at", "N/A"),
                "metrics": {
                    k.replace("round_", ""): v
                    for k, v in run_info.get("metrics", {}).items()
                    if k.startswith("round_")
                },
                "stage": round_info.get("stage", "None"),
            }

        return model_info

    except Exception as e:
        return None


def get_local_model_info():
    """Get model info from local registry."""
    registry = load_model_registry()

    if not registry:
        return None

    model_info = {
        "version": registry.get("current_version", "N/A"),
        "source": "local",
        "models": {}
    }

    for model_type, info in registry.get("models", {}).items():
        versions = info.get("versions", [])
        if versions:
            # Get the latest version (first in list, sorted by date)
            latest = versions[0]
            model_info["models"][model_type] = {
                "version": latest.get("version", "N/A"),
                "trained_at": latest.get("created_at", "N/A"),
                "training_samples": latest.get("training_samples", 0),
                "metrics": latest.get("metrics", {}),
            }

    return model_info


st.set_page_config(
    page_title="How It Works - UFC Prediction App",
    page_icon="❓",
    layout="wide",
)

st.title("❓ How It Works")
st.markdown("Learn about our data sources, prediction methodology, and limitations.")

# Initialize services
data_service = DataService()
llm_service = LLMService()
accuracy_service = AccuracyService()

# Data Sources
st.markdown("## 1. Data Sources")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        f"""
        <div style="
            background: {COLORS['card_bg']};
            padding: 20px;
            border-radius: 10px;
            height: 200px;
        ">
            <h4 style="color: {COLORS['primary']};">UFCStats.com</h4>
            <p style="color: {COLORS['text_secondary']};">
                Primary source for fighter statistics, fight results,
                and detailed per-fight metrics.
            </p>
            <p style="color: {COLORS['text_muted']};">Updated after each event</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div style="
            background: {COLORS['card_bg']};
            padding: 20px;
            border-radius: 10px;
            height: 200px;
        ">
            <h4 style="color: {COLORS['primary']};">Kaggle Datasets</h4>
            <p style="color: {COLORS['text_secondary']};">
                Historical fight data for model training,
                including older events and statistics.
            </p>
            <p style="color: {COLORS['text_muted']};">Initial data load</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
        <div style="
            background: {COLORS['card_bg']};
            padding: 20px;
            border-radius: 10px;
            height: 200px;
        ">
            <h4 style="color: {COLORS['primary']};">UFC.com</h4>
            <p style="color: {COLORS['text_secondary']};">
                Fighter images and official profiles
                (external linking only).
            </p>
            <p style="color: {COLORS['text_muted']};">Real-time linking</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Data Statistics
st.markdown("---")
st.markdown("## 2. Data Statistics")

try:
    stats = data_service.get_database_stats()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Fighters", f"{stats.get('fighters', 0):,}")

    with col2:
        st.metric("Historical Fights", f"{stats.get('fights', 0):,}")

    with col3:
        st.metric("UFC Events", f"{stats.get('events', 0):,}")

    with col4:
        st.metric("Active Fighters", f"{stats.get('active_fighters', 0):,}")

except Exception as e:
    st.info("Database statistics not available. Initialize the database first.")

# Prediction Methodology
st.markdown("---")
st.markdown("## 3. Prediction Methodology")

st.markdown(
    """
    Our prediction system uses an **ensemble machine learning approach** combining
    multiple models for robust predictions.

    ### Winner Prediction Model
    - **XGBoost Classifier** (30% weight)
    - **LightGBM Classifier** (30% weight)
    - **Random Forest Classifier** (20% weight)
    - **Logistic Regression** (20% weight)

    ### Method Prediction Model
    - Multi-class classifier predicting KO/TKO, Submission, or Decision
    - Uses fight context and fighter finishing tendencies

    ### Round Prediction Model
    - Regression model estimating expected round
    - Considers finish rates and fight duration history
    """
)

# Unsupervised Learning / Pattern Discovery
st.markdown("---")
st.markdown("## 4. Unsupervised Learning & Pattern Discovery")

st.markdown(
    """
    Beyond predicting fight outcomes, we use **unsupervised machine learning** to discover
    hidden patterns and insights from the entire UFC dataset. These algorithms find structure
    in the data without being told what to look for.
    """
)

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        f"""
        <div style="
            background: {COLORS['card_bg']};
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 15px;
        ">
            <h4 style="color: {COLORS['primary']};">Clustering Algorithms</h4>
            <p style="color: {COLORS['text_secondary']};">
                <strong>K-Means Clustering</strong><br>
                Groups fighters into distinct style archetypes (strikers, grapplers,
                balanced fighters) based on their statistical profiles.
            </p>
            <p style="color: {COLORS['text_secondary']};">
                <strong>DBSCAN</strong><br>
                Density-based clustering that identifies outlier fighters with
                truly unique statistical profiles that don't fit any archetype.
            </p>
            <p style="color: {COLORS['text_secondary']};">
                <strong>Hierarchical Clustering</strong><br>
                Creates a tree of fighter similarities, showing how different
                fighting styles relate to each other.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div style="
            background: {COLORS['card_bg']};
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 15px;
        ">
            <h4 style="color: {COLORS['primary']};">Dimensionality Reduction</h4>
            <p style="color: {COLORS['text_secondary']};">
                <strong>PCA (Principal Component Analysis)</strong><br>
                Identifies the most important statistical dimensions that
                differentiate fighters (e.g., striking vs grappling axis).
            </p>
            <p style="color: {COLORS['text_secondary']};">
                <strong>t-SNE</strong><br>
                Creates 2D visualizations where similar fighters appear close
                together, revealing natural groupings in the roster.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        f"""
        <div style="
            background: {COLORS['card_bg']};
            padding: 20px;
            border-radius: 10px;
        ">
            <h4 style="color: {COLORS['primary']};">Anomaly Detection</h4>
            <p style="color: {COLORS['text_secondary']};">
                <strong>Isolation Forest</strong><br>
                Detects statistically anomalous fighters - those with unusual
                combinations of attributes that set them apart.
            </p>
            <p style="color: {COLORS['text_secondary']};">
                <strong>Local Outlier Factor (LOF)</strong><br>
                Identifies fighters who are outliers relative to their local
                neighborhood of similar competitors.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div style="
            background: {COLORS['card_bg']};
            padding: 20px;
            border-radius: 10px;
        ">
            <h4 style="color: {COLORS['primary']};">Insights Discovered</h4>
            <p style="color: {COLORS['text_secondary']};">
                These algorithms power the <strong>UFC Insights</strong> page, revealing:
            </p>
            <ul style="color: {COLORS['text_secondary']};">
                <li>Fighter style archetypes and clusters</li>
                <li>Age and performance relationships</li>
                <li>Physical attribute advantages by division</li>
                <li>Career longevity patterns</li>
                <li>Champion characteristics</li>
                <li>Style evolution across UFC eras</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    f"""
    <div style="
        background: linear-gradient(135deg, {COLORS['card_bg']} 0%, #2a2a4e 100%);
        padding: 15px 20px;
        border-radius: 10px;
        margin-top: 15px;
        border-left: 4px solid {COLORS['info']};
    ">
        <p style="color: {COLORS['text_secondary']}; margin: 0;">
            <strong style="color: {COLORS['info']};">Supervised vs Unsupervised:</strong>
            Our prediction models (Section 3) are <em>supervised</em> - they learn from labeled
            fight outcomes. The pattern discovery algorithms here are <em>unsupervised</em> -
            they find natural structure without predetermined labels, revealing insights
            humans might not think to look for.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Features
st.markdown("---")
st.markdown("## 5. Key Features Used")

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        ### Physical Attributes
        - Height differential
        - Reach advantage
        - Age difference
        - Stance matchup

        ### Striking Statistics
        - Significant strikes landed/min
        - Striking accuracy
        - Strike defense
        - Strikes absorbed/min
        """
    )

with col2:
    st.markdown(
        """
        ### Grappling Statistics
        - Takedown average
        - Takedown accuracy
        - Takedown defense
        - Submission attempts

        ### Form & Context
        - Win/loss streaks
        - Recent form (last 5 fights)
        - Days since last fight
        - Title fight / Main event
        """
    )

# Model Updates
st.markdown("---")
st.markdown("## 6. Current Model Details")

# Load dynamic model info
model_info = get_current_model_info()
mlflow_service = get_mlflow_service()

if model_info:
    # Show model source
    source = model_info.get("source", "local")
    source_badge = "MLFlow" if source == "mlflow" else "Local Registry"
    source_color = COLORS['success'] if source == "mlflow" else COLORS['info']

    st.markdown(
        f"""
        <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 20px;">
            <h3 style="margin: 0;">Current Model Version: <code>{model_info['version']}</code></h3>
            <span style="
                background: {source_color};
                color: white;
                padding: 4px 12px;
                border-radius: 15px;
                font-size: 12px;
                font-weight: bold;
            ">{source_badge}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Winner Model Details
    winner_info = model_info["models"].get("winner", {})
    method_info = model_info["models"].get("method", {})
    round_info = model_info["models"].get("round", {})

    if winner_info:
        # Format training date
        trained_at = winner_info.get("trained_at", "N/A")
        if trained_at != "N/A":
            try:
                dt = datetime.fromisoformat(trained_at)
                trained_at_formatted = dt.strftime("%B %d, %Y at %H:%M UTC")
            except (ValueError, TypeError):
                trained_at_formatted = trained_at
        else:
            trained_at_formatted = "N/A"

        st.markdown(
            f"""
            <div style="
                background: {COLORS['card_bg']};
                padding: 20px;
                border-radius: 10px;
                border-left: 4px solid {COLORS['primary']};
                margin-bottom: 20px;
            ">
                <h4 style="color: {COLORS['primary']}; margin: 0;">Last Trained</h4>
                <p style="color: {COLORS['text_primary']}; font-size: 18px; margin: 10px 0;">
                    {trained_at_formatted}
                </p>
                <p style="color: {COLORS['text_secondary']}; margin: 0;">
                    Training samples: {winner_info.get('training_samples', 0):,}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Model Performance Metrics
    st.markdown("### Model Performance Metrics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div style="
                background: {COLORS['card_bg']};
                padding: 20px;
                border-radius: 10px;
                text-align: center;
            ">
                <h5 style="color: {COLORS['primary']};">Winner Prediction</h5>
            """,
            unsafe_allow_html=True,
        )
        winner_metrics = winner_info.get("metrics", {})
        st.metric("Accuracy", f"{winner_metrics.get('accuracy', 0):.1%}")
        st.metric("Precision", f"{winner_metrics.get('precision', 0):.1%}")
        st.metric("Recall", f"{winner_metrics.get('recall', 0):.1%}")
        st.metric("F1 Score", f"{winner_metrics.get('f1', 0):.1%}")
        st.metric("AUC-ROC", f"{winner_metrics.get('auc_roc', 0):.3f}")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(
            f"""
            <div style="
                background: {COLORS['card_bg']};
                padding: 20px;
                border-radius: 10px;
                text-align: center;
            ">
                <h5 style="color: {COLORS['primary']};">Method Prediction</h5>
            """,
            unsafe_allow_html=True,
        )
        method_metrics = method_info.get("metrics", {})
        st.metric("Accuracy", f"{method_metrics.get('accuracy', 0):.1%}")
        st.metric("Precision", f"{method_metrics.get('precision', 0):.1%}")
        st.metric("Recall", f"{method_metrics.get('recall', 0):.1%}")
        st.metric("F1 Score", f"{method_metrics.get('f1', 0):.1%}")
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown(
            f"""
            <div style="
                background: {COLORS['card_bg']};
                padding: 20px;
                border-radius: 10px;
                text-align: center;
            ">
                <h5 style="color: {COLORS['primary']};">Round Prediction</h5>
            """,
            unsafe_allow_html=True,
        )
        round_metrics = round_info.get("metrics", {})
        st.metric("Mean Abs Error", f"{round_metrics.get('mae', 0):.2f} rounds")
        st.metric("Within 1 Round", f"{round_metrics.get('within_one_round', 0):.1%}")
        st.markdown("</div>", unsafe_allow_html=True)

    # Model Parameters
    st.markdown("### Model Parameters")

    st.markdown(
        f"""
        <div style="
            background: {COLORS['card_bg']};
            padding: 20px;
            border-radius: 10px;
        ">
            <h5 style="color: {COLORS['primary']};">Ensemble Configuration</h5>
            <table style="width: 100%; color: {COLORS['text_secondary']};">
                <tr>
                    <th style="text-align: left; padding: 8px;">Model</th>
                    <th style="text-align: left; padding: 8px;">Weight</th>
                    <th style="text-align: left; padding: 8px;">Parameters</th>
                </tr>
                <tr>
                    <td style="padding: 8px;">XGBoost</td>
                    <td style="padding: 8px;">30%</td>
                    <td style="padding: 8px; font-family: monospace;">n_estimators=100, max_depth=5, learning_rate=0.1</td>
                </tr>
                <tr>
                    <td style="padding: 8px;">LightGBM</td>
                    <td style="padding: 8px;">30%</td>
                    <td style="padding: 8px; font-family: monospace;">n_estimators=100, max_depth=5, learning_rate=0.1</td>
                </tr>
                <tr>
                    <td style="padding: 8px;">Random Forest</td>
                    <td style="padding: 8px;">20%</td>
                    <td style="padding: 8px; font-family: monospace;">n_estimators=100, max_depth=5</td>
                </tr>
                <tr>
                    <td style="padding: 8px;">Logistic Regression</td>
                    <td style="padding: 8px;">20%</td>
                    <td style="padding: 8px; font-family: monospace;">max_iter=1000</td>
                </tr>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

else:
    st.markdown(f"### Current Model Version: `{MODEL_VERSION}`")
    st.info("Model registry not found. Train the models to see detailed metrics.")

# MLFlow Status
st.markdown("---")
st.markdown("### Experiment Tracking (MLFlow)")

mlflow_status = mlflow_service.get_status()

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        f"""
        <div style="
            background: {COLORS['card_bg']};
            padding: 20px;
            border-radius: 10px;
        ">
            <h5 style="color: {COLORS['primary']}; margin: 0;">MLFlow Status</h5>
            <table style="width: 100%; margin-top: 15px; color: {COLORS['text_secondary']};">
                <tr>
                    <td style="padding: 5px 0;">Enabled:</td>
                    <td style="padding: 5px 0;">{'Yes' if mlflow_status.get('enabled') else 'No'}</td>
                </tr>
                <tr>
                    <td style="padding: 5px 0;">Initialized:</td>
                    <td style="padding: 5px 0;">{'Yes' if mlflow_status.get('initialized') else 'No'}</td>
                </tr>
                <tr>
                    <td style="padding: 5px 0;">Experiment:</td>
                    <td style="padding: 5px 0; font-family: monospace;">{mlflow_status.get('experiment_name', 'N/A')}</td>
                </tr>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div style="
            background: {COLORS['card_bg']};
            padding: 20px;
            border-radius: 10px;
        ">
            <h5 style="color: {COLORS['primary']}; margin: 0;">Registered Models</h5>
            <table style="width: 100%; margin-top: 15px; color: {COLORS['text_secondary']};">
                <tr>
                    <td style="padding: 5px 0;">Winner:</td>
                    <td style="padding: 5px 0; font-family: monospace;">{mlflow_status.get('models', {}).get('winner', 'N/A')}</td>
                </tr>
                <tr>
                    <td style="padding: 5px 0;">Method:</td>
                    <td style="padding: 5px 0; font-family: monospace;">{mlflow_status.get('models', {}).get('method', 'N/A')}</td>
                </tr>
                <tr>
                    <td style="padding: 5px 0;">Round:</td>
                    <td style="padding: 5px 0; font-family: monospace;">{mlflow_status.get('models', {}).get('round', 'N/A')}</td>
                </tr>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")
st.markdown("## 7. Model Training & Updates")

st.markdown(
    """
    **Update Schedule:**
    - **Automatic**: Every Sunday at 6 AM UTC (via GitHub Actions)
    - **Manual**: Can be triggered anytime via script

    **Retraining Triggers:**
    - Rolling accuracy drops below 55%
    - 90 days since last training
    - 150+ new fights available

    **Version Management:**
    - Latest 3 model versions are kept
    - Older versions are automatically cleaned up
    """
)

# Accuracy
st.markdown("---")
st.markdown("## 8. Prediction Accuracy")

try:
    accuracy_summary = accuracy_service.get_accuracy_summary()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Last 30 Predictions",
            f"{accuracy_summary.get('rolling_30', 0):.0%}",
        )

    with col2:
        st.metric(
            "Last 100 Predictions",
            f"{accuracy_summary.get('rolling_100', 0):.0%}",
        )

    with col3:
        by_conf = accuracy_summary.get("by_confidence", {})
        high_acc = by_conf.get("high", {}).get("accuracy", 0)
        st.metric(
            "High Confidence Accuracy",
            f"{high_acc:.0%}",
        )

    st.markdown(
        """
        **Confidence Levels:**
        - 🟢 **High (>65%)**: Stronger statistical edge detected
        - 🟡 **Medium (55-65%)**: Moderate confidence in prediction
        - 🟠 **Low (<55%)**: Close matchup, could go either way
        """
    )

except Exception as e:
    st.info("Accuracy statistics not yet available.")

# Limitations
st.markdown("---")
st.markdown("## 9. Limitations & Disclaimers")

st.warning(
    """
    **What our model CANNOT account for:**

    - Injuries and health status
    - Training camp quality
    - Weight cut issues
    - Mental state and motivation
    - Recent style changes
    - External factors (travel, altitude, etc.)
    - Referee and judging tendencies
    - Fight-specific strategies
    """
)

# Gambling disclaimer
st.markdown("---")
st.markdown("## 10. Gambling Disclaimer")

st.error(
    """
    **IMPORTANT DISCLAIMER**

    This application is for **entertainment and educational purposes only**.

    - Predictions are based on statistical analysis and are NOT guaranteed
    - Past performance does not guarantee future results
    - **Never bet more than you can afford to lose**
    - If you have a gambling problem, please seek help

    Resources:
    - National Problem Gambling Helpline: 1-800-522-4700
    - www.ncpgambling.org
    """
)

# LLM Integration
st.markdown("---")
st.markdown("## 11. AI/LLM Integration")

llm_status = llm_service.get_status()

st.markdown(
    f"""
    ### Groq LLM Integration

    We use Groq's **Llama 3.3 70B** model for:
    - Fighter analysis narratives
    - Prediction explanations
    - Q&A about UFC trends

    **Current Status:**
    - Service Available: {'✅ Yes' if llm_status.get('available') else '❌ No'}
    - Model: {llm_status.get('model', 'N/A')}
    - Caching: Enabled (24-hour TTL)

    When LLM is unavailable, the app continues to function with
    all core features - only AI narratives are disabled.
    """
)

# App Settings
st.markdown("---")
st.markdown("## 12. App Settings")

st.markdown("Configure display preferences for the application.")

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        f"""
        <div style="
            background: {COLORS['card_bg']};
            padding: 20px;
            border-radius: 10px;
        ">
            <h5 style="color: {COLORS['primary']}; margin: 0 0 15px 0;">News Feed Settings</h5>
        """,
        unsafe_allow_html=True,
    )

    # News article count setting
    current_count = st.session_state.get('news_article_count', 12)
    news_count = st.selectbox(
        "Articles to display",
        options=[8, 12, 16, 24],
        index=[8, 12, 16, 24].index(current_count) if current_count in [8, 12, 16, 24] else 1,
        key="news_count_setting",
        help="Number of news articles to show on the Latest News page"
    )

    # Save to session state
    if news_count != st.session_state.get('news_article_count'):
        st.session_state['news_article_count'] = news_count
        st.success(f"News feed will now show {news_count} articles")

    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown(
        f"""
        <div style="
            background: {COLORS['card_bg']};
            padding: 20px;
            border-radius: 10px;
        ">
            <h5 style="color: {COLORS['primary']}; margin: 0 0 15px 0;">Current Settings</h5>
            <p style="color: {COLORS['text_secondary']}; margin: 0;">
                <strong>News Articles:</strong> {st.session_state.get('news_article_count', 12)}<br>
                <strong>Theme:</strong> Dark (default)<br>
                <strong>Layout:</strong> Wide
            </p>
            <p style="color: {COLORS['text_muted']}; font-size: 12px; margin-top: 15px;">
                Settings are stored in your browser session and will reset when you close the app.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Footer
st.markdown("---")
st.markdown(
    f"""
    <div style="text-align: center; color: {COLORS['text_muted']}; font-size: 12px;">
        <p>UFC Prediction App v1.0.0</p>
        <p>Built with Streamlit | Data from UFCStats.com</p>
        <p>For questions or feedback, please open an issue on GitHub</p>
    </div>
    """,
    unsafe_allow_html=True,
)
