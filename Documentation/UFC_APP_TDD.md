# UFC Fighter Analysis & Prediction App
## Technical Design Document (TDD)
### Version 1.0 | January 2025

---

# TABLE OF CONTENTS

1. [Project Overview](#1-project-overview)
2. [Technology Stack](#2-technology-stack)
3. [Project Structure](#3-project-structure)
4. [Database Schema](#4-database-schema)
5. [Application Pages](#5-application-pages)
6. [Machine Learning System](#6-machine-learning-system)
7. [Services Architecture](#7-services-architecture)
8. [External Integrations](#8-external-integrations)
9. [Data Pipeline](#9-data-pipeline)
10. [Update Automation](#10-update-automation)
11. [UI/UX Specifications](#11-uiux-specifications)
12. [Configuration](#12-configuration)
13. [Implementation Phases](#13-implementation-phases)
14. [Testing Requirements](#14-testing-requirements)

---

# 1. PROJECT OVERVIEW

## 1.1 Description
A comprehensive UFC Fighter Analysis and Prediction application built with Streamlit. The app provides fighter profiles, ML-based fight predictions with explainability, upcoming bout analysis, data-driven trends discovery, and fighter comparison tools.

## 1.2 Key Features
- **Fighter Search & Profiles**: Search UFC fighters, view detailed stats, fight history, win methods
- **Fighter Comparison**: Head-to-head statistical comparison tool
- **Fight Predictions**: ML ensemble predictions with confidence levels and explainability
- **Upcoming Bouts**: Official UFC cards with auto-generated predictions
- **Trends & Facts**: Analytics dashboard, pattern discovery, Q&A system
- **How It Works**: Documentation of data sources, methodology, limitations

## 1.3 Design Decisions Summary

| Decision | Choice |
|----------|--------|
| Frontend | Streamlit |
| Database | SQLite (local dev) → Git LFS (production) |
| ML Models | XGBoost, LightGBM, Random Forest ensemble |
| LLM Provider | Groq (Llama 3.3 70B) |
| Data Source | UFCStats.com (scraping) + Kaggle (historical) |
| Fighter Images | External URL linking to UFC.com |
| Deployment | Streamlit Cloud |
| Updates | GitHub Actions (auto) + Manual script |
| Model Versions | Keep latest 3 versions |
| Theme | UFC colors (Red #D20A0A, Black, White) |

---

# 2. TECHNOLOGY STACK

## 2.1 Core Dependencies

```txt
# requirements.txt

# === Core Framework ===
streamlit>=1.29.0

# === Data Processing ===
pandas>=2.0.0
numpy>=1.24.0

# === Machine Learning ===
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.0.0
shap>=0.43.0
joblib>=1.3.0

# === Web Scraping ===
requests>=2.31.0
beautifulsoup4>=4.12.0
lxml>=4.9.0

# === Visualization ===
plotly>=5.18.0
altair>=5.0.0

# === LLM Integration ===
groq>=0.4.0

# === Utilities ===
python-dotenv>=1.0.0
tenacity>=8.2.0

# === Optional ===
kaggle>=1.5.0
```

## 2.2 Development Dependencies

```txt
# requirements-dev.txt
pytest>=7.4.0
pytest-cov>=4.1.0
black>=23.0.0
flake8>=6.1.0
mypy>=1.5.0
```

---

# 3. PROJECT STRUCTURE

```
ufc_prediction_app/
│
├── .github/
│   └── workflows/
│       └── update_data.yml              # GitHub Actions workflow
│
├── .streamlit/
│   └── config.toml                      # Streamlit theme configuration
│
├── app.py                               # Main Streamlit entry point
├── config.py                            # Centralized configuration
├── requirements.txt
├── requirements-dev.txt
├── .env.template
├── .gitignore
├── .gitattributes                       # Git LFS configuration
├── README.md
│
├── pages/
│   ├── 1_🥊_Fighters.py
│   ├── 2_⚔️_Fighter_Comparison.py
│   ├── 3_🎯_Predictions.py
│   ├── 4_📅_Upcoming_Bouts.py
│   ├── 5_📊_Trends_and_Facts.py
│   └── 6_❓_How_It_Works.py
│
├── data/
│   ├── database/
│   │   └── ufc_database.db
│   ├── cache/
│   │   └── llm_cache.json
│   ├── raw/
│   │   ├── kaggle/
│   │   └── scraped/
│   └── processed/
│
├── models/
│   ├── trained/
│   │   ├── current/                     # Symlinks to active models
│   │   │   ├── winner_model.pkl
│   │   │   ├── method_model.pkl
│   │   │   └── round_model.pkl
│   │   └── versions/
│   │       └── model_registry.json
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train_models.py
│   │   ├── feature_engineering.py
│   │   └── backfill_predictions.py
│   └── inference.py
│
├── services/
│   ├── __init__.py
│   ├── data_service.py
│   ├── scraper_service.py
│   ├── prediction_service.py
│   ├── llm_service.py
│   ├── accuracy_service.py
│   └── cache_service.py
│
├── components/
│   ├── __init__.py
│   ├── fighter_card.py
│   ├── prediction_display.py
│   ├── comparison_charts.py
│   ├── fight_card.py
│   └── stats_charts.py
│
├── utils/
│   ├── __init__.py
│   ├── helpers.py
│   ├── validators.py
│   └── formatters.py
│
├── scripts/
│   ├── manual_update.py
│   ├── check_new_events.py
│   ├── check_retrain_needed.py
│   ├── cleanup_models.py
│   ├── download_kaggle_data.py
│   └── init_database.py
│
├── tests/
│   ├── __init__.py
│   ├── test_data_service.py
│   ├── test_prediction_service.py
│   ├── test_scraper_service.py
│   └── conftest.py
│
└── docs/
    ├── SETUP.md
    ├── DEPLOYMENT.md
    └── GIT_LFS_GUIDE.md
```

---

# 4. DATABASE SCHEMA

## 4.1 SQLite Schema

```sql
-- ============================================================================
-- UFC PREDICTION APP - DATABASE SCHEMA
-- ============================================================================

-- Fighters table
CREATE TABLE IF NOT EXISTS fighters (
    fighter_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    nickname TEXT,
    height_cm REAL,
    weight_kg REAL,
    reach_cm REAL,
    stance TEXT,
    dob DATE,
    nationality TEXT,
    team TEXT,
    wins INTEGER DEFAULT 0,
    losses INTEGER DEFAULT 0,
    draws INTEGER DEFAULT 0,
    no_contests INTEGER DEFAULT 0,
    image_url TEXT,
    ufc_stats_url TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(name, dob)
);

-- Fighter career statistics
CREATE TABLE IF NOT EXISTS fighter_stats (
    stat_id INTEGER PRIMARY KEY AUTOINCREMENT,
    fighter_id INTEGER NOT NULL,
    sig_strikes_landed_per_min REAL,
    sig_strikes_absorbed_per_min REAL,
    sig_strike_accuracy REAL,
    sig_strike_defense REAL,
    takedowns_avg_per_15min REAL,
    takedown_accuracy REAL,
    takedown_defense REAL,
    submissions_avg_per_15min REAL,
    avg_fight_time_seconds INTEGER,
    finish_rate REAL,
    ko_rate REAL,
    submission_rate REAL,
    decision_rate REAL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (fighter_id) REFERENCES fighters(fighter_id) ON DELETE CASCADE
);

-- Events table
CREATE TABLE IF NOT EXISTS events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    date DATE NOT NULL,
    location TEXT,
    venue TEXT,
    country TEXT,
    is_completed BOOLEAN DEFAULT FALSE,
    ufc_stats_url TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(name, date)
);

-- Fights table (historical results)
CREATE TABLE IF NOT EXISTS fights (
    fight_id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id INTEGER NOT NULL,
    fighter_red_id INTEGER NOT NULL,
    fighter_blue_id INTEGER NOT NULL,
    winner_id INTEGER,
    weight_class TEXT NOT NULL,
    is_title_fight BOOLEAN DEFAULT FALSE,
    is_main_event BOOLEAN DEFAULT FALSE,
    method TEXT,
    method_detail TEXT,
    round INTEGER,
    time TEXT,
    referee TEXT,
    bonus TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (event_id) REFERENCES events(event_id) ON DELETE CASCADE,
    FOREIGN KEY (fighter_red_id) REFERENCES fighters(fighter_id),
    FOREIGN KEY (fighter_blue_id) REFERENCES fighters(fighter_id),
    FOREIGN KEY (winner_id) REFERENCES fighters(fighter_id)
);

-- Per-fight statistics
CREATE TABLE IF NOT EXISTS fight_stats (
    stat_id INTEGER PRIMARY KEY AUTOINCREMENT,
    fight_id INTEGER NOT NULL,
    fighter_id INTEGER NOT NULL,
    corner TEXT CHECK(corner IN ('red', 'blue')),
    knockdowns INTEGER DEFAULT 0,
    sig_strikes_landed INTEGER DEFAULT 0,
    sig_strikes_attempted INTEGER DEFAULT 0,
    sig_strikes_head INTEGER DEFAULT 0,
    sig_strikes_body INTEGER DEFAULT 0,
    sig_strikes_leg INTEGER DEFAULT 0,
    total_strikes_landed INTEGER DEFAULT 0,
    total_strikes_attempted INTEGER DEFAULT 0,
    takedowns_landed INTEGER DEFAULT 0,
    takedowns_attempted INTEGER DEFAULT 0,
    submissions_attempted INTEGER DEFAULT 0,
    reversals INTEGER DEFAULT 0,
    control_time_seconds INTEGER DEFAULT 0,
    FOREIGN KEY (fight_id) REFERENCES fights(fight_id) ON DELETE CASCADE,
    FOREIGN KEY (fighter_id) REFERENCES fighters(fighter_id)
);

-- Upcoming fights
CREATE TABLE IF NOT EXISTS upcoming_fights (
    upcoming_id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id INTEGER NOT NULL,
    fighter_red_id INTEGER NOT NULL,
    fighter_blue_id INTEGER NOT NULL,
    weight_class TEXT NOT NULL,
    is_main_event BOOLEAN DEFAULT FALSE,
    is_title_fight BOOLEAN DEFAULT FALSE,
    card_position TEXT CHECK(card_position IN ('main_card', 'prelims', 'early_prelims')),
    bout_order INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (event_id) REFERENCES events(event_id) ON DELETE CASCADE,
    FOREIGN KEY (fighter_red_id) REFERENCES fighters(fighter_id),
    FOREIGN KEY (fighter_blue_id) REFERENCES fighters(fighter_id)
);

-- Predictions
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    fight_id INTEGER,
    upcoming_id INTEGER,
    fighter_red_id INTEGER NOT NULL,
    fighter_blue_id INTEGER NOT NULL,
    predicted_winner_id INTEGER,
    winner_confidence REAL,
    method_ko_prob REAL,
    method_sub_prob REAL,
    method_dec_prob REAL,
    predicted_method TEXT,
    predicted_round REAL,
    feature_importance TEXT,
    top_factors TEXT,
    model_version TEXT,
    is_backfill BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (fight_id) REFERENCES fights(fight_id),
    FOREIGN KEY (upcoming_id) REFERENCES upcoming_fights(upcoming_id)
);

-- Prediction accuracy tracking
CREATE TABLE IF NOT EXISTS prediction_accuracy (
    accuracy_id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER NOT NULL,
    fight_id INTEGER NOT NULL,
    actual_winner_id INTEGER,
    actual_method TEXT,
    actual_round INTEGER,
    winner_correct BOOLEAN,
    method_correct BOOLEAN,
    round_correct BOOLEAN,
    confidence_score REAL,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (prediction_id) REFERENCES predictions(prediction_id),
    FOREIGN KEY (fight_id) REFERENCES fights(fight_id)
);

-- Model performance tracking
CREATE TABLE IF NOT EXISTS model_performance (
    performance_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_version TEXT NOT NULL,
    model_type TEXT NOT NULL CHECK(model_type IN ('winner', 'method', 'round')),
    training_date DATE,
    training_samples INTEGER,
    accuracy REAL,
    precision_score REAL,
    recall_score REAL,
    f1_score REAL,
    auc_roc REAL,
    rolling_accuracy_30 REAL,
    rolling_accuracy_100 REAL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- App metadata
CREATE TABLE IF NOT EXISTS app_metadata (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- INDEXES
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_fighters_name ON fighters(name);
CREATE INDEX IF NOT EXISTS idx_fighters_active ON fighters(is_active);
CREATE INDEX IF NOT EXISTS idx_fights_event ON fights(event_id);
CREATE INDEX IF NOT EXISTS idx_fights_fighters ON fights(fighter_red_id, fighter_blue_id);
CREATE INDEX IF NOT EXISTS idx_fights_date ON fights(fight_id);
CREATE INDEX IF NOT EXISTS idx_fight_stats_fight ON fight_stats(fight_id);
CREATE INDEX IF NOT EXISTS idx_events_date ON events(date);
CREATE INDEX IF NOT EXISTS idx_events_completed ON events(is_completed);
CREATE INDEX IF NOT EXISTS idx_upcoming_event ON upcoming_fights(event_id);
CREATE INDEX IF NOT EXISTS idx_predictions_fight ON predictions(fight_id);
CREATE INDEX IF NOT EXISTS idx_predictions_upcoming ON predictions(upcoming_id);

-- ============================================================================
-- VIEWS
-- ============================================================================

-- Fighter complete profile view
CREATE VIEW IF NOT EXISTS v_fighter_profile AS
SELECT 
    f.*,
    fs.sig_strikes_landed_per_min,
    fs.sig_strikes_absorbed_per_min,
    fs.sig_strike_accuracy,
    fs.sig_strike_defense,
    fs.takedowns_avg_per_15min,
    fs.takedown_accuracy,
    fs.takedown_defense,
    fs.submissions_avg_per_15min,
    fs.finish_rate,
    fs.ko_rate,
    fs.submission_rate,
    fs.decision_rate
FROM fighters f
LEFT JOIN fighter_stats fs ON f.fighter_id = fs.fighter_id;

-- Model accuracy summary view
CREATE VIEW IF NOT EXISTS v_model_accuracy_summary AS
SELECT 
    p.model_version,
    COUNT(*) as total_predictions,
    SUM(CASE WHEN pa.winner_correct THEN 1 ELSE 0 END) as correct_winners,
    ROUND(100.0 * SUM(CASE WHEN pa.winner_correct THEN 1 ELSE 0 END) / COUNT(*), 2) as winner_accuracy_pct,
    SUM(CASE WHEN pa.method_correct THEN 1 ELSE 0 END) as correct_methods,
    ROUND(100.0 * SUM(CASE WHEN pa.method_correct THEN 1 ELSE 0 END) / COUNT(*), 2) as method_accuracy_pct
FROM prediction_accuracy pa
JOIN predictions p ON pa.prediction_id = p.prediction_id
GROUP BY p.model_version;
```

---

# 5. APPLICATION PAGES

## 5.1 Page Overview

| Page | File | Description |
|------|------|-------------|
| Home | `app.py` | Dashboard with stats, next event, recent accuracy |
| Fighters | `pages/1_🥊_Fighters.py` | Search, profiles, stats, fight history |
| Comparison | `pages/2_⚔️_Fighter_Comparison.py` | Head-to-head analysis |
| Predictions | `pages/3_🎯_Predictions.py` | Custom matchup predictions |
| Upcoming | `pages/4_📅_Upcoming_Bouts.py` | Official cards with predictions |
| Trends | `pages/5_📊_Trends_and_Facts.py` | Analytics dashboard, Q&A |
| How It Works | `pages/6_❓_How_It_Works.py` | Documentation, methodology |

## 5.2 Page Specifications

### 5.2.1 Home Page (app.py)

**Components:**
- App logo and title
- Navigation cards (6 feature cards linking to pages)
- Quick stats row (total fighters, fights, events, model accuracy)
- Next UFC event card with countdown
- Recent prediction performance summary

**Data Required:**
- Database stats counts
- Next upcoming event
- Last 30 days prediction accuracy

### 5.2.2 Fighters Page

**Components:**
- Search bar with autocomplete
- Filters: Weight class, Country, Active status
- Fighter profile card (photo, basic info, record)
- Career statistics dashboard (striking, grappling metrics)
- Win methods pie chart
- Fight history timeline (expandable)
- AI-generated fighter analysis (LLM)

**Features:**
- Clicking fighter opens detailed profile
- "Compare" button → navigates to comparison page
- "Predict Matchup" button → navigates to predictions page

### 5.2.3 Fighter Comparison Page

**Components:**
- Two fighter search/select boxes
- Weight class filter
- Side-by-side comparison:
  - Basic info (record, age, height, reach, weight, stance)
  - Striking comparison (bar charts)
  - Grappling comparison (bar charts)
  - Win method comparison (dual pie charts)
- Edge summary (who has advantage in what)
- "Get Prediction" button → navigates to predictions

### 5.2.4 Predictions Page

**Components:**
- Fighter A and Fighter B selection
- Context options (weight class, title fight checkbox)
- "Generate Prediction" button
- Results display:
  - Winner prediction with confidence bar
  - Method probabilities (KO/Sub/Dec)
  - Round prediction
  - Confidence level indicator (High/Medium/Low with colors)
  - Low confidence warning banner (when < 55%)
- Explanation section:
  - Top contributing factors with bars
  - Feature importance chart (expandable)
  - SHAP analysis (expandable)
- AI narrative analysis (LLM)
- Model info footer (version, historical accuracy)

**Confidence Levels:**
- High (Green): > 65%
- Medium (Yellow): 55-65%
- Low (Orange): < 55% → Show warning banner

### 5.2.5 Upcoming Bouts Page

**Components:**
- Event selector (if multiple upcoming)
- Event header (name, location, date, countdown)
- Fight cards organized by:
  - Main Card
  - Prelims
  - Early Prelims
- Each fight card shows:
  - Fighter photos and records
  - Prediction summary (winner, method, round, confidence)
  - Confidence indicator
  - "View Details" expander
- Event prediction summary (total fights, confidence breakdown)
- Export predictions button (CSV)

### 5.2.6 Trends & Facts Page

**Tabs:**
1. **Dashboard** - Visual analytics
2. **Encyclopedia** - Categorized facts
3. **Ask a Question** - LLM-powered Q&A

**Dashboard Components:**
- Key insight cards (top country, most finishes, KO trend, top submission)
- Feature importance chart (what predicts outcomes)
- Fighter style clusters visualization (scatter plot)
- Discovered correlations list
- Outlier fighters section

**Encyclopedia Components:**
- Category sidebar (Countries, Champions, Knockouts, Submissions, etc.)
- Fact cards with charts
- Related facts

**Q&A Components:**
- Text input for questions
- Recent questions list
- Answer display with supporting data

### 5.2.7 How It Works Page

**Sections:**
1. **Data Sources** - UFCStats.com, Kaggle, UFC.com images
2. **Data Statistics** - Counts with last update date
3. **Prediction Methodology** - ML model descriptions
4. **Key Features Used** - Feature categories
5. **Model Training & Updates** - Update schedule, versioning
6. **Prediction Accuracy Tracking** - Historical and recent performance
7. **Limitations & Disclaimers** - What model can't account for
8. **Gambling Disclaimer** - Legal disclaimer
9. **LLM Integration** - Groq usage, status indicator

---

# 6. MACHINE LEARNING SYSTEM

## 6.1 Model Architecture

### 6.1.1 Winner Prediction (Ensemble)

```
Ensemble Components:
├── XGBoost Classifier (weight: 0.30)
├── LightGBM Classifier (weight: 0.30)
├── Random Forest Classifier (weight: 0.20)
└── Logistic Regression (weight: 0.20)

Output: Probability of Fighter A winning
Target Accuracy: 60-65%
```

### 6.1.2 Method Prediction (Multi-class)

```
Ensemble Components:
├── XGBoost Classifier (weight: 0.35)
├── Random Forest Classifier (weight: 0.35)
└── Gradient Boosting Classifier (weight: 0.30)

Classes: KO/TKO, Submission, Decision
Target Accuracy: 48-55%
```

### 6.1.3 Round Prediction (Regression + Classification)

```
Components:
├── XGBoost Regressor (continuous round prediction)
└── Ordinal Classifier (discrete round buckets)

Output: Expected round (float) + most likely round
```

## 6.2 Feature Engineering

### 6.2.1 Feature Categories

```python
DIFFERENTIAL_FEATURES = [
    'height_diff',              # cm
    'reach_diff',               # cm
    'age_diff',                 # years
    'sig_str_acc_diff',         # percentage points
    'sig_str_def_diff',
    'td_acc_diff',
    'td_def_diff',
    'str_landed_pm_diff',
    'str_absorbed_pm_diff',
    'td_avg_diff',
    'sub_avg_diff',
]

RATIO_FEATURES = [
    'win_rate_ratio',
    'finish_rate_ratio',
    'ko_rate_ratio',
    'sub_rate_ratio',
    'experience_ratio',
    'ufc_experience_ratio',
]

FORM_FEATURES = [
    'win_streak_a', 'win_streak_b',
    'lose_streak_a', 'lose_streak_b',
    'recent_form_a', 'recent_form_b',      # Weighted last 5 fights
    'days_since_fight_a', 'days_since_fight_b',
    'momentum_a', 'momentum_b',
]

CONTEXTUAL_FEATURES = [
    'weight_class_encoded',     # One-hot or ordinal
    'is_title_fight',           # Boolean
    'is_main_event',            # Boolean
    'rounds_scheduled',         # 3 or 5
]
```

### 6.2.2 Feature Engineering Pipeline

```python
# models/training/feature_engineering.py

def create_fight_features(fighter_a_stats, fighter_b_stats, context):
    """
    Create feature vector for a matchup.
    
    Args:
        fighter_a_stats: Dict of Fighter A statistics
        fighter_b_stats: Dict of Fighter B statistics
        context: Dict with weight_class, is_title_fight, etc.
    
    Returns:
        np.array: Feature vector
    """
    features = {}
    
    # Differential features
    for stat in ['height_cm', 'reach_cm', 'age', ...]:
        features[f'{stat}_diff'] = fighter_a_stats[stat] - fighter_b_stats[stat]
    
    # Ratio features
    for stat in ['win_rate', 'finish_rate', ...]:
        a_val = fighter_a_stats[stat] or 0.5
        b_val = fighter_b_stats[stat] or 0.5
        features[f'{stat}_ratio'] = a_val / max(b_val, 0.01)
    
    # Form features
    features['win_streak_a'] = fighter_a_stats['current_win_streak']
    features['win_streak_b'] = fighter_b_stats['current_win_streak']
    # ... etc
    
    # Contextual features
    features['is_title_fight'] = int(context.get('is_title_fight', False))
    features['weight_class_encoded'] = encode_weight_class(context['weight_class'])
    
    return np.array([features[f] for f in FEATURE_ORDER])
```

## 6.3 Training Pipeline

```python
# models/training/train_models.py

class ModelTrainer:
    def __init__(self):
        self.models_dir = Path("models/trained/versions")
        self.registry_file = self.models_dir / "model_registry.json"
    
    def train_all_models(self, force=False):
        """Train all prediction models."""
        # Load training data
        X, y_winner, y_method, y_round = self.prepare_training_data()
        
        # Time-based split (no data leakage)
        X_train, X_test, y_train, y_test = self.temporal_split(X, y_winner)
        
        # Train winner ensemble
        winner_model = self.train_winner_model(X_train, y_train['winner'])
        
        # Train method classifier
        method_model = self.train_method_model(X_train, y_train['method'])
        
        # Train round predictor
        round_model = self.train_round_model(X_train, y_train['round'])
        
        # Evaluate
        metrics = self.evaluate_models(X_test, y_test, winner_model, method_model, round_model)
        
        # Save with versioning
        version = self.get_next_version()
        self.save_models(version, winner_model, method_model, round_model, metrics)
        
        # Cleanup old versions (keep 3)
        self.cleanup_old_versions(keep=3)
        
        return version, metrics
```

## 6.4 Inference Pipeline

```python
# models/inference.py

class PredictionEngine:
    def __init__(self):
        self.winner_model = self.load_model('winner')
        self.method_model = self.load_model('method')
        self.round_model = self.load_model('round')
        self.feature_names = self.load_feature_names()
    
    def predict(self, fighter_a_id, fighter_b_id, context=None):
        """
        Generate prediction for a matchup.
        
        Returns:
            PredictionResult with winner, method, round, confidence, explanations
        """
        # Get fighter stats
        fighter_a = self.data_service.get_fighter_stats(fighter_a_id)
        fighter_b = self.data_service.get_fighter_stats(fighter_b_id)
        
        # Create features
        features = create_fight_features(fighter_a, fighter_b, context or {})
        
        # Predict winner
        winner_proba = self.winner_model.predict_proba([features])[0]
        predicted_winner = fighter_a_id if winner_proba[1] > 0.5 else fighter_b_id
        confidence = max(winner_proba)
        
        # Predict method
        method_proba = self.method_model.predict_proba([features])[0]
        method_labels = ['KO/TKO', 'Submission', 'Decision']
        
        # Predict round
        predicted_round = self.round_model.predict([features])[0]
        
        # Calculate feature importance (SHAP)
        feature_importance = self.calculate_shap_values(features)
        
        return PredictionResult(
            predicted_winner_id=predicted_winner,
            winner_confidence=confidence,
            method_ko_prob=method_proba[0],
            method_sub_prob=method_proba[1],
            method_dec_prob=method_proba[2],
            predicted_round=predicted_round,
            feature_importance=feature_importance,
            confidence_level=self.get_confidence_level(confidence)
        )
    
    def get_confidence_level(self, confidence):
        if confidence > 0.65:
            return 'high'
        elif confidence > 0.55:
            return 'medium'
        else:
            return 'low'
```

## 6.5 Model Versioning

```json
// models/trained/versions/model_registry.json
{
  "registry_version": "1.0",
  "max_versions_kept": 3,
  "current_version": "v1.0.0",
  "models": {
    "winner": {
      "current_version": "v1.0.0",
      "versions": [
        {
          "version": "v1.0.0",
          "filename": "winner_v1.0.0.pkl",
          "created_at": "2025-01-29T12:00:00Z",
          "training_samples": 6842,
          "metrics": {
            "accuracy": 0.623,
            "precision": 0.618,
            "recall": 0.631,
            "f1": 0.624,
            "auc_roc": 0.672
          },
          "status": "active"
        }
      ]
    }
  }
}
```

## 6.6 Retraining Strategy

| Trigger | Condition | Action |
|---------|-----------|--------|
| **Scheduled** | Every quarter (90 days) | Full retrain |
| **Performance** | Rolling accuracy < 55% | Full retrain |
| **Data** | 150+ new fights available | Consider retrain |
| **Manual** | Admin trigger | Full retrain |

---

# 7. SERVICES ARCHITECTURE

## 7.1 Service Overview

```
services/
├── data_service.py        # Database operations, queries
├── scraper_service.py     # UFCStats.com scraping
├── prediction_service.py  # ML inference orchestration
├── llm_service.py         # Groq LLM integration
├── accuracy_service.py    # Prediction tracking
└── cache_service.py       # Response caching
```

## 7.2 Data Service

```python
# services/data_service.py

class DataService:
    """Database operations and queries."""
    
    def __init__(self, db_path=None):
        self.db_path = db_path or config.DATABASE_PATH
        self.conn = None
    
    # Fighter operations
    def get_fighter_by_id(self, fighter_id: int) -> dict
    def get_fighter_by_name(self, name: str) -> dict
    def search_fighters(self, query: str, limit: int = 20) -> list
    def get_fighter_stats(self, fighter_id: int) -> dict
    def get_fighter_fight_history(self, fighter_id: int) -> list
    def get_all_fighters(self, filters: dict = None) -> list
    
    # Fight operations
    def get_fight_by_id(self, fight_id: int) -> dict
    def get_fights_by_event(self, event_id: int) -> list
    def get_fighter_record(self, fighter_id: int) -> dict
    
    # Event operations
    def get_upcoming_events(self) -> list
    def get_upcoming_fights(self, event_id: int = None) -> list
    def get_completed_events(self, limit: int = 10) -> list
    
    # Stats operations
    def get_database_stats(self) -> dict
    def get_weight_classes(self) -> list
    def get_countries(self) -> list
    
    # Prediction operations
    def save_prediction(self, prediction: dict) -> int
    def get_prediction(self, fight_id: int = None, upcoming_id: int = None) -> dict
    
    # Trends operations
    def get_win_method_distribution(self, filters: dict = None) -> dict
    def get_country_statistics(self) -> list
    def get_weight_class_statistics(self) -> list
```

## 7.3 Scraper Service

```python
# services/scraper_service.py

class UFCStatsScraper:
    """Scrapes data from UFCStats.com"""
    
    BASE_URL = "http://ufcstats.com"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': '...'})
    
    # Scraping methods
    def scrape_all_fighters(self) -> list
    def scrape_fighter_details(self, fighter_url: str) -> dict
    def scrape_all_events(self) -> list
    def scrape_event_details(self, event_url: str) -> dict
    def scrape_fight_details(self, fight_url: str) -> dict
    def scrape_upcoming_events(self) -> list
    
    # Incremental updates
    def scrape_new_events(self, since_date: date = None) -> list
    def scrape_new_fighters(self) -> list
    
    # Parsing helpers
    def parse_fighter_page(self, html: str) -> dict
    def parse_event_page(self, html: str) -> dict
    def parse_fight_stats(self, html: str) -> dict
```

## 7.4 LLM Service

```python
# services/llm_service.py

class LLMService:
    """Groq LLM integration for natural language generation."""
    
    def __init__(self):
        self.client = Groq(api_key=config.GROQ_API_KEY)
        self.model = config.LLM_MODEL
        self.cache = CacheService()
        self.enabled = config.LLM_ENABLED
    
    def is_available(self) -> bool:
        """Check if LLM service is available."""
        if not self.enabled:
            return False
        try:
            # Simple health check
            return True
        except:
            return False
    
    def generate_fighter_analysis(self, fighter: dict, stats: dict, history: list) -> str:
        """Generate natural language fighter analysis."""
        if not self.is_available():
            return None
        
        prompt = self._build_fighter_prompt(fighter, stats, history)
        return self._generate(prompt, cache_key=f"fighter_{fighter['fighter_id']}")
    
    def generate_prediction_explanation(self, prediction: dict, fighter_a: dict, fighter_b: dict) -> str:
        """Generate explanation for a prediction."""
        if not self.is_available():
            return None
        
        prompt = self._build_prediction_prompt(prediction, fighter_a, fighter_b)
        return self._generate(prompt)
    
    def answer_trends_question(self, question: str, context_data: dict) -> str:
        """Answer a user question about UFC trends."""
        if not self.is_available():
            return None
        
        prompt = self._build_qa_prompt(question, context_data)
        return self._generate(prompt)
    
    def _generate(self, prompt: str, cache_key: str = None) -> str:
        """Generate response from LLM."""
        # Check cache
        if cache_key:
            cached = self.cache.get(cache_key)
            if cached:
                return cached
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=config.LLM_MAX_TOKENS,
                temperature=config.LLM_TEMPERATURE
            )
            result = response.choices[0].message.content
            
            # Cache result
            if cache_key:
                self.cache.set(cache_key, result)
            
            return result
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return None
```

## 7.5 Accuracy Service

```python
# services/accuracy_service.py

class AccuracyService:
    """Track and report prediction accuracy."""
    
    def __init__(self):
        self.data_service = DataService()
    
    def update_completed_predictions(self) -> int:
        """Update accuracy for predictions of completed fights."""
        # Get predictions without accuracy records
        pending = self.data_service.get_pending_accuracy_updates()
        
        updated = 0
        for pred in pending:
            # Get actual fight result
            fight = self.data_service.get_fight_by_id(pred['fight_id'])
            if fight and fight['winner_id']:
                accuracy = self._calculate_accuracy(pred, fight)
                self.data_service.save_accuracy(accuracy)
                updated += 1
        
        return updated
    
    def get_rolling_accuracy(self, window: int = 100) -> float:
        """Get rolling accuracy over last N predictions."""
        # Query accuracy records
        ...
    
    def get_accuracy_by_model(self, model_version: str) -> dict:
        """Get accuracy breakdown for a model version."""
        ...
    
    def get_accuracy_over_time(self) -> list:
        """Get accuracy trend over time for charts."""
        ...
```

---

# 8. EXTERNAL INTEGRATIONS

## 8.1 Groq API (LLM)

**Setup:**
```bash
# .env
GROQ_API_KEY=gsk_your_api_key_here
```

**Usage:**
```python
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=1000,
    temperature=0.7
)
```

**Fallback Behavior:**
- If API unavailable: Hide LLM-dependent features
- Show banner: "AI analysis temporarily unavailable"
- All other features continue working

## 8.2 UFCStats.com (Scraping)

**Target URLs:**
- Fighters list: `http://ufcstats.com/statistics/fighters`
- Fighter details: `http://ufcstats.com/fighter-details/{id}`
- Events list: `http://ufcstats.com/statistics/events/completed`
- Event details: `http://ufcstats.com/event-details/{id}`
- Upcoming: `http://ufcstats.com/statistics/events/upcoming`

**Rate Limiting:**
- 1 request per second
- Retry with exponential backoff on failures
- Cache responses locally

## 8.3 Fighter Images

**Strategy:** External URL linking (no download)

```python
def get_fighter_image_url(fighter_name: str) -> str:
    """Generate UFC.com fighter image URL."""
    # Format: https://www.ufc.com/athlete/{name-slug}
    slug = fighter_name.lower().replace(' ', '-')
    return f"https://www.ufc.com/athlete/{slug}"
```

**Fallback:** Default placeholder silhouette

---

# 9. DATA PIPELINE

## 9.1 Initial Data Load

```
1. Download Kaggle datasets (manual or API)
   └── Place in data/raw/kaggle/

2. Run initialization script
   └── python scripts/init_database.py

3. Script performs:
   ├── Create database schema
   ├── Parse Kaggle CSVs
   ├── Clean and validate data
   ├── Insert into SQLite
   └── Generate initial stats
```

## 9.2 Incremental Updates

```
After each UFC event:

1. Scraper checks for new completed events
2. For each new event:
   ├── Scrape event details
   ├── Scrape all fight results
   ├── Scrape/update fighter stats
   └── Insert into database
3. Update prediction accuracy for completed fights
4. Check if retraining needed
5. Generate predictions for upcoming events
```

## 9.3 Data Validation

```python
def validate_fighter(fighter: dict) -> bool:
    """Validate fighter data before insert."""
    required = ['name']
    for field in required:
        if not fighter.get(field):
            return False
    
    # Validate ranges
    if fighter.get('height_cm') and not (100 < fighter['height_cm'] < 250):
        return False
    
    return True
```

---

# 10. UPDATE AUTOMATION

## 10.1 GitHub Actions Workflow

**File:** `.github/workflows/update_data.yml`

**Triggers:**
- Scheduled: Every Sunday at 6 AM UTC
- Manual: workflow_dispatch with options

**Steps:**
1. Checkout repository with LFS
2. Setup Python environment
3. Install dependencies
4. Run data update script
5. Check for new events
6. Update prediction accuracy
7. Check if retraining needed
8. Retrain models (if needed)
9. Generate upcoming predictions
10. Commit and push changes

## 10.2 Manual Update Script

**File:** `scripts/manual_update.py`

**Usage:**
```bash
# Full update
python scripts/manual_update.py

# Data only (no predictions)
python scripts/manual_update.py --data-only

# Force retrain
python scripts/manual_update.py --force-retrain

# Dry run
python scripts/manual_update.py --dry-run
```

---

# 11. UI/UX SPECIFICATIONS

## 11.1 Theme Configuration

```toml
# .streamlit/config.toml

[theme]
primaryColor = "#D20A0A"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#1A1A2E"
textColor = "#FFFFFF"
font = "sans serif"
```

## 11.2 Color Palette

```python
# config.py

COLORS = {
    'primary': '#D20A0A',        # UFC Red
    'secondary': '#000000',       # Black
    'background': '#0E1117',      # Dark background
    'card_bg': '#1A1A2E',         # Card background
    'success': '#28A745',         # Green (wins, high confidence)
    'warning': '#FFC107',         # Yellow (medium confidence)
    'danger': '#DC3545',          # Red (losses, low confidence)
    'info': '#17A2B8',            # Blue (info)
    'text_primary': '#FFFFFF',    # White text
    'text_secondary': '#B0B0B0',  # Gray text
    'text_muted': '#6C757D',      # Muted text
}
```

## 11.3 Confidence Level Styling

| Level | Confidence | Color | Icon |
|-------|------------|-------|------|
| High | > 65% | Green (#28A745) | 🟢 |
| Medium | 55-65% | Yellow (#FFC107) | 🟡 |
| Low | < 55% | Orange (#FD7E14) | 🟠 |

**Low Confidence Warning:**
```python
if confidence < 0.55:
    st.warning("""
    ⚠️ **Low Confidence Prediction**
    
    This matchup is difficult to predict with high certainty.
    Our model shows only {:.0%} confidence, suggesting this 
    fight could go either way.
    """.format(confidence))
```

## 11.4 Responsive Components

```python
# components/fighter_card.py

def render_fighter_card(fighter: dict, show_actions: bool = True):
    """Render a fighter profile card."""
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Fighter image
        if fighter.get('image_url'):
            st.image(fighter['image_url'], width=150)
        else:
            st.image("assets/images/placeholder.png", width=150)
    
    with col2:
        # Fighter info
        st.subheader(fighter['name'])
        if fighter.get('nickname'):
            st.caption(f'"{fighter["nickname"]}"')
        
        # Record
        record = f"{fighter['wins']}-{fighter['losses']}-{fighter['draws']}"
        st.metric("Record", record)
        
        # Details
        details = []
        if fighter.get('height_cm'):
            details.append(f"Height: {fighter['height_cm']} cm")
        if fighter.get('reach_cm'):
            details.append(f"Reach: {fighter['reach_cm']} cm")
        st.text(" | ".join(details))
    
    if show_actions:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⚔️ Compare", key=f"compare_{fighter['fighter_id']}"):
                st.session_state.compare_fighter = fighter['fighter_id']
                st.switch_page("pages/2_⚔️_Fighter_Comparison.py")
        with col2:
            if st.button("🎯 Predict", key=f"predict_{fighter['fighter_id']}"):
                st.session_state.predict_fighter = fighter['fighter_id']
                st.switch_page("pages/3_🎯_Predictions.py")
```

---

# 12. CONFIGURATION

## 12.1 Configuration File

```python
# config.py

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# === Paths ===
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DATABASE_PATH = DATA_DIR / "database" / "ufc_database.db"

# === API Keys ===
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# === App Settings ===
APP_ENV = os.getenv("APP_ENV", "development")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# === LLM Settings ===
LLM_ENABLED = os.getenv("LLM_ENABLED", "true").lower() == "true"
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1000"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))

# === Model Settings ===
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1.0.0")
MIN_CONFIDENCE_THRESHOLD = float(os.getenv("MIN_CONFIDENCE_THRESHOLD", "0.55"))
MAX_MODEL_VERSIONS = int(os.getenv("MAX_MODEL_VERSIONS", "3"))

# === Update Settings ===
AUTO_UPDATE_ENABLED = os.getenv("AUTO_UPDATE_ENABLED", "true").lower() == "true"

# === UI Theme ===
COLORS = {
    'primary': '#D20A0A',
    'secondary': '#000000',
    'background': '#0E1117',
    'card_bg': '#1A1A2E',
    'success': '#28A745',
    'warning': '#FFC107',
    'danger': '#DC3545',
    'info': '#17A2B8',
    'text_primary': '#FFFFFF',
    'text_secondary': '#B0B0B0',
}

# === Weight Classes ===
WEIGHT_CLASSES = [
    "Strawweight",
    "Flyweight", 
    "Bantamweight",
    "Featherweight",
    "Lightweight",
    "Welterweight",
    "Middleweight",
    "Light Heavyweight",
    "Heavyweight",
    "Women's Strawweight",
    "Women's Flyweight",
    "Women's Bantamweight",
    "Women's Featherweight",
]
```

## 12.2 Environment Template

```bash
# .env.template

# === REQUIRED ===
GROQ_API_KEY=gsk_your_api_key_here

# === OPTIONAL ===
# KAGGLE_USERNAME=your_username
# KAGGLE_KEY=your_key

# === APP SETTINGS ===
APP_ENV=development
DEBUG=true

# === LLM SETTINGS ===
LLM_ENABLED=true
LLM_MODEL=llama-3.3-70b-versatile
LLM_MAX_TOKENS=1000
LLM_TEMPERATURE=0.7

# === MODEL SETTINGS ===
MIN_CONFIDENCE_THRESHOLD=0.55
MAX_MODEL_VERSIONS=3
```

---

# 13. IMPLEMENTATION PHASES

## Phase 1: Foundation (Days 1-3)
- [ ] Create project structure
- [ ] Setup configuration and environment
- [ ] Implement database schema (`scripts/init_database.py`)
- [ ] Create `DataService` with basic CRUD operations
- [ ] Download and parse Kaggle data
- [ ] Populate database with historical data

## Phase 2: Scraper (Days 4-5)
- [ ] Implement `UFCStatsScraper`
- [ ] Parse fighter pages
- [ ] Parse event pages
- [ ] Parse fight statistics
- [ ] Add rate limiting and error handling
- [ ] Test with sample pages

## Phase 3: ML Pipeline (Days 6-9)
- [ ] Implement feature engineering pipeline
- [ ] Create training data preparation
- [ ] Train winner prediction ensemble
- [ ] Train method classifier
- [ ] Train round predictor
- [ ] Implement SHAP explainability
- [ ] Save models with versioning
- [ ] Backfill historical predictions

## Phase 4: Core Services (Days 10-11)
- [ ] Implement `PredictionService`
- [ ] Implement `LLMService` with Groq
- [ ] Implement `AccuracyService`
- [ ] Implement `CacheService`
- [ ] Add fallback behaviors

## Phase 5: UI - Basic Pages (Days 12-15)
- [ ] Home page (`app.py`)
- [ ] Fighters page with search
- [ ] Fighter Comparison page
- [ ] Predictions page
- [ ] How It Works page

## Phase 6: UI - Advanced Pages (Days 16-18)
- [ ] Upcoming Bouts page
- [ ] Trends & Facts page (Dashboard tab)
- [ ] Trends & Facts page (Encyclopedia tab)
- [ ] Trends & Facts page (Q&A tab)

## Phase 7: Polish (Days 19-21)
- [ ] Apply UFC theme styling
- [ ] Add loading states and spinners
- [ ] Implement error handling UI
- [ ] Add LLM unavailable states
- [ ] Create reusable components

## Phase 8: Automation (Days 22-23)
- [ ] Create manual update script
- [ ] Setup GitHub Actions workflow
- [ ] Test update pipeline
- [ ] Add model cleanup

## Phase 9: Testing & Docs (Days 24-25)
- [ ] Write unit tests for services
- [ ] Write integration tests
- [ ] Create SETUP.md
- [ ] Create DEPLOYMENT.md
- [ ] Final review and cleanup

---

# 14. TESTING REQUIREMENTS

## 14.1 Unit Tests

```python
# tests/test_data_service.py

def test_get_fighter_by_id():
    service = DataService(":memory:")
    # Insert test fighter
    fighter = service.get_fighter_by_id(1)
    assert fighter is not None
    assert fighter['name'] == "Test Fighter"

def test_search_fighters():
    service = DataService(":memory:")
    results = service.search_fighters("Khabib")
    assert len(results) > 0

# tests/test_prediction_service.py

def test_prediction_output_format():
    service = PredictionService()
    result = service.predict(fighter_a_id=1, fighter_b_id=2)
    assert 'predicted_winner_id' in result
    assert 'winner_confidence' in result
    assert 0 <= result['winner_confidence'] <= 1

def test_confidence_levels():
    service = PredictionService()
    assert service.get_confidence_level(0.70) == 'high'
    assert service.get_confidence_level(0.60) == 'medium'
    assert service.get_confidence_level(0.50) == 'low'
```

## 14.2 Integration Tests

```python
# tests/test_integration.py

def test_full_prediction_flow():
    """Test complete prediction flow from search to result."""
    data_service = DataService()
    prediction_service = PredictionService()
    
    # Search for fighters
    fighters = data_service.search_fighters("Jones")
    assert len(fighters) > 0
    
    fighter_a = fighters[0]
    fighter_b = data_service.search_fighters("Aspinall")[0]
    
    # Generate prediction
    result = prediction_service.predict(
        fighter_a['fighter_id'],
        fighter_b['fighter_id']
    )
    
    # Validate result
    assert result.predicted_winner_id in [fighter_a['fighter_id'], fighter_b['fighter_id']]
    assert result.winner_confidence > 0
```

---

# APPENDIX A: Sample Prompts for LLM

## Fighter Analysis Prompt

```
You are a UFC analyst. Analyze this fighter based on their statistics:

Fighter: {name}
Record: {wins}-{losses}-{draws}
Division: {weight_class}

Statistics:
- Significant Strikes Landed/Min: {sig_str_pm}
- Striking Accuracy: {str_acc}%
- Takedowns/15min: {td_avg}
- Takedown Defense: {td_def}%
- Submission Average: {sub_avg}

Recent Fights:
{recent_fights}

Provide a 2-3 paragraph analysis covering:
1. Fighting style and strengths
2. Areas of concern or weaknesses
3. What type of opponent gives them trouble

Keep the tone professional and analytical.
```

## Prediction Explanation Prompt

```
Explain this UFC fight prediction in 2-3 paragraphs:

Fighter A: {fighter_a_name} ({fighter_a_record})
Fighter B: {fighter_b_name} ({fighter_b_record})

Prediction: {predicted_winner} wins by {predicted_method} in Round {predicted_round}
Confidence: {confidence}%

Top factors influencing this prediction:
{top_factors}

Explain WHY these factors matter and how they translate to the predicted outcome.
Be specific about the stylistic matchup.
```

---

# APPENDIX B: Git LFS Configuration

```gitattributes
# .gitattributes

# Database files
data/database/*.db filter=lfs diff=lfs merge=lfs -text

# Model files  
models/trained/versions/*.pkl filter=lfs diff=lfs merge=lfs -text
models/trained/versions/*.joblib filter=lfs diff=lfs merge=lfs -text

# Large data files
data/raw/**/*.csv filter=lfs diff=lfs merge=lfs -text
data/processed/*.csv filter=lfs diff=lfs merge=lfs -text
```

**Setup Commands:**
```bash
git lfs install
git lfs track "*.db"
git lfs track "*.pkl"
git add .gitattributes
git commit -m "Configure Git LFS"
```

---

# APPENDIX C: Streamlit Cloud Deployment

1. Push code to GitHub repository
2. Go to share.streamlit.io
3. Connect GitHub account
4. Select repository and branch
5. Set main file: `app.py`
6. Add secrets in Settings → Secrets:
   ```toml
   GROQ_API_KEY = "gsk_xxx"
   ```
7. Deploy

---

**END OF TECHNICAL DESIGN DOCUMENT**
