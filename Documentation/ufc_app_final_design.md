# UFC Fighter Analysis & Prediction App
## Final Project Design Document

---

## 1. Executive Summary

A comprehensive UFC Fighter Analysis and Prediction application built with Streamlit, featuring fighter profiles, ML-based fight predictions with explainability, upcoming bout analysis, data-driven trends discovery, and fighter comparison tools. The app will be deployed publicly on Streamlit Cloud with UFC-inspired theming.

---

## 2. Final Design Decisions

| Decision | Choice | Notes |
|----------|--------|-------|
| Historical Data | All UFC data (1993-present) | Complete dataset |
| Fighter Images | External URL linking | Computationally efficient |
| Data Updates | Hybrid (auto + manual) | After each event |
| Model Retraining | Hybrid (event updates + quarterly full retrain) | Or when accuracy < 55% |
| Confidence Display | Show all predictions with low-confidence warnings | User transparency |
| Accuracy Tracking | Backfill historical + ongoing tracking | From deployment |
| User Accounts | None | Stateless public app |
| Comparison Page | Standalone 6th page | Dedicated functionality |
| Weight Class Filter | All systems | Consistent filtering |
| Mobile Support | Desktop-first | Mobile not required initially |
| LLM Fallback | Hide features with declaration | Graceful degradation |
| Deployment | Streamlit Cloud (public) | Low traffic expected |
| Theme | UFC-inspired (red/black) | Official colors |
| Error Handling | Cached data + warning banner | User-friendly |
| Upcoming Events | Official announcements only | Reliable data |

---

## 3. Application Pages

### Page Structure (6 Pages)

```
┌─────────────────────────────────────────────────────────────────┐
│                     UFC PREDICTION APP                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  🏠 Home                                                         │
│  │                                                               │
│  ├── 🥊 Fighters           (Search, profiles, stats)            │
│  │                                                               │
│  ├── ⚔️ Fighter Comparison  (Head-to-head analysis)             │
│  │                                                               │
│  ├── 🎯 Predictions         (Custom matchup predictions)        │
│  │                                                               │
│  ├── 📅 Upcoming Bouts      (Official cards + predictions)      │
│  │                                                               │
│  ├── 📊 Trends & Facts      (Analytics dashboard + Q&A)         │
│  │                                                               │
│  └── ❓ How It Works        (Data, methodology, limitations)    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Data Architecture

### 4.1 Data Sources

| Source | URL | Data Type | Usage |
|--------|-----|-----------|-------|
| **UFCStats.com** | http://ufcstats.com | Live scraping | Primary source for all data |
| **Kaggle UFC Dataset** | kaggle.com/datasets/rajeevw/ufcdata | CSV download | Initial historical data + training |
| **Kaggle UFC 2024** | kaggle.com/datasets/maksbasher/ufc-complete-dataset | CSV download | Extended historical data |

### 4.2 Data Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                      DATA PIPELINE                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  SOURCES                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │   Kaggle     │  │  UFCStats    │  │   UFC.com Images     │   │
│  │  Historical  │  │   Scraper    │  │   (External URLs)    │   │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘   │
│         │                 │                      │               │
│         │                 │                      │               │
│         ▼                 ▼                      │               │
│  ┌─────────────────────────────────┐            │               │
│  │     Data Cleaning & Merging     │            │               │
│  │  • Deduplication                │            │               │
│  │  • Missing value handling       │            │               │
│  │  • Feature standardization      │            │               │
│  └──────────────┬──────────────────┘            │               │
│                 │                               │               │
│                 ▼                               │               │
│  ┌─────────────────────────────────┐            │               │
│  │       SQLite Database           │◄───────────┘               │
│  │  • fighters                     │  (image URLs stored)       │
│  │  • fighter_stats                │                            │
│  │  • events                       │                            │
│  │  • fights                       │                            │
│  │  • fight_stats                  │                            │
│  │  • upcoming_fights              │                            │
│  │  • predictions                  │                            │
│  │  • prediction_accuracy          │                            │
│  └─────────────────────────────────┘                            │
│                                                                  │
│  UPDATE MECHANISMS                                               │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  AUTOMATIC: Scheduled check after UFC events                ││
│  │  MANUAL: Admin refresh button in app                        ││
│  │  TRIGGER: Detects new completed events → scrapes → updates  ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.3 Database Schema

```sql
-- Core Tables
----------------------------------------------------------------------

-- Fighters table
CREATE TABLE fighters (
    fighter_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    nickname TEXT,
    height_cm REAL,
    weight_kg REAL,
    reach_cm REAL,
    stance TEXT,  -- Orthodox, Southpaw, Switch
    dob DATE,
    nationality TEXT,
    team TEXT,
    wins INTEGER DEFAULT 0,
    losses INTEGER DEFAULT 0,
    draws INTEGER DEFAULT 0,
    no_contests INTEGER DEFAULT 0,
    image_url TEXT,  -- External UFC.com URL
    ufc_stats_url TEXT,  -- Link to UFCStats profile
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(name, dob)
);

-- Fighter career statistics (aggregated)
CREATE TABLE fighter_stats (
    stat_id INTEGER PRIMARY KEY AUTOINCREMENT,
    fighter_id INTEGER NOT NULL,
    -- Striking
    sig_strikes_landed_per_min REAL,
    sig_strikes_absorbed_per_min REAL,
    sig_strike_accuracy REAL,  -- Percentage
    sig_strike_defense REAL,   -- Percentage
    -- Grappling
    takedowns_avg_per_15min REAL,
    takedown_accuracy REAL,    -- Percentage
    takedown_defense REAL,     -- Percentage
    submissions_avg_per_15min REAL,
    -- Calculated
    avg_fight_time_seconds INTEGER,
    finish_rate REAL,
    decision_rate REAL,
    ko_rate REAL,
    submission_rate REAL,
    -- Meta
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (fighter_id) REFERENCES fighters(fighter_id)
);

-- Events table
CREATE TABLE events (
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
CREATE TABLE fights (
    fight_id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id INTEGER NOT NULL,
    fighter_red_id INTEGER NOT NULL,  -- Red corner
    fighter_blue_id INTEGER NOT NULL, -- Blue corner
    winner_id INTEGER,  -- NULL for draws/NC
    weight_class TEXT NOT NULL,
    is_title_fight BOOLEAN DEFAULT FALSE,
    is_main_event BOOLEAN DEFAULT FALSE,
    method TEXT,  -- KO/TKO, Submission, Decision, etc.
    method_detail TEXT,  -- Rear Naked Choke, Unanimous, etc.
    round INTEGER,
    time TEXT,  -- "4:35"
    referee TEXT,
    bonus TEXT,  -- POTN, FOTN, etc.
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (event_id) REFERENCES events(event_id),
    FOREIGN KEY (fighter_red_id) REFERENCES fighters(fighter_id),
    FOREIGN KEY (fighter_blue_id) REFERENCES fighters(fighter_id),
    FOREIGN KEY (winner_id) REFERENCES fighters(fighter_id)
);

-- Per-fight statistics
CREATE TABLE fight_stats (
    stat_id INTEGER PRIMARY KEY AUTOINCREMENT,
    fight_id INTEGER NOT NULL,
    fighter_id INTEGER NOT NULL,
    corner TEXT,  -- 'red' or 'blue'
    -- Strikes
    knockdowns INTEGER DEFAULT 0,
    sig_strikes_landed INTEGER DEFAULT 0,
    sig_strikes_attempted INTEGER DEFAULT 0,
    sig_strikes_head_landed INTEGER DEFAULT 0,
    sig_strikes_body_landed INTEGER DEFAULT 0,
    sig_strikes_leg_landed INTEGER DEFAULT 0,
    total_strikes_landed INTEGER DEFAULT 0,
    total_strikes_attempted INTEGER DEFAULT 0,
    -- Grappling
    takedowns_landed INTEGER DEFAULT 0,
    takedowns_attempted INTEGER DEFAULT 0,
    submissions_attempted INTEGER DEFAULT 0,
    reversals INTEGER DEFAULT 0,
    control_time_seconds INTEGER DEFAULT 0,
    FOREIGN KEY (fight_id) REFERENCES fights(fight_id),
    FOREIGN KEY (fighter_id) REFERENCES fighters(fighter_id)
);

-- Upcoming fights (official announcements only)
CREATE TABLE upcoming_fights (
    upcoming_id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id INTEGER NOT NULL,
    fighter_red_id INTEGER NOT NULL,
    fighter_blue_id INTEGER NOT NULL,
    weight_class TEXT NOT NULL,
    is_main_event BOOLEAN DEFAULT FALSE,
    is_title_fight BOOLEAN DEFAULT FALSE,
    card_position TEXT,  -- 'main_card', 'prelims', 'early_prelims'
    bout_order INTEGER,  -- Order on card
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (event_id) REFERENCES events(event_id),
    FOREIGN KEY (fighter_red_id) REFERENCES fighters(fighter_id),
    FOREIGN KEY (fighter_blue_id) REFERENCES fighters(fighter_id)
);

-- Predictions & Tracking Tables
----------------------------------------------------------------------

-- Stored predictions (for accuracy tracking)
CREATE TABLE predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    fight_id INTEGER,  -- NULL for upcoming fights
    upcoming_id INTEGER,  -- NULL for historical
    fighter_red_id INTEGER NOT NULL,
    fighter_blue_id INTEGER NOT NULL,
    -- Winner prediction
    predicted_winner_id INTEGER,
    winner_confidence REAL,  -- 0.0 to 1.0
    -- Method prediction
    method_ko_prob REAL,
    method_sub_prob REAL,
    method_dec_prob REAL,
    predicted_method TEXT,
    -- Round prediction
    predicted_round REAL,
    -- Explainability
    feature_importance JSON,  -- {"feature": importance, ...}
    top_factors JSON,  -- ["factor1", "factor2", ...]
    -- Model metadata
    model_version TEXT,
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (fight_id) REFERENCES fights(fight_id),
    FOREIGN KEY (upcoming_id) REFERENCES upcoming_fights(upcoming_id)
);

-- Prediction accuracy tracking
CREATE TABLE prediction_accuracy (
    accuracy_id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER NOT NULL,
    fight_id INTEGER NOT NULL,
    -- Actual outcomes
    actual_winner_id INTEGER,
    actual_method TEXT,
    actual_round INTEGER,
    -- Correctness
    winner_correct BOOLEAN,
    method_correct BOOLEAN,
    round_correct BOOLEAN,  -- Within 1 round
    -- Scoring
    confidence_score REAL,  -- How confident was the prediction
    brier_score REAL,  -- Probability calibration metric
    -- Timestamps
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (prediction_id) REFERENCES predictions(prediction_id),
    FOREIGN KEY (fight_id) REFERENCES fights(fight_id)
);

-- Model performance tracking
CREATE TABLE model_performance (
    performance_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_version TEXT NOT NULL,
    model_type TEXT NOT NULL,  -- 'winner', 'method', 'round'
    -- Training info
    training_date DATE,
    training_samples INTEGER,
    -- Metrics
    accuracy REAL,
    precision_score REAL,
    recall_score REAL,
    f1_score REAL,
    auc_roc REAL,
    -- Rolling performance
    rolling_accuracy_30 REAL,  -- Last 30 predictions
    rolling_accuracy_100 REAL, -- Last 100 predictions
    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
----------------------------------------------------------------------
CREATE INDEX idx_fighters_name ON fighters(name);
CREATE INDEX idx_fights_event ON fights(event_id);
CREATE INDEX idx_fights_fighters ON fights(fighter_red_id, fighter_blue_id);
CREATE INDEX idx_fight_stats_fight ON fight_stats(fight_id);
CREATE INDEX idx_predictions_fight ON predictions(fight_id);
CREATE INDEX idx_events_date ON events(date);
CREATE INDEX idx_upcoming_event ON upcoming_fights(event_id);

-- Views for common queries
----------------------------------------------------------------------

-- Fighter records with stats
CREATE VIEW v_fighter_full AS
SELECT 
    f.*,
    fs.sig_strikes_landed_per_min,
    fs.sig_strike_accuracy,
    fs.takedowns_avg_per_15min,
    fs.takedown_accuracy,
    fs.submission_rate,
    fs.finish_rate
FROM fighters f
LEFT JOIN fighter_stats fs ON f.fighter_id = fs.fighter_id;

-- Prediction accuracy summary
CREATE VIEW v_model_accuracy AS
SELECT 
    model_version,
    COUNT(*) as total_predictions,
    SUM(CASE WHEN winner_correct THEN 1 ELSE 0 END) as correct_winners,
    ROUND(100.0 * SUM(CASE WHEN winner_correct THEN 1 ELSE 0 END) / COUNT(*), 2) as winner_accuracy,
    SUM(CASE WHEN method_correct THEN 1 ELSE 0 END) as correct_methods,
    ROUND(100.0 * SUM(CASE WHEN method_correct THEN 1 ELSE 0 END) / COUNT(*), 2) as method_accuracy
FROM prediction_accuracy pa
JOIN predictions p ON pa.prediction_id = p.prediction_id
GROUP BY model_version;
```

---

## 5. Machine Learning Architecture

### 5.1 Model Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    ML PREDICTION SYSTEM                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  INPUT: Fighter A Profile + Fighter B Profile + Context         │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                 FEATURE ENGINEERING                          ││
│  │                                                              ││
│  │  Differential Features:        Ratio Features:               ││
│  │  • height_diff                 • win_rate_ratio              ││
│  │  • reach_diff                  • finish_rate_ratio           ││
│  │  • age_diff                    • experience_ratio            ││
│  │  • strike_accuracy_diff                                      ││
│  │  • takedown_accuracy_diff      Contextual:                   ││
│  │  • defense_diff                • weight_class_encoded        ││
│  │                                • is_title_fight              ││
│  │  Form Features:                • days_since_last_fight       ││
│  │  • recent_win_streak           • career_stage                ││
│  │  • recent_form_score                                         ││
│  │  • momentum_indicator                                        ││
│  └─────────────────────────────────────────────────────────────┘│
│                          │                                       │
│          ┌───────────────┼───────────────┐                      │
│          ▼               ▼               ▼                      │
│   ┌─────────────┐ ┌─────────────┐ ┌─────────────┐              │
│   │   WINNER    │ │   METHOD    │ │   ROUND     │              │
│   │  ENSEMBLE   │ │  CLASSIFIER │ │  PREDICTOR  │              │
│   └──────┬──────┘ └──────┬──────┘ └──────┬──────┘              │
│          │               │               │                      │
│          ▼               ▼               ▼                      │
│   ┌─────────────┐ ┌─────────────┐ ┌─────────────┐              │
│   │ • XGBoost   │ │ • XGBoost   │ │ • XGBoost   │              │
│   │ • LightGBM  │ │ • RandomFor │ │   Regressor │              │
│   │ • RandomFor │ │ • GradBoost │ │ • Ordinal   │              │
│   │ • LogRegres │ │             │ │   Classif.  │              │
│   │             │ │             │ │             │              │
│   │ Weights:    │ │ Weights:    │ │             │              │
│   │ 30/30/20/20 │ │ 35/35/30    │ │             │              │
│   └─────────────┘ └─────────────┘ └─────────────┘              │
│          │               │               │                      │
│          └───────────────┴───────────────┘                      │
│                          │                                       │
│                          ▼                                       │
│   ┌─────────────────────────────────────────────────────────────┐│
│   │                    OUTPUT                                   ││
│   │                                                             ││
│   │  Winner:  Fighter A (67.3% confidence)                     ││
│   │  Method:  Submission (42%) | KO/TKO (31%) | Decision (27%) ││
│   │  Round:   2.4 (Most likely Round 2)                        ││
│   │                                                             ││
│   │  Confidence Level: ████████░░ HIGH (>60%)                  ││
│   │                                                             ││
│   │  Top Factors:                                               ││
│   │  1. Takedown differential (+3.2/15min)                     ││
│   │  2. Submission rate advantage (+18%)                       ││
│   │  3. Experience (12 more UFC fights)                        ││
│   └─────────────────────────────────────────────────────────────┘│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Feature Engineering Details

```python
# Feature Categories

DIFFERENTIAL_FEATURES = [
    'height_diff',           # Fighter A height - Fighter B height (cm)
    'reach_diff',            # Fighter A reach - Fighter B reach (cm)
    'age_diff',              # Fighter A age - Fighter B age (years)
    'sig_str_acc_diff',      # Striking accuracy differential
    'sig_str_def_diff',      # Striking defense differential
    'td_acc_diff',           # Takedown accuracy differential
    'td_def_diff',           # Takedown defense differential
    'str_landed_pm_diff',    # Strikes landed per minute differential
    'str_absorbed_pm_diff',  # Strikes absorbed per minute differential
    'td_avg_diff',           # Takedowns per 15min differential
    'sub_avg_diff',          # Submissions per 15min differential
]

RATIO_FEATURES = [
    'win_rate_ratio',        # Fighter A win% / Fighter B win%
    'finish_rate_ratio',     # Finish rate comparison
    'ko_rate_ratio',         # KO rate comparison
    'sub_rate_ratio',        # Submission rate comparison
    'experience_ratio',      # Total fights comparison
    'ufc_experience_ratio',  # UFC fights comparison
]

FORM_FEATURES = [
    'win_streak_a',          # Current win streak
    'win_streak_b',
    'lose_streak_a',         # Current losing streak
    'lose_streak_b',
    'recent_form_a',         # Weighted score of last 5 fights
    'recent_form_b',
    'days_since_fight_a',    # Ring rust indicator
    'days_since_fight_b',
    'momentum_a',            # Win/loss trajectory
    'momentum_b',
]

CONTEXTUAL_FEATURES = [
    'weight_class_encoded',  # One-hot or ordinal encoding
    'is_title_fight',        # Boolean
    'is_main_event',         # Boolean
    'rounds_scheduled',      # 3 or 5
]

STYLE_FEATURES = [
    'striker_vs_grappler',   # Style matchup indicator
    'orthodox_vs_southpaw',  # Stance matchup
    'pressure_vs_counter',   # Derived from fight patterns
]
```

### 5.3 Model Training Strategy

| Phase | Trigger | Action |
|-------|---------|--------|
| **Initial Training** | App deployment | Train on all historical data |
| **Light Update** | After each UFC event | Add new fights to dataset, no retrain |
| **Full Retrain** | Quarterly OR accuracy < 55% | Complete model retraining |
| **Version Control** | Each retrain | Save model version, track performance |

### 5.4 Backfill Strategy for Historical Accuracy

```
┌─────────────────────────────────────────────────────────────────┐
│              HISTORICAL ACCURACY BACKFILL                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Step 1: Time-Based Split                                        │
│  ───────────────────────────────────────────────────────────────│
│  • Split data chronologically (not random)                      │
│  • For each fight, only use data available BEFORE that fight    │
│  • Prevents data leakage                                        │
│                                                                  │
│  Step 2: Walk-Forward Validation                                 │
│  ───────────────────────────────────────────────────────────────│
│  • Train on fights 1-1000                                       │
│  • Predict fight 1001, record accuracy                          │
│  • Train on fights 1-1001                                       │
│  • Predict fight 1002, record accuracy                          │
│  • Continue through all historical fights                       │
│                                                                  │
│  Step 3: Store Backfilled Predictions                           │
│  ───────────────────────────────────────────────────────────────│
│  • Save predictions with model version "backfill_v1"            │
│  • Calculate historical accuracy metrics                        │
│  • Display in "How It Works" page                               │
│                                                                  │
│  Expected Output:                                                │
│  • "Our model has been tested on 5,000+ historical fights"      │
│  • "Winner prediction accuracy: 62.3%"                          │
│  • "Method prediction accuracy: 48.7%"                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.5 Confidence Thresholds

| Confidence Level | Range | Display | Color |
|------------------|-------|---------|-------|
| **High** | > 65% | Normal display | Green |
| **Medium** | 55-65% | Normal display | Yellow |
| **Low** | < 55% | ⚠️ Warning banner | Orange |
| **Very Low** | < 50% | ⚠️ "Toss-up" disclaimer | Red |

Low confidence warning example:
```
⚠️ Low Confidence Prediction
This matchup is difficult to predict with high certainty. 
Our model shows only 52% confidence, suggesting this fight 
could go either way. Key factors are closely matched.
```

---

## 6. Page Designs

### 6.1 Home Page

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│              ███╗   ██╗███████╗ ██████╗                         │
│              ████╗  ██║██╔════╝██╔════╝                         │
│              ██╔██╗ ██║█████╗  ██║                              │
│              ██║╚██╗██║██╔══╝  ██║                              │
│              ██║ ╚████║██║     ╚██████╗                         │
│              ╚═╝  ╚═══╝╚═╝      ╚═════╝                         │
│                                                                  │
│                 UFC FIGHT PREDICTOR                              │
│           Data-Driven MMA Analysis & Predictions                 │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  🥊 FIGHTERS    │  │  🎯 PREDICT     │  │  📅 UPCOMING    │  │
│  │                 │  │                 │  │                 │  │
│  │  Search and     │  │  Create custom  │  │  View official  │  │
│  │  explore UFC    │  │  matchups and   │  │  fight cards    │  │
│  │  fighter        │  │  get AI-powered │  │  with auto      │  │
│  │  profiles       │  │  predictions    │  │  predictions    │  │
│  │                 │  │                 │  │                 │  │
│  │  [Explore →]    │  │  [Predict →]    │  │  [View →]       │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
│                                                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  ⚔️ COMPARE     │  │  📊 TRENDS      │  │  ❓ HOW IT      │  │
│  │                 │  │                 │  │     WORKS       │  │
│  │  Head-to-head   │  │  Discover       │  │                 │  │
│  │  fighter        │  │  patterns and   │  │  Learn about    │  │
│  │  comparison     │  │  interesting    │  │  our data and   │  │
│  │  tool           │  │  UFC facts      │  │  methodology    │  │
│  │                 │  │                 │  │                 │  │
│  │  [Compare →]    │  │  [Explore →]    │  │  [Learn →]      │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  📊 QUICK STATS                                                  │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌────────────┐ │
│  │   4,200+    │ │   7,500+    │ │    650+     │ │   62.3%    │ │
│  │  Fighters   │ │   Fights    │ │   Events    │ │  Accuracy  │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └────────────┘ │
│                                                                  │
│  🔥 NEXT UFC EVENT                                               │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  UFC 315: JONES VS ASPINALL                                 ││
│  │  📍 Las Vegas | 🗓 March 15, 2025 | ⏰ 7 days                ││
│  │  Main Event: Jon Jones (C) vs Tom Aspinall                  ││
│  │  [View Full Card →]                                         ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  📈 RECENT PREDICTION PERFORMANCE                                │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  Last Event: UFC 314 | Correct: 8/12 (66.7%)               ││
│  │  Last 30 Days: 45/72 correct (62.5%)                        ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Fighters Page

```
┌─────────────────────────────────────────────────────────────────┐
│  🥊 FIGHTERS                                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────┐                    │
│  │ 🔍 Search fighter...                    │ [Search]           │
│  └─────────────────────────────────────────┘                    │
│                                                                  │
│  Filter: [All Weight Classes ▼] [All Countries ▼] [Active ▼]   │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                                                              ││
│  │  ┌──────────────┐  KHABIB NURMAGOMEDOV                      ││
│  │  │              │  "The Eagle" 🇷🇺                           ││
│  │  │   [Photo     │  ────────────────────────────────────     ││
│  │  │    from      │  Record: 29-0-0 (Retired)                 ││
│  │  │    UFC.com]  │  Division: Lightweight                    ││
│  │  │              │  Height: 5'10" (178 cm) | Reach: 70"      ││
│  │  │              │  Age: 35 | Stance: Orthodox               ││
│  │  └──────────────┘  Team: American Kickboxing Academy        ││
│  │                                                              ││
│  │  [⚔️ Compare] [🎯 Predict Matchup]                          ││
│  │                                                              ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  ═══════════════════════════════════════════════════════════    │
│                                                                  │
│  📊 CAREER STATISTICS                                            │
│  ┌──────────────────────────┐ ┌────────────────────────────────┐│
│  │  STRIKING                │ │  GRAPPLING                     ││
│  │  ────────────────────    │ │  ────────────────────          ││
│  │  Sig. Str. Landed: 4.10/m│ │  Takedowns: 5.32/15min        ││
│  │  Sig. Str. Accuracy: 53% │ │  Takedown Acc: 48%            ││
│  │  Sig. Str. Absorbed: 2.2 │ │  Takedown Def: 84%            ││
│  │  Sig. Str. Defense: 51%  │ │  Submissions: 0.6/15min       ││
│  └──────────────────────────┘ └────────────────────────────────┘│
│                                                                  │
│  🏆 WIN METHODS                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  [Pie Chart: KO/TKO: 8, Submission: 11, Decision: 10]       ││
│  │                                                              ││
│  │  🥇 Top Method: SUBMISSION (38%)                            ││
│  │  Most Common: Rear Naked Choke (4 wins)                     ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  📜 FIGHT HISTORY                                                │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  UFC 254 | Oct 24, 2020                                     ││
│  │  ✅ WIN vs Justin Gaethje                                   ││
│  │  Submission (Triangle Choke) | R2, 1:34                     ││
│  │  🏆 Lightweight Title Defense                               ││
│  │  ─────────────────────────────────────────────────────────  ││
│  │  UFC 242 | Sep 7, 2019                                      ││
│  │  ✅ WIN vs Dustin Poirier                                   ││
│  │  Submission (Rear Naked Choke) | R3, 2:06                   ││
│  │  🏆 Lightweight Title Defense                               ││
│  │  ─────────────────────────────────────────────────────────  ││
│  │  [Load More...]                                             ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  🤖 AI ANALYSIS                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  Khabib Nurmagomedov retired as one of the most dominant    ││
│  │  fighters in UFC history. His relentless pressure,          ││
│  │  suffocating grappling, and cardio allowed him to...        ││
│  │  [Read More]                                                ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.3 Fighter Comparison Page (NEW)

```
┌─────────────────────────────────────────────────────────────────┐
│  ⚔️ FIGHTER COMPARISON                                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────┐     ┌─────────────────────────┐   │
│  │ Fighter A               │     │ Fighter B               │   │
│  │ ┌─────────────────────┐ │     │ ┌─────────────────────┐ │   │
│  │ │ Search...           │ │     │ │ Search...           │ │   │
│  │ └─────────────────────┘ │     │ └─────────────────────┘ │   │
│  │ Selected: Jon Jones    │     │ Selected: Tom Aspinall  │   │
│  └─────────────────────────┘     └─────────────────────────┘   │
│                                                                  │
│  Weight Class Filter: [Heavyweight ▼]                           │
│                                                                  │
│                    [⚔️ COMPARE FIGHTERS]                        │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  HEAD-TO-HEAD COMPARISON                                         │
│  ═══════════════════════════════════════════════════════════    │
│                                                                  │
│        JON JONES          STAT           TOM ASPINALL           │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                                                              ││
│  │  ┌────────┐                              ┌────────┐         ││
│  │  │ [Photo]│         VS                   │ [Photo]│         ││
│  │  └────────┘                              └────────┘         ││
│  │                                                              ││
│  │   27-1-0           RECORD            15-3-0                 ││
│  │   "Bones"         NICKNAME           "The Gunner"          ││
│  │   37 years          AGE              31 years               ││
│  │   6'4" (193cm)     HEIGHT            6'5" (196cm)          ││
│  │   84.5" (215cm)    REACH             80" (203cm)           ││
│  │   248 lbs          WEIGHT            260 lbs               ││
│  │   Orthodox         STANCE            Orthodox              ││
│  │                                                              ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  STRIKING COMPARISON                                             │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                                                              ││
│  │  Sig. Strikes/Min                                           ││
│  │  Jones: 4.31 ████████████████████░░░░░░ Aspinall: 6.53     ││
│  │                                                              ││
│  │  Striking Accuracy                                          ││
│  │  Jones: 57% ████████████████████████░░░ Aspinall: 63%      ││
│  │                                                              ││
│  │  Striking Defense                                           ││
│  │  Jones: 60% ██████████████████████████░ Aspinall: 55%      ││
│  │                                                              ││
│  │  Strikes Absorbed/Min                                       ││
│  │  Jones: 2.14 ████████░░░░░░░░░░░░░░░░░░ Aspinall: 2.88     ││
│  │                                                              ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  GRAPPLING COMPARISON                                            │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                                                              ││
│  │  Takedowns/15min                                            ││
│  │  Jones: 1.87 ████████████████████████░░ Aspinall: 0.87     ││
│  │                                                              ││
│  │  Takedown Accuracy                                          ││
│  │  Jones: 44% ██████████████████░░░░░░░░░ Aspinall: 40%      ││
│  │                                                              ││
│  │  Takedown Defense                                           ││
│  │  Jones: 94% ██████████████████████████░ Aspinall: 73%      ││
│  │                                                              ││
│  │  Submissions/15min                                          ││
│  │  Jones: 0.4 ██████████████████░░░░░░░░░ Aspinall: 0.6      ││
│  │                                                              ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  WIN METHOD COMPARISON                                           │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  [Side-by-side pie charts showing KO/Sub/Dec breakdown]     ││
│  │                                                              ││
│  │  Jones: KO 37% | Sub 26% | Dec 37%                         ││
│  │  Aspinall: KO 73% | Sub 13% | Dec 7%                       ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  EDGE SUMMARY                                                    │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  ✅ Jones has the edge in: Experience, Takedown Defense,    ││
│  │     Striking Defense, Cardio                                ││
│  │  ✅ Aspinall has the edge in: Power, Finishing Rate,        ││
│  │     Striking Volume, Youth                                  ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  [🎯 Get Prediction for This Matchup]                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.4 Predictions Page

```
┌─────────────────────────────────────────────────────────────────┐
│  🎯 FIGHT PREDICTION                                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  CREATE A MATCHUP                                                │
│  ┌─────────────────────────┐   VS   ┌─────────────────────────┐│
│  │ Red Corner              │        │ Blue Corner             ││
│  │ ┌─────────────────────┐ │        │ ┌─────────────────────┐ ││
│  │ │ Search fighter...   │ │        │ │ Search fighter...   │ ││
│  │ └─────────────────────┘ │        │ └─────────────────────┘ ││
│  │                         │        │                         ││
│  │ ✓ Jon Jones            │        │ ✓ Tom Aspinall          ││
│  │   Record: 27-1         │        │   Record: 15-3          ││
│  │   [Photo]              │        │   [Photo]               ││
│  └─────────────────────────┘        └─────────────────────────┘│
│                                                                  │
│  [Weight Class: Heavyweight ▼]  [☑ Title Fight] [☐ Main Event] │
│                                                                  │
│                    [🔮 GENERATE PREDICTION]                     │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  📊 PREDICTION RESULTS                                          │
│  ═══════════════════════════════════════════════════════════   │
│                                                                  │
│  PREDICTED WINNER                                                │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                                                              ││
│  │   JON JONES                                                 ││
│  │   ████████████████████████████░░░░░░░░░░░░ 68.3%           ││
│  │                                                              ││
│  │   TOM ASPINALL                                              ││
│  │   ░░░░░░░░░░░░██████████████████░░░░░░░░░░ 31.7%           ││
│  │                                                              ││
│  │   Confidence Level: 🟢 HIGH                                 ││
│  │                                                              ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  ┌──────────────────────────┐ ┌────────────────────────────────┐│
│  │  METHOD OF VICTORY       │ │  PREDICTED ROUND              ││
│  │  ─────────────────────   │ │  ─────────────────────        ││
│  │                          │ │                               ││
│  │  🤼 Submission: 42%      │ │  Expected: Round 2.8          ││
│  │  🥊 KO/TKO:     31%      │ │                               ││
│  │  📋 Decision:   27%      │ │  Most Likely: Round 3         ││
│  │                          │ │  [████████████████░░░░]       ││
│  │  Likely: Rear Naked      │ │                               ││
│  │          Choke           │ │  Goes Distance: 27%           ││
│  └──────────────────────────┘ └────────────────────────────────┘│
│                                                                  │
│  🔍 PREDICTION EXPLANATION                                       │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                                                              ││
│  │  TOP CONTRIBUTING FACTORS                                   ││
│  │  ────────────────────────────────────────────────────────── ││
│  │                                                              ││
│  │  1. Takedown Defense Differential (+21%)      [██████████]  ││
│  │     Jones: 94% vs Aspinall: 73%                             ││
│  │                                                              ││
│  │  2. Experience Advantage (+12 UFC fights)     [█████████░]  ││
│  │     Jones has faced significantly more elite competition    ││
│  │                                                              ││
│  │  3. Submission Threat Level                   [████████░░]  ││
│  │     Jones has 7 career submissions                          ││
│  │                                                              ││
│  │  4. Cardio/Later Round Performance            [███████░░░]  ││
│  │     Jones historically improves in later rounds             ││
│  │                                                              ││
│  │  [📊 View Full Feature Importance Chart]                    ││
│  │  [📈 View SHAP Analysis]                                    ││
│  │                                                              ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  🤖 AI ANALYSIS                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                                                              ││
│  │  "Jon Jones enters this fight as the favorite primarily    ││
│  │  due to his elite wrestling credentials and championship   ││
│  │  experience. While Aspinall possesses significant knockout ││
│  │  power, Jones's takedown defense (94%) should neutralize   ││
│  │  any grappling exchanges, while his own offensive          ││
│  │  wrestling could put Aspinall in uncomfortable positions.  ││
│  │                                                              ││
│  │  The model predicts a submission finish based on Jones's   ││
│  │  tendency to utilize ground control against younger,       ││
│  │  less experienced opponents. If the fight stays standing,  ││
│  │  Aspinall's power remains a threat, but Jones's counter-   ││
│  │  striking and footwork should limit clean opportunities."  ││
│  │                                                              ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  MODEL INFORMATION                                               │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  Model Version: v2.3.1 | Trained: Jan 15, 2025             ││
│  │  Historical Accuracy: 62.3% (5,234 predictions)            ││
│  │  Recent Performance: 64.1% (last 100 predictions)          ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.5 Upcoming Bouts Page

```
┌─────────────────────────────────────────────────────────────────┐
│  📅 UPCOMING BOUTS                                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Filter: [All Weight Classes ▼]  [All Cards ▼]                  │
│                                                                  │
│  ═══════════════════════════════════════════════════════════    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  🏆 UFC 315: JONES VS ASPINALL                              ││
│  │  📍 T-Mobile Arena, Las Vegas, Nevada                       ││
│  │  🗓 Saturday, March 15, 2025 | 10:00 PM ET                  ││
│  │  ⏰ Countdown: 7 days 14 hours                              ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  MAIN CARD                                                       │
│  ───────────────────────────────────────────────────────────    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  🏆 MAIN EVENT - Heavyweight Championship                   ││
│  │  ┌────────────────────────────────────────────────────────┐ ││
│  │  │                                                        │ ││
│  │  │  JON JONES (C)              TOM ASPINALL               │ ││
│  │  │  27-1-0                  VS 15-3-0                     │ ││
│  │  │  [Photo]                    [Photo]                    │ ││
│  │  │                                                        │ ││
│  │  │  🔮 PREDICTION: Jones by Submission (R3) - 68%         │ ││
│  │  │  Confidence: 🟢 HIGH                                   │ ││
│  │  │                                                        │ ││
│  │  │  [View Detailed Analysis]  [Compare Fighters]          │ ││
│  │  │                                                        │ ││
│  │  └────────────────────────────────────────────────────────┘ ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  CO-MAIN EVENT - Light Heavyweight                          ││
│  │  Alex Pereira (11-2) vs Magomed Ankalaev (19-1)            ││
│  │  🔮 Pereira by KO (R2) - 54%                               ││
│  │  Confidence: 🟡 MEDIUM                                      ││
│  │  [Details]                                                  ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  Bantamweight                                               ││
│  │  Sean O'Malley (18-1) vs Merab Dvalishvili (17-4)          ││
│  │  🔮 Dvalishvili by Decision - 51%                          ││
│  │  Confidence: 🟠 LOW ⚠️ Toss-up fight                       ││
│  │  [Details]                                                  ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  [+ 2 more main card fights...]                                 │
│                                                                  │
│  PRELIMS                                                         │
│  ───────────────────────────────────────────────────────────    │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  Welterweight: Fighter A vs Fighter B | Pred: A - 63%      ││
│  │  Middleweight: Fighter C vs Fighter D | Pred: D - 58%      ││
│  │  [+ 4 more prelim fights...]                               ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  ═══════════════════════════════════════════════════════════    │
│                                                                  │
│  📊 EVENT PREDICTION SUMMARY                                     │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  Total Fights: 12                                           ││
│  │  High Confidence Predictions: 5 (42%)                       ││
│  │  Medium Confidence: 4 (33%)                                 ││
│  │  Low Confidence (Toss-ups): 3 (25%)                         ││
│  │                                                              ││
│  │  [📥 Export All Predictions as CSV]                         ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  ═══════════════════════════════════════════════════════════    │
│                                                                  │
│  🔜 FUTURE EVENTS                                                │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  UFC Fight Night - March 22, 2025 | Card TBA               ││
│  │  UFC 316 - April 5, 2025 | Partial card announced          ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.6 Trends & Facts Page

```
┌─────────────────────────────────────────────────────────────────┐
│  📊 TRENDS & FACTS                                              │
│                                                                  │
│  [Dashboard] [Encyclopedia] [Ask a Question]                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Filter: [All Weight Classes ▼] [All Time ▼] [All Genders ▼]   │
│                                                                  │
│  ═══════════════════════════════════════════════════════════    │
│                        DASHBOARD VIEW                            │
│  ═══════════════════════════════════════════════════════════    │
│                                                                  │
│  KEY INSIGHTS                                                    │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌────────────┐ │
│  │  🇺🇸 #1      │ │  68%       │ │  52%       │ │  Rear      │ │
│  │  Country    │ │  HW Finish │ │  KO in R1  │ │  Naked     │ │
│  │  USA: 412   │ │  Rate      │ │  Occur     │ │  Choke     │ │
│  │  Fighters   │ │  (Highest) │ │  (All KOs) │ │  #1 Sub    │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └────────────┘ │
│                                                                  │
│  WHAT PREDICTS FIGHT OUTCOMES?                                   │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  Feature Importance (from ML Model)                         ││
│  │  ────────────────────────────────────────────────────────── ││
│  │                                                              ││
│  │  Win Streak         ████████████████████████░░ 0.18        ││
│  │  Strike Accuracy    ████████████████████░░░░░░ 0.15        ││
│  │  Takedown Accuracy  ██████████████████░░░░░░░░ 0.13        ││
│  │  Age               █████████████████░░░░░░░░░ 0.12         ││
│  │  Reach Advantage   ██████████████░░░░░░░░░░░░ 0.10         ││
│  │  Experience        █████████████░░░░░░░░░░░░░ 0.09         ││
│  │  Takedown Defense  ████████████░░░░░░░░░░░░░░ 0.08         ││
│  │  Strike Defense    ██████████░░░░░░░░░░░░░░░░ 0.07         ││
│  │  Height Advantage  ████████░░░░░░░░░░░░░░░░░░ 0.05         ││
│  │  Finish Rate       ██████░░░░░░░░░░░░░░░░░░░░ 0.03         ││
│  │                                                              ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  FIGHTER STYLE CLUSTERS                                          │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  [Interactive Scatter Plot - t-SNE Visualization]           ││
│  │                                                              ││
│  │  🔵 Cluster 1: "Wrestlers" (32% of roster)                  ││
│  │     Examples: Khabib, Usman, Covington, Merab               ││
│  │                                                              ││
│  │  🔴 Cluster 2: "Strikers" (28% of roster)                   ││
│  │     Examples: Adesanya, Pereira, Holloway, Volkanovski      ││
│  │                                                              ││
│  │  🟢 Cluster 3: "Submission Specialists" (18% of roster)     ││
│  │     Examples: Oliveira, Makhachev, Burns                    ││
│  │                                                              ││
│  │  🟡 Cluster 4: "Brawlers" (22% of roster)                   ││
│  │     Examples: Gaethje, Chandler, O'Malley                   ││
│  │                                                              ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  DISCOVERED CORRELATIONS                                         │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                                                              ││
│  │  ✓ Takedown attempts → Submission wins: r = 0.67 (Strong)  ││
│  │  ✓ Height advantage → Win rate: r = 0.23 (Weak)            ││
│  │  ✓ Age > 35 → Decision losses: r = 0.41 (Moderate)         ││
│  │  ✓ Southpaw vs Orthodox → Southpaw edge: +4.2% win rate    ││
│  │  ✗ Reach advantage → KO rate: r = 0.08 (No correlation)    ││
│  │                                                              ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  OUTLIER FIGHTERS                                                │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  Fighters with unusual statistical profiles:                 ││
│  │                                                              ││
│  │  • Khabib: Highest TD accuracy + defense combo ever        ││
│  │  • Israel Adesanya: Highest strike defense in MW history   ││
│  │  • Max Holloway: Most significant strikes landed (career)  ││
│  │  • Derrick Lewis: Most KO wins in UFC history              ││
│  │                                                              ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│  🤖 ASK A QUESTION                                               │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  [Which weight class has the most submissions?           ]  ││
│  │                                              [Ask Question] ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  💬 Recent Questions:                                            │
│  • "Does reach advantage lead to more knockouts?" → [View]     │
│  • "Who has the longest win streak?" → [View]                   │
│  • "How has KO rate changed over time?" → [View]               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.7 How It Works Page

```
┌─────────────────────────────────────────────────────────────────┐
│  ❓ HOW IT WORKS                                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Welcome! This page explains how the UFC Prediction App works,  │
│  including our data sources, methodology, and limitations.      │
│                                                                  │
│  ═══════════════════════════════════════════════════════════    │
│                                                                  │
│  📊 DATA SOURCES                                                 │
│  ───────────────────────────────────────────────────────────    │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                                                              ││
│  │  PRIMARY SOURCE: UFCStats.com                               ││
│  │  ─────────────────────────────────────────────────────────  ││
│  │  The official statistics partner of the UFC. We scrape     ││
│  │  fighter profiles, fight results, and per-round statistics ││
│  │  after each UFC event.                                      ││
│  │                                                              ││
│  │  SUPPLEMENTARY: Kaggle UFC Datasets                         ││
│  │  ─────────────────────────────────────────────────────────  ││
│  │  Historical data from 1993-2024 used for initial model     ││
│  │  training and validation.                                   ││
│  │                                                              ││
│  │  FIGHTER IMAGES: UFC.com                                    ││
│  │  ─────────────────────────────────────────────────────────  ││
│  │  Fighter photos are linked from official UFC profiles.     ││
│  │                                                              ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  📈 DATA STATISTICS                                              │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌────────────┐ │
│  │   4,215     │ │   7,842     │ │    682      │ │  1993-     │ │
│  │  Fighters   │ │   Fights    │ │   Events    │ │  Present   │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └────────────┘ │
│  │  Last Updated: January 28, 2025                              │
│                                                                  │
│  ═══════════════════════════════════════════════════════════    │
│                                                                  │
│  🤖 PREDICTION METHODOLOGY                                       │
│  ───────────────────────────────────────────────────────────    │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                                                              ││
│  │  Our prediction system uses an ENSEMBLE of machine learning ││
│  │  models to predict fight outcomes:                          ││
│  │                                                              ││
│  │  WINNER PREDICTION                                          ││
│  │  • Ensemble of: XGBoost, LightGBM, Random Forest, Logistic ││
│  │  • Features: 50+ fighter statistics and differentials      ││
│  │  • Historical accuracy: 62.3%                               ││
│  │                                                              ││
│  │  METHOD PREDICTION                                          ││
│  │  • Predicts: KO/TKO, Submission, or Decision               ││
│  │  • Uses fighter finish rates and style matchups            ││
│  │  • Historical accuracy: 48.7%                               ││
│  │                                                              ││
│  │  ROUND PREDICTION                                           ││
│  │  • Estimates which round the fight will end                ││
│  │  • Uses average fight times and finish rates               ││
│  │                                                              ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  🔧 KEY FEATURES USED                                            │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                                                              ││
│  │  Physical Attributes     Performance Metrics                ││
│  │  • Height differential   • Striking accuracy               ││
│  │  • Reach differential    • Takedown accuracy               ││
│  │  • Age differential      • Defense percentages             ││
│  │                          • Submission rate                  ││
│  │  Experience Factors      Contextual Factors                 ││
│  │  • Total UFC fights      • Weight class                    ││
│  │  • Win streak            • Title fight flag                ││
│  │  • Recent form (last 5)  • Time since last fight           ││
│  │                                                              ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  🔄 MODEL TRAINING & UPDATES                                     │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                                                              ││
│  │  • LIGHT UPDATE: After each UFC event, new fight data is   ││
│  │    added to the training set                                ││
│  │                                                              ││
│  │  • FULL RETRAIN: Quarterly, or when rolling accuracy       ││
│  │    drops below 55%                                          ││
│  │                                                              ││
│  │  • Current Model Version: v2.3.1                           ││
│  │  • Last Trained: January 15, 2025                          ││
│  │  • Training Samples: 6,842 fights                          ││
│  │                                                              ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  ═══════════════════════════════════════════════════════════    │
│                                                                  │
│  📊 PREDICTION ACCURACY TRACKING                                 │
│  ───────────────────────────────────────────────────────────    │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                                                              ││
│  │  HISTORICAL PERFORMANCE (Backfilled)                        ││
│  │  ─────────────────────────────────────────────────────────  ││
│  │  Tested on 5,234 historical fights using walk-forward      ││
│  │  validation (no data leakage).                              ││
│  │                                                              ││
│  │  Winner Accuracy:   62.3%   [██████████████████░░░░]       ││
│  │  Method Accuracy:   48.7%   [█████████████░░░░░░░░░]       ││
│  │  Round (±1) Acc:    54.2%   [███████████████░░░░░░░]       ││
│  │                                                              ││
│  │  RECENT PERFORMANCE (Last 100 Predictions)                  ││
│  │  ─────────────────────────────────────────────────────────  ││
│  │  Winner Accuracy:   64.1%   (Slight improvement)           ││
│  │  Method Accuracy:   51.3%                                   ││
│  │                                                              ││
│  │  [📊 View Detailed Accuracy Charts]                         ││
│  │                                                              ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  ═══════════════════════════════════════════════════════════    │
│                                                                  │
│  ⚠️ LIMITATIONS & DISCLAIMERS                                    │
│  ───────────────────────────────────────────────────────────    │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                                                              ││
│  │  WHAT OUR MODEL CANNOT ACCOUNT FOR:                         ││
│  │                                                              ││
│  │  ❌ Injuries and training camp issues                       ││
│  │  ❌ Weight cut problems                                     ││
│  │  ❌ Psychological factors and motivation                    ││
│  │  ❌ Style evolution between fights                          ││
│  │  ❌ Referee tendencies                                      ││
│  │  ❌ Environmental factors (altitude, climate)               ││
│  │  ❌ Lucky or unlucky outcomes (flash KOs)                   ││
│  │                                                              ││
│  │  MMA is inherently unpredictable. Even the best models     ││
│  │  rarely exceed 65% accuracy. Our predictions should be     ││
│  │  used for entertainment and analysis only.                  ││
│  │                                                              ││
│  │  ─────────────────────────────────────────────────────────  ││
│  │                                                              ││
│  │  🎰 GAMBLING DISCLAIMER                                     ││
│  │                                                              ││
│  │  This application is for EDUCATIONAL and ENTERTAINMENT     ││
│  │  purposes only. We do not encourage gambling. If you       ││
│  │  choose to bet, please do so responsibly and within        ││
│  │  your means. Past prediction performance does not          ││
│  │  guarantee future results.                                  ││
│  │                                                              ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  ═══════════════════════════════════════════════════════════    │
│                                                                  │
│  🤖 LLM INTEGRATION                                              │
│  ───────────────────────────────────────────────────────────    │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                                                              ││
│  │  This app uses Groq's Llama 3.3 70B model to:              ││
│  │  • Generate natural language fighter analyses               ││
│  │  • Explain prediction reasoning in plain English           ││
│  │  • Answer questions about UFC trends and statistics        ││
│  │                                                              ││
│  │  LLM Status: 🟢 Online                                      ││
│  │                                                              ││
│  │  Note: If the LLM service is unavailable, AI-generated     ││
│  │  content will be hidden, but all other features will       ││
│  │  continue to work normally.                                 ││
│  │                                                              ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  ═══════════════════════════════════════════════════════════    │
│                                                                  │
│  📞 FEEDBACK & CONTACT                                           │
│  ───────────────────────────────────────────────────────────    │
│  Have suggestions or found a bug? We'd love to hear from you!  │
│  [GitHub Repository] [Submit Feedback]                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. Technology Stack

### 7.1 Final Tech Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Frontend** | Streamlit | UI framework |
| **Backend** | Python 3.10+ | Core application |
| **Database** | SQLite | Data storage |
| **ML - Training** | Scikit-learn, XGBoost, LightGBM | Model training |
| **ML - Explainability** | SHAP | Feature importance |
| **Data Processing** | Pandas, NumPy | Data manipulation |
| **Web Scraping** | BeautifulSoup, Requests | UFCStats scraper |
| **Visualization** | Plotly, Altair | Interactive charts |
| **LLM** | Groq API (Llama 3.3 70B) | Natural language generation |
| **Deployment** | Streamlit Cloud | Public hosting |
| **Scheduling** | APScheduler | Data update scheduling |

### 7.2 Color Theme (UFC-Inspired)

```python
# UFC Theme Colors
COLORS = {
    'primary': '#D20A0A',      # UFC Red
    'secondary': '#000000',    # Black
    'accent': '#FFFFFF',       # White
    'background': '#1A1A1A',   # Dark gray
    'card_bg': '#2D2D2D',      # Card background
    'success': '#28A745',      # Green (wins)
    'danger': '#DC3545',       # Red (losses)
    'warning': '#FFC107',      # Yellow (warnings)
    'info': '#17A2B8',         # Blue (info)
    'text_primary': '#FFFFFF', # White text
    'text_secondary': '#B0B0B0', # Gray text
}
```

---

## 8. Project Structure

```
ufc_prediction_app/
├── app.py                              # Main Streamlit entry
├── requirements.txt
├── config.py                           # App configuration
├── .env                                # API keys (gitignored)
├── .streamlit/
│   └── config.toml                     # Streamlit theme config
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
│   ├── raw/                            # Raw data files
│   ├── processed/                      # Cleaned data
│   ├── cache/                          # Cached responses
│   └── ufc_database.db                 # SQLite database
│
├── models/
│   ├── trained/                        # Saved model files
│   │   ├── winner_ensemble_v2.3.pkl
│   │   ├── method_classifier_v2.3.pkl
│   │   └── round_predictor_v2.3.pkl
│   ├── training/
│   │   ├── train_models.py
│   │   ├── feature_engineering.py
│   │   ├── hyperparameter_tuning.py
│   │   └── backfill_predictions.py
│   └── inference.py
│
├── services/
│   ├── data_service.py                 # Database operations
│   ├── scraper_service.py              # UFCStats scraping
│   ├── prediction_service.py           # ML inference
│   ├── llm_service.py                  # Groq integration
│   ├── accuracy_service.py             # Prediction tracking
│   └── update_service.py               # Data update logic
│
├── components/
│   ├── fighter_card.py
│   ├── prediction_display.py
│   ├── comparison_charts.py
│   ├── fight_card.py
│   └── stats_charts.py
│
├── utils/
│   ├── helpers.py
│   ├── validators.py
│   └── formatters.py
│
└── assets/
    ├── images/
    └── styles/
```

---

## 9. Future LLM Integrations (Documented)

The following LLM integrations are planned for future development:

| Feature | Description | Priority |
|---------|-------------|----------|
| **Fight Preview Generator** | Narrative previews for upcoming bouts with storyline context | High |
| **Post-Event Analysis** | Automated analysis comparing predictions vs actual results | High |
| **Tale of the Tape Narrator** | Convert stat comparisons into engaging narratives | Medium |
| **Training Camp Intel** | Summarize recent news/updates about fighters | Medium |
| **Historical Matchup Context** | "These fighters met before..." or similar matchup history | Medium |
| **Extended Chatbot** | Free-form Q&A beyond just trends (e.g., "Who should I watch?") | Low |

---

## 10. Implementation Roadmap

### Phase 1: Foundation (Week 1)
- [ ] Set up project structure
- [ ] Implement database schema
- [ ] Download and process Kaggle data
- [ ] Build UFCStats scraper
- [ ] Initial data population

### Phase 2: ML Pipeline (Week 2)
- [ ] Feature engineering pipeline
- [ ] Train winner ensemble model
- [ ] Train method classifier
- [ ] Train round predictor
- [ ] Backfill historical predictions
- [ ] Implement accuracy tracking

### Phase 3: Core Pages (Week 3)
- [ ] Home page
- [ ] Fighters page with search
- [ ] Fighter Comparison page
- [ ] Predictions page
- [ ] How It Works page

### Phase 4: Advanced Features (Week 4)
- [ ] Upcoming Bouts page
- [ ] Trends & Facts page
- [ ] LLM integration (Groq)
- [ ] Explainability features (SHAP)

### Phase 5: Polish & Deploy (Week 5)
- [ ] UFC theme styling
- [ ] Error handling & caching
- [ ] Testing
- [ ] Streamlit Cloud deployment
- [ ] Documentation

---

## 11. Confirmation

This document represents the complete, final design based on all your inputs. Before I begin implementation, please confirm:

1. ✅ All design decisions are correct
2. ✅ Page structure is approved
3. ✅ Feature scope is appropriate
4. ✅ Ready to proceed with development

**Shall I begin building the application?**
