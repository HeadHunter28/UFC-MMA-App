# UFC Fighter Analysis & Prediction App

A comprehensive UFC Fighter Analysis and Prediction application built with Streamlit. Features ML-based fight predictions, fighter profiles, head-to-head comparisons, and data-driven insights.

## Features

- **Fighter Search & Profiles**: Search UFC fighters, view detailed stats, fight history
- **Fighter Comparison**: Head-to-head statistical comparison tool
- **Fight Predictions**: ML ensemble predictions with confidence levels and explainability
- **Upcoming Fights**: Official UFC cards with predictions, fighter analysis, and strategy recommendations
- **Trends & Facts**: Analytics dashboard, pattern discovery, Q&A system
- **How It Works**: Documentation of data sources, methodology, limitations
- **Fight Simulator**: Simulate hypothetical fights between any two active fighters with multi-model analysis

## Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| Database | SQLite |
| ML Models | XGBoost, LightGBM, Random Forest |
| MLOps | MLFlow (Experiment Tracking & Model Registry) |
| LLM | Groq (Llama 3.3 70B) |
| Data Source | UFCStats.com |
| Deployment | Streamlit Cloud |
| CI/CD | GitHub Actions |

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/ufc-prediction-app.git
cd ufc-prediction-app
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

```bash
cp .env.template .env
# Edit .env and add your GROQ_API_KEY
```

### 5. Initialize database

```bash
python scripts/init_database.py
```

### 6. Run the app

```bash
streamlit run app.py
```

## Project Structure

```
ufc_prediction_app/
├── app.py                     # Main Streamlit entry point
├── config.py                  # Configuration settings
├── requirements.txt           # Dependencies
├── .env.template              # Environment template
│
├── pages/                     # Streamlit pages
│   ├── 1_Fighters.py
│   ├── 2_Fighter_Comparison.py
│   ├── 3_Predictions.py
│   ├── 4_Upcoming_Fights.py
│   ├── 5_Trends_and_Facts.py
│   ├── 6_How_It_Works.py
│   └── 7_Fight_Simulator.py
│
├── services/                  # Business logic
│   ├── data_service.py
│   ├── scraper_service.py
│   ├── prediction_service.py
│   ├── simulation_service.py
│   ├── mlflow_service.py
│   ├── llm_service.py
│   └── accuracy_service.py
│
├── models/                    # ML models
│   ├── inference.py
│   └── training/
│       ├── feature_engineering.py
│       └── train_models.py
│
├── components/                # UI components
│   ├── fighter_card.py
│   ├── prediction_display.py
│   └── comparison_charts.py
│
├── utils/                     # Utilities
│   ├── helpers.py
│   ├── formatters.py
│   └── validators.py
│
├── scripts/                   # CLI scripts
│   ├── init_database.py
│   ├── manual_update.py
│   └── cleanup_models.py
│
└── data/                      # Data files
    ├── database/
    ├── cache/
    └── raw/
```

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GROQ_API_KEY` | Groq API key for LLM features | Yes |
| `LLM_ENABLED` | Enable/disable LLM features | No (default: true) |
| `DEBUG` | Enable debug mode | No (default: false) |
| `FIGHTER_ACTIVITY_CUTOFF_YEARS` | Years since last fight to consider active | No (default: 3) |
| `MLFLOW_ENABLED` | Enable MLFlow tracking | No (default: true) |
| `MLFLOW_TRACKING_URI` | MLFlow tracking server URI | No (default: ./mlruns) |
| `MLFLOW_EXPERIMENT_NAME` | MLFlow experiment name | No (default: ufc-prediction-models) |
| `MIN_HOURS_BEFORE_EVENT` | Min hours before event for valid prediction | No (default: 1) |
| `INCLUDE_BACKFILL_IN_ACCURACY` | Include backfills in accuracy metrics | No (default: false) |

### Model Settings

- Models are stored in `models/trained/versions/`
- Latest 3 versions are kept automatically
- Retraining triggers at <55% accuracy or every 90 days

## Data Updates

### Automatic (GitHub Actions)

- **Sunday 6 AM & 2 PM UTC** - After Saturday night main events
- **Thursday 6 AM UTC** - After Wednesday Fight Nights
- **Daily 7 AM UTC** - Lightweight check for any new results
- Scrapes new event data with data quality validation
- Updates prediction accuracy with timestamp validation
- Retrains models if accuracy drops below threshold

### Manual

```bash
# Full update
python scripts/manual_update.py

# Data only
python scripts/manual_update.py --data-only

# Force retrain
python scripts/manual_update.py --force-retrain

# Backfill predictions for historical fights
python scripts/backfill_predictions.py --events 20 --verbose
```

## Ground Truth Collection

The app implements a comprehensive ground truth collection system for continuous model improvement:

### Prediction Validation
- **Timestamp Validation**: Predictions are validated to ensure they were made BEFORE the fight
- **Edge Case Handling**: NC (No Contest), DQ, and Draw outcomes are properly tracked and excluded from accuracy metrics
- **Stats Snapshots**: Fighter statistics are captured at prediction time for historical analysis

### Data Quality
- **Automated Validation**: Scraped data is validated for completeness and correctness
- **Method Normalization**: Fight methods are normalized for consistent accuracy tracking
- **Backfill Support**: Historical predictions can be generated for model testing

### Accuracy Tracking
| Metric | Description |
|--------|-------------|
| `rolling_30` | Accuracy over last 30 predictions |
| `rolling_100` | Accuracy over last 100 predictions |
| `by_confidence` | Breakdown by confidence level (high/medium/low) |
| `by_method` | Accuracy for each predicted method type |
| `edge_cases` | Count of NC/DQ/Draw outcomes |

### Database Tables
- `predictions` - All predictions with timestamps and event dates
- `prediction_accuracy` - Accuracy records with validation flags
- `prediction_stats_snapshot` - Fighter stats at prediction time

## Prediction Methodology

### Winner Prediction (Ensemble)
- XGBoost (30%)
- LightGBM (30%)
- Random Forest (20%)
- Logistic Regression (20%)

### Features Used
- Physical differentials (height, reach, age)
- Striking stats (accuracy, defense, output)
- Grappling stats (takedowns, submissions)
- Form indicators (win streak, recent results)
- Context (title fight, main event)

### Confidence Levels
- **High (>65%)**: Strong statistical edge
- **Medium (55-65%)**: Moderate confidence
- **Low (<55%)**: Close matchup

## Fight Simulator

The Fight Simulator allows you to simulate hypothetical matchups between any two active UFC fighters.

### Features
- **Multi-Model Simulation**: Runs simulations across 5 different models (Statistical, Momentum, Stylistic, Historical, Ensemble)
- **Realism Scoring**: Automatically selects the most realistic outcome based on fighter profiles
- **Round-by-Round Breakdown**: Detailed statistics for each round including strikes, takedowns, and control time
- **Fighter Analysis**: Shows strengths, weaknesses, fighting style, and keys to victory for each fighter
- **Alternative Outcomes**: Displays the second most likely simulation result

### Simulation Models
| Model | Focus |
|-------|-------|
| Statistical | Pure statistical comparison |
| Momentum | Factors in winning/losing streaks |
| Stylistic | Considers style matchups (striker vs wrestler) |
| Historical | Weights experience heavily |
| Ensemble | Combines all model insights |

### Fighter Activity Cutoff
Only active fighters are available for simulation. A fighter is considered **active** if they have competed within the last **3 years**. This ensures simulations use current, relevant fighter data.

## MLFlow Integration

The app uses MLFlow for comprehensive experiment tracking and model versioning.

### Features
- **Experiment Tracking**: All training runs are logged with parameters, metrics, and artifacts
- **Model Registry**: Trained models are registered with version control and stage management
- **Automatic Promotion**: New models are automatically promoted to "Production" stage
- **Fallback Support**: Falls back to local model files if MLFlow is unavailable

### Viewing Experiments

Start the MLFlow UI to view experiment runs and model versions:

```bash
mlflow ui --backend-store-uri ./mlruns
```

Then open http://localhost:5000 in your browser.

### Registered Models
| Model | Name | Description |
|-------|------|-------------|
| Winner | `ufc-winner-predictor` | Ensemble model for fight winner prediction |
| Method | `ufc-method-predictor` | Classifier for method of victory |
| Round | `ufc-round-predictor` | Regressor for predicted finish round |

### Logged Metrics
Each training run logs:
- **Winner Model**: accuracy, precision, recall, f1, auc_roc
- **Method Model**: accuracy, precision, recall, f1
- **Round Model**: mae, within_one_round

## Disclaimer

This application is for **entertainment purposes only**. Predictions are based on statistical analysis and are not guaranteed. Never bet more than you can afford to lose.

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions welcome! Please read CONTRIBUTING.md first.
