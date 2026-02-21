# UFC Prediction App - Storage & Infrastructure Specification

## 1. Storage Configuration

### 1.1 Local Development Setup

```
ufc_prediction_app/
├── data/
│   ├── database/
│   │   └── ufc_database.db          # SQLite database (~20 MB)
│   ├── cache/
│   │   ├── llm_cache.json           # Cached LLM responses
│   │   └── scraper_cache.json       # Cached scraper data
│   ├── raw/
│   │   ├── kaggle/                  # Downloaded Kaggle CSVs
│   │   │   ├── ufc_fights.csv
│   │   │   ├── ufc_fighters.csv
│   │   │   └── ufc_events.csv
│   │   └── scraped/                 # Raw scraped data
│   │       └── ufcstats_YYYYMMDD/
│   └── processed/
│       └── training_data.csv        # Processed ML training data
│
├── models/
│   └── trained/
│       ├── current/                 # Active models (symlinks)
│       │   ├── winner_model.pkl -> ../versions/winner_v1.0.0.pkl
│       │   ├── method_model.pkl -> ../versions/method_v1.0.0.pkl
│       │   └── round_model.pkl -> ../versions/round_v1.0.0.pkl
│       └── versions/                # Version history (keep latest 3)
│           ├── winner_v1.0.0.pkl
│           ├── method_v1.0.0.pkl
│           ├── round_v1.0.0.pkl
│           └── model_registry.json  # Model metadata & performance
│
└── logs/
    ├── scraper.log
    ├── training.log
    └── predictions.log
```

### 1.2 Git LFS Configuration

**.gitattributes** (for production deployment):
```gitattributes
# Database files
data/database/*.db filter=lfs diff=lfs merge=lfs -text

# Model files
models/trained/versions/*.pkl filter=lfs diff=lfs merge=lfs -text
models/trained/versions/*.joblib filter=lfs diff=lfs merge=lfs -text

# Large data files
data/raw/**/*.csv filter=lfs diff=lfs merge=lfs -text
data/processed/*.csv filter=lfs diff=lfs merge=lfs -text
```

**Git LFS Setup Commands:**
```bash
# Install Git LFS (one-time)
git lfs install

# Track large files
git lfs track "*.db"
git lfs track "*.pkl"
git lfs track "*.joblib"
git lfs track "data/raw/**/*.csv"

# Verify tracking
git lfs ls-files
```

### 1.3 .gitignore Configuration

```gitignore
# === LOCAL DEVELOPMENT ONLY ===
# These are ignored during development
# Remove entries when deploying to production with Git LFS

# Environment
.env
.env.local
*.pyc
__pycache__/
.venv/
venv/

# IDE
.vscode/
.idea/
*.swp

# Logs
logs/
*.log

# Cache (always ignored)
data/cache/
.streamlit/secrets.toml

# === TOGGLE FOR PRODUCTION ===
# Comment these out when deploying with Git LFS
# Uncomment for local development to avoid committing large files

# Database (local dev - comment out for prod)
# data/database/*.db

# Models (local dev - comment out for prod)
# models/trained/versions/*.pkl

# Raw data (local dev - comment out for prod)
# data/raw/

# Processed data (local dev - comment out for prod)
# data/processed/
```

---

## 2. Model Version Management

### 2.1 Model Registry Schema

**models/trained/versions/model_registry.json:**
```json
{
  "registry_version": "1.0",
  "max_versions_kept": 3,
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
          "rolling_accuracy_30": 0.641,
          "rolling_accuracy_100": 0.628,
          "feature_count": 52,
          "status": "active"
        }
      ]
    },
    "method": {
      "current_version": "v1.0.0",
      "versions": [...]
    },
    "round": {
      "current_version": "v1.0.0",
      "versions": [...]
    }
  }
}
```

### 2.2 Model Cleanup Script

**scripts/cleanup_models.py:**
```python
"""
Model Version Cleanup Script
Keeps only the latest N versions of each model type.
"""

import os
import json
from pathlib import Path
from datetime import datetime

MAX_VERSIONS = 3
MODELS_DIR = Path("models/trained/versions")
REGISTRY_FILE = MODELS_DIR / "model_registry.json"


def cleanup_old_versions():
    """Remove old model versions, keeping only the latest MAX_VERSIONS."""
    
    with open(REGISTRY_FILE, 'r') as f:
        registry = json.load(f)
    
    for model_type, model_info in registry['models'].items():
        versions = model_info['versions']
        
        # Sort by creation date (newest first)
        versions.sort(key=lambda x: x['created_at'], reverse=True)
        
        # Keep only latest MAX_VERSIONS
        versions_to_keep = versions[:MAX_VERSIONS]
        versions_to_remove = versions[MAX_VERSIONS:]
        
        # Delete old model files
        for old_version in versions_to_remove:
            old_file = MODELS_DIR / old_version['filename']
            if old_file.exists():
                old_file.unlink()
                print(f"Deleted: {old_file}")
        
        # Update registry
        model_info['versions'] = versions_to_keep
    
    # Save updated registry
    with open(REGISTRY_FILE, 'w') as f:
        json.dump(registry, f, indent=2)
    
    print(f"Cleanup complete. Kept latest {MAX_VERSIONS} versions of each model.")


if __name__ == "__main__":
    cleanup_old_versions()
```

---

## 3. Update Automation

### 3.1 GitHub Actions Workflow

**.github/workflows/update_data.yml:**
```yaml
name: UFC Data Update

on:
  # Scheduled: Run every Sunday at 6 AM UTC (after Saturday night events)
  schedule:
    - cron: '0 6 * * 0'
  
  # Manual trigger
  workflow_dispatch:
    inputs:
      force_retrain:
        description: 'Force model retraining'
        required: false
        default: 'false'
        type: boolean
      update_type:
        description: 'Update type'
        required: false
        default: 'full'
        type: choice
        options:
          - full
          - data_only
          - predictions_only

jobs:
  update-data:
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
        with:
          lfs: true
          token: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'
          cache: 'pip'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      
      - name: Setup Git LFS
        run: |
          git lfs install
          git lfs pull
      
      - name: Run data update
        env:
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
        run: |
          python scripts/update_data.py \
            --update-type ${{ github.event.inputs.update_type || 'full' }}
      
      - name: Check for new completed events
        id: check_events
        run: |
          python scripts/check_new_events.py
          echo "new_events=$(cat .new_events_flag)" >> $GITHUB_OUTPUT
      
      - name: Update prediction accuracy
        if: steps.check_events.outputs.new_events == 'true'
        run: |
          python scripts/update_accuracy.py
      
      - name: Check if retraining needed
        id: check_retrain
        run: |
          python scripts/check_retrain_needed.py
          echo "needs_retrain=$(cat .retrain_flag)" >> $GITHUB_OUTPUT
      
      - name: Retrain models (if needed)
        if: steps.check_retrain.outputs.needs_retrain == 'true' || github.event.inputs.force_retrain == 'true'
        run: |
          python scripts/train_models.py --full-retrain
          python scripts/cleanup_models.py
      
      - name: Generate predictions for upcoming events
        run: |
          python scripts/generate_upcoming_predictions.py
      
      - name: Commit and push changes
        run: |
          git config --local user.email "github-actions[bot]@users.noreply.github.com"
          git config --local user.name "github-actions[bot]"
          
          git add data/database/ufc_database.db
          git add models/trained/
          git add data/processed/
          
          # Check if there are changes to commit
          if git diff --staged --quiet; then
            echo "No changes to commit"
          else
            git commit -m "🤖 Auto-update: $(date +'%Y-%m-%d')"
            git push
          fi
      
      - name: Create update summary
        run: |
          python scripts/generate_update_summary.py > update_summary.md
          cat update_summary.md >> $GITHUB_STEP_SUMMARY

  notify-on-failure:
    runs-on: ubuntu-latest
    needs: update-data
    if: failure()
    steps:
      - name: Log failure
        run: |
          echo "Data update failed at $(date)"
          # Add notification logic here (email, Slack, etc.)
```

### 3.2 Manual Update Script

**scripts/manual_update.py:**
```python
#!/usr/bin/env python3
"""
Manual Data Update Script

Usage:
    python scripts/manual_update.py [OPTIONS]

Options:
    --data-only         Only update data, skip predictions
    --predictions-only  Only regenerate predictions
    --force-retrain     Force model retraining
    --dry-run          Show what would be done without making changes
    --verbose          Verbose output
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.scraper_service import UFCStatsScraper
from services.data_service import DataService
from services.prediction_service import PredictionService
from services.accuracy_service import AccuracyService
from models.training.train_models import ModelTrainer
from utils.helpers import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description='Manual UFC Data Update')
    parser.add_argument('--data-only', action='store_true',
                        help='Only update data, skip predictions')
    parser.add_argument('--predictions-only', action='store_true',
                        help='Only regenerate predictions')
    parser.add_argument('--force-retrain', action='store_true',
                        help='Force model retraining')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be done')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logging(verbose=args.verbose)
    
    print("=" * 60)
    print("UFC PREDICTION APP - MANUAL UPDATE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    if args.dry_run:
        print("\n🔍 DRY RUN MODE - No changes will be made\n")
    
    # Initialize services
    scraper = UFCStatsScraper()
    data_service = DataService()
    prediction_service = PredictionService()
    accuracy_service = AccuracyService()
    trainer = ModelTrainer()
    
    try:
        # Step 1: Scrape new data
        if not args.predictions_only:
            print("\n📥 Step 1: Scraping UFCStats.com...")
            if not args.dry_run:
                new_events = scraper.scrape_new_events()
                new_fighters = scraper.scrape_new_fighters()
                print(f"   Found {len(new_events)} new events")
                print(f"   Found {len(new_fighters)} new/updated fighters")
            else:
                print("   [DRY RUN] Would scrape for new data")
        
        # Step 2: Update database
        if not args.predictions_only:
            print("\n💾 Step 2: Updating database...")
            if not args.dry_run:
                data_service.process_and_store(new_events, new_fighters)
                print("   Database updated successfully")
            else:
                print("   [DRY RUN] Would update database")
        
        # Step 3: Update prediction accuracy
        if not args.predictions_only:
            print("\n📊 Step 3: Updating prediction accuracy...")
            if not args.dry_run:
                accuracy_updates = accuracy_service.update_completed_predictions()
                print(f"   Updated accuracy for {accuracy_updates} predictions")
            else:
                print("   [DRY RUN] Would update accuracy tracking")
        
        # Step 4: Check if retraining needed
        if not args.data_only:
            print("\n🔍 Step 4: Checking if retraining needed...")
            needs_retrain = trainer.check_retrain_needed()
            
            if args.force_retrain:
                print("   Force retrain flag set - will retrain")
                needs_retrain = True
            elif needs_retrain:
                print("   Retraining needed (accuracy below threshold)")
            else:
                print("   No retraining needed")
            
            # Step 5: Retrain if needed
            if needs_retrain:
                print("\n🤖 Step 5: Retraining models...")
                if not args.dry_run:
                    trainer.train_all_models()
                    trainer.cleanup_old_versions(keep=3)
                    print("   Models retrained successfully")
                else:
                    print("   [DRY RUN] Would retrain models")
        
        # Step 6: Generate predictions for upcoming events
        if not args.data_only:
            print("\n🔮 Step 6: Generating predictions for upcoming events...")
            if not args.dry_run:
                upcoming = prediction_service.predict_upcoming_events()
                print(f"   Generated predictions for {len(upcoming)} fights")
            else:
                print("   [DRY RUN] Would generate predictions")
        
        print("\n" + "=" * 60)
        print("✅ UPDATE COMPLETE")
        print("=" * 60)
        
        # Summary
        print("\n📋 SUMMARY:")
        print(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if not args.dry_run:
            stats = data_service.get_database_stats()
            print(f"   Total Fighters: {stats['fighters']}")
            print(f"   Total Fights: {stats['fights']}")
            print(f"   Total Events: {stats['events']}")
            print(f"   Model Version: {trainer.get_current_version()}")
        
        print("\n💡 Next steps:")
        print("   1. Review changes: git status")
        print("   2. Commit: git add -A && git commit -m 'Data update'")
        print("   3. Push: git push")
        
    except Exception as e:
        logger.error(f"Update failed: {e}")
        print(f"\n❌ ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

### 3.3 Helper Scripts

**scripts/check_new_events.py:**
```python
#!/usr/bin/env python3
"""Check for newly completed UFC events since last update."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from services.data_service import DataService
from services.scraper_service import UFCStatsScraper


def main():
    data_service = DataService()
    scraper = UFCStatsScraper()
    
    # Get last update timestamp
    last_update = data_service.get_last_update_timestamp()
    
    # Check for new completed events
    new_events = scraper.get_completed_events_since(last_update)
    
    has_new_events = len(new_events) > 0
    
    # Write flag for GitHub Actions
    with open('.new_events_flag', 'w') as f:
        f.write('true' if has_new_events else 'false')
    
    if has_new_events:
        print(f"Found {len(new_events)} new completed events:")
        for event in new_events:
            print(f"  - {event['name']} ({event['date']})")
    else:
        print("No new completed events found.")
    
    return 0 if has_new_events else 1


if __name__ == "__main__":
    sys.exit(main())
```

**scripts/check_retrain_needed.py:**
```python
#!/usr/bin/env python3
"""Check if model retraining is needed based on accuracy thresholds."""

import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from services.accuracy_service import AccuracyService
from models.training.train_models import ModelTrainer


# Thresholds
MIN_ACCURACY = 0.55  # Retrain if rolling accuracy drops below 55%
MAX_DAYS_SINCE_RETRAIN = 90  # Quarterly retrain
MIN_NEW_FIGHTS_FOR_RETRAIN = 150  # ~3 months of data


def main():
    accuracy_service = AccuracyService()
    trainer = ModelTrainer()
    
    needs_retrain = False
    reasons = []
    
    # Check 1: Rolling accuracy below threshold
    rolling_accuracy = accuracy_service.get_rolling_accuracy(window=100)
    if rolling_accuracy < MIN_ACCURACY:
        needs_retrain = True
        reasons.append(f"Rolling accuracy ({rolling_accuracy:.1%}) below threshold ({MIN_ACCURACY:.0%})")
    
    # Check 2: Time since last retrain
    last_retrain = trainer.get_last_training_date()
    if last_retrain:
        days_since = (datetime.now() - last_retrain).days
        if days_since >= MAX_DAYS_SINCE_RETRAIN:
            needs_retrain = True
            reasons.append(f"Quarterly retrain due ({days_since} days since last)")
    
    # Check 3: Sufficient new data
    new_fights = trainer.get_new_fights_since_training()
    if new_fights >= MIN_NEW_FIGHTS_FOR_RETRAIN:
        reasons.append(f"{new_fights} new fights available for training")
    
    # Write flag for GitHub Actions
    with open('.retrain_flag', 'w') as f:
        f.write('true' if needs_retrain else 'false')
    
    # Output
    if needs_retrain:
        print("🔄 Model retraining NEEDED:")
        for reason in reasons:
            print(f"  - {reason}")
    else:
        print("✅ Model retraining NOT needed:")
        print(f"  - Rolling accuracy: {rolling_accuracy:.1%}")
        print(f"  - Days since retrain: {(datetime.now() - last_retrain).days if last_retrain else 'N/A'}")
        print(f"  - New fights available: {new_fights}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

---

## 4. Environment Configuration

### 4.1 Environment File Template

**.env.template:**
```bash
# =============================================================================
# UFC PREDICTION APP - ENVIRONMENT CONFIGURATION
# =============================================================================
# Copy this file to .env and fill in your values
# NEVER commit .env to version control!
# =============================================================================

# === REQUIRED ===

# Groq API Key (get from https://console.groq.com)
GROQ_API_KEY=gsk_your_api_key_here

# === OPTIONAL ===

# Kaggle API (for automated dataset downloads)
# Get from https://www.kaggle.com/settings → API → Create New Token
# KAGGLE_USERNAME=your_username
# KAGGLE_KEY=your_key

# === APP SETTINGS ===

# Environment: development, staging, production
APP_ENV=development

# Debug mode (true/false)
DEBUG=true

# Database path (relative to project root)
DATABASE_PATH=data/database/ufc_database.db

# === DATA UPDATE SETTINGS ===

# Enable automatic updates
AUTO_UPDATE_ENABLED=true

# How often to check for updates (hours)
UPDATE_CHECK_INTERVAL=24

# === MODEL SETTINGS ===

# Current model version (managed automatically)
MODEL_VERSION=v1.0.0

# Minimum confidence threshold for predictions
MIN_CONFIDENCE_THRESHOLD=0.55

# Maximum model versions to keep
MAX_MODEL_VERSIONS=3

# === LLM SETTINGS ===

# Enable LLM features
LLM_ENABLED=true

# LLM provider (groq, openai, google)
LLM_PROVIDER=groq

# Model to use
LLM_MODEL=llama-3.3-70b-versatile

# Max tokens per request
LLM_MAX_TOKENS=1000

# Temperature (0.0 - 1.0)
LLM_TEMPERATURE=0.7

# === LOGGING ===

# Log level: DEBUG, INFO, WARNING, ERROR
LOG_LEVEL=INFO

# Log file path
LOG_FILE=logs/app.log

# === CACHING ===

# Enable LLM response caching
CACHE_LLM_RESPONSES=true

# Cache TTL in seconds (24 hours)
CACHE_TTL=86400
```

### 4.2 Streamlit Cloud Secrets

For Streamlit Cloud deployment, add these to `.streamlit/secrets.toml` (or via the web UI):

```toml
# .streamlit/secrets.toml (DO NOT COMMIT)

# Required
GROQ_API_KEY = "gsk_your_api_key_here"

# Optional
# KAGGLE_USERNAME = "your_username"
# KAGGLE_KEY = "your_key"

# App settings
APP_ENV = "production"
DEBUG = false
LLM_ENABLED = true
```

---

## 5. Complete File Structure (Final)

```
ufc_prediction_app/
│
├── .github/
│   └── workflows/
│       └── update_data.yml              # Automated update workflow
│
├── .streamlit/
│   ├── config.toml                      # Streamlit theme config
│   └── secrets.toml                     # Secrets (gitignored)
│
├── app.py                               # Main Streamlit entry point
├── config.py                            # App configuration
├── requirements.txt                     # Production dependencies
├── requirements-dev.txt                 # Development dependencies
│
├── .env.template                        # Environment template
├── .env                                 # Local env (gitignored)
├── .gitignore
├── .gitattributes                       # Git LFS config
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
│   │   └── ufc_database.db              # SQLite database
│   ├── cache/
│   │   └── llm_cache.json
│   ├── raw/
│   │   ├── kaggle/
│   │   └── scraped/
│   └── processed/
│       └── training_data.csv
│
├── models/
│   ├── trained/
│   │   ├── current/                     # Symlinks to active models
│   │   └── versions/                    # Versioned model files
│   │       └── model_registry.json
│   ├── training/
│   │   ├── train_models.py
│   │   ├── feature_engineering.py
│   │   ├── hyperparameter_tuning.py
│   │   └── backfill_predictions.py
│   └── inference.py
│
├── services/
│   ├── data_service.py
│   ├── scraper_service.py
│   ├── prediction_service.py
│   ├── llm_service.py
│   ├── accuracy_service.py
│   └── update_service.py
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
├── scripts/
│   ├── manual_update.py                 # Manual update script
│   ├── check_new_events.py
│   ├── check_retrain_needed.py
│   ├── update_accuracy.py
│   ├── generate_upcoming_predictions.py
│   ├── cleanup_models.py
│   └── generate_update_summary.py
│
├── tests/
│   ├── test_data_service.py
│   ├── test_prediction_service.py
│   └── test_scraper.py
│
├── logs/                                # Gitignored
│   ├── scraper.log
│   ├── training.log
│   └── predictions.log
│
├── docs/
│   ├── SETUP.md                         # Setup instructions
│   ├── DEPLOYMENT.md                    # Deployment guide
│   ├── GIT_LFS_GUIDE.md                # Git LFS documentation
│   └── API_REFERENCE.md                 # API documentation
│
└── assets/
    ├── images/
    │   └── logo.png
    └── styles/
        └── custom.css
```

---

## 6. Quick Reference Commands

### Local Development
```bash
# Setup
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
cp .env.template .env
# Edit .env with your API keys

# Run app
streamlit run app.py

# Manual data update
python scripts/manual_update.py --verbose

# Force model retrain
python scripts/manual_update.py --force-retrain
```

### Git LFS (Production)
```bash
# Initial setup
git lfs install
git lfs track "*.db" "*.pkl"
git add .gitattributes
git commit -m "Configure Git LFS"

# Push with LFS
git add data/database/ufc_database.db
git add models/trained/versions/*.pkl
git commit -m "Add data and models"
git push
```

### GitHub Actions
```bash
# Trigger manual update via GitHub CLI
gh workflow run update_data.yml

# Trigger with options
gh workflow run update_data.yml -f force_retrain=true -f update_type=full
```

---

## Summary

| Aspect | Decision |
|--------|----------|
| **Local Storage** | SQLite + file-based models |
| **Production Storage** | Git LFS for database and models |
| **Model Versions** | Keep latest 3 versions |
| **Auto Updates** | GitHub Actions (weekly + manual trigger) |
| **Manual Updates** | Python script with CLI options |
| **Secrets** | .env locally, Streamlit secrets in cloud |
