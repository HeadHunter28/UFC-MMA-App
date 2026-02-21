#!/usr/bin/env python3
"""
Check if model retraining is needed based on accuracy thresholds.

Writes a flag file for use with GitHub Actions.
"""

import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    MIN_ACCURACY_THRESHOLD,
    MAX_DAYS_SINCE_RETRAIN,
    MIN_NEW_FIGHTS_FOR_RETRAIN,
)
from services.accuracy_service import AccuracyService
from models.training.train_models import ModelTrainer


def main():
    accuracy_service = AccuracyService()
    trainer = ModelTrainer()

    needs_retrain = False
    reasons = []

    # Check 1: Rolling accuracy below threshold
    try:
        rolling_accuracy = accuracy_service.get_rolling_accuracy(window=100)
        if rolling_accuracy < MIN_ACCURACY_THRESHOLD:
            needs_retrain = True
            reasons.append(
                f"Rolling accuracy ({rolling_accuracy:.1%}) below threshold ({MIN_ACCURACY_THRESHOLD:.0%})"
            )
    except Exception as e:
        print(f"Warning: Could not check rolling accuracy: {e}")

    # Check 2: Time since last retrain
    try:
        last_retrain = trainer.get_last_training_date()
        if last_retrain:
            days_since = (datetime.now() - last_retrain).days
            if days_since >= MAX_DAYS_SINCE_RETRAIN:
                needs_retrain = True
                reasons.append(f"Quarterly retrain due ({days_since} days since last)")
        else:
            needs_retrain = True
            reasons.append("No previous training found")
    except Exception as e:
        print(f"Warning: Could not check last training date: {e}")

    # Check 3: Sufficient new data available
    try:
        new_fights = trainer.get_new_fights_since_training()
        if new_fights >= MIN_NEW_FIGHTS_FOR_RETRAIN:
            reasons.append(f"{new_fights} new fights available for training")
    except Exception as e:
        print(f"Warning: Could not check new fights count: {e}")

    # Write flag for GitHub Actions
    flag_file = Path(".retrain_flag")
    with open(flag_file, "w") as f:
        f.write("true" if needs_retrain else "false")

    # Output
    if needs_retrain:
        print("Model retraining NEEDED:")
        for reason in reasons:
            print(f"  - {reason}")
    else:
        print("Model retraining NOT needed:")
        try:
            print(f"  - Rolling accuracy: {rolling_accuracy:.1%}")
            if last_retrain:
                print(f"  - Days since retrain: {(datetime.now() - last_retrain).days}")
            print(f"  - New fights available: {new_fights}")
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
