#!/usr/bin/env python3
"""
Manual Data Update Script.

Usage:
    python scripts/manual_update.py [OPTIONS]

Options:
    --data-only         Only update data, skip predictions
    --predictions-only  Only regenerate predictions
    --force-retrain     Force model retraining
    --dry-run           Show what would be done without making changes
    --verbose           Verbose output
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.scraper_service import UFCStatsScraper
from services.data_service import DataService
from services.prediction_service import PredictionService
from services.accuracy_service import AccuracyService
from models.training.train_models import ModelTrainer
from utils.helpers import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Manual UFC Data Update")
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="Only update data, skip predictions",
    )
    parser.add_argument(
        "--predictions-only",
        action="store_true",
        help="Only regenerate predictions",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Force model retraining",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logging(verbose=args.verbose)

    print("=" * 60)
    print("UFC PREDICTION APP - MANUAL UPDATE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    if args.dry_run:
        print("\n[DRY RUN MODE - No changes will be made]\n")

    # Initialize services
    scraper = UFCStatsScraper()
    data_service = DataService()
    prediction_service = PredictionService()
    accuracy_service = AccuracyService()
    trainer = ModelTrainer()

    try:
        # Step 1: Scrape new data
        if not args.predictions_only:
            print("\n[1/7] Scraping UFCStats.com...")
            if not args.dry_run:
                last_update = data_service.get_last_update_timestamp()
                new_events = scraper.scrape_new_events(
                    since_date=last_update.date() if last_update else None
                )
                print(f"      Found {len(new_events)} new/updated events")
            else:
                print("      [DRY RUN] Would scrape for new data")

        # Step 2: Scrape upcoming fight cards
        if not args.predictions_only:
            print("\n[2/7] Scraping upcoming fight cards...")
            if not args.dry_run:
                try:
                    upcoming_fights_saved = scraper.scrape_and_save_upcoming_fights(data_service)
                    print(f"      Saved {upcoming_fights_saved} upcoming fights")
                except Exception as e:
                    print(f"      Warning: Could not scrape upcoming fights: {e}")
            else:
                print("      [DRY RUN] Would scrape upcoming fight cards")

        # Step 3: Update database metadata
        if not args.predictions_only:
            print("\n[3/7] Updating database...")
            if not args.dry_run:
                # Update last scrape timestamp
                data_service.set_metadata(
                    "last_scrape_date",
                    datetime.now().isoformat()
                )
                print("      Database updated successfully")
            else:
                print("      [DRY RUN] Would update database")

        # Step 4: Update prediction accuracy
        if not args.predictions_only:
            print("\n[4/7] Updating prediction accuracy...")
            if not args.dry_run:
                accuracy_updates = accuracy_service.update_completed_predictions()
                print(f"      Updated accuracy for {accuracy_updates} predictions")
            else:
                print("      [DRY RUN] Would update accuracy tracking")

        # Step 5: Check if retraining needed
        if not args.data_only:
            print("\n[5/7] Checking if retraining needed...")
            needs_retrain = trainer.check_retrain_needed()

            if args.force_retrain:
                print("      Force retrain flag set - will retrain")
                needs_retrain = True
            elif needs_retrain:
                print("      Retraining needed (accuracy below threshold)")
            else:
                print("      No retraining needed")

            # Step 6: Retrain if needed
            if needs_retrain:
                print("\n[6/7] Retraining models...")
                if not args.dry_run:
                    version, metrics = trainer.train_all_models()
                    print(f"      Models retrained successfully (version: {version})")
                    print(f"      Winner accuracy: {metrics['winner']['accuracy']:.1%}")
                else:
                    print("      [DRY RUN] Would retrain models")
            else:
                print("\n[6/7] Skipping model training (not needed)")

        # Step 7: Generate predictions for upcoming events
        if not args.data_only:
            print("\n[7/7] Generating predictions for upcoming events...")
            if not args.dry_run:
                predictions = prediction_service.predict_upcoming_events()
                print(f"      Generated predictions for {len(predictions)} fights")
            else:
                print("      [DRY RUN] Would generate predictions")

        print("\n" + "=" * 60)
        print("UPDATE COMPLETE")
        print("=" * 60)

        # Summary
        print("\nSUMMARY:")
        print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if not args.dry_run:
            try:
                stats = data_service.get_database_stats()
                print(f"  Total Fighters: {stats.get('fighters', 0)}")
                print(f"  Total Fights: {stats.get('fights', 0)}")
                print(f"  Total Events: {stats.get('events', 0)}")
                print(f"  Model Version: {trainer.get_current_version()}")
            except Exception:
                pass

        print("\nNEXT STEPS:")
        print("  1. Review changes: git status")
        print("  2. Commit: git add -A && git commit -m 'Data update'")
        print("  3. Push: git push")

    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
