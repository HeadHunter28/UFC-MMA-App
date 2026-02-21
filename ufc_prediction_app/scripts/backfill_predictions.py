#!/usr/bin/env python3
"""
Backfill Predictions Script.

Generates predictions for historical fights to build ground truth data.
Marks backfilled predictions appropriately so they can be filtered from
accuracy metrics if needed.

Usage:
    python scripts/backfill_predictions.py [OPTIONS]

Options:
    --events N         Number of recent events to backfill (default: 10)
    --from-date DATE   Start date for backfill (YYYY-MM-DD)
    --to-date DATE     End date for backfill (YYYY-MM-DD)
    --dry-run          Don't save predictions, just preview
    --verbose          Verbose output
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.data_service import DataService
from services.prediction_service import PredictionService
from services.accuracy_service import AccuracyService

logger = logging.getLogger(__name__)


def get_historical_fights(
    data_service: DataService,
    num_events: int = 10,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Get historical fights for backfilling.

    Args:
        data_service: Data service instance
        num_events: Number of recent events to include
        from_date: Start date filter (YYYY-MM-DD)
        to_date: End date filter (YYYY-MM-DD)

    Returns:
        List of fights with event data
    """
    fights = []

    with data_service.get_connection() as conn:
        # Build query
        sql = """
            SELECT
                f.fight_id,
                f.fighter_red_id,
                f.fighter_blue_id,
                f.winner_id,
                f.method,
                f.round,
                f.weight_class,
                f.is_title_fight,
                f.is_main_event,
                e.event_id,
                e.name as event_name,
                e.date as event_date,
                fr.name as fighter_red_name,
                fb.name as fighter_blue_name
            FROM fights f
            JOIN events e ON f.event_id = e.event_id
            JOIN fighters fr ON f.fighter_red_id = fr.fighter_id
            JOIN fighters fb ON f.fighter_blue_id = fb.fighter_id
            WHERE e.is_completed = TRUE
              AND f.winner_id IS NOT NULL
        """

        params = []

        if from_date:
            sql += " AND e.date >= ?"
            params.append(from_date)

        if to_date:
            sql += " AND e.date <= ?"
            params.append(to_date)

        sql += " ORDER BY e.date DESC"

        if not from_date and not to_date:
            # Limit by number of events
            sql = f"""
                WITH recent_events AS (
                    SELECT event_id FROM events
                    WHERE is_completed = TRUE
                    ORDER BY date DESC
                    LIMIT {num_events}
                )
                SELECT
                    f.fight_id,
                    f.fighter_red_id,
                    f.fighter_blue_id,
                    f.winner_id,
                    f.method,
                    f.round,
                    f.weight_class,
                    f.is_title_fight,
                    f.is_main_event,
                    e.event_id,
                    e.name as event_name,
                    e.date as event_date,
                    fr.name as fighter_red_name,
                    fb.name as fighter_blue_name
                FROM fights f
                JOIN events e ON f.event_id = e.event_id
                JOIN fighters fr ON f.fighter_red_id = fr.fighter_id
                JOIN fighters fb ON f.fighter_blue_id = fb.fighter_id
                WHERE f.event_id IN (SELECT event_id FROM recent_events)
                  AND f.winner_id IS NOT NULL
                ORDER BY e.date DESC, f.fight_id
            """
            params = []

        cursor = conn.execute(sql, params)
        fights = [dict(row) for row in cursor.fetchall()]

    return fights


def check_existing_prediction(
    data_service: DataService,
    fight_id: int
) -> bool:
    """
    Check if a prediction already exists for a fight.

    Args:
        data_service: Data service instance
        fight_id: Fight ID to check

    Returns:
        True if prediction exists
    """
    with data_service.get_connection() as conn:
        cursor = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE fight_id = ?",
            (fight_id,)
        )
        return cursor.fetchone()[0] > 0


def backfill_fight(
    prediction_service: PredictionService,
    data_service: DataService,
    fight: Dict[str, Any],
    dry_run: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Generate and save a backfill prediction for a fight.

    Args:
        prediction_service: Prediction service instance
        data_service: Data service instance
        fight: Fight data dict
        dry_run: If True, don't save to database

    Returns:
        Prediction result dict or None
    """
    # Generate prediction
    result = prediction_service.predict(
        fight["fighter_red_id"],
        fight["fighter_blue_id"],
        context={
            "is_title_fight": fight.get("is_title_fight", False),
            "is_main_event": fight.get("is_main_event", False),
            "weight_class": fight.get("weight_class"),
        }
    )

    if not result:
        return None

    # Build prediction dict
    pred_dict = result.to_dict()
    pred_dict["fight_id"] = fight["fight_id"]
    pred_dict["fighter_red_id"] = fight["fighter_red_id"]
    pred_dict["fighter_blue_id"] = fight["fighter_blue_id"]
    pred_dict["event_date"] = fight["event_date"]
    pred_dict["is_backfill"] = True
    pred_dict["is_valid_timing"] = False  # Backfills are not valid for timing

    if not dry_run:
        # Save prediction
        prediction_id = data_service.save_prediction(pred_dict)
        pred_dict["prediction_id"] = prediction_id

        # Save stats snapshot (important for ground truth)
        fighter_red = data_service.get_fighter_by_id(fight["fighter_red_id"])
        fighter_blue = data_service.get_fighter_by_id(fight["fighter_blue_id"])
        prediction_service._save_stats_snapshot(prediction_id, fighter_red, fighter_blue)

    return pred_dict


def run_backfill(
    num_events: int = 10,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    dry_run: bool = False,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run the backfill process.

    Args:
        num_events: Number of recent events to backfill
        from_date: Start date filter
        to_date: End date filter
        dry_run: Don't save, just preview
        verbose: Verbose output

    Returns:
        Summary dict with results
    """
    data_service = DataService()
    prediction_service = PredictionService()
    accuracy_service = AccuracyService()

    # Get fights to backfill
    fights = get_historical_fights(
        data_service,
        num_events=num_events,
        from_date=from_date,
        to_date=to_date
    )

    logger.info(f"Found {len(fights)} historical fights to process")

    results = {
        "total_fights": len(fights),
        "already_predicted": 0,
        "backfilled": 0,
        "failed": 0,
        "correct_predictions": 0,
        "events_processed": set(),
    }

    for fight in fights:
        event_name = fight["event_name"]
        results["events_processed"].add(event_name)

        # Check if already has prediction
        if check_existing_prediction(data_service, fight["fight_id"]):
            results["already_predicted"] += 1
            if verbose:
                logger.debug(f"Skip {fight['fighter_red_name']} vs {fight['fighter_blue_name']} - already predicted")
            continue

        # Generate backfill prediction
        prediction = backfill_fight(
            prediction_service,
            data_service,
            fight,
            dry_run=dry_run
        )

        if prediction:
            results["backfilled"] += 1

            # Check if prediction was correct
            if prediction["predicted_winner_id"] == fight["winner_id"]:
                results["correct_predictions"] += 1

            if verbose:
                winner_name = fight["fighter_red_name"] if fight["winner_id"] == fight["fighter_red_id"] else fight["fighter_blue_name"]
                predicted_name = fight["fighter_red_name"] if prediction["predicted_winner_id"] == fight["fighter_red_id"] else fight["fighter_blue_name"]
                correct = "OK" if prediction["predicted_winner_id"] == fight["winner_id"] else "WRONG"

                logger.info(
                    f"{fight['fighter_red_name']} vs {fight['fighter_blue_name']} | "
                    f"Predicted: {predicted_name} ({prediction['winner_confidence']*100:.0f}%) | "
                    f"Actual: {winner_name} | {correct}"
                )
        else:
            results["failed"] += 1
            if verbose:
                logger.warning(f"Failed to generate prediction for {fight['fighter_red_name']} vs {fight['fighter_blue_name']}")

    results["events_processed"] = len(results["events_processed"])

    # Update accuracy records for backfilled predictions
    if not dry_run and results["backfilled"] > 0:
        logger.info("Updating accuracy records for backfilled predictions...")
        accuracy_service.update_completed_predictions()

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Backfill predictions for historical fights"
    )
    parser.add_argument(
        "--events",
        type=int,
        default=10,
        help="Number of recent events to backfill (default: 10)"
    )
    parser.add_argument(
        "--from-date",
        type=str,
        help="Start date for backfill (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--to-date",
        type=str,
        help="End date for backfill (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't save predictions, just preview"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    print("=" * 60)
    print("UFC PREDICTION APP - BACKFILL PREDICTIONS")
    print("=" * 60)

    if args.dry_run:
        print("DRY RUN MODE - No predictions will be saved")
        print()

    # Run backfill
    results = run_backfill(
        num_events=args.events,
        from_date=args.from_date,
        to_date=args.to_date,
        dry_run=args.dry_run,
        verbose=args.verbose
    )

    # Print summary
    print()
    print("=" * 60)
    print("BACKFILL SUMMARY")
    print("=" * 60)
    print(f"Total fights found:    {results['total_fights']}")
    print(f"Already had prediction:{results['already_predicted']}")
    print(f"Backfilled:            {results['backfilled']}")
    print(f"Failed:                {results['failed']}")
    print(f"Events processed:      {results['events_processed']}")
    print()

    if results['backfilled'] > 0:
        accuracy = results['correct_predictions'] / results['backfilled'] * 100
        print(f"Backfill accuracy:     {accuracy:.1f}% ({results['correct_predictions']}/{results['backfilled']})")
        print()
        print("Note: Backfilled predictions are marked with is_backfill=True")
        print("      and is_valid_timing=False in the database.")

    if args.dry_run:
        print()
        print("Run without --dry-run to save predictions to database.")


if __name__ == "__main__":
    main()
