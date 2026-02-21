#!/usr/bin/env python3
"""
Update Fight History from UFCStats.com.

Scrapes recent completed events and adds missing fights to the database.

Usage:
    python scripts/update_fight_history.py [OPTIONS]

Options:
    --since DATE    Only scrape events after this date (YYYY-MM-DD)
    --limit N       Limit to N most recent events
    --verbose       Show detailed output
"""

import argparse
import sys
from datetime import datetime, date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from services.scraper_service import UFCStatsScraper
from services.data_service import DataService


def update_fight_history(since_date: str = None, limit: int = None, verbose: bool = False):
    """Update fight history from UFCStats.com."""
    scraper = UFCStatsScraper()
    data_service = DataService()

    print("=" * 60)
    print("UFC PREDICTION APP - UPDATE FIGHT HISTORY")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Get current latest event in DB
    with data_service.get_connection() as conn:
        cursor = conn.execute("SELECT MAX(date) FROM events WHERE is_completed = 1")
        db_latest = cursor.fetchone()[0]
        print(f"Latest event in DB: {db_latest}")

    # Use since_date if provided, otherwise use DB latest
    if since_date:
        cutoff = since_date
    elif db_latest:
        cutoff = db_latest
    else:
        cutoff = "2020-01-01"

    print(f"Fetching events after: {cutoff}")

    # Scrape all completed events
    print("\nScraping completed events from UFCStats...")
    all_events = scraper.scrape_all_events()
    print(f"Total events on UFCStats: {len(all_events)}")

    # Filter to events after cutoff
    new_events = [e for e in all_events if e.get("date") and e.get("date") > cutoff]
    print(f"New events to process: {len(new_events)}")

    if limit:
        new_events = new_events[:limit]
        print(f"Limited to: {limit} events")

    if not new_events:
        print("\nNo new events to process!")
        return 0

    # Sort by date (oldest first)
    new_events.sort(key=lambda x: x.get("date", ""))

    events_added = 0
    fights_added = 0

    for idx, event in enumerate(new_events):
        event_name = event.get("name", "Unknown")
        event_date = event.get("date")
        event_url = event.get("ufc_stats_url")

        if verbose or (idx + 1) % 5 == 0:
            print(f"\n[{idx + 1}/{len(new_events)}] {event_name} ({event_date})")

        if not event_url:
            print(f"  - No URL, skipping")
            continue

        # Scrape event details
        try:
            details = scraper.scrape_event_details(event_url)
            if not details:
                print(f"  - Failed to scrape details")
                continue
        except Exception as e:
            print(f"  - Error scraping: {e}")
            continue

        fights = details.get("fights", [])
        if verbose:
            print(f"  - Found {len(fights)} fights")

        # Get or create event in DB
        with data_service.get_connection() as conn:
            cursor = conn.execute(
                "SELECT event_id FROM events WHERE name = ? AND date = ?",
                (event_name, event_date)
            )
            row = cursor.fetchone()

            if row:
                event_id = row[0]
                # Mark as completed
                conn.execute(
                    "UPDATE events SET is_completed = 1 WHERE event_id = ?",
                    (event_id,)
                )
            else:
                cursor = conn.execute(
                    """
                    INSERT INTO events (name, date, location, is_completed, ufc_stats_url)
                    VALUES (?, ?, ?, 1, ?)
                    """,
                    (event_name, event_date, event.get("location"), event_url)
                )
                event_id = cursor.lastrowid
                events_added += 1

            conn.commit()

        # Process each fight
        for fight in fights:
            red_name = fight.get("fighter_red_name")
            blue_name = fight.get("fighter_blue_name")
            winner_name = fight.get("winner_name")

            if not red_name or not blue_name:
                continue

            # Get fighter IDs
            with data_service.get_connection() as conn:
                cursor = conn.execute(
                    "SELECT fighter_id FROM fighters WHERE name = ?", (red_name,)
                )
                red_row = cursor.fetchone()

                cursor = conn.execute(
                    "SELECT fighter_id FROM fighters WHERE name = ?", (blue_name,)
                )
                blue_row = cursor.fetchone()

                if not red_row or not blue_row:
                    if verbose:
                        print(f"  - Fighter not found: {red_name} or {blue_name}")
                    continue

                red_id = red_row[0]
                blue_id = blue_row[0]

                # Get winner ID
                winner_id = None
                if winner_name:
                    if winner_name == red_name:
                        winner_id = red_id
                    elif winner_name == blue_name:
                        winner_id = blue_id

                # Check if fight exists
                cursor = conn.execute(
                    """
                    SELECT fight_id FROM fights
                    WHERE event_id = ? AND fighter_red_id = ? AND fighter_blue_id = ?
                    """,
                    (event_id, red_id, blue_id)
                )
                if cursor.fetchone():
                    continue  # Fight already exists

                # Insert fight
                method = fight.get("method")
                method_normalized = None
                if method:
                    method_upper = method.upper()
                    if "KO" in method_upper or "TKO" in method_upper:
                        method_normalized = "KO/TKO"
                    elif "SUB" in method_upper:
                        method_normalized = "Submission"
                    elif "DEC" in method_upper:
                        method_normalized = "Decision"
                    else:
                        method_normalized = method

                try:
                    fight_round = int(fight.get("round")) if fight.get("round") else None
                except:
                    fight_round = None

                cursor = conn.execute(
                    """
                    INSERT INTO fights (
                        event_id, fighter_red_id, fighter_blue_id, winner_id,
                        weight_class, method, round, time
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event_id, red_id, blue_id, winner_id,
                        fight.get("weight_class"), method_normalized,
                        fight_round, fight.get("time")
                    )
                )
                fight_id = cursor.lastrowid
                fights_added += 1
                conn.commit()

                if verbose:
                    result = "W" if winner_id == red_id else "L" if winner_id else "?"
                    print(f"  - Added: {red_name} ({result}) vs {blue_name}")

                # Scrape and store per-fight statistics if fight URL available
                fight_url = fight.get("fight_url")
                if fight_url and fight_id:
                    try:
                        fight_stats = scraper.scrape_fight_stats(fight_url)
                        if fight_stats:
                            # Insert stats for red corner
                            red_stats = fight_stats.get("red", {})
                            conn.execute(
                                """
                                INSERT INTO fight_stats (
                                    fight_id, fighter_id, sig_strikes_landed, sig_strikes_attempted,
                                    total_strikes_landed, total_strikes_attempted, takedowns_landed,
                                    takedowns_attempted, submissions_attempted, knockdowns, control_time_seconds
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    fight_id, red_id,
                                    red_stats.get("sig_strikes_landed", 0),
                                    red_stats.get("sig_strikes_attempted", 0),
                                    red_stats.get("total_strikes_landed", 0),
                                    red_stats.get("total_strikes_attempted", 0),
                                    red_stats.get("takedowns_landed", 0),
                                    red_stats.get("takedowns_attempted", 0),
                                    red_stats.get("submission_attempts", 0),
                                    red_stats.get("knockdowns", 0),
                                    red_stats.get("control_time", 0)
                                )
                            )
                            # Insert stats for blue corner
                            blue_stats = fight_stats.get("blue", {})
                            conn.execute(
                                """
                                INSERT INTO fight_stats (
                                    fight_id, fighter_id, sig_strikes_landed, sig_strikes_attempted,
                                    total_strikes_landed, total_strikes_attempted, takedowns_landed,
                                    takedowns_attempted, submissions_attempted, knockdowns, control_time_seconds
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    fight_id, blue_id,
                                    blue_stats.get("sig_strikes_landed", 0),
                                    blue_stats.get("sig_strikes_attempted", 0),
                                    blue_stats.get("total_strikes_landed", 0),
                                    blue_stats.get("total_strikes_attempted", 0),
                                    blue_stats.get("takedowns_landed", 0),
                                    blue_stats.get("takedowns_attempted", 0),
                                    blue_stats.get("submission_attempts", 0),
                                    blue_stats.get("knockdowns", 0),
                                    blue_stats.get("control_time", 0)
                                )
                            )
                            conn.commit()
                            if verbose:
                                print(f"    -> Stats: Red {red_stats.get('sig_strikes_landed', 0)} sig.str vs Blue {blue_stats.get('sig_strikes_landed', 0)}")
                    except Exception as e:
                        if verbose:
                            print(f"    -> Failed to get fight stats: {e}")

    # Update metadata
    data_service.set_metadata("last_fight_history_update", datetime.now().isoformat())

    print("\n" + "=" * 60)
    print("UPDATE COMPLETE")
    print("=" * 60)
    print(f"Events added: {events_added}")
    print(f"Fights added: {fights_added}")

    # Show new latest
    with data_service.get_connection() as conn:
        cursor = conn.execute("SELECT MAX(date) FROM events WHERE is_completed = 1")
        new_latest = cursor.fetchone()[0]
        print(f"New latest event: {new_latest}")

    return fights_added


def main():
    parser = argparse.ArgumentParser(description="Update fight history from UFCStats.com")
    parser.add_argument("--since", type=str, help="Only scrape events after this date (YYYY-MM-DD)")
    parser.add_argument("--limit", type=int, help="Limit to N most recent events")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    update_fight_history(
        since_date=args.since,
        limit=args.limit,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
