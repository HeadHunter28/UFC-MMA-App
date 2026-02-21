#!/usr/bin/env python3
"""
Backfill Fight Stats from UFCStats.com.

Scrapes per-fight statistics for existing fights that don't have stats yet.
This enables features like "Comeback Kings" (submission wins while being out-struck).

Usage:
    python scripts/backfill_fight_stats.py [OPTIONS]

Options:
    --limit N       Limit to N fights
    --verbose       Show detailed output
    --event NAME    Only process fights from events matching NAME
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from services.scraper_service import UFCStatsScraper
from services.data_service import DataService


def backfill_fight_stats(limit: int = None, verbose: bool = False, event_filter: str = None):
    """Backfill fight stats for existing fights."""
    scraper = UFCStatsScraper()
    data_service = DataService()

    print("=" * 60)
    print("UFC PREDICTION APP - BACKFILL FIGHT STATS")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Get fights that don't have stats yet
    with data_service.get_connection() as conn:
        # Count total fights
        cursor = conn.execute("SELECT COUNT(*) FROM fights")
        total_fights = cursor.fetchone()[0]

        # Count fights with stats
        cursor = conn.execute("SELECT COUNT(DISTINCT fight_id) FROM fight_stats")
        fights_with_stats = cursor.fetchone()[0]

        print(f"Total fights in database: {total_fights}")
        print(f"Fights with stats: {fights_with_stats}")
        print(f"Fights needing stats: {total_fights - fights_with_stats}")

    # Get events with their UFCStats URLs
    with data_service.get_connection() as conn:
        query = """
            SELECT DISTINCT e.event_id, e.name, e.date, e.ufc_stats_url
            FROM events e
            JOIN fights f ON e.event_id = f.event_id
            WHERE e.ufc_stats_url IS NOT NULL
            AND f.fight_id NOT IN (SELECT DISTINCT fight_id FROM fight_stats)
        """
        if event_filter:
            query += f" AND e.name LIKE '%{event_filter}%'"
        query += " ORDER BY e.date DESC"

        cursor = conn.execute(query)
        events_needing_stats = cursor.fetchall()

    print(f"\nEvents with fights needing stats: {len(events_needing_stats)}")

    if not events_needing_stats:
        print("No fights need stats backfilling!")
        return 0

    stats_added = 0
    events_processed = 0

    for event_id, event_name, event_date, event_url in events_needing_stats:
        print(f"\n[{events_processed + 1}/{len(events_needing_stats)}] {event_name} ({event_date})")

        # Scrape event to get fight URLs
        try:
            event_details = scraper.scrape_event_details(event_url)
            if not event_details:
                print(f"  - Failed to scrape event")
                continue
        except Exception as e:
            print(f"  - Error: {e}")
            continue

        fights_in_event = event_details.get("fights", [])
        if verbose:
            print(f"  - Found {len(fights_in_event)} fights in event")

        # Get fights from DB that need stats
        with data_service.get_connection() as conn:
            cursor = conn.execute("""
                SELECT f.fight_id, f.fighter_red_id, f.fighter_blue_id,
                       fr.name as red_name, fb.name as blue_name
                FROM fights f
                JOIN fighters fr ON f.fighter_red_id = fr.fighter_id
                JOIN fighters fb ON f.fighter_blue_id = fb.fighter_id
                WHERE f.event_id = ?
                AND f.fight_id NOT IN (SELECT DISTINCT fight_id FROM fight_stats)
            """, (event_id,))
            db_fights = cursor.fetchall()

        if verbose:
            print(f"  - {len(db_fights)} fights need stats")

        # Match scraped fights with DB fights
        for fight_id, red_id, blue_id, red_name, blue_name in db_fights:
            # Find matching fight in scraped data
            fight_url = None
            for scraped_fight in fights_in_event:
                scraped_red = scraped_fight.get("fighter_red_name", "").strip()
                scraped_blue = scraped_fight.get("fighter_blue_name", "").strip()

                # Match by fighter names
                if (scraped_red == red_name and scraped_blue == blue_name) or \
                   (scraped_red == blue_name and scraped_blue == red_name):
                    fight_url = scraped_fight.get("fight_url")
                    break

            if not fight_url:
                if verbose:
                    print(f"    - No URL for {red_name} vs {blue_name}")
                continue

            # Scrape fight stats
            try:
                fight_stats = scraper.scrape_fight_stats(fight_url)
                if not fight_stats:
                    if verbose:
                        print(f"    - No stats for {red_name} vs {blue_name}")
                    continue

                # Determine which corner is which
                stats_red_name = fight_stats.get("red", {}).get("name", "")
                if stats_red_name == red_name:
                    red_stats = fight_stats.get("red", {})
                    blue_stats = fight_stats.get("blue", {})
                else:
                    # Names are swapped
                    red_stats = fight_stats.get("blue", {})
                    blue_stats = fight_stats.get("red", {})

                # Insert stats
                with data_service.get_connection() as conn:
                    # Red corner stats
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO fight_stats (
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
                    # Blue corner stats
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO fight_stats (
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

                stats_added += 1
                if verbose:
                    print(f"    + {red_name} ({red_stats.get('sig_strikes_landed', 0)} sig.str) vs {blue_name} ({blue_stats.get('sig_strikes_landed', 0)} sig.str)")

                # Rate limit to avoid overloading the server
                time.sleep(0.5)

            except Exception as e:
                if verbose:
                    print(f"    - Error for {red_name} vs {blue_name}: {e}")

            if limit and stats_added >= limit:
                print(f"\nReached limit of {limit} fights")
                break

        events_processed += 1

        if limit and stats_added >= limit:
            break

    # Update metadata
    data_service.set_metadata("last_fight_stats_backfill", datetime.now().isoformat())

    print("\n" + "=" * 60)
    print("BACKFILL COMPLETE")
    print("=" * 60)
    print(f"Events processed: {events_processed}")
    print(f"Fight stats added: {stats_added}")

    # Show updated counts
    with data_service.get_connection() as conn:
        cursor = conn.execute("SELECT COUNT(DISTINCT fight_id) FROM fight_stats")
        new_count = cursor.fetchone()[0]
        print(f"Total fights with stats now: {new_count}")

    return stats_added


def main():
    parser = argparse.ArgumentParser(description="Backfill fight stats from UFCStats.com")
    parser.add_argument("--limit", type=int, help="Limit to N fights")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--event", type=str, help="Only process events matching this name")

    args = parser.parse_args()

    backfill_fight_stats(
        limit=args.limit,
        verbose=args.verbose,
        event_filter=args.event
    )


if __name__ == "__main__":
    main()
