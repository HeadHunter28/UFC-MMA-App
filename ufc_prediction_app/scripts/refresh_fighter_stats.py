#!/usr/bin/env python3
"""
Refresh Fighter Stats from UFCStats.com.

Updates all fighter statistics by scraping their individual pages.

Usage:
    python scripts/refresh_fighter_stats.py [OPTIONS]

Options:
    --limit N       Only update N fighters (for testing)
    --fighter NAME  Update only a specific fighter
    --verbose       Show detailed output
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from services.data_service import DataService
from services.scraper_service import UFCStatsScraper


def refresh_all_fighters(limit: int = None, verbose: bool = False):
    """Refresh all fighter stats from UFCStats.com."""
    data_service = DataService()
    scraper = UFCStatsScraper()

    print("=" * 60)
    print("UFC PREDICTION APP - REFRESH FIGHTER STATS")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Get all fighters with UFC URLs
    with data_service.get_connection() as conn:
        cursor = conn.execute("""
            SELECT fighter_id, name, ufc_stats_url
            FROM fighters
            WHERE ufc_stats_url IS NOT NULL AND ufc_stats_url != ''
            ORDER BY updated_at ASC
        """)
        fighters = cursor.fetchall()

    if limit:
        fighters = fighters[:limit]

    print(f"\nFound {len(fighters)} fighters to update")

    updated = 0
    errors = 0

    for idx, (fighter_id, name, url) in enumerate(fighters):
        if verbose or (idx + 1) % 50 == 0:
            print(f"[{idx + 1}/{len(fighters)}] Updating: {name}")

        try:
            # Scrape fighter details
            details = scraper.scrape_fighter_details(url)
            if not details:
                if verbose:
                    print(f"  - No data found for {name}")
                errors += 1
                continue

            # Update fighter record
            with data_service.get_connection() as conn:
                # Update basic info
                conn.execute("""
                    UPDATE fighters SET
                        wins = COALESCE(?, wins),
                        losses = COALESCE(?, losses),
                        draws = COALESCE(?, draws),
                        height_cm = COALESCE(?, height_cm),
                        reach_cm = COALESCE(?, reach_cm),
                        stance = COALESCE(?, stance),
                        dob = COALESCE(?, dob),
                        updated_at = CURRENT_TIMESTAMP
                    WHERE fighter_id = ?
                """, (
                    details.get('wins'),
                    details.get('losses'),
                    details.get('draws'),
                    details.get('height_cm'),
                    details.get('reach_cm'),
                    details.get('stance'),
                    details.get('dob'),
                    fighter_id
                ))

                # Update stats
                conn.execute("""
                    INSERT INTO fighter_stats (fighter_id)
                    SELECT ? WHERE NOT EXISTS (
                        SELECT 1 FROM fighter_stats WHERE fighter_id = ?
                    )
                """, (fighter_id, fighter_id))

                conn.execute("""
                    UPDATE fighter_stats SET
                        sig_strikes_landed_per_min = COALESCE(?, sig_strikes_landed_per_min),
                        sig_strikes_absorbed_per_min = COALESCE(?, sig_strikes_absorbed_per_min),
                        sig_strike_accuracy = COALESCE(?, sig_strike_accuracy),
                        sig_strike_defense = COALESCE(?, sig_strike_defense),
                        takedowns_avg_per_15min = COALESCE(?, takedowns_avg_per_15min),
                        takedown_accuracy = COALESCE(?, takedown_accuracy),
                        takedown_defense = COALESCE(?, takedown_defense),
                        submissions_avg_per_15min = COALESCE(?, submissions_avg_per_15min),
                        updated_at = CURRENT_TIMESTAMP
                    WHERE fighter_id = ?
                """, (
                    details.get('sig_strikes_landed_per_min'),
                    details.get('sig_strikes_absorbed_per_min'),
                    details.get('sig_strike_accuracy'),
                    details.get('sig_strike_defense'),
                    details.get('takedowns_avg_per_15min'),
                    details.get('takedown_accuracy'),
                    details.get('takedown_defense'),
                    details.get('submissions_avg_per_15min'),
                    fighter_id
                ))

                conn.commit()

            updated += 1

            if verbose:
                print(f"  - Updated: {details.get('wins', '?')}-{details.get('losses', '?')}-{details.get('draws', 0)}")
                print(f"  - SLpM: {details.get('sig_strikes_landed_per_min', 'N/A')}, Str Acc: {details.get('sig_strike_accuracy', 'N/A')}")

        except Exception as e:
            if verbose:
                print(f"  - ERROR: {e}")
            errors += 1

    # Update metadata
    data_service.set_metadata("last_fighter_stats_refresh", datetime.now().isoformat())

    print("\n" + "=" * 60)
    print("REFRESH COMPLETE")
    print("=" * 60)
    print(f"Updated: {updated} fighters")
    print(f"Errors:  {errors} fighters")

    return updated


def refresh_single_fighter(name: str, verbose: bool = True):
    """Refresh stats for a single fighter."""
    data_service = DataService()
    scraper = UFCStatsScraper()

    print(f"Looking up: {name}")

    # Find fighter
    with data_service.get_connection() as conn:
        cursor = conn.execute("""
            SELECT fighter_id, name, ufc_stats_url
            FROM fighters
            WHERE name LIKE ?
            LIMIT 1
        """, (f"%{name}%",))
        row = cursor.fetchone()

    if not row:
        print(f"Fighter not found: {name}")
        return False

    fighter_id, full_name, url = row
    print(f"Found: {full_name} (ID: {fighter_id})")

    if not url:
        print("No UFC Stats URL - searching...")
        # Try to find on UFCStats
        all_fighters = scraper.scrape_all_fighters()
        matching = [f for f in all_fighters if name.lower() in f['name'].lower()]
        if matching:
            url = matching[0].get('ufc_stats_url')
            print(f"Found URL: {url}")
            with data_service.get_connection() as conn:
                conn.execute("UPDATE fighters SET ufc_stats_url = ? WHERE fighter_id = ?", (url, fighter_id))
                conn.commit()
        else:
            print("Could not find fighter on UFCStats")
            return False

    # Scrape details
    print(f"Scraping: {url}")
    details = scraper.scrape_fighter_details(url)

    if not details:
        print("Failed to scrape details")
        return False

    print(f"\nScraped data:")
    print(f"  Record: {details.get('wins', '?')}-{details.get('losses', '?')}-{details.get('draws', 0)}")
    print(f"  Height: {details.get('height_cm')} cm")
    print(f"  Reach: {details.get('reach_cm')} cm")
    print(f"  SLpM: {details.get('sig_strikes_landed_per_min')}")
    print(f"  SApM: {details.get('sig_strikes_absorbed_per_min')}")
    print(f"  Str Acc: {details.get('sig_strike_accuracy')}")
    print(f"  Str Def: {details.get('sig_strike_defense')}")
    print(f"  TD Avg: {details.get('takedowns_avg_per_15min')}")
    print(f"  TD Acc: {details.get('takedown_accuracy')}")
    print(f"  TD Def: {details.get('takedown_defense')}")
    print(f"  Sub Avg: {details.get('submissions_avg_per_15min')}")

    # Update database
    with data_service.get_connection() as conn:
        conn.execute("""
            UPDATE fighters SET
                wins = COALESCE(?, wins),
                losses = COALESCE(?, losses),
                draws = COALESCE(?, draws),
                height_cm = COALESCE(?, height_cm),
                reach_cm = COALESCE(?, reach_cm),
                stance = COALESCE(?, stance),
                dob = COALESCE(?, dob),
                updated_at = CURRENT_TIMESTAMP
            WHERE fighter_id = ?
        """, (
            details.get('wins'),
            details.get('losses'),
            details.get('draws'),
            details.get('height_cm'),
            details.get('reach_cm'),
            details.get('stance'),
            details.get('dob'),
            fighter_id
        ))

        conn.execute("""
            INSERT INTO fighter_stats (fighter_id)
            SELECT ? WHERE NOT EXISTS (
                SELECT 1 FROM fighter_stats WHERE fighter_id = ?
            )
        """, (fighter_id, fighter_id))

        conn.execute("""
            UPDATE fighter_stats SET
                sig_strikes_landed_per_min = COALESCE(?, sig_strikes_landed_per_min),
                sig_strikes_absorbed_per_min = COALESCE(?, sig_strikes_absorbed_per_min),
                sig_strike_accuracy = COALESCE(?, sig_strike_accuracy),
                sig_strike_defense = COALESCE(?, sig_strike_defense),
                takedowns_avg_per_15min = COALESCE(?, takedowns_avg_per_15min),
                takedown_accuracy = COALESCE(?, takedown_accuracy),
                takedown_defense = COALESCE(?, takedown_defense),
                submissions_avg_per_15min = COALESCE(?, submissions_avg_per_15min),
                updated_at = CURRENT_TIMESTAMP
            WHERE fighter_id = ?
        """, (
            details.get('sig_strikes_landed_per_min'),
            details.get('sig_strikes_absorbed_per_min'),
            details.get('sig_strike_accuracy'),
            details.get('sig_strike_defense'),
            details.get('takedowns_avg_per_15min'),
            details.get('takedown_accuracy'),
            details.get('takedown_defense'),
            details.get('submissions_avg_per_15min'),
            fighter_id
        ))

        conn.commit()

    print(f"\nUpdated {full_name} successfully!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Refresh fighter stats from UFCStats.com")
    parser.add_argument("--limit", type=int, help="Only update N fighters")
    parser.add_argument("--fighter", type=str, help="Update a specific fighter by name")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.fighter:
        refresh_single_fighter(args.fighter, verbose=args.verbose)
    else:
        refresh_all_fighters(limit=args.limit, verbose=args.verbose)


if __name__ == "__main__":
    main()
