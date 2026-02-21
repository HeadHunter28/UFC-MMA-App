#!/usr/bin/env python3
"""
Sync Fighter Data from UFCStats.com.

This script:
1. Scrapes all fighters from UFCStats.com
2. Matches them to existing database fighters by name
3. Updates URLs and stats for matched fighters
4. Creates new fighters for unmatched ones

Usage:
    python scripts/sync_ufcstats.py [OPTIONS]

Options:
    --urls-only     Only update URLs, don't refresh stats
    --stats-only    Only refresh stats (assumes URLs exist)
    --limit N       Limit to N fighters for testing
    --verbose       Show detailed output
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from services.data_service import DataService
from services.scraper_service import UFCStatsScraper


def normalize_name(name: str) -> str:
    """Normalize fighter name for matching."""
    if not name:
        return ""
    # Lowercase, remove extra spaces, handle special chars
    name = name.lower().strip()
    name = " ".join(name.split())  # normalize whitespace
    return name


def match_fighters(db_fighters: list, ufc_fighters: list) -> dict:
    """Match UFC fighters to database fighters."""
    # Create lookup by normalized name
    db_by_name = {normalize_name(f[1]): f for f in db_fighters}

    matches = {}  # ufc_fighter -> db_fighter_id
    unmatched_ufc = []

    for ufc in ufc_fighters:
        ufc_name = normalize_name(ufc.get("name", ""))

        if ufc_name in db_by_name:
            matches[ufc_name] = (db_by_name[ufc_name][0], ufc)
        else:
            # Try partial matching (last name)
            ufc_parts = ufc_name.split()
            if ufc_parts:
                ufc_last = ufc_parts[-1]
                for db_norm, db_fighter in db_by_name.items():
                    db_parts = db_norm.split()
                    if db_parts and db_parts[-1] == ufc_last:
                        # Check first name initial or full match
                        if len(ufc_parts) > 1 and len(db_parts) > 1:
                            if ufc_parts[0] == db_parts[0] or ufc_parts[0][0] == db_parts[0][0]:
                                matches[ufc_name] = (db_fighter[0], ufc)
                                break

            if ufc_name not in matches:
                unmatched_ufc.append(ufc)

    return matches, unmatched_ufc


def sync_urls(verbose: bool = False):
    """Sync fighter URLs from UFCStats."""
    data_service = DataService()
    scraper = UFCStatsScraper()

    print("=" * 60)
    print("STEP 1: SYNCING FIGHTER URLs FROM UFCStats")
    print("=" * 60)

    # Get all fighters from database
    with data_service.get_connection() as conn:
        cursor = conn.execute("SELECT fighter_id, name FROM fighters")
        db_fighters = cursor.fetchall()
    print(f"Database fighters: {len(db_fighters)}")

    # Scrape all fighters from UFCStats
    print("\nScraping UFCStats.com (this takes a few minutes)...")
    ufc_fighters = scraper.scrape_all_fighters()
    print(f"UFCStats fighters: {len(ufc_fighters)}")

    # Match fighters
    print("\nMatching fighters...")
    matches, unmatched = match_fighters(db_fighters, ufc_fighters)
    print(f"Matched: {len(matches)}")
    print(f"Unmatched: {len(unmatched)}")

    # Update URLs
    print("\nUpdating URLs...")
    updated = 0
    with data_service.get_connection() as conn:
        for name, (fighter_id, ufc_data) in matches.items():
            url = ufc_data.get("ufc_stats_url")
            if url:
                conn.execute(
                    "UPDATE fighters SET ufc_stats_url = ?, updated_at = CURRENT_TIMESTAMP WHERE fighter_id = ?",
                    (url, fighter_id)
                )
                updated += 1

                if verbose:
                    print(f"  Updated URL for: {ufc_data.get('name')}")

        conn.commit()

    print(f"\nUpdated URLs for {updated} fighters")

    # Create new fighters for unmatched
    if unmatched:
        print(f"\nCreating {len(unmatched)} new fighters...")
        created = 0
        with data_service.get_connection() as conn:
            for ufc in unmatched:
                try:
                    cursor = conn.execute("""
                        INSERT INTO fighters (name, height_cm, weight_kg, reach_cm, stance, wins, losses, draws, ufc_stats_url, is_active)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, TRUE)
                    """, (
                        ufc.get("name"),
                        ufc.get("height_cm"),
                        ufc.get("weight_kg"),
                        ufc.get("reach_cm"),
                        ufc.get("stance"),
                        ufc.get("wins", 0),
                        ufc.get("losses", 0),
                        ufc.get("draws", 0),
                        ufc.get("ufc_stats_url"),
                    ))
                    created += 1
                except Exception as e:
                    if verbose:
                        print(f"  Could not create: {ufc.get('name')} - {e}")
            conn.commit()
        print(f"Created {created} new fighters")

    return updated


def sync_stats(limit: int = None, verbose: bool = False):
    """Sync fighter stats from UFCStats."""
    data_service = DataService()
    scraper = UFCStatsScraper()

    print("\n" + "=" * 60)
    print("STEP 2: REFRESHING FIGHTER STATS")
    print("=" * 60)

    # Get fighters with URLs
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

    print(f"Refreshing stats for {len(fighters)} fighters...")

    updated = 0
    errors = 0

    for idx, (fighter_id, name, url) in enumerate(fighters):
        if (idx + 1) % 100 == 0 or verbose:
            print(f"[{idx + 1}/{len(fighters)}] {name}")

        try:
            details = scraper.scrape_fighter_details(url)
            if not details:
                errors += 1
                continue

            with data_service.get_connection() as conn:
                # Update fighter record
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

                # Ensure stats row exists
                conn.execute("""
                    INSERT INTO fighter_stats (fighter_id)
                    SELECT ? WHERE NOT EXISTS (SELECT 1 FROM fighter_stats WHERE fighter_id = ?)
                """, (fighter_id, fighter_id))

                # Update stats
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
                print(f"  -> {details.get('wins', '?')}-{details.get('losses', '?')}, SLpM: {details.get('sig_strikes_landed_per_min', 'N/A')}")

        except Exception as e:
            errors += 1
            if verbose:
                print(f"  ERROR: {e}")

    print(f"\nUpdated: {updated} fighters")
    print(f"Errors: {errors} fighters")

    return updated


def main():
    parser = argparse.ArgumentParser(description="Sync fighter data from UFCStats.com")
    parser.add_argument("--urls-only", action="store_true", help="Only sync URLs, skip stats")
    parser.add_argument("--stats-only", action="store_true", help="Only sync stats (assumes URLs exist)")
    parser.add_argument("--limit", type=int, help="Limit stats refresh to N fighters")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    print("=" * 60)
    print("UFC PREDICTION APP - SYNC FROM UFCStats.com")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    if not args.stats_only:
        sync_urls(verbose=args.verbose)

    if not args.urls_only:
        sync_stats(limit=args.limit, verbose=args.verbose)

    # Update metadata
    data_service = DataService()
    data_service.set_metadata("last_ufcstats_sync", datetime.now().isoformat())

    print("\n" + "=" * 60)
    print("SYNC COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
