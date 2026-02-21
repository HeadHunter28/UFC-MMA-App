#!/usr/bin/env python3
"""
Load Kaggle UFC Data Script.

Loads the Ultimate UFC Dataset from Kaggle into the application database.

Usage:
    python scripts/load_kaggle_data.py
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.data_service import DataService
from config import RAW_DATA_DIR


def load_kaggle_data():
    """Load Kaggle UFC data into the database."""
    data_service = DataService()

    csv_path = RAW_DATA_DIR / "kaggle" / "ufc-master.csv"
    if not csv_path.exists():
        print(f"ERROR: Dataset not found at {csv_path}")
        print("Run: kaggle datasets download -d mdabbert/ultimate-ufc-dataset")
        return False

    print("=" * 60)
    print("UFC PREDICTION APP - LOAD KAGGLE DATA")
    print("=" * 60)

    # Load the CSV
    print(f"\nLoading {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Found {len(df)} fights")

    # Replace NaN with None for database
    df = df.replace({np.nan: None})

    # Track progress
    fighters_created = 0
    events_created = 0
    fights_created = 0
    fighter_cache = {}  # name -> fighter_id
    event_cache = {}    # (date, location) -> event_id

    with data_service.get_connection() as conn:
        # Process each fight
        print("\nProcessing fights...")
        for idx, row in df.iterrows():
            if idx % 500 == 0:
                print(f"  Progress: {idx}/{len(df)}")

            # Extract fighter data
            red_name = row['RedFighter']
            blue_name = row['BlueFighter']

            # Get or create Red fighter
            if red_name not in fighter_cache:
                fighter_id = _get_or_create_fighter(
                    conn, red_name,
                    height_cm=row.get('RedHeightCms'),
                    reach_cm=row.get('RedReachCms'),
                    stance=row.get('RedStance'),
                    wins=row.get('RedWins', 0) or 0,
                    losses=row.get('RedLosses', 0) or 0,
                    draws=row.get('RedDraws', 0) or 0,
                )
                if fighter_id:
                    fighter_cache[red_name] = fighter_id
                    fighters_created += 1
            red_id = fighter_cache.get(red_name)

            # Get or create Blue fighter
            if blue_name not in fighter_cache:
                fighter_id = _get_or_create_fighter(
                    conn, blue_name,
                    height_cm=row.get('BlueHeightCms'),
                    reach_cm=row.get('BlueReachCms'),
                    stance=row.get('BlueStance'),
                    wins=row.get('BlueWins', 0) or 0,
                    losses=row.get('BlueLosses', 0) or 0,
                    draws=row.get('BlueDraws', 0) or 0,
                )
                if fighter_id:
                    fighter_cache[blue_name] = fighter_id
                    fighters_created += 1
            blue_id = fighter_cache.get(blue_name)

            if not red_id or not blue_id:
                continue

            # Get or create Event
            fight_date = row['Date']
            location = row.get('Location', 'Unknown')
            event_key = (fight_date, location)

            if event_key not in event_cache:
                event_id = _get_or_create_event(
                    conn,
                    date=fight_date,
                    location=location,
                    country=row.get('Country'),
                )
                if event_id:
                    event_cache[event_key] = event_id
                    events_created += 1
            event_id = event_cache.get(event_key)

            if not event_id:
                continue

            # Determine winner
            winner_id = None
            winner = row.get('Winner')
            if winner == 'Red':
                winner_id = red_id
            elif winner == 'Blue':
                winner_id = blue_id

            # Determine method
            method = _normalize_method(row.get('Finish'))
            method_detail = row.get('FinishDetails')

            # Create fight
            is_title = row.get('TitleBout', False)
            weight_class = row.get('WeightClass')
            fight_round = row.get('FinishRound')
            fight_time = row.get('FinishRoundTime')

            # Check if fight already exists
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
            cursor = conn.execute(
                """
                INSERT INTO fights (
                    event_id, fighter_red_id, fighter_blue_id, winner_id,
                    weight_class, is_title_fight, method, method_detail,
                    round, time
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id, red_id, blue_id, winner_id,
                    weight_class, is_title or False, method, method_detail,
                    fight_round, fight_time
                )
            )
            fights_created += 1

            # Update fighter stats
            _update_fighter_stats(conn, red_id, row, 'Red')
            _update_fighter_stats(conn, blue_id, row, 'Blue')

        conn.commit()

    # Update events as completed
    with data_service.get_connection() as conn:
        conn.execute("UPDATE events SET is_completed = TRUE WHERE date < date('now')")
        conn.commit()

    # Set last update timestamp
    data_service.set_metadata("last_kaggle_load", datetime.now().isoformat())
    data_service.set_metadata("data_source", "kaggle:mdabbert/ultimate-ufc-dataset")

    print("\n" + "=" * 60)
    print("LOAD COMPLETE")
    print("=" * 60)
    print(f"Fighters created: {fighters_created}")
    print(f"Events created:   {events_created}")
    print(f"Fights created:   {fights_created}")

    # Show database stats
    stats = data_service.get_database_stats()
    print("\nDatabase Stats:")
    print(f"  Total Fighters: {stats.get('fighters', 0)}")
    print(f"  Total Events:   {stats.get('events', 0)}")
    print(f"  Total Fights:   {stats.get('fights', 0)}")

    return True


def _get_or_create_fighter(conn, name, height_cm=None, reach_cm=None,
                           stance=None, wins=0, losses=0, draws=0):
    """Get existing fighter or create new one."""
    # Check if exists
    cursor = conn.execute(
        "SELECT fighter_id FROM fighters WHERE name = ?",
        (name,)
    )
    row = cursor.fetchone()
    if row:
        return row[0]

    # Create new fighter
    cursor = conn.execute(
        """
        INSERT INTO fighters (name, height_cm, reach_cm, stance, wins, losses, draws, is_active)
        VALUES (?, ?, ?, ?, ?, ?, ?, TRUE)
        """,
        (name, height_cm, reach_cm, stance, wins, losses, draws)
    )
    return cursor.lastrowid


def _get_or_create_event(conn, date, location, country=None):
    """Get existing event or create new one."""
    # Check if exists by date and location
    cursor = conn.execute(
        "SELECT event_id FROM events WHERE date = ? AND location = ?",
        (date, location)
    )
    row = cursor.fetchone()
    if row:
        return row[0]

    # Generate unique event name
    try:
        dt = datetime.strptime(date, "%Y-%m-%d")
        base_name = f"UFC Event {dt.strftime('%B %d, %Y')}"
    except ValueError:
        base_name = f"UFC Event {date}"

    # Check if name already exists and make unique
    cursor = conn.execute(
        "SELECT COUNT(*) FROM events WHERE name LIKE ?",
        (f"{base_name}%",)
    )
    count = cursor.fetchone()[0]
    event_name = f"{base_name} - {location[:20]}" if count > 0 else base_name

    # Create new event
    cursor = conn.execute(
        """
        INSERT INTO events (name, date, location, country, is_completed)
        VALUES (?, ?, ?, ?, FALSE)
        """,
        (event_name, date, location, country)
    )
    return cursor.lastrowid


def _normalize_method(method):
    """Normalize finish method to standard format."""
    if not method:
        return None
    method = str(method).upper()
    if method in ['KO', 'TKO']:
        return 'KO/TKO'
    elif method == 'SUB':
        return 'Submission'
    elif method in ['DEC', 'U-DEC', 'S-DEC', 'M-DEC']:
        return 'Decision'
    elif method == 'NC':
        return 'No Contest'
    elif method == 'DQ':
        return 'DQ'
    return method


def _update_fighter_stats(conn, fighter_id, row, corner):
    """Update fighter stats from fight data."""
    prefix = corner  # 'Red' or 'Blue'

    # Get stats from row
    sig_str_landed = row.get(f'{prefix}AvgSigStrLanded')
    sig_str_pct = row.get(f'{prefix}AvgSigStrPct')
    sub_att = row.get(f'{prefix}AvgSubAtt')
    td_landed = row.get(f'{prefix}AvgTDLanded')
    td_pct = row.get(f'{prefix}AvgTDPct')

    # Check if stats exist
    cursor = conn.execute(
        "SELECT stat_id FROM fighter_stats WHERE fighter_id = ?",
        (fighter_id,)
    )

    if cursor.fetchone():
        # Update existing stats (if new data is better)
        conn.execute(
            """
            UPDATE fighter_stats SET
                sig_strikes_landed_per_min = COALESCE(?, sig_strikes_landed_per_min),
                sig_strike_accuracy = COALESCE(?, sig_strike_accuracy),
                submissions_avg_per_15min = COALESCE(?, submissions_avg_per_15min),
                takedowns_avg_per_15min = COALESCE(?, takedowns_avg_per_15min),
                takedown_accuracy = COALESCE(?, takedown_accuracy),
                updated_at = CURRENT_TIMESTAMP
            WHERE fighter_id = ?
            """,
            (sig_str_landed, sig_str_pct, sub_att, td_landed, td_pct, fighter_id)
        )
    else:
        # Insert new stats
        conn.execute(
            """
            INSERT INTO fighter_stats (
                fighter_id, sig_strikes_landed_per_min, sig_strike_accuracy,
                submissions_avg_per_15min, takedowns_avg_per_15min, takedown_accuracy
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (fighter_id, sig_str_landed, sig_str_pct, sub_att, td_landed, td_pct)
        )


if __name__ == "__main__":
    success = load_kaggle_data()
    sys.exit(0 if success else 1)
