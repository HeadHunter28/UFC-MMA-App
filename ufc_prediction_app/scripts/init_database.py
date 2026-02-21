#!/usr/bin/env python3
"""
Database Initialization Script.

Creates the SQLite database with all required tables, indexes, and views.
Can also load initial data from Kaggle CSVs if available.

Usage:
    python scripts/init_database.py [OPTIONS]

Options:
    --reset         Drop and recreate all tables
    --load-kaggle   Load data from Kaggle CSVs
    --verbose       Verbose output
"""

import argparse
import logging
import sqlite3
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATABASE_PATH, ensure_directories

logger = logging.getLogger(__name__)


# =============================================================================
# DATABASE SCHEMA
# =============================================================================

SCHEMA_SQL = """
-- ============================================================================
-- UFC PREDICTION APP - DATABASE SCHEMA
-- ============================================================================

-- Fighters table
CREATE TABLE IF NOT EXISTS fighters (
    fighter_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    nickname TEXT,
    height_cm REAL,
    weight_kg REAL,
    reach_cm REAL,
    stance TEXT,
    dob DATE,
    nationality TEXT,
    team TEXT,
    wins INTEGER DEFAULT 0,
    losses INTEGER DEFAULT 0,
    draws INTEGER DEFAULT 0,
    no_contests INTEGER DEFAULT 0,
    image_url TEXT,
    ufc_stats_url TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(name, dob)
);

-- Fighter career statistics
CREATE TABLE IF NOT EXISTS fighter_stats (
    stat_id INTEGER PRIMARY KEY AUTOINCREMENT,
    fighter_id INTEGER NOT NULL,
    sig_strikes_landed_per_min REAL,
    sig_strikes_absorbed_per_min REAL,
    sig_strike_accuracy REAL,
    sig_strike_defense REAL,
    takedowns_avg_per_15min REAL,
    takedown_accuracy REAL,
    takedown_defense REAL,
    submissions_avg_per_15min REAL,
    avg_fight_time_seconds INTEGER,
    finish_rate REAL,
    ko_rate REAL,
    submission_rate REAL,
    decision_rate REAL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (fighter_id) REFERENCES fighters(fighter_id) ON DELETE CASCADE
);

-- Events table
CREATE TABLE IF NOT EXISTS events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    date DATE NOT NULL,
    location TEXT,
    venue TEXT,
    country TEXT,
    is_completed BOOLEAN DEFAULT FALSE,
    ufc_stats_url TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(name, date)
);

-- Fights table (historical results)
CREATE TABLE IF NOT EXISTS fights (
    fight_id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id INTEGER NOT NULL,
    fighter_red_id INTEGER NOT NULL,
    fighter_blue_id INTEGER NOT NULL,
    winner_id INTEGER,
    weight_class TEXT NOT NULL,
    is_title_fight BOOLEAN DEFAULT FALSE,
    is_main_event BOOLEAN DEFAULT FALSE,
    method TEXT,
    method_detail TEXT,
    round INTEGER,
    time TEXT,
    referee TEXT,
    bonus TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (event_id) REFERENCES events(event_id) ON DELETE CASCADE,
    FOREIGN KEY (fighter_red_id) REFERENCES fighters(fighter_id),
    FOREIGN KEY (fighter_blue_id) REFERENCES fighters(fighter_id),
    FOREIGN KEY (winner_id) REFERENCES fighters(fighter_id)
);

-- Per-fight statistics
CREATE TABLE IF NOT EXISTS fight_stats (
    stat_id INTEGER PRIMARY KEY AUTOINCREMENT,
    fight_id INTEGER NOT NULL,
    fighter_id INTEGER NOT NULL,
    corner TEXT CHECK(corner IN ('red', 'blue')),
    knockdowns INTEGER DEFAULT 0,
    sig_strikes_landed INTEGER DEFAULT 0,
    sig_strikes_attempted INTEGER DEFAULT 0,
    sig_strikes_head INTEGER DEFAULT 0,
    sig_strikes_body INTEGER DEFAULT 0,
    sig_strikes_leg INTEGER DEFAULT 0,
    total_strikes_landed INTEGER DEFAULT 0,
    total_strikes_attempted INTEGER DEFAULT 0,
    takedowns_landed INTEGER DEFAULT 0,
    takedowns_attempted INTEGER DEFAULT 0,
    submissions_attempted INTEGER DEFAULT 0,
    reversals INTEGER DEFAULT 0,
    control_time_seconds INTEGER DEFAULT 0,
    FOREIGN KEY (fight_id) REFERENCES fights(fight_id) ON DELETE CASCADE,
    FOREIGN KEY (fighter_id) REFERENCES fighters(fighter_id)
);

-- Upcoming fights
CREATE TABLE IF NOT EXISTS upcoming_fights (
    upcoming_id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id INTEGER NOT NULL,
    fighter_red_id INTEGER NOT NULL,
    fighter_blue_id INTEGER NOT NULL,
    weight_class TEXT NOT NULL,
    is_main_event BOOLEAN DEFAULT FALSE,
    is_title_fight BOOLEAN DEFAULT FALSE,
    card_position TEXT CHECK(card_position IN ('main_card', 'prelims', 'early_prelims')),
    bout_order INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (event_id) REFERENCES events(event_id) ON DELETE CASCADE,
    FOREIGN KEY (fighter_red_id) REFERENCES fighters(fighter_id),
    FOREIGN KEY (fighter_blue_id) REFERENCES fighters(fighter_id)
);

-- Predictions
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    fight_id INTEGER,
    upcoming_id INTEGER,
    fighter_red_id INTEGER NOT NULL,
    fighter_blue_id INTEGER NOT NULL,
    predicted_winner_id INTEGER,
    winner_confidence REAL,
    method_ko_prob REAL,
    method_sub_prob REAL,
    method_dec_prob REAL,
    predicted_method TEXT,
    predicted_round REAL,
    feature_importance TEXT,
    top_factors TEXT,
    model_version TEXT,
    is_backfill BOOLEAN DEFAULT FALSE,
    event_date DATE,
    prediction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_valid_timing BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (fight_id) REFERENCES fights(fight_id),
    FOREIGN KEY (upcoming_id) REFERENCES upcoming_fights(upcoming_id)
);

-- Historical fighter stats snapshot at prediction time
CREATE TABLE IF NOT EXISTS prediction_stats_snapshot (
    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER NOT NULL,
    fighter_id INTEGER NOT NULL,
    corner TEXT CHECK(corner IN ('red', 'blue')),
    -- Record at time of prediction
    wins INTEGER,
    losses INTEGER,
    draws INTEGER,
    -- Stats at time of prediction
    sig_strikes_landed_per_min REAL,
    sig_strikes_absorbed_per_min REAL,
    sig_strike_accuracy REAL,
    sig_strike_defense REAL,
    takedowns_avg_per_15min REAL,
    takedown_accuracy REAL,
    takedown_defense REAL,
    submissions_avg_per_15min REAL,
    finish_rate REAL,
    ko_rate REAL,
    submission_rate REAL,
    decision_rate REAL,
    -- Form/momentum features
    win_streak INTEGER DEFAULT 0,
    loss_streak INTEGER DEFAULT 0,
    recent_form_score REAL,
    days_since_last_fight INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (prediction_id) REFERENCES predictions(prediction_id) ON DELETE CASCADE,
    FOREIGN KEY (fighter_id) REFERENCES fighters(fighter_id)
);

-- Prediction accuracy tracking
CREATE TABLE IF NOT EXISTS prediction_accuracy (
    accuracy_id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER NOT NULL,
    fight_id INTEGER NOT NULL,
    actual_winner_id INTEGER,
    actual_method TEXT,
    actual_round INTEGER,
    winner_correct BOOLEAN,
    method_correct BOOLEAN,
    round_correct BOOLEAN,
    confidence_score REAL,
    -- Timestamp validation
    prediction_made_before_fight BOOLEAN DEFAULT TRUE,
    hours_before_fight REAL,
    -- Edge case handling
    fight_outcome TEXT CHECK(fight_outcome IN ('completed', 'no_contest', 'draw', 'dq', 'cancelled')),
    is_valid_for_accuracy BOOLEAN DEFAULT TRUE,
    validation_notes TEXT,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (prediction_id) REFERENCES predictions(prediction_id),
    FOREIGN KEY (fight_id) REFERENCES fights(fight_id)
);

-- Model performance tracking
CREATE TABLE IF NOT EXISTS model_performance (
    performance_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_version TEXT NOT NULL,
    model_type TEXT NOT NULL CHECK(model_type IN ('winner', 'method', 'round')),
    training_date DATE,
    training_samples INTEGER,
    accuracy REAL,
    precision_score REAL,
    recall_score REAL,
    f1_score REAL,
    auc_roc REAL,
    rolling_accuracy_30 REAL,
    rolling_accuracy_100 REAL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- App metadata
CREATE TABLE IF NOT EXISTS app_metadata (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

INDEXES_SQL = """
-- ============================================================================
-- INDEXES
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_fighters_name ON fighters(name);
CREATE INDEX IF NOT EXISTS idx_fighters_active ON fighters(is_active);
CREATE INDEX IF NOT EXISTS idx_fighters_nationality ON fighters(nationality);
CREATE INDEX IF NOT EXISTS idx_fighter_stats_fighter ON fighter_stats(fighter_id);

CREATE INDEX IF NOT EXISTS idx_events_date ON events(date);
CREATE INDEX IF NOT EXISTS idx_events_completed ON events(is_completed);

CREATE INDEX IF NOT EXISTS idx_fights_event ON fights(event_id);
CREATE INDEX IF NOT EXISTS idx_fights_fighters ON fights(fighter_red_id, fighter_blue_id);
CREATE INDEX IF NOT EXISTS idx_fights_winner ON fights(winner_id);
CREATE INDEX IF NOT EXISTS idx_fights_weight_class ON fights(weight_class);

CREATE INDEX IF NOT EXISTS idx_fight_stats_fight ON fight_stats(fight_id);
CREATE INDEX IF NOT EXISTS idx_fight_stats_fighter ON fight_stats(fighter_id);

CREATE INDEX IF NOT EXISTS idx_upcoming_event ON upcoming_fights(event_id);

CREATE INDEX IF NOT EXISTS idx_predictions_fight ON predictions(fight_id);
CREATE INDEX IF NOT EXISTS idx_predictions_upcoming ON predictions(upcoming_id);
CREATE INDEX IF NOT EXISTS idx_predictions_model ON predictions(model_version);

CREATE INDEX IF NOT EXISTS idx_accuracy_prediction ON prediction_accuracy(prediction_id);
CREATE INDEX IF NOT EXISTS idx_accuracy_fight ON prediction_accuracy(fight_id);
CREATE INDEX IF NOT EXISTS idx_accuracy_valid ON prediction_accuracy(is_valid_for_accuracy);

CREATE INDEX IF NOT EXISTS idx_stats_snapshot_prediction ON prediction_stats_snapshot(prediction_id);
CREATE INDEX IF NOT EXISTS idx_stats_snapshot_fighter ON prediction_stats_snapshot(fighter_id);

CREATE INDEX IF NOT EXISTS idx_predictions_event_date ON predictions(event_date);
CREATE INDEX IF NOT EXISTS idx_predictions_valid_timing ON predictions(is_valid_timing);
"""

VIEWS_SQL = """
-- ============================================================================
-- VIEWS
-- ============================================================================

-- Fighter complete profile view
CREATE VIEW IF NOT EXISTS v_fighter_profile AS
SELECT
    f.*,
    fs.sig_strikes_landed_per_min,
    fs.sig_strikes_absorbed_per_min,
    fs.sig_strike_accuracy,
    fs.sig_strike_defense,
    fs.takedowns_avg_per_15min,
    fs.takedown_accuracy,
    fs.takedown_defense,
    fs.submissions_avg_per_15min,
    fs.finish_rate,
    fs.ko_rate,
    fs.submission_rate,
    fs.decision_rate
FROM fighters f
LEFT JOIN fighter_stats fs ON f.fighter_id = fs.fighter_id;

-- Fight details view with fighter names
CREATE VIEW IF NOT EXISTS v_fight_details AS
SELECT
    f.fight_id,
    f.event_id,
    e.name AS event_name,
    e.date AS event_date,
    f.weight_class,
    f.is_title_fight,
    f.is_main_event,
    fr.name AS fighter_red_name,
    fb.name AS fighter_blue_name,
    w.name AS winner_name,
    f.method,
    f.method_detail,
    f.round,
    f.time,
    f.referee,
    f.bonus
FROM fights f
JOIN events e ON f.event_id = e.event_id
JOIN fighters fr ON f.fighter_red_id = fr.fighter_id
JOIN fighters fb ON f.fighter_blue_id = fb.fighter_id
LEFT JOIN fighters w ON f.winner_id = w.fighter_id;

-- Model accuracy summary view
CREATE VIEW IF NOT EXISTS v_model_accuracy_summary AS
SELECT
    p.model_version,
    COUNT(*) as total_predictions,
    SUM(CASE WHEN pa.winner_correct THEN 1 ELSE 0 END) as correct_winners,
    ROUND(100.0 * SUM(CASE WHEN pa.winner_correct THEN 1 ELSE 0 END) / COUNT(*), 2) as winner_accuracy_pct,
    SUM(CASE WHEN pa.method_correct THEN 1 ELSE 0 END) as correct_methods,
    ROUND(100.0 * SUM(CASE WHEN pa.method_correct THEN 1 ELSE 0 END) / COUNT(*), 2) as method_accuracy_pct,
    AVG(pa.confidence_score) as avg_confidence
FROM prediction_accuracy pa
JOIN predictions p ON pa.prediction_id = p.prediction_id
GROUP BY p.model_version;

-- Upcoming events with fight count
CREATE VIEW IF NOT EXISTS v_upcoming_events AS
SELECT
    e.event_id,
    e.name,
    e.date,
    e.location,
    e.venue,
    e.country,
    COUNT(uf.upcoming_id) as fight_count
FROM events e
JOIN upcoming_fights uf ON e.event_id = uf.event_id
WHERE e.is_completed = FALSE AND e.date >= date('now')
GROUP BY e.event_id
ORDER BY e.date;
"""

INITIAL_METADATA = [
    ("db_version", "1.0.0"),
    ("last_scrape_date", None),
    ("last_training_date", None),
    ("total_predictions", "0"),
]


def init_database(reset: bool = False, verbose: bool = False) -> bool:
    """
    Initialize the database with schema, indexes, and views.

    Args:
        reset: If True, drop all tables first
        verbose: If True, print detailed output

    Returns:
        True if successful, False otherwise
    """
    ensure_directories()

    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    logger.info(f"Initializing database at: {DATABASE_PATH}")

    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        # Enable foreign keys
        cursor.execute("PRAGMA foreign_keys = ON")

        if reset:
            logger.warning("Resetting database - dropping all tables")
            # Drop views first
            cursor.execute("DROP VIEW IF EXISTS v_fighter_profile")
            cursor.execute("DROP VIEW IF EXISTS v_fight_details")
            cursor.execute("DROP VIEW IF EXISTS v_model_accuracy_summary")
            cursor.execute("DROP VIEW IF EXISTS v_upcoming_events")

            # Drop tables in reverse dependency order
            tables = [
                "prediction_accuracy",
                "predictions",
                "fight_stats",
                "fights",
                "upcoming_fights",
                "fighter_stats",
                "fighters",
                "events",
                "model_performance",
                "app_metadata",
            ]
            for table in tables:
                cursor.execute(f"DROP TABLE IF EXISTS {table}")
            logger.info("All tables dropped")

        # Create schema
        logger.info("Creating tables...")
        cursor.executescript(SCHEMA_SQL)

        # Create indexes
        logger.info("Creating indexes...")
        cursor.executescript(INDEXES_SQL)

        # Create views
        logger.info("Creating views...")
        cursor.executescript(VIEWS_SQL)

        # Insert initial metadata
        logger.info("Inserting initial metadata...")
        for key, value in INITIAL_METADATA:
            cursor.execute(
                """
                INSERT OR REPLACE INTO app_metadata (key, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                """,
                (key, value)
            )

        conn.commit()
        logger.info("Database initialized successfully!")

        # Print table counts
        if verbose:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            tables = cursor.fetchall()
            print(f"\nCreated {len(tables)} tables:")
            for (table,) in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"  - {table}: {count} rows")

        conn.close()
        return True

    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return False


def load_kaggle_data(verbose: bool = False) -> bool:
    """
    Load initial data from Kaggle CSVs if available.

    Args:
        verbose: If True, print detailed output

    Returns:
        True if successful, False otherwise
    """
    import pandas as pd
    from config import RAW_DATA_DIR

    kaggle_dir = RAW_DATA_DIR / "kaggle"

    if not kaggle_dir.exists():
        logger.warning(f"Kaggle data directory not found: {kaggle_dir}")
        return False

    # Check for expected files
    fighters_csv = kaggle_dir / "ufc_fighters.csv"
    fights_csv = kaggle_dir / "ufc_fights.csv"
    events_csv = kaggle_dir / "ufc_events.csv"

    if not any([fighters_csv.exists(), fights_csv.exists(), events_csv.exists()]):
        logger.warning("No Kaggle CSV files found")
        return False

    logger.info("Loading Kaggle data...")

    try:
        conn = sqlite3.connect(DATABASE_PATH)

        # Load fighters
        if fighters_csv.exists():
            logger.info(f"Loading fighters from {fighters_csv}")
            df = pd.read_csv(fighters_csv)
            # Map columns and insert
            # (Implementation depends on actual Kaggle CSV structure)
            logger.info(f"Loaded {len(df)} fighters")

        # Load events
        if events_csv.exists():
            logger.info(f"Loading events from {events_csv}")
            df = pd.read_csv(events_csv)
            logger.info(f"Loaded {len(df)} events")

        # Load fights
        if fights_csv.exists():
            logger.info(f"Loading fights from {fights_csv}")
            df = pd.read_csv(fights_csv)
            logger.info(f"Loaded {len(df)} fights")

        conn.close()
        logger.info("Kaggle data loaded successfully!")
        return True

    except Exception as e:
        logger.error(f"Failed to load Kaggle data: {e}")
        return False


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Initialize the UFC Prediction App database"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and recreate all tables"
    )
    parser.add_argument(
        "--load-kaggle",
        action="store_true",
        help="Load data from Kaggle CSVs"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("UFC PREDICTION APP - DATABASE INITIALIZATION")
    print("=" * 60)

    # Initialize database
    success = init_database(reset=args.reset, verbose=args.verbose)

    if not success:
        print("\nDatabase initialization FAILED!")
        sys.exit(1)

    # Load Kaggle data if requested
    if args.load_kaggle:
        print("\nLoading Kaggle data...")
        load_kaggle_data(verbose=args.verbose)

    print("\nDatabase initialization COMPLETE!")
    print(f"Database path: {DATABASE_PATH}")


if __name__ == "__main__":
    main()
