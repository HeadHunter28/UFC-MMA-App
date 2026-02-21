"""
Data Service Module.

Handles all database operations and queries for the UFC Prediction App.
"""

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATABASE_PATH, ensure_directories

logger = logging.getLogger(__name__)


class DataService:
    """
    Database operations and queries service.

    Provides methods for CRUD operations on fighters, fights, events,
    predictions, and related data.
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the data service.

        Args:
            db_path: Path to SQLite database. Defaults to config DATABASE_PATH.
        """
        self.db_path = db_path or str(DATABASE_PATH)
        ensure_directories()

    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
        finally:
            conn.close()

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert a sqlite3.Row to a dictionary."""
        if row is None:
            return None
        return dict(row)

    def _rows_to_list(self, rows: List[sqlite3.Row]) -> List[Dict[str, Any]]:
        """Convert a list of sqlite3.Row to a list of dictionaries."""
        return [self._row_to_dict(row) for row in rows]

    # =========================================================================
    # FIGHTER OPERATIONS
    # =========================================================================

    def get_fighter_by_id(self, fighter_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a fighter by their ID.

        Args:
            fighter_id: The fighter's database ID

        Returns:
            Fighter data as dict or None if not found
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM v_fighter_profile WHERE fighter_id = ?",
                (fighter_id,)
            )
            row = cursor.fetchone()
            return self._row_to_dict(row)

    def get_fighter_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a fighter by their name (exact match).

        Args:
            name: The fighter's name

        Returns:
            Fighter data as dict or None if not found
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM v_fighter_profile WHERE name = ?",
                (name,)
            )
            row = cursor.fetchone()
            return self._row_to_dict(row)

    def search_fighters(
        self,
        query: str,
        limit: int = 20,
        weight_class: Optional[str] = None,
        is_active: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """
        Search fighters by name with optional filters.

        Args:
            query: Search query (partial match)
            limit: Maximum results to return
            weight_class: Filter by weight class
            is_active: Filter by active status

        Returns:
            List of matching fighters
        """
        with self.get_connection() as conn:
            sql = "SELECT * FROM v_fighter_profile WHERE name LIKE ?"
            params = [f"%{query}%"]

            if is_active is not None:
                sql += " AND is_active = ?"
                params.append(is_active)

            sql += " ORDER BY wins DESC LIMIT ?"
            params.append(limit)

            cursor = conn.execute(sql, params)
            return self._rows_to_list(cursor.fetchall())

    def get_all_fighters(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get all fighters with optional filters and pagination.

        Args:
            filters: Dict of filter conditions (weight_class, nationality, is_active)
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of fighters
        """
        with self.get_connection() as conn:
            sql = "SELECT * FROM v_fighter_profile WHERE 1=1"
            params = []

            if filters:
                if "nationality" in filters:
                    sql += " AND nationality = ?"
                    params.append(filters["nationality"])
                if "is_active" in filters:
                    sql += " AND is_active = ?"
                    params.append(filters["is_active"])

            sql += " ORDER BY wins DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            cursor = conn.execute(sql, params)
            return self._rows_to_list(cursor.fetchall())

    def get_fighter_stats(self, fighter_id: int) -> Optional[Dict[str, Any]]:
        """
        Get detailed stats for a fighter.

        Args:
            fighter_id: The fighter's database ID

        Returns:
            Fighter stats as dict or None
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM fighter_stats WHERE fighter_id = ?",
                (fighter_id,)
            )
            row = cursor.fetchone()
            return self._row_to_dict(row)

    def get_fighter_fight_history(
        self,
        fighter_id: int,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get a fighter's fight history.

        Args:
            fighter_id: The fighter's database ID
            limit: Maximum fights to return

        Returns:
            List of fights
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT
                    f.*,
                    e.name as event_name,
                    e.date as event_date,
                    fr.name as opponent_name,
                    CASE
                        WHEN f.winner_id = ? THEN 'Win'
                        WHEN f.winner_id IS NULL THEN 'Draw/NC'
                        ELSE 'Loss'
                    END as result
                FROM fights f
                JOIN events e ON f.event_id = e.event_id
                JOIN fighters fr ON (
                    CASE
                        WHEN f.fighter_red_id = ? THEN f.fighter_blue_id
                        ELSE f.fighter_red_id
                    END = fr.fighter_id
                )
                WHERE f.fighter_red_id = ? OR f.fighter_blue_id = ?
                ORDER BY e.date DESC
                LIMIT ?
                """,
                (fighter_id, fighter_id, fighter_id, fighter_id, limit)
            )
            return self._rows_to_list(cursor.fetchall())

    def get_head_to_head_history(
        self,
        fighter_a_id: int,
        fighter_b_id: int
    ) -> List[Dict[str, Any]]:
        """
        Get head-to-head fight history between two fighters.

        Args:
            fighter_a_id: First fighter's database ID
            fighter_b_id: Second fighter's database ID

        Returns:
            List of past fights between the two fighters, ordered by date (most recent first)
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT
                    f.*,
                    e.name as event_name,
                    e.date as event_date,
                    fr.name as fighter_red_name,
                    fb.name as fighter_blue_name,
                    CASE
                        WHEN f.winner_id = ? THEN 'fighter_a'
                        WHEN f.winner_id = ? THEN 'fighter_b'
                        ELSE 'draw'
                    END as winner_key
                FROM fights f
                JOIN events e ON f.event_id = e.event_id
                JOIN fighters fr ON f.fighter_red_id = fr.fighter_id
                JOIN fighters fb ON f.fighter_blue_id = fb.fighter_id
                WHERE (f.fighter_red_id = ? AND f.fighter_blue_id = ?)
                   OR (f.fighter_red_id = ? AND f.fighter_blue_id = ?)
                ORDER BY e.date DESC
                """,
                (fighter_a_id, fighter_b_id, fighter_a_id, fighter_b_id, fighter_b_id, fighter_a_id)
            )
            return self._rows_to_list(cursor.fetchall())

    def get_fighter_record(self, fighter_id: int) -> Dict[str, int]:
        """
        Get a fighter's record (wins, losses, draws, no contests).

        Args:
            fighter_id: The fighter's database ID

        Returns:
            Dict with wins, losses, draws, no_contests
        """
        fighter = self.get_fighter_by_id(fighter_id)
        if not fighter:
            return {"wins": 0, "losses": 0, "draws": 0, "no_contests": 0}
        return {
            "wins": fighter.get("wins", 0),
            "losses": fighter.get("losses", 0),
            "draws": fighter.get("draws", 0),
            "no_contests": fighter.get("no_contests", 0),
        }

    def insert_fighter(self, fighter_data: Dict[str, Any]) -> int:
        """
        Insert a new fighter.

        Args:
            fighter_data: Dict with fighter attributes

        Returns:
            The new fighter's ID
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO fighters (
                    name, nickname, height_cm, weight_kg, reach_cm,
                    stance, dob, nationality, team, wins, losses,
                    draws, no_contests, image_url, ufc_stats_url, is_active
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fighter_data.get("name"),
                    fighter_data.get("nickname"),
                    fighter_data.get("height_cm"),
                    fighter_data.get("weight_kg"),
                    fighter_data.get("reach_cm"),
                    fighter_data.get("stance"),
                    fighter_data.get("dob"),
                    fighter_data.get("nationality"),
                    fighter_data.get("team"),
                    fighter_data.get("wins", 0),
                    fighter_data.get("losses", 0),
                    fighter_data.get("draws", 0),
                    fighter_data.get("no_contests", 0),
                    fighter_data.get("image_url"),
                    fighter_data.get("ufc_stats_url"),
                    fighter_data.get("is_active", True),
                )
            )
            conn.commit()
            return cursor.lastrowid

    def update_fighter(self, fighter_id: int, fighter_data: Dict[str, Any]) -> bool:
        """
        Update an existing fighter.

        Args:
            fighter_id: The fighter's ID
            fighter_data: Dict with updated attributes

        Returns:
            True if successful
        """
        with self.get_connection() as conn:
            # Build dynamic UPDATE statement
            updates = []
            params = []
            for key, value in fighter_data.items():
                if key != "fighter_id":
                    updates.append(f"{key} = ?")
                    params.append(value)

            if not updates:
                return False

            updates.append("updated_at = CURRENT_TIMESTAMP")
            params.append(fighter_id)

            sql = f"UPDATE fighters SET {', '.join(updates)} WHERE fighter_id = ?"
            conn.execute(sql, params)
            conn.commit()
            return True

    # =========================================================================
    # EVENT OPERATIONS
    # =========================================================================

    def get_event_by_id(self, event_id: int) -> Optional[Dict[str, Any]]:
        """Get an event by its ID."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM events WHERE event_id = ?",
                (event_id,)
            )
            return self._row_to_dict(cursor.fetchone())

    def get_event_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get an event by its name (case-insensitive partial match)."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM events WHERE name LIKE ? ORDER BY date DESC LIMIT 1",
                (f"%{name}%",)
            )
            return self._row_to_dict(cursor.fetchone())

    def get_events_by_date(self, date_str: str) -> List[Dict[str, Any]]:
        """Get events on a specific date."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM events WHERE date = ? ORDER BY event_id DESC",
                (date_str,)
            )
            return self._rows_to_list(cursor.fetchall())

    def get_upcoming_events(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get upcoming UFC events.

        Args:
            limit: Maximum events to return

        Returns:
            List of upcoming events
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM events
                WHERE is_completed = FALSE AND date >= date('now')
                ORDER BY date ASC
                LIMIT ?
                """,
                (limit,)
            )
            return self._rows_to_list(cursor.fetchall())

    def get_completed_events(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recently completed UFC events.

        Args:
            limit: Maximum events to return

        Returns:
            List of completed events
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM events
                WHERE is_completed = TRUE
                ORDER BY date DESC
                LIMIT ?
                """,
                (limit,)
            )
            return self._rows_to_list(cursor.fetchall())

    def insert_event(self, event_data: Dict[str, Any]) -> int:
        """Insert a new event."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO events (
                    name, date, location, venue, country, is_completed, ufc_stats_url
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_data.get("name"),
                    event_data.get("date"),
                    event_data.get("location"),
                    event_data.get("venue"),
                    event_data.get("country"),
                    event_data.get("is_completed", False),
                    event_data.get("ufc_stats_url"),
                )
            )
            conn.commit()
            return cursor.lastrowid

    # =========================================================================
    # FIGHT OPERATIONS
    # =========================================================================

    def get_fight_by_id(self, fight_id: int) -> Optional[Dict[str, Any]]:
        """Get a fight by its ID."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM v_fight_details WHERE fight_id = ?",
                (fight_id,)
            )
            return self._row_to_dict(cursor.fetchone())

    def get_fights_by_event(self, event_id: int) -> List[Dict[str, Any]]:
        """Get all fights for an event."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM v_fight_details
                WHERE event_id = ?
                ORDER BY is_main_event DESC, fight_id
                """,
                (event_id,)
            )
            return self._rows_to_list(cursor.fetchall())

    def get_upcoming_fights(self, event_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get upcoming fights, optionally filtered by event.

        Args:
            event_id: Optional event ID filter

        Returns:
            List of upcoming fights with fighter details
        """
        with self.get_connection() as conn:
            sql = """
                SELECT
                    uf.*,
                    e.name as event_name,
                    e.date as event_date,
                    e.location,
                    fr.name as fighter_red_name,
                    fr.wins as fighter_red_wins,
                    fr.losses as fighter_red_losses,
                    fb.name as fighter_blue_name,
                    fb.wins as fighter_blue_wins,
                    fb.losses as fighter_blue_losses
                FROM upcoming_fights uf
                JOIN events e ON uf.event_id = e.event_id
                JOIN fighters fr ON uf.fighter_red_id = fr.fighter_id
                JOIN fighters fb ON uf.fighter_blue_id = fb.fighter_id
                WHERE e.date >= date('now')
            """
            params = []

            if event_id:
                sql += " AND uf.event_id = ?"
                params.append(event_id)

            sql += " ORDER BY e.date, uf.bout_order DESC"

            cursor = conn.execute(sql, params)
            return self._rows_to_list(cursor.fetchall())

    def insert_fight(self, fight_data: Dict[str, Any]) -> int:
        """Insert a new fight."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO fights (
                    event_id, fighter_red_id, fighter_blue_id, winner_id,
                    weight_class, is_title_fight, is_main_event, method,
                    method_detail, round, time, referee, bonus
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fight_data.get("event_id"),
                    fight_data.get("fighter_red_id"),
                    fight_data.get("fighter_blue_id"),
                    fight_data.get("winner_id"),
                    fight_data.get("weight_class"),
                    fight_data.get("is_title_fight", False),
                    fight_data.get("is_main_event", False),
                    fight_data.get("method"),
                    fight_data.get("method_detail"),
                    fight_data.get("round"),
                    fight_data.get("time"),
                    fight_data.get("referee"),
                    fight_data.get("bonus"),
                )
            )
            conn.commit()
            return cursor.lastrowid

    def save_upcoming_fight(self, fight_data: Dict[str, Any]) -> Optional[int]:
        """
        Save an upcoming fight to the database.

        Args:
            fight_data: Dict with upcoming fight data including:
                - event_id: ID of the event
                - fighter_red_id: ID of fighter in red corner
                - fighter_blue_id: ID of fighter in blue corner
                - weight_class: Weight class of the fight
                - is_main_event: Whether this is the main event
                - is_title_fight: Whether this is a title fight
                - card_position: 'main_card', 'prelims', or 'early_prelims'
                - bout_order: Order of the fight on the card

        Returns:
            The new upcoming fight's ID or None if already exists
        """
        with self.get_connection() as conn:
            # Check if fight already exists
            cursor = conn.execute(
                """
                SELECT upcoming_id FROM upcoming_fights
                WHERE event_id = ? AND fighter_red_id = ? AND fighter_blue_id = ?
                """,
                (
                    fight_data.get("event_id"),
                    fight_data.get("fighter_red_id"),
                    fight_data.get("fighter_blue_id"),
                )
            )
            existing = cursor.fetchone()
            if existing:
                return existing[0]  # Return existing ID

            cursor = conn.execute(
                """
                INSERT INTO upcoming_fights (
                    event_id, fighter_red_id, fighter_blue_id, weight_class,
                    is_main_event, is_title_fight, card_position, bout_order
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fight_data.get("event_id"),
                    fight_data.get("fighter_red_id"),
                    fight_data.get("fighter_blue_id"),
                    fight_data.get("weight_class"),
                    fight_data.get("is_main_event", False),
                    fight_data.get("is_title_fight", False),
                    fight_data.get("card_position", "main_card"),
                    fight_data.get("bout_order"),
                )
            )
            conn.commit()
            return cursor.lastrowid

    def save_upcoming_fights_batch(self, event_id: int, fights: List[Dict[str, Any]]) -> int:
        """
        Save multiple upcoming fights for an event.

        Args:
            event_id: ID of the event
            fights: List of fight data dicts

        Returns:
            Number of fights saved
        """
        saved_count = 0
        for i, fight in enumerate(fights):
            fight["event_id"] = event_id
            if "bout_order" not in fight:
                fight["bout_order"] = len(fights) - i  # Higher = earlier on card
            result = self.save_upcoming_fight(fight)
            if result:
                saved_count += 1
        return saved_count

    def clear_upcoming_fights(self, event_id: int) -> int:
        """
        Clear all upcoming fights for an event (for re-scraping).

        Args:
            event_id: ID of the event

        Returns:
            Number of fights deleted
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM upcoming_fights WHERE event_id = ?",
                (event_id,)
            )
            conn.commit()
            return cursor.rowcount

    # =========================================================================
    # PREDICTION OPERATIONS
    # =========================================================================

    def save_prediction(self, prediction: Dict[str, Any]) -> int:
        """
        Save a prediction to the database.

        Args:
            prediction: Dict with prediction data

        Returns:
            The new prediction's ID
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO predictions (
                    fight_id, upcoming_id, fighter_red_id, fighter_blue_id,
                    predicted_winner_id, winner_confidence, method_ko_prob,
                    method_sub_prob, method_dec_prob, predicted_method,
                    predicted_round, feature_importance, top_factors,
                    model_version, is_backfill, event_date, is_valid_timing
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    prediction.get("fight_id"),
                    prediction.get("upcoming_id"),
                    prediction.get("fighter_red_id"),
                    prediction.get("fighter_blue_id"),
                    prediction.get("predicted_winner_id"),
                    prediction.get("winner_confidence"),
                    prediction.get("method_ko_prob"),
                    prediction.get("method_sub_prob"),
                    prediction.get("method_dec_prob"),
                    prediction.get("predicted_method"),
                    prediction.get("predicted_round"),
                    prediction.get("feature_importance"),
                    prediction.get("top_factors"),
                    prediction.get("model_version"),
                    prediction.get("is_backfill", False),
                    prediction.get("event_date"),
                    prediction.get("is_valid_timing", True),
                )
            )
            conn.commit()
            return cursor.lastrowid

    def save_prediction_stats_snapshot(
        self,
        prediction_id: int,
        fighter_stats: Dict[str, Any],
        corner: str
    ) -> int:
        """
        Save a snapshot of fighter stats at prediction time.

        Args:
            prediction_id: ID of the prediction
            fighter_stats: Dict with fighter statistics
            corner: 'red' or 'blue'

        Returns:
            The new snapshot ID
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO prediction_stats_snapshot (
                    prediction_id, fighter_id, corner,
                    wins, losses, draws,
                    sig_strikes_landed_per_min, sig_strikes_absorbed_per_min,
                    sig_strike_accuracy, sig_strike_defense,
                    takedowns_avg_per_15min, takedown_accuracy, takedown_defense,
                    submissions_avg_per_15min, finish_rate, ko_rate,
                    submission_rate, decision_rate,
                    win_streak, loss_streak, recent_form_score, days_since_last_fight
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    prediction_id,
                    fighter_stats.get("fighter_id"),
                    corner,
                    fighter_stats.get("wins"),
                    fighter_stats.get("losses"),
                    fighter_stats.get("draws"),
                    fighter_stats.get("sig_strikes_landed_per_min"),
                    fighter_stats.get("sig_strikes_absorbed_per_min"),
                    fighter_stats.get("sig_strike_accuracy"),
                    fighter_stats.get("sig_strike_defense"),
                    fighter_stats.get("takedowns_avg_per_15min"),
                    fighter_stats.get("takedown_accuracy"),
                    fighter_stats.get("takedown_defense"),
                    fighter_stats.get("submissions_avg_per_15min"),
                    fighter_stats.get("finish_rate"),
                    fighter_stats.get("ko_rate"),
                    fighter_stats.get("submission_rate"),
                    fighter_stats.get("decision_rate"),
                    fighter_stats.get("win_streak", 0),
                    fighter_stats.get("loss_streak", 0),
                    fighter_stats.get("recent_form_score"),
                    fighter_stats.get("days_since_last_fight"),
                )
            )
            conn.commit()
            return cursor.lastrowid

    def get_prediction_stats_snapshot(
        self,
        prediction_id: int
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get stats snapshot for a prediction.

        Args:
            prediction_id: ID of the prediction

        Returns:
            Dict with 'red' and 'blue' corner stats
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM prediction_stats_snapshot
                WHERE prediction_id = ?
                """,
                (prediction_id,)
            )
            rows = cursor.fetchall()

            result = {}
            for row in rows:
                row_dict = self._row_to_dict(row)
                corner = row_dict.get("corner")
                if corner:
                    result[corner] = row_dict

            return result

    def get_prediction(
        self,
        fight_id: Optional[int] = None,
        upcoming_id: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get a prediction for a fight.

        Args:
            fight_id: Historical fight ID
            upcoming_id: Upcoming fight ID

        Returns:
            Prediction data or None
        """
        with self.get_connection() as conn:
            if fight_id:
                cursor = conn.execute(
                    "SELECT * FROM predictions WHERE fight_id = ? ORDER BY created_at DESC LIMIT 1",
                    (fight_id,)
                )
            elif upcoming_id:
                cursor = conn.execute(
                    "SELECT * FROM predictions WHERE upcoming_id = ? ORDER BY created_at DESC LIMIT 1",
                    (upcoming_id,)
                )
            else:
                return None

            return self._row_to_dict(cursor.fetchone())

    # =========================================================================
    # STATISTICS OPERATIONS
    # =========================================================================

    def get_database_stats(self) -> Dict[str, int]:
        """
        Get database statistics (counts of all major tables).

        Returns:
            Dict with table counts
        """
        with self.get_connection() as conn:
            stats = {}

            tables = ["fighters", "events", "fights", "predictions"]
            for table in tables:
                cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                stats[table] = cursor.fetchone()[0]

            # Active fighters
            cursor = conn.execute(
                "SELECT COUNT(*) FROM fighters WHERE is_active = TRUE"
            )
            stats["active_fighters"] = cursor.fetchone()[0]

            return stats

    def get_weight_classes(self) -> List[str]:
        """Get distinct weight classes from fights."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT DISTINCT weight_class FROM fights WHERE weight_class IS NOT NULL ORDER BY weight_class"
            )
            return [row[0] for row in cursor.fetchall()]

    def get_countries(self) -> List[str]:
        """Get distinct nationalities from fighters."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT DISTINCT nationality FROM fighters WHERE nationality IS NOT NULL ORDER BY nationality"
            )
            return [row[0] for row in cursor.fetchall()]

    def get_win_method_distribution(
        self,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, int]:
        """
        Get distribution of win methods.

        Args:
            filters: Optional filters (weight_class, date_range)

        Returns:
            Dict with method counts
        """
        with self.get_connection() as conn:
            sql = """
                SELECT method, COUNT(*) as count
                FROM fights
                WHERE method IS NOT NULL
            """
            params = []

            if filters:
                if "weight_class" in filters:
                    sql += " AND weight_class = ?"
                    params.append(filters["weight_class"])

            sql += " GROUP BY method ORDER BY count DESC"

            cursor = conn.execute(sql, params)
            return {row["method"]: row["count"] for row in cursor.fetchall()}

    # =========================================================================
    # METADATA OPERATIONS
    # =========================================================================

    def get_metadata(self, key: str) -> Optional[str]:
        """Get a metadata value."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT value FROM app_metadata WHERE key = ?",
                (key,)
            )
            row = cursor.fetchone()
            return row["value"] if row else None

    def set_metadata(self, key: str, value: str) -> None:
        """Set a metadata value."""
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO app_metadata (key, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                """,
                (key, value)
            )
            conn.commit()

    def get_last_update_timestamp(self) -> Optional[datetime]:
        """Get the last data update timestamp."""
        value = self.get_metadata("last_scrape_date")
        if value:
            return datetime.fromisoformat(value)
        return None
