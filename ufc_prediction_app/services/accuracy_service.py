"""
Accuracy Service Module.

Tracks and reports prediction accuracy for the ML models.
Includes timestamp validation, edge case handling, and comprehensive ground truth collection.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MIN_ACCURACY_THRESHOLD
from services.data_service import DataService

logger = logging.getLogger(__name__)


# Edge case outcome mappings
EDGE_CASE_METHODS = {
    "NC": "no_contest",
    "NO CONTEST": "no_contest",
    "DQ": "dq",
    "DISQUALIFICATION": "dq",
    "DRAW": "draw",
    "MAJORITY DRAW": "draw",
    "SPLIT DRAW": "draw",
    "TECHNICAL DRAW": "draw",
    "CANCELLED": "cancelled",
    "OVERTURNED": "no_contest",
}


class AccuracyService:
    """
    Service for tracking and analyzing prediction accuracy.

    Provides methods for updating accuracy records, calculating rolling
    accuracy, and generating performance reports.
    """

    def __init__(self):
        """Initialize the accuracy service."""
        self.data_service = DataService()

    def update_completed_predictions(self) -> int:
        """
        Update accuracy records for predictions of completed fights.

        Checks for predictions without accuracy records and calculates
        accuracy for fights that have completed. Includes timestamp validation
        and edge case handling.

        Returns:
            Number of predictions updated
        """
        logger.info("Updating prediction accuracy for completed fights...")
        updated = 0
        skipped_edge_cases = 0
        invalid_timing = 0

        with self.data_service.get_connection() as conn:
            # Get predictions without accuracy records - include event date for validation
            cursor = conn.execute("""
                SELECT p.*,
                       f.winner_id as actual_winner,
                       f.method as actual_method,
                       f.round as actual_round,
                       e.date as event_date,
                       p.prediction_timestamp,
                       p.event_date as prediction_event_date
                FROM predictions p
                JOIN fights f ON p.fight_id = f.fight_id
                JOIN events e ON f.event_id = e.event_id
                LEFT JOIN prediction_accuracy pa ON p.prediction_id = pa.prediction_id
                WHERE pa.accuracy_id IS NULL
            """)

            pending = cursor.fetchall()
            logger.info(f"Found {len(pending)} predictions to process")

            for row in pending:
                prediction = dict(row)

                # Determine fight outcome type
                fight_outcome = self._determine_fight_outcome(prediction)
                is_valid_for_accuracy = fight_outcome == "completed"

                # Validate prediction timing
                timing_validation = self._validate_prediction_timing(
                    prediction.get("prediction_timestamp") or prediction.get("created_at"),
                    prediction.get("event_date")
                )

                # Calculate accuracy (only for valid completed fights)
                if is_valid_for_accuracy and timing_validation["is_valid"]:
                    accuracy = self._calculate_accuracy(prediction)
                else:
                    accuracy = {
                        "winner_correct": None,
                        "method_correct": None,
                        "round_correct": None,
                    }
                    if not is_valid_for_accuracy:
                        skipped_edge_cases += 1
                    if not timing_validation["is_valid"]:
                        invalid_timing += 1

                # Build validation notes
                validation_notes = []
                if not timing_validation["is_valid"]:
                    validation_notes.append(f"Prediction made after fight: {timing_validation['hours_before']:.1f}h")
                if fight_outcome != "completed":
                    validation_notes.append(f"Fight outcome: {fight_outcome}")

                conn.execute("""
                    INSERT INTO prediction_accuracy (
                        prediction_id, fight_id, actual_winner_id,
                        actual_method, actual_round, winner_correct,
                        method_correct, round_correct, confidence_score,
                        prediction_made_before_fight, hours_before_fight,
                        fight_outcome, is_valid_for_accuracy, validation_notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    prediction["prediction_id"],
                    prediction["fight_id"],
                    prediction.get("actual_winner"),
                    prediction.get("actual_method"),
                    prediction.get("actual_round"),
                    accuracy["winner_correct"],
                    accuracy["method_correct"],
                    accuracy["round_correct"],
                    prediction.get("winner_confidence"),
                    timing_validation["is_valid"],
                    timing_validation["hours_before"],
                    fight_outcome,
                    is_valid_for_accuracy and timing_validation["is_valid"],
                    "; ".join(validation_notes) if validation_notes else None,
                ))
                updated += 1

            conn.commit()

        logger.info(f"Updated accuracy for {updated} predictions")
        if skipped_edge_cases:
            logger.info(f"  - {skipped_edge_cases} edge cases (NC/Draw/DQ)")
        if invalid_timing:
            logger.info(f"  - {invalid_timing} invalid timing (prediction after fight)")
        return updated

    def _determine_fight_outcome(self, prediction: Dict[str, Any]) -> str:
        """
        Determine the fight outcome type for edge case handling.

        Args:
            prediction: Dict with prediction and actual results

        Returns:
            Outcome type: 'completed', 'no_contest', 'draw', 'dq', 'cancelled'
        """
        actual_method = (prediction.get("actual_method") or "").upper()
        actual_winner = prediction.get("actual_winner")

        # Check for edge case methods
        for method_pattern, outcome in EDGE_CASE_METHODS.items():
            if method_pattern in actual_method:
                return outcome

        # No winner but fight happened
        if actual_winner is None and actual_method:
            if "DRAW" in actual_method:
                return "draw"
            return "no_contest"

        # Normal completed fight
        if actual_winner is not None:
            return "completed"

        return "cancelled"

    def _validate_prediction_timing(
        self,
        prediction_time: Optional[str],
        event_date: Optional[str]
    ) -> Dict[str, Any]:
        """
        Validate that prediction was made before the fight.

        Args:
            prediction_time: Timestamp of when prediction was made
            event_date: Date of the event

        Returns:
            Dict with is_valid and hours_before
        """
        result = {
            "is_valid": True,
            "hours_before": None,
        }

        if not prediction_time or not event_date:
            # Can't validate, assume valid
            return result

        try:
            # Parse timestamps
            if isinstance(prediction_time, str):
                pred_dt = datetime.fromisoformat(prediction_time.replace("Z", "+00:00"))
            else:
                pred_dt = prediction_time

            if isinstance(event_date, str):
                # Event date is typically just a date, assume fight is at noon
                event_dt = datetime.fromisoformat(event_date)
                if event_dt.hour == 0 and event_dt.minute == 0:
                    event_dt = event_dt.replace(hour=12)  # Assume noon start
            else:
                event_dt = event_date

            # Calculate hours difference
            time_diff = event_dt - pred_dt
            hours_before = time_diff.total_seconds() / 3600

            result["hours_before"] = hours_before
            result["is_valid"] = hours_before > 0  # Must be made before event

        except Exception as e:
            logger.warning(f"Error validating prediction timing: {e}")

        return result

    def _calculate_accuracy(self, prediction: Dict[str, Any]) -> Dict[str, Optional[bool]]:
        """
        Calculate accuracy for a single prediction.

        Args:
            prediction: Dict with prediction and actual results

        Returns:
            Dict with accuracy flags (None if not applicable)
        """
        actual_winner = prediction.get("actual_winner")
        actual_method = (prediction.get("actual_method") or "").upper()

        # Check for edge cases where accuracy doesn't apply
        for method_pattern in EDGE_CASE_METHODS:
            if method_pattern in actual_method:
                return {
                    "winner_correct": None,
                    "method_correct": None,
                    "round_correct": None,
                }

        # Winner accuracy
        winner_correct = prediction.get("predicted_winner_id") == actual_winner

        # Method accuracy (normalize method names)
        predicted_method = prediction.get("predicted_method", "").upper()

        # Group methods for matching
        ko_methods = ["KO", "TKO", "KO/TKO", "KNOCKOUT"]
        sub_methods = ["SUB", "SUBMISSION"]
        dec_methods = ["DEC", "DECISION", "U-DEC", "S-DEC", "M-DEC", "UNANIMOUS", "SPLIT", "MAJORITY"]
        doc_methods = ["DOCTOR", "DOCTOR STOPPAGE", "DOC"]

        method_correct = False

        # Match KO/TKO
        if any(m in predicted_method for m in ko_methods):
            method_correct = any(m in actual_method for m in ko_methods + doc_methods)
        # Match Submission
        elif any(m in predicted_method for m in sub_methods):
            method_correct = any(m in actual_method for m in sub_methods)
        # Match Decision
        elif any(m in predicted_method for m in dec_methods):
            method_correct = any(m in actual_method for m in dec_methods)

        # Round accuracy (within 1 round)
        predicted_round = prediction.get("predicted_round")
        actual_round = prediction.get("actual_round")
        round_correct = False

        if predicted_round is not None and actual_round is not None:
            try:
                round_correct = abs(float(predicted_round) - float(actual_round)) <= 1
            except (ValueError, TypeError):
                pass

        return {
            "winner_correct": winner_correct,
            "method_correct": method_correct,
            "round_correct": round_correct,
        }

    def get_rolling_accuracy(self, window: int = 100, valid_only: bool = True) -> float:
        """
        Calculate rolling accuracy over the last N predictions.

        Args:
            window: Number of predictions to consider
            valid_only: If True, only count predictions that are valid for accuracy

        Returns:
            Accuracy as a float (0.0 - 1.0)
        """
        with self.data_service.get_connection() as conn:
            # Build query with optional validity filter
            validity_clause = "WHERE is_valid_for_accuracy = 1" if valid_only else ""

            cursor = conn.execute(f"""
                SELECT
                    SUM(CASE WHEN winner_correct THEN 1 ELSE 0 END) as correct,
                    COUNT(*) as total
                FROM (
                    SELECT winner_correct
                    FROM prediction_accuracy
                    {validity_clause}
                    ORDER BY recorded_at DESC
                    LIMIT ?
                )
            """, (window,))

            row = cursor.fetchone()
            if row and row["total"] > 0:
                return row["correct"] / row["total"]
            return 0.0

    def get_accuracy_by_model(self, model_version: str, valid_only: bool = True) -> Dict[str, Any]:
        """
        Get accuracy breakdown for a specific model version.

        Args:
            model_version: The model version string
            valid_only: If True, only count valid predictions

        Returns:
            Dict with accuracy metrics
        """
        with self.data_service.get_connection() as conn:
            validity_clause = "AND pa.is_valid_for_accuracy = 1" if valid_only else ""

            cursor = conn.execute(f"""
                SELECT
                    COUNT(*) as total_predictions,
                    SUM(CASE WHEN pa.winner_correct THEN 1 ELSE 0 END) as winner_correct,
                    SUM(CASE WHEN pa.method_correct THEN 1 ELSE 0 END) as method_correct,
                    SUM(CASE WHEN pa.round_correct THEN 1 ELSE 0 END) as round_correct,
                    AVG(pa.confidence_score) as avg_confidence
                FROM prediction_accuracy pa
                JOIN predictions p ON pa.prediction_id = p.prediction_id
                WHERE p.model_version = ?
                {validity_clause}
            """, (model_version,))

            row = cursor.fetchone()
            if not row or row["total_predictions"] == 0:
                return {
                    "model_version": model_version,
                    "total_predictions": 0,
                    "winner_accuracy": 0.0,
                    "method_accuracy": 0.0,
                    "round_accuracy": 0.0,
                    "avg_confidence": 0.0,
                }

            total = row["total_predictions"]
            return {
                "model_version": model_version,
                "total_predictions": total,
                "winner_accuracy": row["winner_correct"] / total,
                "method_accuracy": row["method_correct"] / total,
                "round_accuracy": row["round_correct"] / total,
                "avg_confidence": row["avg_confidence"] or 0.0,
            }

    def get_accuracy_over_time(
        self,
        days: int = 365,
        window: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get rolling accuracy over time for charting.

        Args:
            days: Number of days to look back
            window: Rolling window size

        Returns:
            List of dicts with date and accuracy
        """
        results = []

        with self.data_service.get_connection() as conn:
            cursor = conn.execute("""
                SELECT
                    date(recorded_at) as date,
                    winner_correct
                FROM prediction_accuracy
                WHERE recorded_at >= date('now', ?)
                ORDER BY recorded_at
            """, (f"-{days} days",))

            records = cursor.fetchall()

        if not records:
            return results

        # Calculate rolling accuracy
        correct_window = []
        for record in records:
            correct_window.append(1 if record["winner_correct"] else 0)
            if len(correct_window) > window:
                correct_window.pop(0)

            if len(correct_window) >= window:
                accuracy = sum(correct_window) / len(correct_window)
                results.append({
                    "date": record["date"],
                    "accuracy": accuracy,
                    "window_size": len(correct_window),
                })

        return results

    def get_accuracy_by_confidence(self, valid_only: bool = True) -> Dict[str, Dict[str, float]]:
        """
        Get accuracy breakdown by confidence level.

        Args:
            valid_only: If True, only count valid predictions

        Returns:
            Dict with accuracy by confidence bucket
        """
        buckets = {
            "high": {"min": 0.65, "max": 1.0},
            "medium": {"min": 0.55, "max": 0.65},
            "low": {"min": 0.0, "max": 0.55},
        }

        results = {}
        validity_clause = "AND is_valid_for_accuracy = 1" if valid_only else ""

        with self.data_service.get_connection() as conn:
            for bucket_name, bounds in buckets.items():
                cursor = conn.execute(f"""
                    SELECT
                        COUNT(*) as total,
                        SUM(CASE WHEN winner_correct THEN 1 ELSE 0 END) as correct
                    FROM prediction_accuracy
                    WHERE confidence_score >= ? AND confidence_score < ?
                    {validity_clause}
                """, (bounds["min"], bounds["max"]))

                row = cursor.fetchone()
                total = row["total"] or 0
                correct = row["correct"] or 0

                results[bucket_name] = {
                    "total": total,
                    "correct": correct,
                    "accuracy": correct / total if total > 0 else 0.0,
                }

        return results

    def get_pending_accuracy_updates(self) -> List[Dict[str, Any]]:
        """
        Get predictions that need accuracy updates.

        Returns:
            List of predictions without accuracy records
        """
        with self.data_service.get_connection() as conn:
            cursor = conn.execute("""
                SELECT p.*
                FROM predictions p
                JOIN fights f ON p.fight_id = f.fight_id
                LEFT JOIN prediction_accuracy pa ON p.prediction_id = pa.prediction_id
                WHERE pa.accuracy_id IS NULL
                  AND f.winner_id IS NOT NULL
            """)
            return [dict(row) for row in cursor.fetchall()]

    def needs_retraining(self) -> bool:
        """
        Check if model retraining is needed based on accuracy.

        Returns:
            True if accuracy is below threshold
        """
        rolling_accuracy = self.get_rolling_accuracy(window=100)
        return rolling_accuracy < MIN_ACCURACY_THRESHOLD

    def get_accuracy_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive accuracy summary.

        Returns:
            Dict with all accuracy metrics
        """
        return {
            "rolling_30": self.get_rolling_accuracy(30),
            "rolling_100": self.get_rolling_accuracy(100),
            "by_confidence": self.get_accuracy_by_confidence(),
            "needs_retraining": self.needs_retraining(),
            "validation_stats": self.get_validation_statistics(),
            "edge_case_stats": self.get_edge_case_statistics(),
            "last_updated": datetime.now().isoformat(),
        }

    def get_edge_case_statistics(self) -> Dict[str, Any]:
        """
        Get statistics on edge case predictions (NC, Draw, DQ).

        Returns:
            Dict with edge case counts by outcome type
        """
        with self.data_service.get_connection() as conn:
            cursor = conn.execute("""
                SELECT
                    fight_outcome,
                    COUNT(*) as count
                FROM prediction_accuracy
                WHERE fight_outcome != 'completed'
                GROUP BY fight_outcome
            """)

            outcomes = {}
            total_edge_cases = 0
            for row in cursor.fetchall():
                outcome = row["fight_outcome"]
                count = row["count"]
                outcomes[outcome] = count
                total_edge_cases += count

            return {
                "total_edge_cases": total_edge_cases,
                "by_outcome": outcomes,
            }

    def get_validation_statistics(self) -> Dict[str, Any]:
        """
        Get statistics on prediction timing validation.

        Returns:
            Dict with validation statistics
        """
        with self.data_service.get_connection() as conn:
            cursor = conn.execute("""
                SELECT
                    COUNT(*) as total_predictions,
                    SUM(CASE WHEN prediction_made_before_fight = 1 THEN 1 ELSE 0 END) as valid_timing,
                    SUM(CASE WHEN prediction_made_before_fight = 0 THEN 1 ELSE 0 END) as invalid_timing,
                    SUM(CASE WHEN is_valid_for_accuracy = 1 THEN 1 ELSE 0 END) as valid_for_accuracy,
                    AVG(CASE WHEN hours_before_fight > 0 THEN hours_before_fight END) as avg_hours_before
                FROM prediction_accuracy
            """)

            row = cursor.fetchone()
            total = row["total_predictions"] or 0

            return {
                "total_predictions": total,
                "valid_timing": row["valid_timing"] or 0,
                "invalid_timing": row["invalid_timing"] or 0,
                "valid_for_accuracy": row["valid_for_accuracy"] or 0,
                "avg_hours_before_fight": round(row["avg_hours_before"] or 0, 1),
                "valid_timing_pct": (row["valid_timing"] or 0) / total * 100 if total > 0 else 0,
            }

    def get_method_accuracy_breakdown(self, valid_only: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        Get detailed accuracy breakdown by predicted method.

        Args:
            valid_only: If True, only count valid predictions

        Returns:
            Dict with accuracy stats per method type
        """
        validity_clause = "AND pa.is_valid_for_accuracy = 1" if valid_only else ""

        with self.data_service.get_connection() as conn:
            cursor = conn.execute(f"""
                SELECT
                    p.predicted_method,
                    COUNT(*) as total,
                    SUM(CASE WHEN pa.winner_correct THEN 1 ELSE 0 END) as winner_correct,
                    SUM(CASE WHEN pa.method_correct THEN 1 ELSE 0 END) as method_correct,
                    AVG(pa.confidence_score) as avg_confidence
                FROM prediction_accuracy pa
                JOIN predictions p ON pa.prediction_id = p.prediction_id
                WHERE p.predicted_method IS NOT NULL
                {validity_clause}
                GROUP BY p.predicted_method
            """)

            results = {}
            for row in cursor.fetchall():
                method = row["predicted_method"]
                total = row["total"]
                results[method] = {
                    "total": total,
                    "winner_accuracy": row["winner_correct"] / total if total > 0 else 0,
                    "method_accuracy": row["method_correct"] / total if total > 0 else 0,
                    "avg_confidence": row["avg_confidence"] or 0,
                }

            return results
