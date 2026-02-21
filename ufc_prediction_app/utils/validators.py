"""
Validation Utilities Module.

Functions for validating data before database operations.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


def validate_fighter_data(fighter: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate fighter data before insert/update.

    Args:
        fighter: Fighter data dictionary

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []

    # Required fields
    if not fighter.get("name"):
        errors.append("Fighter name is required")

    # Validate ranges
    height = fighter.get("height_cm")
    if height is not None:
        try:
            h = float(height)
            if not (100 < h < 250):
                errors.append(f"Height {h}cm is outside valid range (100-250)")
        except (ValueError, TypeError):
            errors.append("Height must be a number")

    weight = fighter.get("weight_kg")
    if weight is not None:
        try:
            w = float(weight)
            if not (40 < w < 200):
                errors.append(f"Weight {w}kg is outside valid range (40-200)")
        except (ValueError, TypeError):
            errors.append("Weight must be a number")

    reach = fighter.get("reach_cm")
    if reach is not None:
        try:
            r = float(reach)
            if not (100 < r < 250):
                errors.append(f"Reach {r}cm is outside valid range (100-250)")
        except (ValueError, TypeError):
            errors.append("Reach must be a number")

    # Validate record
    for field in ["wins", "losses", "draws", "no_contests"]:
        value = fighter.get(field)
        if value is not None:
            try:
                v = int(value)
                if v < 0:
                    errors.append(f"{field} cannot be negative")
            except (ValueError, TypeError):
                errors.append(f"{field} must be an integer")

    # Validate DOB format
    dob = fighter.get("dob")
    if dob:
        try:
            if isinstance(dob, str):
                datetime.strptime(dob, "%Y-%m-%d")
        except ValueError:
            errors.append("DOB must be in YYYY-MM-DD format")

    # Validate stance
    valid_stances = ["Orthodox", "Southpaw", "Switch", None, ""]
    stance = fighter.get("stance")
    if stance and stance not in valid_stances:
        errors.append(f"Invalid stance: {stance}")

    return len(errors) == 0, errors


def validate_prediction_data(prediction: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate prediction data.

    Args:
        prediction: Prediction data dictionary

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []

    # Required fields
    required = ["fighter_red_id", "fighter_blue_id"]
    for field in required:
        if not prediction.get(field):
            errors.append(f"{field} is required")

    # Validate probabilities
    prob_fields = ["winner_confidence", "method_ko_prob", "method_sub_prob", "method_dec_prob"]
    for field in prob_fields:
        value = prediction.get(field)
        if value is not None:
            try:
                p = float(value)
                if not (0 <= p <= 1):
                    errors.append(f"{field} must be between 0 and 1")
            except (ValueError, TypeError):
                errors.append(f"{field} must be a number")

    # Validate method probabilities sum to ~1
    ko_prob = prediction.get("method_ko_prob", 0) or 0
    sub_prob = prediction.get("method_sub_prob", 0) or 0
    dec_prob = prediction.get("method_dec_prob", 0) or 0
    total_prob = ko_prob + sub_prob + dec_prob

    if total_prob > 0 and not (0.99 <= total_prob <= 1.01):
        errors.append(f"Method probabilities should sum to 1.0, got {total_prob:.2f}")

    # Validate round prediction
    predicted_round = prediction.get("predicted_round")
    if predicted_round is not None:
        try:
            r = float(predicted_round)
            if not (1 <= r <= 5):
                errors.append("Predicted round must be between 1 and 5")
        except (ValueError, TypeError):
            errors.append("Predicted round must be a number")

    return len(errors) == 0, errors


def validate_event_data(event: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate event data.

    Args:
        event: Event data dictionary

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []

    # Required fields
    if not event.get("name"):
        errors.append("Event name is required")

    if not event.get("date"):
        errors.append("Event date is required")

    # Validate date format
    date_value = event.get("date")
    if date_value:
        try:
            if isinstance(date_value, str):
                datetime.strptime(date_value, "%Y-%m-%d")
        except ValueError:
            errors.append("Date must be in YYYY-MM-DD format")

    return len(errors) == 0, errors


def validate_fight_data(fight: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate fight data.

    Args:
        fight: Fight data dictionary

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []

    # Required fields
    required = ["event_id", "fighter_red_id", "fighter_blue_id", "weight_class"]
    for field in required:
        if not fight.get(field):
            errors.append(f"{field} is required")

    # Validate fighters are different
    if fight.get("fighter_red_id") == fight.get("fighter_blue_id"):
        errors.append("Red and blue fighters must be different")

    # Validate round
    round_num = fight.get("round")
    if round_num is not None:
        try:
            r = int(round_num)
            if not (1 <= r <= 5):
                errors.append("Round must be between 1 and 5")
        except (ValueError, TypeError):
            errors.append("Round must be an integer")

    # Validate method if winner exists
    if fight.get("winner_id") and not fight.get("method"):
        errors.append("Method is required when winner is specified")

    return len(errors) == 0, errors


def sanitize_string(value: Optional[str], max_length: int = 255) -> Optional[str]:
    """
    Sanitize a string value.

    Args:
        value: String to sanitize
        max_length: Maximum allowed length

    Returns:
        Sanitized string or None
    """
    if value is None:
        return None

    # Strip whitespace
    value = str(value).strip()

    # Remove null characters
    value = value.replace("\x00", "")

    # Truncate if too long
    if len(value) > max_length:
        value = value[:max_length]

    return value if value else None


def validate_weight_class(weight_class: str) -> bool:
    """
    Validate weight class is a known value.

    Args:
        weight_class: Weight class string

    Returns:
        True if valid
    """
    valid_classes = [
        "Strawweight",
        "Flyweight",
        "Bantamweight",
        "Featherweight",
        "Lightweight",
        "Welterweight",
        "Middleweight",
        "Light Heavyweight",
        "Heavyweight",
        "Women's Strawweight",
        "Women's Flyweight",
        "Women's Bantamweight",
        "Women's Featherweight",
        "Catch Weight",
    ]

    return weight_class in valid_classes
