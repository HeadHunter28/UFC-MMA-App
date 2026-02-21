"""
Helper Utilities Module.

General helper functions for the UFC Prediction App.
"""

import logging
import re
from datetime import date, datetime
from typing import Any, Dict, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import LOG_LEVEL, LOG_FILE, LOGS_DIR


def setup_logging(
    verbose: bool = False,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging configuration.

    Args:
        verbose: Enable debug logging
        log_file: Optional log file path

    Returns:
        Configured logger
    """
    # Ensure logs directory exists
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    level = logging.DEBUG if verbose else getattr(logging, LOG_LEVEL, logging.INFO)

    # Configure root logger
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file or LOG_FILE, mode="a"),
        ],
    )

    return logging.getLogger(__name__)


def format_record(
    wins: int,
    losses: int,
    draws: int = 0,
    no_contests: int = 0
) -> str:
    """
    Format a fighter's record.

    Args:
        wins: Number of wins
        losses: Number of losses
        draws: Number of draws
        no_contests: Number of no contests

    Returns:
        Formatted record string (e.g., "20-3-0 (1 NC)")
    """
    record = f"{wins}-{losses}-{draws}"
    if no_contests:
        record += f" ({no_contests} NC)"
    return record


def calculate_age(dob: Optional[str]) -> Optional[int]:
    """
    Calculate age from date of birth.

    Args:
        dob: Date of birth string (YYYY-MM-DD)

    Returns:
        Age in years or None
    """
    if not dob:
        return None

    try:
        if isinstance(dob, str):
            birth_date = datetime.strptime(dob, "%Y-%m-%d").date()
        else:
            birth_date = dob

        today = date.today()
        age = today.year - birth_date.year

        # Adjust if birthday hasn't occurred this year
        if (today.month, today.day) < (birth_date.month, birth_date.day):
            age -= 1

        return age
    except (ValueError, TypeError):
        return None


def format_date(
    date_value: Any,
    format_str: str = "%B %d, %Y"
) -> str:
    """
    Format a date value.

    Args:
        date_value: Date string or date object
        format_str: Output format string

    Returns:
        Formatted date string
    """
    if not date_value:
        return "Unknown"

    try:
        if isinstance(date_value, str):
            dt = datetime.strptime(date_value, "%Y-%m-%d")
        elif isinstance(date_value, datetime):
            dt = date_value
        elif isinstance(date_value, date):
            dt = datetime.combine(date_value, datetime.min.time())
        else:
            return str(date_value)

        return dt.strftime(format_str)
    except (ValueError, TypeError):
        return str(date_value)


def slugify(text: str) -> str:
    """
    Convert text to URL-friendly slug.

    Args:
        text: Input text

    Returns:
        Slugified string
    """
    if not text:
        return ""

    # Convert to lowercase
    slug = text.lower()

    # Replace spaces with hyphens
    slug = slug.replace(" ", "-")

    # Remove special characters
    slug = re.sub(r"[^a-z0-9\-]", "", slug)

    # Remove multiple consecutive hyphens
    slug = re.sub(r"-+", "-", slug)

    # Remove leading/trailing hyphens
    slug = slug.strip("-")

    return slug


def parse_record(record_str: str) -> Dict[str, int]:
    """
    Parse a record string into wins, losses, draws.

    Args:
        record_str: Record string (e.g., "20-3-0")

    Returns:
        Dict with wins, losses, draws, no_contests
    """
    result = {"wins": 0, "losses": 0, "draws": 0, "no_contests": 0}

    if not record_str:
        return result

    # Pattern: W-L-D (NC)
    match = re.match(r"(\d+)-(\d+)-(\d+)(?:\s*\((\d+)\s*NC\))?", record_str.strip())
    if match:
        result["wins"] = int(match.group(1))
        result["losses"] = int(match.group(2))
        result["draws"] = int(match.group(3))
        if match.group(4):
            result["no_contests"] = int(match.group(4))

    return result


def calculate_win_rate(wins: int, losses: int, draws: int = 0) -> float:
    """
    Calculate win rate.

    Args:
        wins: Number of wins
        losses: Number of losses
        draws: Number of draws

    Returns:
        Win rate as float (0.0 - 1.0)
    """
    total = wins + losses + draws
    if total == 0:
        return 0.0
    return wins / total


def get_countdown(event_date: Any) -> str:
    """
    Get countdown string to an event.

    Args:
        event_date: Event date

    Returns:
        Countdown string (e.g., "3 days, 5 hours")
    """
    if not event_date:
        return "Unknown"

    try:
        if isinstance(event_date, str):
            target = datetime.strptime(event_date, "%Y-%m-%d")
        else:
            target = datetime.combine(event_date, datetime.min.time())

        now = datetime.now()
        delta = target - now

        if delta.total_seconds() < 0:
            return "Event has passed"

        days = delta.days
        hours = delta.seconds // 3600

        if days > 0:
            return f"{days} days, {hours} hours"
        elif hours > 0:
            return f"{hours} hours"
        else:
            minutes = delta.seconds // 60
            return f"{minutes} minutes"

    except (ValueError, TypeError):
        return "Unknown"


def normalize_method(method: str) -> str:
    """
    Normalize win method to standard categories.

    Args:
        method: Raw method string

    Returns:
        Normalized method (KO/TKO, Submission, Decision)
    """
    if not method:
        return "Unknown"

    method_upper = method.upper()

    if "KO" in method_upper or "TKO" in method_upper:
        return "KO/TKO"
    elif "SUB" in method_upper:
        return "Submission"
    elif "DEC" in method_upper:
        return "Decision"
    else:
        return method
