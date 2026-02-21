"""
Formatting Utilities Module.

Functions for formatting values for display.
"""

from typing import Optional, Union


def format_percentage(
    value: Optional[float],
    decimal_places: int = 0
) -> str:
    """
    Format a decimal value as percentage.

    Args:
        value: Value to format (0.0 - 1.0)
        decimal_places: Number of decimal places

    Returns:
        Formatted percentage string
    """
    if value is None:
        return "N/A"

    try:
        pct = float(value) * 100
        return f"{pct:.{decimal_places}f}%"
    except (ValueError, TypeError):
        return "N/A"


def format_height(
    height_cm: Optional[float],
    include_cm: bool = True
) -> str:
    """
    Format height in feet/inches with optional cm.

    Args:
        height_cm: Height in centimeters
        include_cm: Whether to include cm in output

    Returns:
        Formatted height string (e.g., "6'2\" (188cm)")
    """
    if height_cm is None:
        return "N/A"

    try:
        total_inches = height_cm / 2.54
        feet = int(total_inches // 12)
        inches = int(total_inches % 12)

        result = f"{feet}'{inches}\""
        if include_cm:
            result += f" ({height_cm:.0f}cm)"

        return result
    except (ValueError, TypeError):
        return "N/A"


def format_reach(
    reach_cm: Optional[float],
    include_cm: bool = True
) -> str:
    """
    Format reach in inches with optional cm.

    Args:
        reach_cm: Reach in centimeters
        include_cm: Whether to include cm in output

    Returns:
        Formatted reach string (e.g., '74" (188cm)')
    """
    if reach_cm is None:
        return "N/A"

    try:
        inches = reach_cm / 2.54
        result = f'{inches:.0f}"'
        if include_cm:
            result += f" ({reach_cm:.0f}cm)"
        return result
    except (ValueError, TypeError):
        return "N/A"


def format_time(time_str: Optional[str]) -> str:
    """
    Format fight time.

    Args:
        time_str: Time string (e.g., "4:35")

    Returns:
        Formatted time string
    """
    if not time_str:
        return "N/A"
    return time_str


def format_confidence(
    confidence: Optional[float],
    include_level: bool = True
) -> str:
    """
    Format confidence score.

    Args:
        confidence: Confidence value (0.0 - 1.0)
        include_level: Whether to include level indicator

    Returns:
        Formatted confidence string
    """
    if confidence is None:
        return "N/A"

    try:
        pct = float(confidence) * 100

        if include_level:
            if confidence > 0.65:
                level = "High"
            elif confidence > 0.55:
                level = "Medium"
            else:
                level = "Low"
            return f"{pct:.0f}% ({level})"

        return f"{pct:.0f}%"
    except (ValueError, TypeError):
        return "N/A"


def format_stat(
    value: Optional[float],
    suffix: str = "",
    decimal_places: int = 2
) -> str:
    """
    Format a general statistic.

    Args:
        value: Value to format
        suffix: Optional suffix (e.g., "/min")
        decimal_places: Number of decimal places

    Returns:
        Formatted stat string
    """
    if value is None:
        return "N/A"

    try:
        return f"{float(value):.{decimal_places}f}{suffix}"
    except (ValueError, TypeError):
        return "N/A"


def format_record_colored(
    wins: int,
    losses: int,
    draws: int = 0
) -> str:
    """
    Format record with HTML colors.

    Args:
        wins: Number of wins
        losses: Number of losses
        draws: Number of draws

    Returns:
        HTML-formatted record string
    """
    return (
        f'<span style="color: #28A745;">{wins}</span>-'
        f'<span style="color: #DC3545;">{losses}</span>-'
        f'{draws}'
    )


def format_weight_class(weight_class: Optional[str]) -> str:
    """
    Format weight class for display.

    Args:
        weight_class: Weight class string

    Returns:
        Formatted weight class
    """
    if not weight_class:
        return "Unknown"

    # Handle common abbreviations
    replacements = {
        "LW": "Lightweight",
        "WW": "Welterweight",
        "MW": "Middleweight",
        "LHW": "Light Heavyweight",
        "HW": "Heavyweight",
        "FW": "Featherweight",
        "BW": "Bantamweight",
        "FLW": "Flyweight",
        "SW": "Strawweight",
    }

    return replacements.get(weight_class.upper(), weight_class)


def format_method_short(method: Optional[str]) -> str:
    """
    Get short abbreviation for win method.

    Args:
        method: Full method string

    Returns:
        Short abbreviation
    """
    if not method:
        return "?"

    method_upper = method.upper()

    if "KO" in method_upper or "TKO" in method_upper:
        return "KO"
    elif "SUB" in method_upper:
        return "SUB"
    elif "U-DEC" in method_upper or "UNANIMOUS" in method_upper:
        return "UD"
    elif "S-DEC" in method_upper or "SPLIT" in method_upper:
        return "SD"
    elif "M-DEC" in method_upper or "MAJORITY" in method_upper:
        return "MD"
    elif "DEC" in method_upper:
        return "DEC"
    else:
        return method[:3].upper()


def format_duration(seconds: Optional[int]) -> str:
    """
    Format duration in seconds to readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string
    """
    if seconds is None:
        return "N/A"

    try:
        seconds = int(seconds)
        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            minutes = seconds // 60
            secs = seconds % 60
            return f"{minutes}m {secs}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours}h {minutes}m"
    except (ValueError, TypeError):
        return "N/A"
