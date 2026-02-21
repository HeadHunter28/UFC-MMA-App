"""
Fight Simulation Service Module.

Provides multi-model fight simulation with round-by-round breakdown.
Enhanced with physical attributes, age factors, and sanity checks.
"""

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    SIMULATION_MODELS,
    SIMULATION_DEFAULT_ROUNDS,
    FINISH_PROB_BY_ROUND,
    FIGHTER_ACTIVITY_CUTOFF_YEARS,
    METHOD_CLASSES,
    WEIGHT_CLASSES,
    get_confidence_level,
)
from services.data_service import DataService

logger = logging.getLogger(__name__)


# Weight class weight limits (in lbs) for comparison
WEIGHT_CLASS_LIMITS = {
    "Strawweight": 115,
    "Women's Strawweight": 115,
    "Flyweight": 125,
    "Women's Flyweight": 125,
    "Bantamweight": 135,
    "Women's Bantamweight": 135,
    "Featherweight": 145,
    "Women's Featherweight": 145,
    "Lightweight": 155,
    "Welterweight": 170,
    "Middleweight": 185,
    "Light Heavyweight": 205,
    "Heavyweight": 265,
    "Catch Weight": 180,  # Approximate
    "Open Weight": 265,
}

# Weight class finish rate modifiers (based on UFC statistics)
# Heavier weight classes have higher finish rates
WEIGHT_CLASS_FINISH_MODIFIER = {
    "Heavyweight": 1.40,           # ~70% finish rate
    "Light Heavyweight": 1.25,     # ~62% finish rate
    "Middleweight": 1.15,          # ~58% finish rate
    "Welterweight": 1.05,          # ~52% finish rate
    "Lightweight": 1.00,           # ~50% finish rate (baseline)
    "Featherweight": 0.95,         # ~48% finish rate
    "Bantamweight": 0.90,          # ~45% finish rate
    "Flyweight": 0.85,             # ~42% finish rate
    "Women's Featherweight": 0.90,
    "Women's Bantamweight": 0.85,
    "Women's Flyweight": 0.80,
    "Women's Strawweight": 0.75,   # ~38% finish rate
    "Strawweight": 0.80,
    "Catch Weight": 1.00,
    "Open Weight": 1.30,
}

# RECOMMENDATION 4: Weight class-specific stat importance adjustments
# Based on test results - some divisions have different dynamics
# Positive = physical stats matter more, Negative = technique/cardio matters more
WEIGHT_CLASS_STAT_ADJUSTMENT = {
    "Heavyweight": 0.08,           # Physical stats very important
    "Light Heavyweight": 0.05,     # Physical stats important
    "Middleweight": -0.03,         # More technical, slight reduction (underperformed at 52.9%)
    "Welterweight": 0.02,          # Well-rounded division
    "Lightweight": -0.05,          # Very technical, cardio important (underperformed at 54.2%)
    "Featherweight": 0.0,          # Balanced
    "Bantamweight": 0.0,           # Balanced
    "Flyweight": -0.02,            # Speed/technique matters more
    "Women's Featherweight": 0.03, # Performed well (83.3%)
    "Women's Bantamweight": -0.08, # Most unpredictable (underperformed at 46.2%)
    "Women's Flyweight": 0.0,
    "Women's Strawweight": 0.0,
    "Strawweight": -0.02,
    "Catch Weight": -0.05,         # Unpredictable (50.0%)
    "Open Weight": 0.05,
}


@dataclass
class SanityCheck:
    """Result of a simulation sanity check."""
    check_type: str  # "weight_class", "age", "experience", "physical", "activity"
    severity: str  # "info", "warning", "critical"
    message: str
    details: Optional[str] = None


@dataclass
class FighterProfile:
    """Complete fighter profile for simulation display."""
    fighter_id: int
    name: str
    nickname: Optional[str]
    record: str  # "W-L-D"
    wins: int
    losses: int
    draws: int
    height_cm: Optional[float]
    height_display: str  # e.g., "6'0\" (183cm)"
    weight_kg: Optional[float]
    reach_cm: Optional[float]
    reach_display: str  # e.g., "74\" (188cm)"
    stance: Optional[str]
    age: Optional[int]
    nationality: Optional[str]
    weight_class: Optional[str]
    last_fight_date: Optional[str]
    is_active: bool
    # Striking Stats
    sig_strike_accuracy: Optional[float]
    sig_strike_defense: Optional[float]
    sig_strikes_landed_per_min: Optional[float]
    sig_strikes_absorbed_per_min: Optional[float]
    # Grappling Stats
    takedown_accuracy: Optional[float]
    takedown_defense: Optional[float]
    takedowns_per_15min: Optional[float]
    submissions_per_15min: Optional[float]
    avg_control_time: Optional[int]  # seconds per fight
    # Finishing Stats
    ko_rate: Optional[float]
    submission_rate: Optional[float]
    decision_rate: Optional[float]
    finish_rate: Optional[float]
    avg_fight_time: Optional[int]  # seconds
    # Recent form
    recent_results: List[str]  # Last 5 fight results
    win_streak: int
    loss_streak: int
    # Fight history details
    total_fights: int
    ufc_fights: int
    ko_wins: int
    sub_wins: int
    dec_wins: int
    # Recent form (last 5 fights) for weighted calculations
    recent_ko_wins: int = 0
    recent_sub_wins: int = 0
    recent_wins: int = 0


@dataclass
class FighterTendencies:
    """Fighter tendencies and style analysis from fight history."""
    fighter_id: int
    name: str
    # Primary style classification
    primary_style: str  # "Striker", "Wrestler", "Grappler", "Balanced"
    style_breakdown: Dict[str, float]  # {"striking": 0.6, "wrestling": 0.25, "submissions": 0.15}
    # Finishing tendencies
    prefers_finish: bool  # True if finish rate > 50%
    early_finisher: bool  # True if most finishes in R1-R2
    late_finisher: bool  # True if finishes spread across rounds
    goes_to_decision: bool  # True if decision rate > 60%
    # Offensive tendencies
    volume_striker: bool  # High strike output
    power_puncher: bool  # High KO rate
    pressure_fighter: bool  # High strikes absorbed but also landed
    counter_striker: bool  # Lower output but high accuracy
    # Grappling tendencies
    active_wrestler: bool  # High takedown attempts
    ground_and_pound: bool  # Gets takedowns but not many subs
    submission_hunter: bool  # High submission attempts
    top_control: bool  # Long control time
    # Defensive profile
    hard_to_finish: bool  # Low times finished
    chin_issues: bool  # Multiple KO losses
    grappling_vulnerable: bool  # Multiple sub losses or low TD defense
    # Performance under pressure
    performs_in_later_rounds: bool  # Good cardio
    fades_in_later_rounds: bool  # Cardio issues
    # Recent trends
    improving: bool
    declining: bool
    # Key metrics for comparison
    avg_strikes_landed: float
    avg_strikes_absorbed: float
    avg_takedowns_landed: float
    avg_takedowns_defended_pct: float
    avg_sub_attempts: float
    avg_control_time: float  # seconds


@dataclass
class RoundResult:
    """Result of a single round simulation."""
    round_number: int
    winner: str  # "fighter_a", "fighter_b", or "even"
    fighter_a_strikes: int
    fighter_b_strikes: int
    fighter_a_takedowns: int
    fighter_b_takedowns: int
    fighter_a_control_time: int  # seconds
    fighter_b_control_time: int  # seconds
    significant_moments: List[str] = field(default_factory=list)
    finish: Optional[str] = None  # None, "KO/TKO", "Submission"
    finish_time: Optional[str] = None  # e.g., "3:42"


@dataclass
class SimulationResult:
    """Complete fight simulation result."""
    model_name: str
    winner_id: int
    winner_name: str
    loser_name: str
    method: str
    finish_round: Optional[int]
    finish_time: Optional[str]
    confidence: float
    total_rounds: int
    rounds: List[RoundResult]
    fighter_a_stats: Dict[str, Any]
    fighter_b_stats: Dict[str, Any]
    key_factors: List[str]
    realism_score: float  # 0-1, how realistic this simulation is
    sanity_checks: List[SanityCheck] = field(default_factory=list)


@dataclass
class FighterAnalysis:
    """Analysis of fighter strengths, trends, and strategy."""
    fighter_id: int
    name: str
    strengths: List[str]
    weaknesses: List[str]
    recent_trend: str  # "improving", "declining", "consistent"
    style_description: str
    keys_to_victory: List[str]


class FightSimulationService:
    """
    Multi-model fight simulation service.

    Simulates fights using multiple models and selects the most realistic outcome.
    Enhanced with physical factors, age consideration, and sanity checks.
    """

    def __init__(self):
        """Initialize the simulation service."""
        self.data_service = DataService()
        self.activity_cutoff_years = FIGHTER_ACTIVITY_CUTOFF_YEARS

    def get_activity_cutoff_date(self) -> date:
        """Get the cutoff date for fighter activity."""
        today = date.today()
        return date(today.year - self.activity_cutoff_years, today.month, today.day)

    def is_fighter_active(self, fighter: Dict[str, Any]) -> bool:
        """Check if a fighter is considered active (fought recently)."""
        if not fighter.get("is_active", True):
            return False

        last_fight = fighter.get("last_fight_date")
        if last_fight:
            try:
                if isinstance(last_fight, str):
                    last_date = datetime.strptime(last_fight, "%Y-%m-%d").date()
                else:
                    last_date = last_fight
                return last_date >= self.get_activity_cutoff_date()
            except (ValueError, TypeError):
                pass

        return fighter.get("is_active", True)

    def search_active_fighters(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for active fighters only."""
        all_results = self.data_service.search_fighters(query, limit=limit * 2)
        active_fighters = [f for f in all_results if self.is_fighter_active(f)]
        return active_fighters[:limit]

    def get_fighter_profile(self, fighter_id: int) -> Optional[FighterProfile]:
        """
        Get complete fighter profile for simulation display.

        Args:
            fighter_id: Fighter ID

        Returns:
            FighterProfile with all details
        """
        fighter = self.data_service.get_fighter_by_id(fighter_id)
        if not fighter:
            return None

        stats = self.data_service.get_fighter_stats(fighter_id) or {}
        # Get recent history for streaks and recent results display
        history = self.data_service.get_fighter_fight_history(fighter_id, limit=10)
        # Get FULL history for accurate win method calculations (ko_wins, sub_wins, etc.)
        full_history = self.data_service.get_fighter_fight_history(fighter_id, limit=100)

        # Calculate age from DOB
        age = None
        dob = fighter.get("dob")
        if dob:
            try:
                if isinstance(dob, str):
                    dob_date = datetime.strptime(dob, "%Y-%m-%d").date()
                else:
                    dob_date = dob
                today = date.today()
                age = today.year - dob_date.year - ((today.month, today.day) < (dob_date.month, dob_date.day))
            except (ValueError, TypeError):
                pass

        # Format height display
        height_cm = fighter.get("height_cm")
        if height_cm:
            inches = height_cm / 2.54
            feet = int(inches // 12)
            remaining_inches = int(inches % 12)
            height_display = f"{feet}'{remaining_inches}\" ({int(height_cm)}cm)"
        else:
            height_display = "N/A"

        # Format reach display
        reach_cm = fighter.get("reach_cm")
        if reach_cm:
            reach_inches = reach_cm / 2.54
            reach_display = f"{int(reach_inches)}\" ({int(reach_cm)}cm)"
        else:
            reach_display = "N/A"

        # Extract record
        wins = fighter.get("wins", 0)
        losses = fighter.get("losses", 0)
        draws = fighter.get("draws", 0)
        record = f"{wins}-{losses}-{draws}"

        # Get recent results and calculate streaks
        recent_results = []
        win_streak = 0
        loss_streak = 0

        if history:
            for fight in history[:5]:
                result = fight.get("result", "")
                recent_results.append(result)

            # Calculate streaks
            for fight in history:
                result = fight.get("result", "")
                if result == "Win":
                    if loss_streak == 0:
                        win_streak += 1
                    else:
                        break
                elif result == "Loss":
                    if win_streak == 0:
                        loss_streak += 1
                    else:
                        break
                else:
                    break

        # Get last fight date
        last_fight_date = None
        if history:
            last_fight_date = history[0].get("event_date")

        # Determine weight class from recent fights
        weight_class = None
        if history:
            for fight in history[:3]:
                wc = fight.get("weight_class")
                if wc:
                    weight_class = wc
                    break

        # Calculate win types from FULL history for accurate rates
        ko_wins = 0
        sub_wins = 0
        dec_wins = 0
        total_control_time = 0
        fight_count = 0

        # IMPROVEMENT 4: Also track recent wins (last 5 fights) for form weighting
        recent_ko_wins = 0
        recent_sub_wins = 0
        recent_wins = 0

        if full_history:
            for idx, fight in enumerate(full_history):
                if fight.get("result") == "Win":
                    method = (fight.get("method") or "").lower()
                    if "ko" in method or "tko" in method:
                        ko_wins += 1
                        if idx < 5:  # Recent fight
                            recent_ko_wins += 1
                    elif "sub" in method:
                        sub_wins += 1
                        if idx < 5:
                            recent_sub_wins += 1
                    elif "dec" in method or "unanimous" in method or "split" in method:
                        dec_wins += 1

                    if idx < 5:
                        recent_wins += 1

                # Sum control time if available
                ctrl_time = fight.get("control_time_seconds", 0)
                if ctrl_time:
                    total_control_time += ctrl_time
                    fight_count += 1

        avg_control_time = total_control_time // max(fight_count, 1) if fight_count > 0 else None

        return FighterProfile(
            fighter_id=fighter_id,
            name=fighter.get("name", "Unknown"),
            nickname=fighter.get("nickname"),
            record=record,
            wins=wins,
            losses=losses,
            draws=draws,
            height_cm=height_cm,
            height_display=height_display,
            weight_kg=fighter.get("weight_kg"),
            reach_cm=reach_cm,
            reach_display=reach_display,
            stance=fighter.get("stance"),
            age=age,
            nationality=fighter.get("nationality"),
            weight_class=weight_class,
            last_fight_date=last_fight_date,
            is_active=fighter.get("is_active", True),
            # Striking stats
            sig_strike_accuracy=stats.get("sig_strike_accuracy"),
            sig_strike_defense=stats.get("sig_strike_defense"),
            sig_strikes_landed_per_min=stats.get("sig_strikes_landed_per_min"),
            sig_strikes_absorbed_per_min=stats.get("sig_strikes_absorbed_per_min"),
            # Grappling stats
            takedown_accuracy=stats.get("takedown_accuracy"),
            takedown_defense=stats.get("takedown_defense"),
            takedowns_per_15min=stats.get("takedowns_avg_per_15min"),
            submissions_per_15min=stats.get("submissions_avg_per_15min"),
            avg_control_time=avg_control_time,
            # Finishing stats
            ko_rate=stats.get("ko_rate"),
            submission_rate=stats.get("submission_rate"),
            decision_rate=stats.get("decision_rate"),
            finish_rate=stats.get("finish_rate"),
            avg_fight_time=stats.get("avg_fight_time_seconds"),
            # Recent form
            recent_results=recent_results,
            win_streak=win_streak,
            loss_streak=loss_streak,
            # Fight history
            total_fights=wins + losses + draws,
            ufc_fights=len(history) if history else 0,
            ko_wins=ko_wins,
            sub_wins=sub_wins,
            dec_wins=dec_wins,
            # Recent form (last 5 fights)
            recent_ko_wins=recent_ko_wins,
            recent_sub_wins=recent_sub_wins,
            recent_wins=recent_wins,
        )

    def run_sanity_checks(
        self,
        fighter_a: Dict[str, Any],
        fighter_b: Dict[str, Any],
        profile_a: Optional[FighterProfile],
        profile_b: Optional[FighterProfile],
    ) -> List[SanityCheck]:
        """
        Run sanity checks on the matchup.

        Args:
            fighter_a: Fighter A data
            fighter_b: Fighter B data
            profile_a: Fighter A profile
            profile_b: Fighter B profile

        Returns:
            List of sanity check results
        """
        checks = []

        # Weight class check
        wc_a = profile_a.weight_class if profile_a else None
        wc_b = profile_b.weight_class if profile_b else None

        if wc_a and wc_b and wc_a != wc_b:
            weight_a = WEIGHT_CLASS_LIMITS.get(wc_a, 170)
            weight_b = WEIGHT_CLASS_LIMITS.get(wc_b, 170)
            weight_diff = abs(weight_a - weight_b)

            if weight_diff > 30:
                checks.append(SanityCheck(
                    check_type="weight_class",
                    severity="critical",
                    message=f"Significant weight class mismatch: {wc_a} vs {wc_b}",
                    details=f"Weight difference of ~{weight_diff} lbs. This matchup is highly unlikely in real UFC."
                ))
            elif weight_diff > 15:
                checks.append(SanityCheck(
                    check_type="weight_class",
                    severity="warning",
                    message=f"Weight class difference: {wc_a} vs {wc_b}",
                    details=f"Weight difference of ~{weight_diff} lbs may affect simulation accuracy."
                ))
            elif weight_diff > 0:
                checks.append(SanityCheck(
                    check_type="weight_class",
                    severity="info",
                    message=f"Different weight classes: {wc_a} vs {wc_b}",
                    details="Fighters compete in different divisions."
                ))

        # Age check
        age_a = profile_a.age if profile_a else None
        age_b = profile_b.age if profile_b else None

        if age_a and age_b:
            age_diff = abs(age_a - age_b)

            if age_diff >= 15:
                younger = profile_a.name if age_a < age_b else profile_b.name
                older = profile_b.name if age_a < age_b else profile_a.name
                checks.append(SanityCheck(
                    check_type="age",
                    severity="warning",
                    message=f"Large age gap: {age_diff} years",
                    details=f"{younger} ({min(age_a, age_b)}) vs {older} ({max(age_a, age_b)}). Age can significantly impact performance."
                ))
            elif age_diff >= 10:
                checks.append(SanityCheck(
                    check_type="age",
                    severity="info",
                    message=f"Notable age difference: {age_diff} years",
                    details=f"Ages: {age_a} vs {age_b}"
                ))

            # Check for older fighters
            if max(age_a, age_b) >= 40:
                older_fighter = profile_a.name if age_a >= 40 else profile_b.name
                older_age = max(age_a, age_b)
                checks.append(SanityCheck(
                    check_type="age",
                    severity="info",
                    message=f"{older_fighter} is {older_age} years old",
                    details="Fighters over 40 may have reduced physical attributes."
                ))

        # Experience check
        exp_a = (fighter_a.get("wins", 0) + fighter_a.get("losses", 0))
        exp_b = (fighter_b.get("wins", 0) + fighter_b.get("losses", 0))

        if exp_a > 0 and exp_b > 0:
            exp_ratio = max(exp_a, exp_b) / max(min(exp_a, exp_b), 1)

            if exp_ratio >= 3:
                more_exp = fighter_a.get("name") if exp_a > exp_b else fighter_b.get("name")
                less_exp = fighter_b.get("name") if exp_a > exp_b else fighter_a.get("name")
                checks.append(SanityCheck(
                    check_type="experience",
                    severity="warning",
                    message=f"Large experience gap",
                    details=f"{more_exp} ({max(exp_a, exp_b)} fights) vs {less_exp} ({min(exp_a, exp_b)} fights)"
                ))
            elif exp_ratio >= 2:
                checks.append(SanityCheck(
                    check_type="experience",
                    severity="info",
                    message=f"Experience difference: {max(exp_a, exp_b)} vs {min(exp_a, exp_b)} fights",
                    details="More experienced fighter may have an advantage."
                ))

        # Physical attributes check
        height_a = fighter_a.get("height_cm") or 0
        height_b = fighter_b.get("height_cm") or 0
        reach_a = fighter_a.get("reach_cm") or 0
        reach_b = fighter_b.get("reach_cm") or 0

        if height_a > 0 and height_b > 0:
            height_diff = abs(height_a - height_b)
            if height_diff >= 15:  # ~6 inches
                taller = profile_a.name if height_a > height_b else profile_b.name
                checks.append(SanityCheck(
                    check_type="physical",
                    severity="info",
                    message=f"Significant height advantage: {taller}",
                    details=f"Height difference of {height_diff:.0f}cm ({height_diff/2.54:.0f} inches)"
                ))

        if reach_a > 0 and reach_b > 0:
            reach_diff = abs(reach_a - reach_b)
            if reach_diff >= 12:  # ~5 inches
                longer_reach = profile_a.name if reach_a > reach_b else profile_b.name
                checks.append(SanityCheck(
                    check_type="physical",
                    severity="info",
                    message=f"Significant reach advantage: {longer_reach}",
                    details=f"Reach difference of {reach_diff:.0f}cm ({reach_diff/2.54:.0f} inches)"
                ))

        # Win/Loss streak checks
        if profile_a and profile_b:
            if profile_a.win_streak >= 5:
                checks.append(SanityCheck(
                    check_type="momentum",
                    severity="info",
                    message=f"{profile_a.name} on a {profile_a.win_streak}-fight win streak",
                    details="Strong momentum coming into this fight."
                ))
            if profile_b.win_streak >= 5:
                checks.append(SanityCheck(
                    check_type="momentum",
                    severity="info",
                    message=f"{profile_b.name} on a {profile_b.win_streak}-fight win streak",
                    details="Strong momentum coming into this fight."
                ))
            if profile_a.loss_streak >= 3:
                checks.append(SanityCheck(
                    check_type="momentum",
                    severity="warning",
                    message=f"{profile_a.name} on a {profile_a.loss_streak}-fight losing streak",
                    details="May be on a decline or facing tough competition."
                ))
            if profile_b.loss_streak >= 3:
                checks.append(SanityCheck(
                    check_type="momentum",
                    severity="warning",
                    message=f"{profile_b.name} on a {profile_b.loss_streak}-fight losing streak",
                    details="May be on a decline or facing tough competition."
                ))

        return checks

    def get_fighter_tendencies(self, fighter_id: int) -> Optional[FighterTendencies]:
        """
        Analyze a fighter's tendencies and style from their fight history.

        Args:
            fighter_id: Fighter ID

        Returns:
            FighterTendencies with style analysis
        """
        fighter = self.data_service.get_fighter_by_id(fighter_id)
        if not fighter:
            return None

        stats = self.data_service.get_fighter_stats(fighter_id) or {}
        history = self.data_service.get_fighter_fight_history(fighter_id, limit=15)

        # Get stats
        str_pm = stats.get("sig_strikes_landed_per_min") or 0
        str_absorbed = stats.get("sig_strikes_absorbed_per_min") or 0
        str_acc = stats.get("sig_strike_accuracy") or 0
        str_def = stats.get("sig_strike_defense") or 0
        td_avg = stats.get("takedowns_avg_per_15min") or 0
        td_acc = stats.get("takedown_accuracy") or 0
        td_def = stats.get("takedown_defense") or 0
        sub_avg = stats.get("submissions_avg_per_15min") or 0
        ko_rate = stats.get("ko_rate") or 0
        sub_rate = stats.get("submission_rate") or 0
        dec_rate = stats.get("decision_rate") or 0
        finish_rate = stats.get("finish_rate") or 0

        # Analyze fight history for tendencies
        total_fights = len(history) if history else 0
        ko_finishes = 0
        sub_finishes = 0
        early_finishes = 0  # R1-R2
        late_finishes = 0   # R3+
        ko_losses = 0
        sub_losses = 0
        total_control = 0

        if history:
            for fight in history:
                method = (fight.get("method") or "").lower()
                round_num = fight.get("round") or 3
                result = fight.get("result", "")

                if result == "Win":
                    if "ko" in method or "tko" in method:
                        ko_finishes += 1
                        if round_num <= 2:
                            early_finishes += 1
                        else:
                            late_finishes += 1
                    elif "sub" in method:
                        sub_finishes += 1
                        if round_num <= 2:
                            early_finishes += 1
                        else:
                            late_finishes += 1
                elif result == "Loss":
                    if "ko" in method or "tko" in method:
                        ko_losses += 1
                    elif "sub" in method:
                        sub_losses += 1

                ctrl = fight.get("control_time_seconds", 0)
                if ctrl:
                    total_control += ctrl

        avg_control = total_control / max(total_fights, 1)

        # Calculate style breakdown
        striking_score = (str_pm / 6.0) * 0.4 + (ko_rate) * 0.3 + (str_acc) * 0.3
        wrestling_score = (td_avg / 4.0) * 0.4 + (td_acc) * 0.3 + (avg_control / 180) * 0.3
        submission_score = (sub_avg / 2.0) * 0.5 + (sub_rate) * 0.5

        total_score = striking_score + wrestling_score + submission_score + 0.01

        style_breakdown = {
            "striking": round(striking_score / total_score, 2),
            "wrestling": round(wrestling_score / total_score, 2),
            "submissions": round(submission_score / total_score, 2),
        }

        # Determine primary style
        max_style = max(style_breakdown, key=style_breakdown.get)
        if style_breakdown[max_style] > 0.5:
            primary_style = {"striking": "Striker", "wrestling": "Wrestler", "submissions": "Grappler"}[max_style]
        else:
            primary_style = "Balanced"

        # Determine tendencies
        prefers_finish = finish_rate > 0.50
        early_finisher = early_finishes > late_finishes and (early_finishes + late_finishes) > 2
        late_finisher = late_finishes >= early_finishes and late_finishes > 1
        goes_to_decision = dec_rate > 0.60

        volume_striker = str_pm > 5.0
        power_puncher = ko_rate > 0.35
        pressure_fighter = str_pm > 4.0 and str_absorbed > 3.5
        counter_striker = str_acc > 0.50 and str_pm < 4.0

        active_wrestler = td_avg > 2.5
        ground_and_pound = td_avg > 2.0 and sub_avg < 0.5
        submission_hunter = sub_avg > 1.0
        top_control = avg_control > 120

        hard_to_finish = (ko_losses + sub_losses) < 2 and total_fights >= 8
        chin_issues = ko_losses >= 3
        grappling_vulnerable = sub_losses >= 2 or td_def < 0.55

        # Check cardio from avg fight time
        avg_time = stats.get("avg_fight_time_seconds") or 600
        performs_in_later_rounds = avg_time > 720  # > 12 min avg
        fades_in_later_rounds = avg_time < 480 and finish_rate < 0.50  # <8 min but not finishing

        # Recent trends
        if history:
            recent_wins = sum(1 for f in history[:5] if f.get("result") == "Win")
            improving = recent_wins >= 4
            declining = recent_wins <= 1
        else:
            improving = False
            declining = False

        return FighterTendencies(
            fighter_id=fighter_id,
            name=fighter.get("name", "Unknown"),
            primary_style=primary_style,
            style_breakdown=style_breakdown,
            prefers_finish=prefers_finish,
            early_finisher=early_finisher,
            late_finisher=late_finisher,
            goes_to_decision=goes_to_decision,
            volume_striker=volume_striker,
            power_puncher=power_puncher,
            pressure_fighter=pressure_fighter,
            counter_striker=counter_striker,
            active_wrestler=active_wrestler,
            ground_and_pound=ground_and_pound,
            submission_hunter=submission_hunter,
            top_control=top_control,
            hard_to_finish=hard_to_finish,
            chin_issues=chin_issues,
            grappling_vulnerable=grappling_vulnerable,
            performs_in_later_rounds=performs_in_later_rounds,
            fades_in_later_rounds=fades_in_later_rounds,
            improving=improving,
            declining=declining,
            avg_strikes_landed=str_pm,
            avg_strikes_absorbed=str_absorbed,
            avg_takedowns_landed=td_avg,
            avg_takedowns_defended_pct=td_def,
            avg_sub_attempts=sub_avg,
            avg_control_time=avg_control,
        )

    def analyze_fighter(self, fighter_id: int) -> Optional[FighterAnalysis]:
        """Analyze a fighter's strengths, trends, and strategy."""
        fighter = self.data_service.get_fighter_by_id(fighter_id)
        if not fighter:
            return None

        stats = self.data_service.get_fighter_stats(fighter_id) or {}
        history = self.data_service.get_fighter_fight_history(fighter_id, limit=5)

        strengths = []
        weaknesses = []

        # Striking analysis
        str_acc = stats.get("sig_strike_accuracy") or 0
        str_def = stats.get("sig_strike_defense") or 0
        str_pm = stats.get("sig_strikes_landed_per_min") or 0

        if str_acc > 0.50:
            strengths.append(f"Accurate striker ({str_acc:.0%} accuracy)")
        elif str_acc < 0.40:
            weaknesses.append("Below average striking accuracy")

        if str_def > 0.60:
            strengths.append(f"Excellent defensive striking ({str_def:.0%} defense)")
        elif str_def < 0.50:
            weaknesses.append("Vulnerable to strikes")

        if str_pm > 5.0:
            strengths.append(f"High-volume striker ({str_pm:.1f} sig. strikes/min)")

        # Grappling analysis
        td_acc = stats.get("takedown_accuracy") or 0
        td_def = stats.get("takedown_defense") or 0
        sub_avg = stats.get("submissions_avg_per_15min") or 0

        if td_acc > 0.45:
            strengths.append(f"Strong takedown game ({td_acc:.0%} accuracy)")
        elif td_acc < 0.30:
            weaknesses.append("Limited takedown ability")

        if td_def > 0.70:
            strengths.append(f"Hard to take down ({td_def:.0%} defense)")
        elif td_def < 0.55:
            weaknesses.append("Susceptible to takedowns")

        if sub_avg > 1.0:
            strengths.append(f"Submission threat ({sub_avg:.1f} attempts/15min)")

        # Finishing ability
        ko_rate = stats.get("ko_rate") or 0
        sub_rate = stats.get("submission_rate") or 0
        finish_rate = stats.get("finish_rate") or 0

        if finish_rate > 0.60:
            strengths.append(f"High finish rate ({finish_rate:.0%})")
        elif finish_rate < 0.30:
            weaknesses.append("Relies heavily on decisions")

        if ko_rate > 0.40:
            strengths.append("Knockout power")
        if sub_rate > 0.30:
            strengths.append("Dangerous on the ground")

        # Physical advantages
        height = fighter.get("height_cm") or 0
        reach = fighter.get("reach_cm") or 0
        if height > 185:  # 6'1"+
            strengths.append("Height advantage in most matchups")
        if reach > 190:  # 75"+
            strengths.append("Long reach for striking range")

        # Determine recent trend
        if history:
            recent_wins = sum(1 for f in history[:3] if f.get("result") == "Win")
            if recent_wins >= 3:
                trend = "improving"
            elif recent_wins <= 1:
                trend = "declining"
            else:
                trend = "consistent"
        else:
            trend = "unknown"

        # Style description
        style_parts = []
        if str_pm > 4.5:
            style_parts.append("volume striker")
        elif ko_rate > 0.35:
            style_parts.append("power puncher")
        if td_acc > 0.40 and sub_avg > 0.5:
            style_parts.append("well-rounded grappler")
        elif td_acc > 0.40:
            style_parts.append("wrestler")
        elif sub_avg > 1.0:
            style_parts.append("submission artist")

        style = ", ".join(style_parts) if style_parts else "balanced fighter"

        # Keys to victory
        keys = []
        if str_acc > 0.45:
            keys.append("Land clean, accurate strikes")
        if td_acc > 0.40:
            keys.append("Utilize wrestling to control the fight")
        if td_def > 0.65:
            keys.append("Keep the fight standing")
        if sub_avg > 0.8:
            keys.append("Look for submission opportunities")
        if finish_rate > 0.50:
            keys.append("Push the pace and look for the finish")
        if str_def > 0.55:
            keys.append("Use superior defense to frustrate opponent")
        if not keys:
            keys.append("Stick to the game plan and stay composed")

        return FighterAnalysis(
            fighter_id=fighter_id,
            name=fighter.get("name", "Unknown"),
            strengths=strengths[:5],
            weaknesses=weaknesses[:3],
            recent_trend=trend,
            style_description=style.capitalize(),
            keys_to_victory=keys[:4],
        )

    def simulate_fight(
        self,
        fighter_a_id: int,
        fighter_b_id: int,
        num_rounds: int = SIMULATION_DEFAULT_ROUNDS,
    ) -> List[SimulationResult]:
        """
        Simulate a fight using multiple models.

        Args:
            fighter_a_id: Fighter A (red corner) ID
            fighter_b_id: Fighter B (blue corner) ID
            num_rounds: Number of scheduled rounds

        Returns:
            List of simulation results from different models
        """
        fighter_a = self.data_service.get_fighter_by_id(fighter_a_id)
        fighter_b = self.data_service.get_fighter_by_id(fighter_b_id)

        if not fighter_a or not fighter_b:
            logger.error("Could not find one or both fighters")
            return []

        stats_a = self.data_service.get_fighter_stats(fighter_a_id) or {}
        stats_b = self.data_service.get_fighter_stats(fighter_b_id) or {}

        # Get fighter profiles for sanity checks
        profile_a = self.get_fighter_profile(fighter_a_id)
        profile_b = self.get_fighter_profile(fighter_b_id)

        # Get fighter tendencies for style analysis
        tendencies_a = self.get_fighter_tendencies(fighter_a_id)
        tendencies_b = self.get_fighter_tendencies(fighter_b_id)

        # Run sanity checks (including style matchup)
        sanity_checks = self.run_sanity_checks(fighter_a, fighter_b, profile_a, profile_b)

        # Add style matchup sanity checks
        if tendencies_a and tendencies_b:
            style_checks = self._check_style_matchup(tendencies_a, tendencies_b)
            sanity_checks.extend(style_checks)

        # Merge fighter data with stats
        fighter_a.update(stats_a)
        fighter_b.update(stats_b)

        # Add profile data for simulation
        if profile_a:
            fighter_a["age"] = profile_a.age
            fighter_a["win_streak"] = profile_a.win_streak
            fighter_a["loss_streak"] = profile_a.loss_streak
            fighter_a["weight_class"] = profile_a.weight_class
            fighter_a["ko_wins"] = profile_a.ko_wins
            fighter_a["sub_wins"] = profile_a.sub_wins
            fighter_a["dec_wins"] = profile_a.dec_wins
            fighter_a["avg_control_time"] = profile_a.avg_control_time
            # IMPROVEMENT 4: Recent form data for weighted calculations
            fighter_a["recent_ko_wins"] = profile_a.recent_ko_wins
            fighter_a["recent_sub_wins"] = profile_a.recent_sub_wins
            fighter_a["recent_wins"] = profile_a.recent_wins
        if profile_b:
            fighter_b["age"] = profile_b.age
            fighter_b["win_streak"] = profile_b.win_streak
            fighter_b["loss_streak"] = profile_b.loss_streak
            fighter_b["weight_class"] = profile_b.weight_class
            fighter_b["ko_wins"] = profile_b.ko_wins
            fighter_b["sub_wins"] = profile_b.sub_wins
            fighter_b["dec_wins"] = profile_b.dec_wins
            fighter_b["avg_control_time"] = profile_b.avg_control_time
            # IMPROVEMENT 4: Recent form data for weighted calculations
            fighter_b["recent_ko_wins"] = profile_b.recent_ko_wins
            fighter_b["recent_sub_wins"] = profile_b.recent_sub_wins
            fighter_b["recent_wins"] = profile_b.recent_wins

        # Add tendencies for simulation
        if tendencies_a:
            fighter_a["tendencies"] = tendencies_a
        if tendencies_b:
            fighter_b["tendencies"] = tendencies_b

        # IMPROVEMENT 7: Get head-to-head history for rematches
        h2h_history = self.data_service.get_head_to_head_history(fighter_a_id, fighter_b_id)
        if h2h_history:
            fighter_a["h2h_history"] = h2h_history
            fighter_b["h2h_history"] = h2h_history
            # Add sanity check for rematch
            h2h_wins_a = sum(1 for f in h2h_history if f.get("winner_key") == "fighter_a")
            h2h_wins_b = sum(1 for f in h2h_history if f.get("winner_key") == "fighter_b")
            sanity_checks.append(SanityCheck(
                check_type="head_to_head",
                severity="info",
                message=f"Rematch: {len(h2h_history)} previous fight(s)",
                details=f"Previous record: {fighter_a.get('name')} {h2h_wins_a} - {h2h_wins_b} {fighter_b.get('name')}"
            ))

        simulations = []

        for model_name in SIMULATION_MODELS:
            try:
                result = self._run_simulation(
                    model_name,
                    fighter_a,
                    fighter_b,
                    num_rounds,
                )
                if result:
                    result.sanity_checks = sanity_checks
                    simulations.append(result)
            except Exception as e:
                logger.error(f"Simulation failed for model {model_name}: {e}")

        # Calculate realism scores
        self._calculate_realism_scores(simulations, fighter_a, fighter_b)

        # Sort by realism score
        simulations.sort(key=lambda x: x.realism_score, reverse=True)

        return simulations

    def _run_simulation(
        self,
        model_name: str,
        fighter_a: Dict[str, Any],
        fighter_b: Dict[str, Any],
        num_rounds: int,
    ) -> Optional[SimulationResult]:
        """Run a single simulation with a specific model."""

        # Get base probabilities with enhanced factors
        base_probs = self._calculate_base_probabilities(fighter_a, fighter_b, model_name)

        rounds = []
        cumulative_damage_a = 0
        cumulative_damage_b = 0
        finish_occurred = False
        finish_round = None
        finish_time = None
        method = "Decision"

        for round_num in range(1, num_rounds + 1):
            round_result = self._simulate_round(
                round_num,
                fighter_a,
                fighter_b,
                base_probs,
                cumulative_damage_a,
                cumulative_damage_b,
                model_name,
            )
            rounds.append(round_result)

            if round_result.finish:
                finish_occurred = True
                finish_round = round_num
                finish_time = round_result.finish_time
                method = round_result.finish
                break

            cumulative_damage_a += round_result.fighter_b_strikes * 0.1
            cumulative_damage_b += round_result.fighter_a_strikes * 0.1

        # Determine winner
        if finish_occurred:
            last_round = rounds[-1]
            if last_round.winner == "fighter_a":
                winner_id = fighter_a.get("fighter_id")
                winner_name = fighter_a.get("name")
                loser_name = fighter_b.get("name")
            else:
                winner_id = fighter_b.get("fighter_id")
                winner_name = fighter_b.get("name")
                loser_name = fighter_a.get("name")
        else:
            a_rounds = sum(1 for r in rounds if r.winner == "fighter_a")
            b_rounds = sum(1 for r in rounds if r.winner == "fighter_b")

            if a_rounds >= b_rounds:
                winner_id = fighter_a.get("fighter_id")
                winner_name = fighter_a.get("name")
                loser_name = fighter_b.get("name")
            else:
                winner_id = fighter_b.get("fighter_id")
                winner_name = fighter_b.get("name")
                loser_name = fighter_a.get("name")

        confidence = self._calculate_confidence(
            rounds, base_probs, finish_occurred, model_name
        )

        key_factors = self._get_key_factors(
            fighter_a, fighter_b, rounds, method, model_name
        )

        return SimulationResult(
            model_name=model_name,
            winner_id=winner_id,
            winner_name=winner_name,
            loser_name=loser_name,
            method=method,
            finish_round=finish_round,
            finish_time=finish_time,
            confidence=confidence,
            total_rounds=len(rounds),
            rounds=rounds,
            fighter_a_stats=self._summarize_fighter_performance(rounds, "a"),
            fighter_b_stats=self._summarize_fighter_performance(rounds, "b"),
            key_factors=key_factors,
            realism_score=0.0,
            sanity_checks=[],
        )

    def _calculate_base_probabilities(
        self,
        fighter_a: Dict[str, Any],
        fighter_b: Dict[str, Any],
        model_name: str,
    ) -> Dict[str, float]:
        """Calculate base probabilities with enhanced physical and age factors."""

        # Base stats
        a_str_pm = fighter_a.get("sig_strikes_landed_per_min") or 3.5
        b_str_pm = fighter_b.get("sig_strikes_landed_per_min") or 3.5
        a_str_acc = fighter_a.get("sig_strike_accuracy") or 0.45
        b_str_acc = fighter_b.get("sig_strike_accuracy") or 0.45
        a_str_def = fighter_a.get("sig_strike_defense") or 0.55
        b_str_def = fighter_b.get("sig_strike_defense") or 0.55
        a_td_acc = fighter_a.get("takedown_accuracy") or 0.35
        b_td_acc = fighter_b.get("takedown_accuracy") or 0.35
        a_td_def = fighter_a.get("takedown_defense") or 0.60
        b_td_def = fighter_b.get("takedown_defense") or 0.60

        # Calculate KO/Sub rates from fighter's actual win history if database rates are NULL
        # This ensures each fighter's method distribution reflects their real tendencies
        a_wins = fighter_a.get("wins") or 0
        b_wins = fighter_b.get("wins") or 0
        a_ko_wins = fighter_a.get("ko_wins") or 0
        b_ko_wins = fighter_b.get("ko_wins") or 0
        a_sub_wins = fighter_a.get("sub_wins") or 0
        b_sub_wins = fighter_b.get("sub_wins") or 0

        # IMPROVEMENT 4: Calculate weighted KO/Sub rates using recent form
        # Recent fights (last 5) are weighted more heavily than career stats
        # Blend: 60% recent form + 40% career (if sufficient data)
        a_recent_ko_wins = fighter_a.get("recent_ko_wins") or 0
        b_recent_ko_wins = fighter_b.get("recent_ko_wins") or 0
        a_recent_sub_wins = fighter_a.get("recent_sub_wins") or 0
        b_recent_sub_wins = fighter_b.get("recent_sub_wins") or 0
        a_recent_wins = fighter_a.get("recent_wins") or 0
        b_recent_wins = fighter_b.get("recent_wins") or 0

        # Calculate career rates
        if fighter_a.get("ko_rate") is not None:
            a_career_ko_rate = fighter_a.get("ko_rate")
        elif a_wins > 0:
            a_career_ko_rate = a_ko_wins / a_wins
        else:
            a_career_ko_rate = 0.25

        if fighter_b.get("ko_rate") is not None:
            b_career_ko_rate = fighter_b.get("ko_rate")
        elif b_wins > 0:
            b_career_ko_rate = b_ko_wins / b_wins
        else:
            b_career_ko_rate = 0.25

        # Calculate recent rates (if sufficient recent wins)
        a_recent_ko_rate = a_recent_ko_wins / max(a_recent_wins, 1) if a_recent_wins >= 2 else a_career_ko_rate
        b_recent_ko_rate = b_recent_ko_wins / max(b_recent_wins, 1) if b_recent_wins >= 2 else b_career_ko_rate

        # Blend recent and career (60% recent, 40% career if sufficient recent data)
        if a_recent_wins >= 3:
            a_ko_rate = a_recent_ko_rate * 0.6 + a_career_ko_rate * 0.4
        else:
            a_ko_rate = a_career_ko_rate

        if b_recent_wins >= 3:
            b_ko_rate = b_recent_ko_rate * 0.6 + b_career_ko_rate * 0.4
        else:
            b_ko_rate = b_career_ko_rate

        # Same for submission rates
        if fighter_a.get("submission_rate") is not None:
            a_career_sub_rate = fighter_a.get("submission_rate")
        elif a_wins > 0:
            a_career_sub_rate = a_sub_wins / a_wins
        else:
            a_career_sub_rate = 0.15

        if fighter_b.get("submission_rate") is not None:
            b_career_sub_rate = fighter_b.get("submission_rate")
        elif b_wins > 0:
            b_career_sub_rate = b_sub_wins / b_wins
        else:
            b_career_sub_rate = 0.15

        # Calculate recent sub rates
        a_recent_sub_rate = a_recent_sub_wins / max(a_recent_wins, 1) if a_recent_wins >= 2 else a_career_sub_rate
        b_recent_sub_rate = b_recent_sub_wins / max(b_recent_wins, 1) if b_recent_wins >= 2 else b_career_sub_rate

        # Blend recent and career
        if a_recent_wins >= 3:
            a_sub_rate = a_recent_sub_rate * 0.6 + a_career_sub_rate * 0.4
        else:
            a_sub_rate = a_career_sub_rate

        if b_recent_wins >= 3:
            b_sub_rate = b_recent_sub_rate * 0.6 + b_career_sub_rate * 0.4
        else:
            b_sub_rate = b_career_sub_rate

        # Get tendencies for style analysis
        tendencies_a = fighter_a.get("tendencies")
        tendencies_b = fighter_b.get("tendencies")

        # IMPROVEMENT 1: Boost rates for fighters with clear style tendencies
        # RECOMMENDATION 1 & 2: Increased power_puncher and submission_hunter weights
        if tendencies_a:
            if tendencies_a.power_puncher:
                a_ko_rate = max(a_ko_rate, 0.45)  # Power punchers have high KO threat (increased from 0.35)
            if tendencies_a.submission_hunter:
                a_sub_rate = max(a_sub_rate, 0.35)  # Sub hunters have high sub threat (increased from 0.25)

        if tendencies_b:
            if tendencies_b.power_puncher:
                b_ko_rate = max(b_ko_rate, 0.45)  # Increased from 0.35
            if tendencies_b.submission_hunter:
                b_sub_rate = max(b_sub_rate, 0.35)  # Increased from 0.25

        # IMPROVEMENT 2: Opponent vulnerability factors
        # chin_issues = multiple KO losses, grappling_vulnerable = sub losses or low TD defense
        # RECOMMENDATION 1 & 2: Increased vulnerability weights for better method prediction
        opponent_chin_vulnerability_a = 1.0  # Fighter A's KO chance boost vs B
        opponent_chin_vulnerability_b = 1.0  # Fighter B's KO chance boost vs A
        opponent_grappling_vulnerability_a = 1.0  # Fighter A's sub chance boost vs B
        opponent_grappling_vulnerability_b = 1.0  # Fighter B's sub chance boost vs A

        if tendencies_b:
            if tendencies_b.chin_issues:
                opponent_chin_vulnerability_a = 1.50  # 50% boost vs chinny opponent (increased from 1.35)
            if tendencies_b.grappling_vulnerable:
                opponent_grappling_vulnerability_a = 1.45  # 45% boost vs bad grappler (increased from 1.30)

        if tendencies_a:
            if tendencies_a.chin_issues:
                opponent_chin_vulnerability_b = 1.50  # Increased from 1.35
            if tendencies_a.grappling_vulnerable:
                opponent_grappling_vulnerability_b = 1.45  # Increased from 1.30

        # IMPROVEMENT 3: Style matchup method prediction
        # RECOMMENDATION 1 & 2: Increased style matchup modifiers for better method prediction
        style_ko_modifier_a = 1.0
        style_ko_modifier_b = 1.0
        style_sub_modifier_a = 1.0
        style_sub_modifier_b = 1.0

        if tendencies_a and tendencies_b:
            # Striker vs Striker = higher KO likelihood for both
            if tendencies_a.primary_style == "Striker" and tendencies_b.primary_style == "Striker":
                style_ko_modifier_a = 1.25  # Increased from 1.15
                style_ko_modifier_b = 1.25

            # Wrestler brings fight to ground - increases both fighters' sub chances
            if tendencies_a.active_wrestler or tendencies_b.active_wrestler:
                style_sub_modifier_a = 1.30  # Increased from 1.20
                style_sub_modifier_b = 1.30

            # Wrestler vs submission specialist = higher sub chance for sub hunter
            if tendencies_a.active_wrestler and tendencies_b.submission_hunter:
                style_sub_modifier_b = 1.50  # Sub hunter capitalizes on takedowns (increased from 1.35)
            if tendencies_b.active_wrestler and tendencies_a.submission_hunter:
                style_sub_modifier_a = 1.50

            # Ground and pound specialist vs someone = higher GnP TKO chance
            if tendencies_a.ground_and_pound:
                style_ko_modifier_a = max(style_ko_modifier_a, 1.25)  # Increased from 1.15
            if tendencies_b.ground_and_pound:
                style_ko_modifier_b = max(style_ko_modifier_b, 1.25)

            # NEW: Power puncher vs power puncher = even higher KO likelihood
            if tendencies_a.power_puncher and tendencies_b.power_puncher:
                style_ko_modifier_a = max(style_ko_modifier_a, 1.30)
                style_ko_modifier_b = max(style_ko_modifier_b, 1.30)

            # NEW: Submission hunter vs submission hunter on ground
            if tendencies_a.submission_hunter and tendencies_b.submission_hunter:
                style_sub_modifier_a = max(style_sub_modifier_a, 1.35)
                style_sub_modifier_b = max(style_sub_modifier_b, 1.35)

        # IMPROVEMENT 4: Weight class finish rate modifier
        weight_class = fighter_a.get("weight_class") or fighter_b.get("weight_class") or "Lightweight"
        weight_class_modifier = WEIGHT_CLASS_FINISH_MODIFIER.get(weight_class, 1.0)

        # RECOMMENDATION 4: Weight class-specific stat adjustment
        # Adjusts how much physical stats matter vs technique/cardio
        wc_stat_adjustment = WEIGHT_CLASS_STAT_ADJUSTMENT.get(weight_class, 0.0)

        # Physical attributes
        height_a = fighter_a.get("height_cm") or 175
        height_b = fighter_b.get("height_cm") or 175
        reach_a = fighter_a.get("reach_cm") or 175
        reach_b = fighter_b.get("reach_cm") or 175

        # Calculate physical advantage factors
        # Apply weight class adjustment - higher divisions favor physical stats more
        height_weight = 0.05 + wc_stat_adjustment
        reach_weight = 0.08 + wc_stat_adjustment
        height_advantage = (height_a - height_b) / 100  # Normalize
        reach_advantage = (reach_a - reach_b) / 100
        physical_factor = 1.0 + (height_advantage * height_weight) + (reach_advantage * reach_weight)

        # Age factor (prime is ~28-32)
        age_a = fighter_a.get("age") or 30
        age_b = fighter_b.get("age") or 30

        def age_factor(age):
            if age < 24:
                return 0.95  # Less experienced
            elif 24 <= age <= 27:
                return 1.0  # Coming into prime
            elif 28 <= age <= 32:
                return 1.05  # Prime
            elif 33 <= age <= 36:
                return 1.0  # Post-prime but experienced
            elif 37 <= age <= 40:
                return 0.95  # Declining
            else:
                return 0.90  # Past prime

        age_modifier_a = age_factor(age_a)
        age_modifier_b = age_factor(age_b)

        # Momentum factors (win/loss streaks)
        win_streak_a = fighter_a.get("win_streak", 0)
        win_streak_b = fighter_b.get("win_streak", 0)
        loss_streak_a = fighter_a.get("loss_streak", 0)
        loss_streak_b = fighter_b.get("loss_streak", 0)

        momentum_a = 1.0 + (win_streak_a * 0.02) - (loss_streak_a * 0.03)
        momentum_b = 1.0 + (win_streak_b * 0.02) - (loss_streak_b * 0.03)

        # IMPROVEMENT 7: Head-to-head history factor for rematches
        # Previous wins against the same opponent provide psychological and tactical edge
        h2h_history = fighter_a.get("h2h_history", [])
        h2h_modifier_a = 1.0
        h2h_modifier_b = 1.0
        h2h_method_boost_ko_a = 1.0
        h2h_method_boost_ko_b = 1.0
        h2h_method_boost_sub_a = 1.0
        h2h_method_boost_sub_b = 1.0

        if h2h_history:
            h2h_wins_a = 0
            h2h_wins_b = 0
            h2h_ko_wins_a = 0
            h2h_ko_wins_b = 0
            h2h_sub_wins_a = 0
            h2h_sub_wins_b = 0

            for fight in h2h_history:
                winner_key = fight.get("winner_key")
                method = (fight.get("method") or "").lower()

                if winner_key == "fighter_a":
                    h2h_wins_a += 1
                    if "ko" in method or "tko" in method:
                        h2h_ko_wins_a += 1
                    elif "sub" in method:
                        h2h_sub_wins_a += 1
                elif winner_key == "fighter_b":
                    h2h_wins_b += 1
                    if "ko" in method or "tko" in method:
                        h2h_ko_wins_b += 1
                    elif "sub" in method:
                        h2h_sub_wins_b += 1

            # RECOMMENDATION 3: Refined head-to-head history factor
            # Reduced psychological edge (was 5%/3%, now 3%/2%) - past results matter less than current form
            # But method-specific boosts remain high - if you've finished them before, you likely can again
            h2h_modifier_a = 1.0 + (h2h_wins_a * 0.03) - (h2h_wins_b * 0.02)
            h2h_modifier_b = 1.0 + (h2h_wins_b * 0.03) - (h2h_wins_a * 0.02)

            # Method-specific boost: If you've KO'd/subbed this opponent before, higher chance again
            # Kept high because finishing the same opponent twice is common (proven vulnerability)
            if h2h_ko_wins_a > 0:
                h2h_method_boost_ko_a = 1.25  # 25% boost if previously KO'd this opponent (increased from 1.20)
            if h2h_ko_wins_b > 0:
                h2h_method_boost_ko_b = 1.25
            if h2h_sub_wins_a > 0:
                h2h_method_boost_sub_a = 1.30  # 30% boost if previously subbed this opponent (increased from 1.25)
            if h2h_sub_wins_b > 0:
                h2h_method_boost_sub_b = 1.30

        # Model-specific adjustments
        if model_name == "statistical":
            weight_factor = 1.0
        elif model_name == "momentum":
            weight_factor = momentum_a / max(momentum_b, 0.5)
        elif model_name == "stylistic":
            # Consider style matchups (wrestler vs striker, etc)
            a_is_wrestler = a_td_acc > 0.40
            b_is_striker = b_str_pm > 4.0
            if a_is_wrestler and b_is_striker:
                weight_factor = 1.05  # Wrestlers often beat pure strikers
            else:
                weight_factor = 1.0
        elif model_name == "historical":
            a_exp = (fighter_a.get("wins") or 0) + (fighter_a.get("losses") or 0)
            b_exp = (fighter_b.get("wins") or 0) + (fighter_b.get("losses") or 0)
            weight_factor = 1.0 + (a_exp - b_exp) * 0.01
        else:
            weight_factor = 1.0

        # Combined modifiers (including head-to-head history)
        total_modifier_a = weight_factor * physical_factor * age_modifier_a * momentum_a * h2h_modifier_a
        total_modifier_b = (1/weight_factor) * (1/physical_factor) * age_modifier_b * momentum_b * h2h_modifier_b

        # Calculate strike probabilities
        a_strike_power = (a_str_pm * a_str_acc) * total_modifier_a
        b_strike_power = (b_str_pm * b_str_acc) * total_modifier_b
        a_strike_prob = a_strike_power / (a_strike_power + b_strike_power + 0.01)

        # Calculate effective takedown probability
        a_td_eff = a_td_acc * (1 - b_td_def) * total_modifier_a
        b_td_eff = b_td_acc * (1 - a_td_def) * total_modifier_b

        # Get finish rates and tendencies for more realistic method prediction
        a_finish_rate = fighter_a.get("finish_rate") or 0.40
        b_finish_rate = fighter_b.get("finish_rate") or 0.40

        # Calculate KO probability per round - scaled to match UFC finish rates
        # UFC stats: ~30% of fights end in KO/TKO, ~20% in submission
        # For 3-round fight, need ~12-15% per round to achieve 30% cumulative KO rate
        a_ko_base = a_ko_rate * 0.40
        b_ko_base = b_ko_rate * 0.40

        # Factor in opponent's defensive striking (poor defense = higher KO chance)
        # Defense ranges 40-70%, so (1.3 - def) ranges 0.6 to 0.9
        # Also apply head-to-head method boost (if previously KO'd this opponent)
        a_ko_final = (a_ko_base * (1.3 - b_str_def) * total_modifier_a *
                      opponent_chin_vulnerability_a * style_ko_modifier_a * weight_class_modifier *
                      h2h_method_boost_ko_a)
        b_ko_final = (b_ko_base * (1.3 - a_str_def) * total_modifier_b *
                      opponent_chin_vulnerability_b * style_ko_modifier_b * weight_class_modifier *
                      h2h_method_boost_ko_b)

        # Calculate submission probability per round
        # Factor in submission rate, ability to get to the ground, and opponent vulnerability
        # Also apply head-to-head method boost (if previously subbed this opponent)
        a_grappling_threat = max(a_td_eff, 0.20)  # Minimum grappling engagement
        b_grappling_threat = max(b_td_eff, 0.20)

        a_sub_final = (a_sub_rate * 0.45 * (0.5 + a_grappling_threat) * total_modifier_a *
                       opponent_grappling_vulnerability_a * style_sub_modifier_a * weight_class_modifier *
                       h2h_method_boost_sub_a)
        b_sub_final = (b_sub_rate * 0.45 * (0.5 + b_grappling_threat) * total_modifier_b *
                       opponent_grappling_vulnerability_b * style_sub_modifier_b * weight_class_modifier *
                       h2h_method_boost_sub_b)

        # IMPROVEMENT 5: Track early/late finisher tendencies for round-specific patterns
        a_early_finisher = tendencies_a.early_finisher if tendencies_a else False
        b_early_finisher = tendencies_b.early_finisher if tendencies_b else False
        a_late_finisher = tendencies_a.late_finisher if tendencies_a else False
        b_late_finisher = tendencies_b.late_finisher if tendencies_b else False

        # IMPROVEMENT 6: Calculate stat disparity for confidence calibration
        stat_disparity = abs(a_strike_power - b_strike_power) / max(a_strike_power, b_strike_power, 0.01)

        return {
            "a_win_round": min(0.70, max(0.30, 0.5 + (a_strike_prob - 0.5) * 0.8)),
            "a_ko_prob": a_ko_final,
            "b_ko_prob": b_ko_final,
            "a_sub_prob": a_sub_final,
            "b_sub_prob": b_sub_final,
            "a_str_pm": a_str_pm * total_modifier_a,
            "b_str_pm": b_str_pm * total_modifier_b,
            "a_td_eff": a_td_eff,
            "b_td_eff": b_td_eff,
            "physical_advantage": physical_factor,
            "age_a": age_a,
            "age_b": age_b,
            "a_finish_rate": a_finish_rate,
            "b_finish_rate": b_finish_rate,
            "a_ko_rate": a_ko_rate,
            "b_ko_rate": b_ko_rate,
            "a_sub_rate": a_sub_rate,
            "b_sub_rate": b_sub_rate,
            # New fields for round-specific patterns
            "a_early_finisher": a_early_finisher,
            "b_early_finisher": b_early_finisher,
            "a_late_finisher": a_late_finisher,
            "b_late_finisher": b_late_finisher,
            # For confidence calibration
            "stat_disparity": stat_disparity,
            "weight_class_modifier": weight_class_modifier,
        }

    def _simulate_round(
        self,
        round_num: int,
        fighter_a: Dict[str, Any],
        fighter_b: Dict[str, Any],
        base_probs: Dict[str, float],
        cumulative_damage_a: float,
        cumulative_damage_b: float,
        model_name: str,
    ) -> RoundResult:
        """Simulate a single round with enhanced factors."""

        variance = random.gauss(0, 0.1)

        # Calculate strikes with fatigue factor for later rounds
        fatigue_factor = 1.0 - (round_num - 1) * 0.05
        a_strikes = int(max(0, random.gauss(base_probs["a_str_pm"] * 5 * fatigue_factor, 3)))
        b_strikes = int(max(0, random.gauss(base_probs["b_str_pm"] * 5 * fatigue_factor, 3)))

        # Calculate takedowns
        a_td = int(random.random() < base_probs["a_td_eff"] * 2)
        b_td = int(random.random() < base_probs["b_td_eff"] * 2)

        # Control time based on takedowns
        a_control = a_td * random.randint(30, 120)
        b_control = b_td * random.randint(30, 120)

        # Determine round winner
        a_score = a_strikes * 0.5 + a_td * 3 + a_control * 0.02
        b_score = b_strikes * 0.5 + b_td * 3 + b_control * 0.02

        if a_score > b_score + 2:
            winner = "fighter_a"
        elif b_score > a_score + 2:
            winner = "fighter_b"
        else:
            winner = "even"

        # Check for finish
        finish = None
        finish_time = None

        round_modifier = FINISH_PROB_BY_ROUND.get(round_num, 1.0)
        # Damage accumulation increases finish likelihood in later rounds
        damage_modifier = 1.0 + (cumulative_damage_a * 0.3) + (cumulative_damage_b * 0.3)

        # Age affects chin - older fighters more susceptible to KO
        age_a = base_probs.get("age_a", 30)
        age_b = base_probs.get("age_b", 30)
        chin_modifier_a = 1.0 + max(0, (age_a - 35) * 0.03)
        chin_modifier_b = 1.0 + max(0, (age_b - 35) * 0.03)

        # Strike differential affects KO chance - landing more = higher finish chance
        strike_ratio_a = a_strikes / max(b_strikes, 1)
        strike_ratio_b = b_strikes / max(a_strikes, 1)
        strike_modifier_a = 1.0 + min(0.3, (strike_ratio_a - 1) * 0.15) if strike_ratio_a > 1 else 1.0
        strike_modifier_b = 1.0 + min(0.3, (strike_ratio_b - 1) * 0.15) if strike_ratio_b > 1 else 1.0

        # IMPROVEMENT 5: Round-specific finish patterns (early/late finisher)
        # Early finishers get boosted KO/Sub chances in rounds 1-2
        # Late finishers get boosted chances in rounds 3+
        early_round_modifier_a = 1.0
        early_round_modifier_b = 1.0

        if round_num <= 2:  # Early rounds
            if base_probs.get("a_early_finisher"):
                early_round_modifier_a = 1.30  # 30% boost for early finishers in R1-R2
            if base_probs.get("b_early_finisher"):
                early_round_modifier_b = 1.30
        else:  # Later rounds (R3+)
            if base_probs.get("a_late_finisher"):
                early_round_modifier_a = 1.25  # 25% boost for late finishers in R3+
            if base_probs.get("b_late_finisher"):
                early_round_modifier_b = 1.25

        # Calculate effective KO probability for this round
        a_ko_chance = (base_probs["a_ko_prob"] * round_modifier * damage_modifier *
                       chin_modifier_b * strike_modifier_a * early_round_modifier_a)
        b_ko_chance = (base_probs["b_ko_prob"] * round_modifier * damage_modifier *
                       chin_modifier_a * strike_modifier_b * early_round_modifier_b)

        # KO check - separate rolls for each fighter
        if random.random() < a_ko_chance:
            finish = "KO/TKO"
            winner = "fighter_a"
            finish_time = f"{random.randint(1, 4)}:{random.randint(10, 59):02d}"
        elif random.random() < b_ko_chance:
            finish = "KO/TKO"
            winner = "fighter_b"
            finish_time = f"{random.randint(1, 4)}:{random.randint(10, 59):02d}"

        # Submission check (if no KO)
        if not finish:
            # Takedowns increase submission chance significantly
            td_bonus_a = 1.0 + (a_td * 0.4)  # Each takedown increases sub chance
            td_bonus_b = 1.0 + (b_td * 0.4)
            # Control time also affects submission likelihood
            control_bonus_a = 1.0 + (a_control / 300)  # Bonus for ground control
            control_bonus_b = 1.0 + (b_control / 300)

            # Apply early/late finisher modifier to submissions too
            a_sub_chance = (base_probs["a_sub_prob"] * round_modifier * td_bonus_a *
                           control_bonus_a * early_round_modifier_a)
            b_sub_chance = (base_probs["b_sub_prob"] * round_modifier * td_bonus_b *
                           control_bonus_b * early_round_modifier_b)

            if random.random() < a_sub_chance:
                finish = "Submission"
                winner = "fighter_a"
                finish_time = f"{random.randint(1, 4)}:{random.randint(10, 59):02d}"
            elif random.random() < b_sub_chance:
                finish = "Submission"
                winner = "fighter_b"
                finish_time = f"{random.randint(1, 4)}:{random.randint(10, 59):02d}"

        # TKO via ground and pound check (if no KO or submission and there was ground control)
        if not finish and (a_control > 90 or b_control > 90):
            gnp_roll = random.random()
            if a_control > 90 and gnp_roll < 0.08 * (a_control / 120) * base_probs.get("a_ko_rate", 0.25):
                finish = "KO/TKO"
                winner = "fighter_a"
                finish_time = f"{random.randint(2, 4)}:{random.randint(10, 59):02d}"
            elif b_control > 90 and gnp_roll < 0.08 * (b_control / 120) * base_probs.get("b_ko_rate", 0.25):
                finish = "KO/TKO"
                winner = "fighter_b"
                finish_time = f"{random.randint(2, 4)}:{random.randint(10, 59):02d}"

        # Generate significant moments
        moments = []
        if a_strikes > 20:
            moments.append(f"{fighter_a.get('name', 'Fighter A')} lands heavy combinations")
        if b_strikes > 20:
            moments.append(f"{fighter_b.get('name', 'Fighter B')} lands heavy combinations")
        if a_td > 0:
            moments.append(f"{fighter_a.get('name', 'Fighter A')} secures a takedown")
        if b_td > 0:
            moments.append(f"{fighter_b.get('name', 'Fighter B')} secures a takedown")
        if a_control > 60:
            moments.append(f"{fighter_a.get('name', 'Fighter A')} dominates on the ground")
        if b_control > 60:
            moments.append(f"{fighter_b.get('name', 'Fighter B')} dominates on the ground")

        return RoundResult(
            round_number=round_num,
            winner=winner,
            fighter_a_strikes=a_strikes,
            fighter_b_strikes=b_strikes,
            fighter_a_takedowns=a_td,
            fighter_b_takedowns=b_td,
            fighter_a_control_time=a_control,
            fighter_b_control_time=b_control,
            significant_moments=moments,
            finish=finish,
            finish_time=finish_time,
        )

    def _calculate_confidence(
        self,
        rounds: List[RoundResult],
        base_probs: Dict[str, float],
        finish_occurred: bool,
        model_name: str,
    ) -> float:
        """Calculate confidence in the simulation result with improved calibration."""

        # RECOMMENDATION 6: Improved confidence calibration
        # Test results showed low/medium confidence was overconfident
        # Reduced base and made scaling more conservative

        # Base confidence from round win probability spread
        prob_spread = abs(base_probs["a_win_round"] - 0.5) * 2
        # Start lower (0.45 instead of 0.50) and scale less (0.20 instead of 0.25)
        base_confidence = 0.45 + prob_spread * 0.20

        # Stat disparity - larger gap = higher confidence (reduced from 0.15 to 0.10)
        stat_disparity = base_probs.get("stat_disparity", 0)
        base_confidence += stat_disparity * 0.10

        # Finish bonus - decisive victories are more confident (kept same)
        if finish_occurred:
            base_confidence += 0.08

        # Round dominance bonus (reduced from 0.12 to 0.08)
        a_rounds = sum(1 for r in rounds if r.winner == "fighter_a")
        b_rounds = sum(1 for r in rounds if r.winner == "fighter_b")
        dominance = abs(a_rounds - b_rounds) / max(len(rounds), 1)
        base_confidence += dominance * 0.08

        # Physical advantage factor (reduced from 0.10 to 0.06)
        physical_advantage = abs(base_probs.get("physical_advantage", 1.0) - 1.0)
        base_confidence += physical_advantage * 0.06

        # Model-specific adjustments (ensemble should be more confident)
        if model_name == "ensemble":
            base_confidence += 0.05  # Increased from 0.03 since ensemble performs best

        # Apply non-linear scaling to compress low confidence values
        # This helps calibrate: predictions that should be ~50% stay closer to 50%
        if base_confidence < 0.65:
            # Reduce confidence for uncertain predictions more aggressively
            base_confidence = 0.45 + (base_confidence - 0.45) * 0.7

        # Clamp to reasonable range (lowered max from 0.88 to 0.85)
        return min(0.85, max(0.45, base_confidence))

    def _calculate_realism_scores(
        self,
        simulations: List[SimulationResult],
        fighter_a: Dict[str, Any],
        fighter_b: Dict[str, Any],
    ):
        """Calculate realism scores for all simulations."""

        a_finish_rate = fighter_a.get("finish_rate") or 0.5
        b_finish_rate = fighter_b.get("finish_rate") or 0.5
        avg_finish_rate = (a_finish_rate + b_finish_rate) / 2

        for sim in simulations:
            score = 0.5

            if sim.method in ["KO/TKO"]:
                a_ko = fighter_a.get("ko_rate") or 0
                b_ko = fighter_b.get("ko_rate") or 0
                if sim.winner_id == fighter_a.get("fighter_id"):
                    score += a_ko * 0.3
                else:
                    score += b_ko * 0.3

            elif sim.method == "Submission":
                a_sub = fighter_a.get("submission_rate") or 0
                b_sub = fighter_b.get("submission_rate") or 0
                if sim.winner_id == fighter_a.get("fighter_id"):
                    score += a_sub * 0.3
                else:
                    score += b_sub * 0.3

            else:
                score += (1 - avg_finish_rate) * 0.2

            score += sim.confidence * 0.2

            # RECOMMENDATION 5: Increased ensemble model weighting (68.7% accuracy - best performer)
            if sim.model_name == "ensemble":
                score += 0.12  # Increased from 0.05
            elif sim.model_name == "historical":
                score += 0.08  # Second best performer (68.2%)

            # Penalize unrealistic outcomes based on sanity checks
            critical_checks = sum(1 for c in sim.sanity_checks if c.severity == "critical")
            score -= critical_checks * 0.1

            sim.realism_score = min(1.0, max(0.0, score))

    def _get_key_factors(
        self,
        fighter_a: Dict[str, Any],
        fighter_b: Dict[str, Any],
        rounds: List[RoundResult],
        method: str,
        model_name: str,
    ) -> List[str]:
        """Get key factors that determined the outcome."""
        factors = []

        total_a_strikes = sum(r.fighter_a_strikes for r in rounds)
        total_b_strikes = sum(r.fighter_b_strikes for r in rounds)
        total_a_td = sum(r.fighter_a_takedowns for r in rounds)
        total_b_td = sum(r.fighter_b_takedowns for r in rounds)
        total_a_control = sum(r.fighter_a_control_time for r in rounds)
        total_b_control = sum(r.fighter_b_control_time for r in rounds)

        if total_a_strikes > total_b_strikes * 1.3:
            factors.append(f"{fighter_a.get('name')} dominated on the feet with superior volume")
        elif total_b_strikes > total_a_strikes * 1.3:
            factors.append(f"{fighter_b.get('name')} dominated on the feet with superior volume")

        if total_a_td > total_b_td + 1:
            factors.append(f"{fighter_a.get('name')}'s wrestling made the difference")
        elif total_b_td > total_a_td + 1:
            factors.append(f"{fighter_b.get('name')}'s wrestling made the difference")

        if total_a_control > total_b_control + 120:
            factors.append(f"{fighter_a.get('name')}'s ground control was decisive")
        elif total_b_control > total_a_control + 120:
            factors.append(f"{fighter_b.get('name')}'s ground control was decisive")

        # Physical factors
        height_a = fighter_a.get("height_cm") or 0
        height_b = fighter_b.get("height_cm") or 0
        reach_a = fighter_a.get("reach_cm") or 0
        reach_b = fighter_b.get("reach_cm") or 0

        if abs(reach_a - reach_b) > 10:
            longer_reach = fighter_a.get("name") if reach_a > reach_b else fighter_b.get("name")
            factors.append(f"{longer_reach}'s reach advantage was a factor")

        if method == "KO/TKO":
            factors.append("Power striking proved to be the deciding factor")
        elif method == "Submission":
            factors.append("Ground game dominance led to the finish")
        else:
            factors.append("Superior cardio and volume led to a clear decision")

        return factors[:5]

    def _summarize_fighter_performance(
        self,
        rounds: List[RoundResult],
        fighter: str,
    ) -> Dict[str, Any]:
        """Summarize a fighter's performance across all rounds."""

        if fighter == "a":
            return {
                "total_strikes": sum(r.fighter_a_strikes for r in rounds),
                "total_takedowns": sum(r.fighter_a_takedowns for r in rounds),
                "total_control_time": sum(r.fighter_a_control_time for r in rounds),
                "rounds_won": sum(1 for r in rounds if r.winner == "fighter_a"),
            }
        else:
            return {
                "total_strikes": sum(r.fighter_b_strikes for r in rounds),
                "total_takedowns": sum(r.fighter_b_takedowns for r in rounds),
                "total_control_time": sum(r.fighter_b_control_time for r in rounds),
                "rounds_won": sum(1 for r in rounds if r.winner == "fighter_b"),
            }

    def _check_style_matchup(
        self,
        tendencies_a: FighterTendencies,
        tendencies_b: FighterTendencies,
    ) -> List[SanityCheck]:
        """
        Check style matchup and generate relevant sanity checks.

        Args:
            tendencies_a: Fighter A tendencies
            tendencies_b: Fighter B tendencies

        Returns:
            List of sanity checks related to style matchup
        """
        checks = []

        # Striker vs Wrestler matchup
        if tendencies_a.primary_style == "Striker" and tendencies_b.primary_style == "Wrestler":
            if tendencies_a.avg_takedowns_defended_pct < 0.60:
                checks.append(SanityCheck(
                    check_type="style_matchup",
                    severity="warning",
                    message=f"{tendencies_a.name}'s takedown defense may be tested",
                    details=f"{tendencies_b.name} averages {tendencies_b.avg_takedowns_landed:.1f} TDs/15min vs {tendencies_a.name}'s {tendencies_a.avg_takedowns_defended_pct:.0%} TD defense"
                ))
            else:
                checks.append(SanityCheck(
                    check_type="style_matchup",
                    severity="info",
                    message=f"Classic striker vs wrestler matchup",
                    details=f"{tendencies_a.name} (Striker) has strong TD defense ({tendencies_a.avg_takedowns_defended_pct:.0%}) against {tendencies_b.name} (Wrestler)"
                ))
        elif tendencies_b.primary_style == "Striker" and tendencies_a.primary_style == "Wrestler":
            if tendencies_b.avg_takedowns_defended_pct < 0.60:
                checks.append(SanityCheck(
                    check_type="style_matchup",
                    severity="warning",
                    message=f"{tendencies_b.name}'s takedown defense may be tested",
                    details=f"{tendencies_a.name} averages {tendencies_a.avg_takedowns_landed:.1f} TDs/15min vs {tendencies_b.name}'s {tendencies_b.avg_takedowns_defended_pct:.0%} TD defense"
                ))
            else:
                checks.append(SanityCheck(
                    check_type="style_matchup",
                    severity="info",
                    message=f"Classic striker vs wrestler matchup",
                    details=f"{tendencies_b.name} (Striker) has strong TD defense ({tendencies_b.avg_takedowns_defended_pct:.0%}) against {tendencies_a.name} (Wrestler)"
                ))

        # Power puncher vs chin issues
        if tendencies_a.power_puncher and tendencies_b.chin_issues:
            checks.append(SanityCheck(
                check_type="style_matchup",
                severity="warning",
                message=f"{tendencies_b.name} may be vulnerable to {tendencies_a.name}'s power",
                details=f"{tendencies_a.name} has high KO rate and {tendencies_b.name} has been stopped before"
            ))
        if tendencies_b.power_puncher and tendencies_a.chin_issues:
            checks.append(SanityCheck(
                check_type="style_matchup",
                severity="warning",
                message=f"{tendencies_a.name} may be vulnerable to {tendencies_b.name}'s power",
                details=f"{tendencies_b.name} has high KO rate and {tendencies_a.name} has been stopped before"
            ))

        # Submission hunter vs grappling vulnerable
        if tendencies_a.submission_hunter and tendencies_b.grappling_vulnerable:
            checks.append(SanityCheck(
                check_type="style_matchup",
                severity="warning",
                message=f"{tendencies_b.name} may be at risk of submission",
                details=f"{tendencies_a.name} averages {tendencies_a.avg_sub_attempts:.1f} sub attempts/15min"
            ))
        if tendencies_b.submission_hunter and tendencies_a.grappling_vulnerable:
            checks.append(SanityCheck(
                check_type="style_matchup",
                severity="warning",
                message=f"{tendencies_a.name} may be at risk of submission",
                details=f"{tendencies_b.name} averages {tendencies_b.avg_sub_attempts:.1f} sub attempts/15min"
            ))

        # Cardio concerns
        if tendencies_a.fades_in_later_rounds and tendencies_b.performs_in_later_rounds:
            checks.append(SanityCheck(
                check_type="style_matchup",
                severity="info",
                message=f"Cardio advantage for {tendencies_b.name}",
                details=f"{tendencies_b.name} tends to be strong in later rounds while {tendencies_a.name} may fade"
            ))
        if tendencies_b.fades_in_later_rounds and tendencies_a.performs_in_later_rounds:
            checks.append(SanityCheck(
                check_type="style_matchup",
                severity="info",
                message=f"Cardio advantage for {tendencies_a.name}",
                details=f"{tendencies_a.name} tends to be strong in later rounds while {tendencies_b.name} may fade"
            ))

        # Early finisher vs hard to finish
        if tendencies_a.early_finisher and tendencies_b.hard_to_finish:
            checks.append(SanityCheck(
                check_type="style_matchup",
                severity="info",
                message=f"{tendencies_a.name}'s early finish tendencies vs {tendencies_b.name}'s durability",
                details=f"May go to decision if {tendencies_a.name} can't finish early"
            ))
        if tendencies_b.early_finisher and tendencies_a.hard_to_finish:
            checks.append(SanityCheck(
                check_type="style_matchup",
                severity="info",
                message=f"{tendencies_b.name}'s early finish tendencies vs {tendencies_a.name}'s durability",
                details=f"May go to decision if {tendencies_b.name} can't finish early"
            ))

        # Both fighters declining or improving
        if tendencies_a.declining and tendencies_b.declining:
            checks.append(SanityCheck(
                check_type="style_matchup",
                severity="info",
                message="Both fighters on recent losing streaks",
                details="Form may be unpredictable for both competitors"
            ))
        elif tendencies_a.improving and tendencies_b.improving:
            checks.append(SanityCheck(
                check_type="style_matchup",
                severity="info",
                message="Both fighters on winning streaks",
                details="High-stakes matchup between two fighters in good form"
            ))
        elif tendencies_a.declining and tendencies_b.improving:
            checks.append(SanityCheck(
                check_type="style_matchup",
                severity="info",
                message=f"Momentum favors {tendencies_b.name}",
                details=f"{tendencies_b.name} on winning streak while {tendencies_a.name} has struggled recently"
            ))
        elif tendencies_b.declining and tendencies_a.improving:
            checks.append(SanityCheck(
                check_type="style_matchup",
                severity="info",
                message=f"Momentum favors {tendencies_a.name}",
                details=f"{tendencies_a.name} on winning streak while {tendencies_b.name} has struggled recently"
            ))

        # Volume striker vs counter striker
        if tendencies_a.volume_striker and tendencies_b.counter_striker:
            checks.append(SanityCheck(
                check_type="style_matchup",
                severity="info",
                message="Volume vs precision matchup on the feet",
                details=f"{tendencies_a.name}'s output ({tendencies_a.avg_strikes_landed:.1f} SLpM) vs {tendencies_b.name}'s countering"
            ))
        if tendencies_b.volume_striker and tendencies_a.counter_striker:
            checks.append(SanityCheck(
                check_type="style_matchup",
                severity="info",
                message="Volume vs precision matchup on the feet",
                details=f"{tendencies_b.name}'s output ({tendencies_b.avg_strikes_landed:.1f} SLpM) vs {tendencies_a.name}'s countering"
            ))

        return checks

    def get_most_realistic_simulation(
        self,
        simulations: List[SimulationResult],
    ) -> Tuple[Optional[SimulationResult], Optional[SimulationResult]]:
        """Get the most and second most realistic simulations."""
        if not simulations:
            return None, None

        sorted_sims = sorted(simulations, key=lambda x: x.realism_score, reverse=True)

        most_realistic = sorted_sims[0] if len(sorted_sims) > 0 else None
        second_realistic = sorted_sims[1] if len(sorted_sims) > 1 else None

        return most_realistic, second_realistic
