"""
UFC Prediction App - Configuration Module.

Centralized configuration for the entire application.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# === Paths ===
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DATABASE_DIR = DATA_DIR / "database"
CACHE_DIR = DATA_DIR / "cache"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRAINED_MODELS_DIR = MODELS_DIR / "trained"
MODEL_VERSIONS_DIR = TRAINED_MODELS_DIR / "versions"
CURRENT_MODELS_DIR = TRAINED_MODELS_DIR / "current"
LOGS_DIR = BASE_DIR / "logs"

# Database path
DATABASE_PATH = DATABASE_DIR / os.getenv("DATABASE_NAME", "ufc_database.db")

# === API Keys ===
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
KAGGLE_KEY = os.getenv("KAGGLE_KEY")

# === App Settings ===
APP_ENV = os.getenv("APP_ENV", "development")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# === LLM Settings ===
LLM_ENABLED = os.getenv("LLM_ENABLED", "true").lower() == "true"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1000"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))

# === Model Settings ===
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1.0.0")
MIN_CONFIDENCE_THRESHOLD = float(os.getenv("MIN_CONFIDENCE_THRESHOLD", "0.55"))
MAX_MODEL_VERSIONS = int(os.getenv("MAX_MODEL_VERSIONS", "3"))

# Retraining thresholds
MIN_ACCURACY_THRESHOLD = float(os.getenv("MIN_ACCURACY_THRESHOLD", "0.55"))
MAX_DAYS_SINCE_RETRAIN = int(os.getenv("MAX_DAYS_SINCE_RETRAIN", "90"))
MIN_NEW_FIGHTS_FOR_RETRAIN = int(os.getenv("MIN_NEW_FIGHTS_FOR_RETRAIN", "150"))

# === Ground Truth Collection Settings ===
# Minimum hours before event a prediction must be made to be valid
MIN_HOURS_BEFORE_EVENT = int(os.getenv("MIN_HOURS_BEFORE_EVENT", "1"))
# Whether to include backfilled predictions in accuracy calculations
INCLUDE_BACKFILL_IN_ACCURACY = os.getenv("INCLUDE_BACKFILL_IN_ACCURACY", "false").lower() == "true"
# Edge case outcomes that are excluded from accuracy metrics
EXCLUDED_OUTCOMES = ["no_contest", "dq", "draw", "cancelled"]

# === Update Settings ===
AUTO_UPDATE_ENABLED = os.getenv("AUTO_UPDATE_ENABLED", "true").lower() == "true"
UPDATE_CHECK_INTERVAL = int(os.getenv("UPDATE_CHECK_INTERVAL", "24"))

# === Caching ===
CACHE_LLM_RESPONSES = os.getenv("CACHE_LLM_RESPONSES", "true").lower() == "true"
CACHE_TTL = int(os.getenv("CACHE_TTL", "86400"))  # 24 hours

# === Logging ===
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = LOGS_DIR / os.getenv("LOG_FILE", "app.log")

# === MLFlow Settings ===
# Convert path to proper file:// URI for MLFlow compatibility
_mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", str(BASE_DIR / "mlruns"))
if _mlflow_uri.startswith("./") or not _mlflow_uri.startswith(("http", "file:", "sqlite", "postgresql", "mysql")):
    # Local path - convert to file:// URI
    _mlflow_path = Path(_mlflow_uri).resolve()
    MLFLOW_TRACKING_URI = _mlflow_path.as_uri()
else:
    MLFLOW_TRACKING_URI = _mlflow_uri
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "ufc-prediction-models")
MLFLOW_MODEL_NAME_WINNER = "ufc-winner-predictor"
MLFLOW_MODEL_NAME_METHOD = "ufc-method-predictor"
MLFLOW_MODEL_NAME_ROUND = "ufc-round-predictor"
MLFLOW_ENABLED = os.getenv("MLFLOW_ENABLED", "true").lower() == "true"

# === Scraper Settings ===
SCRAPER_BASE_URL = "http://ufcstats.com"
SCRAPER_RATE_LIMIT = float(os.getenv("SCRAPER_RATE_LIMIT", "1.0"))  # seconds between requests
SCRAPER_MAX_RETRIES = int(os.getenv("SCRAPER_MAX_RETRIES", "3"))
SCRAPER_USER_AGENT = "UFC-Prediction-App/1.0"

# === News Scraper Settings ===
NEWS_CACHE_TTL_HOURS = int(os.getenv("NEWS_CACHE_TTL_HOURS", "12"))  # Refresh news every 12 hours
NEWS_MAX_ITEMS = int(os.getenv("NEWS_MAX_ITEMS", "10"))  # Max news items to display

# === UI Theme ===
COLORS = {
    "primary": "#D20A0A",           # UFC Red
    "secondary": "#000000",          # Black
    "background": "#0E1117",         # Dark background
    "card_bg": "#1A1A2E",            # Card background
    "success": "#28A745",            # Green (wins, high confidence)
    "warning": "#FFC107",            # Yellow (medium confidence)
    "danger": "#DC3545",             # Red (losses, low confidence)
    "info": "#17A2B8",               # Blue (info)
    "text_primary": "#FFFFFF",       # White text
    "text_secondary": "#B0B0B0",     # Gray text
    "text_muted": "#6C757D",         # Muted text
}

# Confidence level colors
CONFIDENCE_COLORS = {
    "high": "#28A745",    # Green (> 65%)
    "medium": "#FFC107",  # Yellow (55-65%)
    "low": "#FD7E14",     # Orange (< 55%)
}

# === Weight Classes ===
WEIGHT_CLASSES = [
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
]

# Weight class limits (in lbs)
WEIGHT_CLASS_LIMITS = {
    "Strawweight": 115,
    "Flyweight": 125,
    "Bantamweight": 135,
    "Featherweight": 145,
    "Lightweight": 155,
    "Welterweight": 170,
    "Middleweight": 185,
    "Light Heavyweight": 205,
    "Heavyweight": 265,
    "Women's Strawweight": 115,
    "Women's Flyweight": 125,
    "Women's Bantamweight": 135,
    "Women's Featherweight": 145,
}

# === Feature Engineering ===
DIFFERENTIAL_FEATURES = [
    "height_diff",
    "reach_diff",
    "age_diff",
    "sig_str_acc_diff",
    "sig_str_def_diff",
    "td_acc_diff",
    "td_def_diff",
    "str_landed_pm_diff",
    "str_absorbed_pm_diff",
    "td_avg_diff",
    "sub_avg_diff",
]

RATIO_FEATURES = [
    "win_rate_ratio",
    "finish_rate_ratio",
    "ko_rate_ratio",
    "sub_rate_ratio",
    "experience_ratio",
    "ufc_experience_ratio",
]

FORM_FEATURES = [
    "win_streak_a",
    "win_streak_b",
    "lose_streak_a",
    "lose_streak_b",
    "recent_form_a",
    "recent_form_b",
    "days_since_fight_a",
    "days_since_fight_b",
    "momentum_a",
    "momentum_b",
]

CONTEXTUAL_FEATURES = [
    "weight_class_encoded",
    "is_title_fight",
    "is_main_event",
    "rounds_scheduled",
]

# All features in order
ALL_FEATURES = DIFFERENTIAL_FEATURES + RATIO_FEATURES + FORM_FEATURES + CONTEXTUAL_FEATURES

# === Model Ensemble Weights ===
WINNER_MODEL_WEIGHTS = {
    "xgboost": 0.30,
    "lightgbm": 0.30,
    "random_forest": 0.20,
    "logistic_regression": 0.20,
}

METHOD_MODEL_WEIGHTS = {
    "xgboost": 0.35,
    "random_forest": 0.35,
    "gradient_boosting": 0.30,
}

# Method classes
METHOD_CLASSES = ["KO/TKO", "Submission", "Decision"]

# === Fighter Data Settings ===
# Only include fighters who have fought within this many years
FIGHTER_ACTIVITY_CUTOFF_YEARS = int(os.getenv("FIGHTER_ACTIVITY_CUTOFF_YEARS", "3"))
# This means fighters who haven't fought since 2022 won't be included in simulations

# === Fight Simulator Settings ===
SIMULATION_MODELS = ["statistical", "momentum", "stylistic", "historical", "ensemble"]
SIMULATION_DEFAULT_ROUNDS = 3
SIMULATION_TITLE_ROUNDS = 5
MAX_SIMULATIONS_PER_REQUEST = 5

# Round-by-round simulation parameters
ROUND_DURATION_SECONDS = 300  # 5 minutes
SIGNIFICANT_STRIKE_THRESHOLD = 20  # per round
TAKEDOWN_ATTEMPT_THRESHOLD = 3  # per round
SUBMISSION_ATTEMPT_THRESHOLD = 2  # per round

# Finish probability modifiers by round
FINISH_PROB_BY_ROUND = {
    1: 1.0,   # Normal
    2: 1.1,   # Slightly higher
    3: 1.2,   # Higher
    4: 1.3,   # Title fight rounds
    5: 1.4,   # Championship rounds
}


def ensure_directories():
    """Create all necessary directories if they don't exist."""
    directories = [
        DATA_DIR,
        DATABASE_DIR,
        CACHE_DIR,
        RAW_DATA_DIR / "kaggle",
        RAW_DATA_DIR / "scraped",
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        TRAINED_MODELS_DIR,
        MODEL_VERSIONS_DIR,
        CURRENT_MODELS_DIR,
        LOGS_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def get_confidence_level(confidence: float) -> str:
    """Get confidence level string based on probability."""
    if confidence > 0.65:
        return "high"
    elif confidence > 0.55:
        return "medium"
    else:
        return "low"


def get_confidence_color(confidence: float) -> str:
    """Get color for confidence level."""
    level = get_confidence_level(confidence)
    return CONFIDENCE_COLORS[level]
