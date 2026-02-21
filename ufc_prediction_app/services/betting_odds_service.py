"""
Betting Odds Service Module.

Fetches betting odds and popular predictions from various sources.
Supports multiple sources: The-Odds-API, BestFightOdds, and OddsShark.
"""

import logging
import re
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


@dataclass
class BettingOdds:
    """Container for betting odds data."""
    fighter_a_name: str
    fighter_b_name: str
    fighter_a_odds: Optional[int]  # American odds (e.g., -150, +200)
    fighter_b_odds: Optional[int]
    fighter_a_implied_prob: Optional[float]  # Implied probability (0-1)
    fighter_b_implied_prob: Optional[float]
    favorite: str  # "fighter_a", "fighter_b", or "pick_em"
    source: str
    last_updated: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "fighter_a_name": self.fighter_a_name,
            "fighter_b_name": self.fighter_b_name,
            "fighter_a_odds": self.fighter_a_odds,
            "fighter_b_odds": self.fighter_b_odds,
            "fighter_a_implied_prob": self.fighter_a_implied_prob,
            "fighter_b_implied_prob": self.fighter_b_implied_prob,
            "favorite": self.favorite,
            "source": self.source,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
        }


class BettingOddsService:
    """
    Service for fetching betting odds from various sources.

    Supports multiple sources with fallback:
    1. The-Odds-API (if API key provided) - Most reliable, free tier available
    2. BestFightOdds (web scraping) - Backup source

    To use The Odds API:
    1. Get a free API key at https://the-odds-api.com/
    2. Set environment variable ODDS_API_KEY or pass to constructor
    """

    # The Odds API sport key for MMA
    ODDS_API_SPORT = "mma_mixed_martial_arts"

    def __init__(self, odds_api_key: Optional[str] = None):
        """Initialize the betting odds service."""
        self.odds_api_key = odds_api_key or os.environ.get("ODDS_API_KEY")
        if not self.odds_api_key:
            logger.info(
                "No ODDS_API_KEY set. Get a free key at https://the-odds-api.com/ "
                "for reliable betting odds data."
            )
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Cache-Control": "max-age=0",
        })
        self._cache: Dict[str, Tuple[Any, datetime]] = {}
        self._cache_duration = timedelta(minutes=30)
        self._event_cache: Dict[str, Tuple[List[BettingOdds], datetime]] = {}

    def get_fight_odds(
        self,
        fighter_a_name: str,
        fighter_b_name: str,
        event_name: Optional[str] = None
    ) -> Optional[BettingOdds]:
        """
        Get betting odds for a specific fight.

        Args:
            fighter_a_name: Name of fighter A (red corner)
            fighter_b_name: Name of fighter B (blue corner)
            event_name: Optional event name for better matching

        Returns:
            BettingOdds or None if not found
        """
        cache_key = f"{fighter_a_name.lower()}_{fighter_b_name.lower()}"

        # Check cache
        if cache_key in self._cache:
            cached_data, cached_time = self._cache[cache_key]
            if datetime.now() - cached_time < self._cache_duration:
                logger.debug(f"Returning cached odds for {fighter_a_name} vs {fighter_b_name}")
                return cached_data

        odds = None

        # Try sources in order of preference
        # 1. The Odds API (most reliable if key available)
        if self.odds_api_key:
            odds = self._fetch_from_odds_api(fighter_a_name, fighter_b_name)
            if odds:
                logger.info(f"Got odds from The Odds API for {fighter_a_name} vs {fighter_b_name}")
                self._cache[cache_key] = (odds, datetime.now())
                return odds

        # 2. Try BestFightOdds
        odds = self._scrape_bestfightodds(fighter_a_name, fighter_b_name)
        if odds:
            logger.info(f"Got odds from BestFightOdds for {fighter_a_name} vs {fighter_b_name}")
            self._cache[cache_key] = (odds, datetime.now())
            return odds

        # 3. Try to get from event page data
        if event_name:
            odds = self._get_odds_from_event_page(fighter_a_name, fighter_b_name, event_name)
            if odds:
                logger.info(f"Got odds from event page for {fighter_a_name} vs {fighter_b_name}")
                self._cache[cache_key] = (odds, datetime.now())
                return odds

        logger.warning(f"Could not find odds for {fighter_a_name} vs {fighter_b_name}")
        return None

    def get_event_odds(self, event_name: str) -> List[BettingOdds]:
        """
        Get all betting odds for an event.

        Args:
            event_name: Name of the UFC event

        Returns:
            List of BettingOdds for all fights in the event
        """
        # Check event cache
        if event_name in self._event_cache:
            cached_odds, cached_time = self._event_cache[event_name]
            if datetime.now() - cached_time < self._cache_duration:
                return cached_odds

        all_odds = []

        # Try The Odds API first
        if self.odds_api_key:
            api_odds = self._fetch_event_from_odds_api()
            if api_odds:
                all_odds.extend(api_odds)
                self._event_cache[event_name] = (all_odds, datetime.now())
                return all_odds

        # Try BestFightOdds
        try:
            event_odds = self._scrape_bestfightodds_event(event_name)
            if event_odds:
                all_odds.extend(event_odds)
        except Exception as e:
            logger.warning(f"Failed to get event odds from BestFightOdds: {e}")

        if all_odds:
            self._event_cache[event_name] = (all_odds, datetime.now())

        return all_odds

    def _fetch_from_odds_api(
        self,
        fighter_a_name: str,
        fighter_b_name: str
    ) -> Optional[BettingOdds]:
        """Fetch odds from The Odds API."""
        if not self.odds_api_key:
            return None

        try:
            url = f"https://api.the-odds-api.com/v4/sports/{self.ODDS_API_SPORT}/odds"
            params = {
                "apiKey": self.odds_api_key,
                "regions": "us",
                "markets": "h2h",
                "oddsFormat": "american",
            }

            response = requests.get(url, params=params, timeout=10)
            if response.status_code != 200:
                logger.warning(f"The Odds API returned status {response.status_code}")
                return None

            data = response.json()

            # Search for our fight
            a_search = self._normalize_name(fighter_a_name)
            b_search = self._normalize_name(fighter_b_name)

            for event in data:
                home_team = event.get("home_team", "")
                away_team = event.get("away_team", "")

                home_norm = self._normalize_name(home_team)
                away_norm = self._normalize_name(away_team)

                # Check if this is our fight using fuzzy matching
                match_a_home = self._names_match_fuzzy(a_search, home_norm)
                match_a_away = self._names_match_fuzzy(a_search, away_norm)
                match_b_home = self._names_match_fuzzy(b_search, home_norm)
                match_b_away = self._names_match_fuzzy(b_search, away_norm)

                if (match_a_home and match_b_away) or (match_a_away and match_b_home):

                    # Get odds from first bookmaker
                    bookmakers = event.get("bookmakers", [])
                    if not bookmakers:
                        continue

                    bookmaker = bookmakers[0]
                    markets = bookmaker.get("markets", [])

                    for market in markets:
                        if market.get("key") == "h2h":
                            outcomes = market.get("outcomes", [])
                            odds_dict = {}
                            for outcome in outcomes:
                                name_norm = self._normalize_name(outcome.get("name", ""))
                                odds_dict[name_norm] = outcome.get("price")

                            # Match to our fighters using fuzzy matching
                            odds_a = None
                            odds_b = None

                            for name, odds in odds_dict.items():
                                if self._names_match_fuzzy(a_search, name):
                                    odds_a = odds
                                elif self._names_match_fuzzy(b_search, name):
                                    odds_b = odds

                            if odds_a is not None or odds_b is not None:
                                return self._create_odds_object(
                                    fighter_a_name, fighter_b_name,
                                    odds_a, odds_b, f"The Odds API ({bookmaker.get('title', 'Unknown')})"
                                )

        except Exception as e:
            logger.error(f"Error fetching from The Odds API: {e}")

        return None

    def _fetch_event_from_odds_api(self) -> List[BettingOdds]:
        """Fetch all UFC odds from The Odds API."""
        odds_list = []

        if not self.odds_api_key:
            return odds_list

        try:
            url = f"https://api.the-odds-api.com/v4/sports/{self.ODDS_API_SPORT}/odds"
            params = {
                "apiKey": self.odds_api_key,
                "regions": "us",
                "markets": "h2h",
                "oddsFormat": "american",
            }

            response = requests.get(url, params=params, timeout=10)
            if response.status_code != 200:
                return odds_list

            data = response.json()

            for event in data:
                home_team = event.get("home_team", "")
                away_team = event.get("away_team", "")

                bookmakers = event.get("bookmakers", [])
                if not bookmakers:
                    continue

                bookmaker = bookmakers[0]
                markets = bookmaker.get("markets", [])

                for market in markets:
                    if market.get("key") == "h2h":
                        outcomes = market.get("outcomes", [])
                        if len(outcomes) >= 2:
                            fighter_a = outcomes[0].get("name", "")
                            fighter_b = outcomes[1].get("name", "")
                            odds_a = outcomes[0].get("price")
                            odds_b = outcomes[1].get("price")

                            odds = self._create_odds_object(
                                fighter_a, fighter_b,
                                odds_a, odds_b,
                                f"The Odds API ({bookmaker.get('title', 'Unknown')})"
                            )
                            if odds:
                                odds_list.append(odds)

        except Exception as e:
            logger.error(f"Error fetching event from The Odds API: {e}")

        return odds_list

    def _scrape_bestfightodds(
        self,
        fighter_a_name: str,
        fighter_b_name: str
    ) -> Optional[BettingOdds]:
        """Scrape BestFightOdds for a specific fight."""
        try:
            a_search = self._normalize_name(fighter_a_name)
            b_search = self._normalize_name(fighter_b_name)

            url = "https://www.bestfightodds.com/"
            response = self.session.get(url, timeout=15)

            if response.status_code != 200:
                logger.warning(f"BestFightOdds returned status {response.status_code}")
                return None

            soup = BeautifulSoup(response.text, "html.parser")

            # Look for content-container or main-content
            tables = soup.find_all("table")

            for table in tables:
                rows = table.find_all("tr")

                for i, row in enumerate(rows):
                    cells = row.find_all(["td", "th"])
                    if len(cells) < 2:
                        continue

                    # Try to find fighter name in first cell
                    first_cell = cells[0]
                    name_elem = first_cell.find("a") or first_cell.find("span", class_="f-name")

                    if not name_elem:
                        # Try getting text directly
                        name_text = first_cell.get_text(strip=True)
                        if not name_text:
                            continue
                    else:
                        name_text = name_elem.get_text(strip=True)

                    normalized = self._normalize_name(name_text)

                    # Check if this matches fighter A
                    if self._names_match(a_search, normalized):
                        odds_a = self._parse_odds_from_row(cells)

                        # Check next row for fighter B
                        if i + 1 < len(rows):
                            next_row = rows[i + 1]
                            next_cells = next_row.find_all(["td", "th"])

                            if len(next_cells) >= 2:
                                next_name_elem = next_cells[0].find("a") or next_cells[0].find("span", class_="f-name")
                                if next_name_elem:
                                    next_name = next_name_elem.get_text(strip=True)
                                else:
                                    next_name = next_cells[0].get_text(strip=True)

                                next_normalized = self._normalize_name(next_name)

                                if self._names_match(b_search, next_normalized):
                                    odds_b = self._parse_odds_from_row(next_cells)
                                    return self._create_odds_object(
                                        fighter_a_name, fighter_b_name,
                                        odds_a, odds_b, "BestFightOdds"
                                    )

                    # Also check if this matches fighter B (reversed order)
                    elif self._names_match(b_search, normalized):
                        odds_b = self._parse_odds_from_row(cells)

                        if i + 1 < len(rows):
                            next_row = rows[i + 1]
                            next_cells = next_row.find_all(["td", "th"])

                            if len(next_cells) >= 2:
                                next_name_elem = next_cells[0].find("a") or next_cells[0].find("span", class_="f-name")
                                if next_name_elem:
                                    next_name = next_name_elem.get_text(strip=True)
                                else:
                                    next_name = next_cells[0].get_text(strip=True)

                                next_normalized = self._normalize_name(next_name)

                                if self._names_match(a_search, next_normalized):
                                    odds_a = self._parse_odds_from_row(next_cells)
                                    return self._create_odds_object(
                                        fighter_a_name, fighter_b_name,
                                        odds_a, odds_b, "BestFightOdds"
                                    )

        except Exception as e:
            logger.error(f"Error scraping BestFightOdds: {e}")

        return None

    def _scrape_bestfightodds_event(self, event_name: str) -> List[BettingOdds]:
        """Scrape BestFightOdds for all fights in an event."""
        odds_list = []

        try:
            # Try the main page first (has upcoming events)
            url = "https://www.bestfightodds.com/"
            response = self.session.get(url, timeout=15)

            if response.status_code != 200:
                return odds_list

            soup = BeautifulSoup(response.text, "html.parser")

            # Find all tables
            tables = soup.find_all("table")

            for table in tables:
                rows = table.find_all("tr")
                i = 0

                while i < len(rows) - 1:
                    row_a = rows[i]
                    row_b = rows[i + 1]

                    cells_a = row_a.find_all(["td", "th"])
                    cells_b = row_b.find_all(["td", "th"])

                    if len(cells_a) >= 2 and len(cells_b) >= 2:
                        # Get fighter names
                        name_a_elem = cells_a[0].find("a") or cells_a[0].find("span", class_="f-name")
                        name_b_elem = cells_b[0].find("a") or cells_b[0].find("span", class_="f-name")

                        if name_a_elem and name_b_elem:
                            fighter_a = name_a_elem.get_text(strip=True)
                            fighter_b = name_b_elem.get_text(strip=True)
                        else:
                            fighter_a = cells_a[0].get_text(strip=True)
                            fighter_b = cells_b[0].get_text(strip=True)

                        if fighter_a and fighter_b and len(fighter_a) > 2 and len(fighter_b) > 2:
                            odds_a = self._parse_odds_from_row(cells_a)
                            odds_b = self._parse_odds_from_row(cells_b)

                            odds = self._create_odds_object(
                                fighter_a, fighter_b,
                                odds_a, odds_b, "BestFightOdds"
                            )
                            if odds:
                                odds_list.append(odds)

                    i += 2

        except Exception as e:
            logger.error(f"Error scraping BestFightOdds event: {e}")

        return odds_list

    def _get_odds_from_event_page(
        self,
        fighter_a_name: str,
        fighter_b_name: str,
        event_name: Optional[str]
    ) -> Optional[BettingOdds]:
        """Try to get odds from cached event data."""
        if not event_name:
            return None

        event_odds = self.get_event_odds(event_name)

        a_search = self._normalize_name(fighter_a_name)
        b_search = self._normalize_name(fighter_b_name)

        for odds in event_odds:
            odds_a_norm = self._normalize_name(odds.fighter_a_name)
            odds_b_norm = self._normalize_name(odds.fighter_b_name)

            # Check if this is our fight (in either order)
            if self._names_match(a_search, odds_a_norm) and self._names_match(b_search, odds_b_norm):
                return odds
            elif self._names_match(a_search, odds_b_norm) and self._names_match(b_search, odds_a_norm):
                # Swap fighters
                return BettingOdds(
                    fighter_a_name=fighter_a_name,
                    fighter_b_name=fighter_b_name,
                    fighter_a_odds=odds.fighter_b_odds,
                    fighter_b_odds=odds.fighter_a_odds,
                    fighter_a_implied_prob=odds.fighter_b_implied_prob,
                    fighter_b_implied_prob=odds.fighter_a_implied_prob,
                    favorite="fighter_a" if odds.favorite == "fighter_b" else ("fighter_b" if odds.favorite == "fighter_a" else "pick_em"),
                    source=odds.source,
                    last_updated=odds.last_updated,
                )

        return None

    def _parse_odds_from_row(self, cells) -> Optional[int]:
        """Parse odds from table row cells."""
        # Try cells 1-3 for odds (skip name cell)
        for cell in cells[1:4]:
            odds = self._parse_odds_cell(cell)
            if odds is not None:
                return odds
        return None

    def _parse_odds_cell(self, cell) -> Optional[int]:
        """Parse odds from a table cell."""
        try:
            text = cell.get_text(strip=True)
            # Extract American odds (e.g., -150, +200)
            # Look for patterns like -150, +200, 150, etc.
            match = re.search(r'([+-]?\d{2,4})', text)
            if match:
                odds_val = int(match.group(1))
                # Validate it looks like odds (not a year or other number)
                if -1000 <= odds_val <= 1000 and odds_val != 0:
                    return odds_val
        except Exception:
            pass
        return None

    def _names_match(self, name1: str, name2: str) -> bool:
        """Check if two normalized names match (allowing partial matches)."""
        if not name1 or not name2:
            return False

        # Exact match
        if name1 == name2:
            return True

        # One contains the other
        if name1 in name2 or name2 in name1:
            return True

        # Remove spaces and compare (handles "Yi Zha" vs "Yizha")
        name1_no_space = name1.replace(" ", "")
        name2_no_space = name2.replace(" ", "")
        if name1_no_space == name2_no_space:
            return True
        if name1_no_space in name2_no_space or name2_no_space in name1_no_space:
            return True

        # Check last name match (common case)
        parts1 = name1.split()
        parts2 = name2.split()

        if parts1 and parts2:
            # Last names match
            if parts1[-1] == parts2[-1] and len(parts1[-1]) > 3:
                return True

            # First names match and last name partial
            if len(parts1) >= 2 and len(parts2) >= 2:
                if parts1[0] == parts2[0] and (parts1[-1] in parts2[-1] or parts2[-1] in parts1[-1]):
                    return True

            # Last name of one matches any part of other (handles nicknames)
            if parts1[-1] in parts2 or parts2[-1] in parts1:
                return True

        return False

    def _create_odds_object(
        self,
        fighter_a_name: str,
        fighter_b_name: str,
        odds_a: Optional[int],
        odds_b: Optional[int],
        source: str
    ) -> Optional[BettingOdds]:
        """Create a BettingOdds object from parsed data."""
        if odds_a is None and odds_b is None:
            return None

        # Calculate implied probabilities
        prob_a = self._american_to_implied_prob(odds_a) if odds_a else None
        prob_b = self._american_to_implied_prob(odds_b) if odds_b else None

        # Determine favorite
        if odds_a is not None and odds_b is not None:
            if odds_a < odds_b:
                favorite = "fighter_a"
            elif odds_b < odds_a:
                favorite = "fighter_b"
            else:
                favorite = "pick_em"
        elif odds_a is not None:
            favorite = "fighter_a" if odds_a < 0 else "fighter_b"
        elif odds_b is not None:
            favorite = "fighter_b" if odds_b < 0 else "fighter_a"
        else:
            favorite = "pick_em"

        return BettingOdds(
            fighter_a_name=fighter_a_name,
            fighter_b_name=fighter_b_name,
            fighter_a_odds=odds_a,
            fighter_b_odds=odds_b,
            fighter_a_implied_prob=prob_a,
            fighter_b_implied_prob=prob_b,
            favorite=favorite,
            source=source,
            last_updated=datetime.now(),
        )

    def _american_to_implied_prob(self, odds: int) -> float:
        """Convert American odds to implied probability."""
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return abs(odds) / (abs(odds) + 100)

    def _normalize_name(self, name: str) -> str:
        """Normalize a fighter name for comparison."""
        if not name:
            return ""

        # Remove common suffixes
        name = re.sub(r'\s+Jr\.?$', '', name, flags=re.IGNORECASE)
        name = re.sub(r'\s+III?$', '', name, flags=re.IGNORECASE)
        name = re.sub(r'\s+Sr\.?$', '', name, flags=re.IGNORECASE)

        # Convert to lowercase
        name = name.lower()

        # Replace hyphens with spaces (Saint-Denis -> Saint Denis)
        name = name.replace('-', ' ')

        # Remove other special characters (but keep spaces)
        name = re.sub(r'[^a-z\s]', '', name)
        name = re.sub(r'\s+', ' ', name).strip()

        return name

    def _get_last_name(self, name: str) -> str:
        """Extract last name from a normalized name."""
        parts = name.split()
        return parts[-1] if parts else ""

    def _names_match_fuzzy(self, name1: str, name2: str) -> bool:
        """
        Check if two names likely refer to the same fighter.
        Handles nicknames (Alex/Alexander, Dan/Daniel) and name variations.
        """
        if not name1 or not name2:
            return False

        # Exact match
        if name1 == name2:
            return True

        # One contains the other
        if name1 in name2 or name2 in name1:
            return True

        # Remove all spaces and compare (handles "Yi Zha" vs "Yizha")
        name1_nospace = name1.replace(" ", "")
        name2_nospace = name2.replace(" ", "")
        if name1_nospace == name2_nospace:
            return True
        if name1_nospace in name2_nospace or name2_nospace in name1_nospace:
            return True

        # Common nickname mappings
        nickname_map = {
            'alex': 'alexander',
            'dan': 'daniel',
            'mike': 'michael',
            'chris': 'christopher',
            'matt': 'matthew',
            'nick': 'nicholas',
            'tony': 'anthony',
            'joe': 'joseph',
            'ben': 'benjamin',
            'rob': 'robert',
            'ed': 'edward',
            'will': 'william',
            'jim': 'james',
            'tom': 'thomas',
            'steve': 'steven',
            'dave': 'david',
            'jon': 'jonathan',
            'pat': 'patrick',
            'tj': 'thomas',
            'tj': 'tyler',
        }

        parts1 = name1.split()
        parts2 = name2.split()

        if not parts1 or not parts2:
            return False

        # Get last names
        last1 = parts1[-1]
        last2 = parts2[-1]

        # Last names must match (or be very similar)
        if last1 != last2:
            # Check if one contains the other for last name
            if not (last1 in last2 or last2 in last1) or len(min(last1, last2, key=len)) < 4:
                return False

        # Now check first names with nickname handling
        if len(parts1) >= 1 and len(parts2) >= 1:
            first1 = parts1[0]
            first2 = parts2[0]

            # Direct match
            if first1 == first2:
                return True

            # One contains the other
            if first1 in first2 or first2 in first1:
                return True

            # Check nickname mappings
            first1_expanded = nickname_map.get(first1, first1)
            first2_expanded = nickname_map.get(first2, first2)

            if first1_expanded == first2_expanded:
                return True
            if first1 == first2_expanded or first2 == first1_expanded:
                return True

        return False

    def format_odds_display(self, odds: int) -> str:
        """Format American odds for display."""
        if odds > 0:
            return f"+{odds}"
        return str(odds)

    def get_odds_color(self, odds: int) -> str:
        """Get color based on odds value (for UI)."""
        if odds < -200:
            return "success"  # Heavy favorite
        elif odds < -100:
            return "info"  # Slight favorite
        elif odds <= 100:
            return "warning"  # Pick 'em
        elif odds <= 200:
            return "info"  # Slight underdog
        else:
            return "danger"  # Heavy underdog

    def clear_cache(self) -> None:
        """Clear all cached odds data."""
        self._cache.clear()
        self._event_cache.clear()
        logger.info("Betting odds cache cleared")


def get_betting_odds_service() -> BettingOddsService:
    """Get a singleton instance of the betting odds service."""
    return BettingOddsService()
