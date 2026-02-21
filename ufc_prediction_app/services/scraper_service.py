"""
Scraper Service Module.

Handles web scraping from UFCStats.com for fighter and event data.
"""

import logging
import re
import time
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    SCRAPER_BASE_URL,
    SCRAPER_RATE_LIMIT,
    SCRAPER_MAX_RETRIES,
    SCRAPER_USER_AGENT,
)

logger = logging.getLogger(__name__)


class UFCStatsScraper:
    """
    Web scraper for UFCStats.com.

    Provides methods to scrape fighters, events, and fight data.
    """

    BASE_URL = SCRAPER_BASE_URL
    FIGHTERS_URL = f"{BASE_URL}/statistics/fighters"
    EVENTS_COMPLETED_URL = f"{BASE_URL}/statistics/events/completed"
    EVENTS_UPCOMING_URL = f"{BASE_URL}/statistics/events/upcoming"

    def __init__(self):
        """Initialize the scraper with a requests session."""
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": SCRAPER_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9",
        })
        self.rate_limit = SCRAPER_RATE_LIMIT
        self._last_request_time = 0

    def _rate_limit_wait(self):
        """Wait to respect rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_request_time = time.time()

    @retry(
        stop=stop_after_attempt(SCRAPER_MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def _get_page(self, url: str) -> Optional[BeautifulSoup]:
        """
        Fetch a page and return parsed BeautifulSoup.

        Args:
            url: URL to fetch

        Returns:
            BeautifulSoup object or None
        """
        self._rate_limit_wait()

        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return BeautifulSoup(response.text, "lxml")
        except requests.RequestException as e:
            logger.error(f"Failed to fetch {url}: {e}")
            raise

    def _parse_height(self, height_str: str) -> Optional[float]:
        """Parse height string (e.g., '6' 2"') to cm."""
        if not height_str or height_str == "--":
            return None
        match = re.match(r"(\d+)'\s*(\d+)\"?", height_str.strip())
        if match:
            feet, inches = int(match.group(1)), int(match.group(2))
            return round((feet * 12 + inches) * 2.54, 1)
        return None

    def _parse_reach(self, reach_str: str) -> Optional[float]:
        """Parse reach string (e.g., '74"') to cm."""
        if not reach_str or reach_str == "--":
            return None
        match = re.match(r"(\d+)\"?", reach_str.strip())
        if match:
            return round(int(match.group(1)) * 2.54, 1)
        return None

    def _parse_weight(self, weight_str: str) -> Optional[float]:
        """Parse weight string (e.g., '170 lbs.') to kg."""
        if not weight_str or weight_str == "--":
            return None
        match = re.match(r"(\d+)", weight_str.strip())
        if match:
            return round(int(match.group(1)) * 0.453592, 1)
        return None

    def _parse_record(self, record_str: str) -> Dict[str, int]:
        """Parse record string (e.g., '20-3-0') to wins/losses/draws."""
        result = {"wins": 0, "losses": 0, "draws": 0, "no_contests": 0}
        if not record_str or record_str == "--":
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

    def _parse_percentage(self, pct_str: str) -> Optional[float]:
        """Parse percentage string (e.g., '52%') to float."""
        if not pct_str or pct_str == "--":
            return None
        match = re.match(r"(\d+)%?", pct_str.strip())
        if match:
            return int(match.group(1)) / 100.0
        return None

    def _parse_date(self, date_str: str) -> Optional[str]:
        """Parse date string to ISO format."""
        if not date_str or date_str == "--":
            return None

        # Try common formats
        formats = [
            "%b %d, %Y",  # Jan 15, 2025
            "%B %d, %Y",  # January 15, 2025
            "%m/%d/%Y",   # 01/15/2025
        ]

        for fmt in formats:
            try:
                dt = datetime.strptime(date_str.strip(), fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue

        return None

    # =========================================================================
    # FIGHTER SCRAPING
    # =========================================================================

    def scrape_all_fighters(self) -> List[Dict[str, Any]]:
        """
        Scrape all fighters from UFCStats.

        Returns:
            List of fighter dicts
        """
        logger.info("Scraping all fighters...")
        fighters = []

        # UFCStats has fighters organized alphabetically
        for letter in "abcdefghijklmnopqrstuvwxyz":
            url = f"{self.FIGHTERS_URL}?char={letter}&page=all"
            logger.debug(f"Fetching fighters: {letter.upper()}")

            try:
                soup = self._get_page(url)
                if soup:
                    page_fighters = self._parse_fighters_list(soup)
                    fighters.extend(page_fighters)
                    logger.debug(f"Found {len(page_fighters)} fighters for '{letter}'")
            except Exception as e:
                logger.error(f"Failed to scrape fighters for '{letter}': {e}")

        logger.info(f"Scraped {len(fighters)} fighters total")
        return fighters

    def _parse_fighters_list(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Parse the fighters list page."""
        fighters = []

        table = soup.find("table", class_="b-statistics__table")
        if not table:
            return fighters

        rows = table.find_all("tr", class_="b-statistics__table-row")
        for row in rows[1:]:  # Skip header
            cols = row.find_all("td")
            if len(cols) < 10:
                continue

            # Extract data from columns
            name_link = cols[0].find("a")
            if not name_link:
                continue

            name = name_link.text.strip()
            detail_url = name_link.get("href", "")

            record = self._parse_record(cols[2].text.strip())

            fighter = {
                "name": f"{cols[0].text.strip()} {cols[1].text.strip()}".strip(),
                "nickname": cols[1].text.strip() if len(cols) > 1 else None,
                "height_cm": self._parse_height(cols[3].text.strip()),
                "weight_kg": self._parse_weight(cols[4].text.strip()),
                "reach_cm": self._parse_reach(cols[5].text.strip()),
                "stance": cols[6].text.strip() if cols[6].text.strip() != "--" else None,
                "wins": record["wins"],
                "losses": record["losses"],
                "draws": record["draws"],
                "ufc_stats_url": detail_url,
            }
            fighters.append(fighter)

        return fighters

    def scrape_fighter_details(self, fighter_url: str) -> Optional[Dict[str, Any]]:
        """
        Scrape detailed information for a single fighter.

        Args:
            fighter_url: URL to the fighter's detail page

        Returns:
            Fighter details dict or None
        """
        try:
            soup = self._get_page(fighter_url)
            if not soup:
                return None

            return self._parse_fighter_page(soup)
        except Exception as e:
            logger.error(f"Failed to scrape fighter details: {e}")
            return None

    def _parse_fighter_page(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Parse a fighter's detail page."""
        fighter = {}

        # Name and nickname
        name_elem = soup.find("span", class_="b-content__title-highlight")
        if name_elem:
            fighter["name"] = name_elem.text.strip()

        nickname_elem = soup.find("p", class_="b-content__Nickname")
        if nickname_elem:
            fighter["nickname"] = nickname_elem.text.strip().strip('"')

        # Record
        record_elem = soup.find("span", class_="b-content__title-record")
        if record_elem:
            record = self._parse_record(record_elem.text.replace("Record:", "").strip())
            fighter.update(record)

        # Stats boxes
        stats_box = soup.find("ul", class_="b-list__box-list")
        if stats_box:
            items = stats_box.find_all("li")
            for item in items:
                text = item.text.strip()
                if "Height:" in text:
                    fighter["height_cm"] = self._parse_height(text.replace("Height:", ""))
                elif "Weight:" in text:
                    fighter["weight_kg"] = self._parse_weight(text.replace("Weight:", ""))
                elif "Reach:" in text:
                    fighter["reach_cm"] = self._parse_reach(text.replace("Reach:", ""))
                elif "STANCE:" in text:
                    stance = text.replace("STANCE:", "").strip()
                    fighter["stance"] = stance if stance != "--" else None
                elif "DOB:" in text:
                    fighter["dob"] = self._parse_date(text.replace("DOB:", ""))

        # Career statistics
        career_stats = soup.find_all("div", class_="b-list__info-box-left")
        for box in career_stats:
            items = box.find_all("li")
            for item in items:
                text = item.text.strip()
                if "SLpM:" in text:
                    val = text.replace("SLpM:", "").strip()
                    fighter["sig_strikes_landed_per_min"] = float(val) if val != "--" else None
                elif "Str. Acc.:" in text:
                    fighter["sig_strike_accuracy"] = self._parse_percentage(
                        text.replace("Str. Acc.:", "")
                    )
                elif "SApM:" in text:
                    val = text.replace("SApM:", "").strip()
                    fighter["sig_strikes_absorbed_per_min"] = float(val) if val != "--" else None
                elif "Str. Def:" in text:
                    fighter["sig_strike_defense"] = self._parse_percentage(
                        text.replace("Str. Def:", "")
                    )
                elif "TD Avg.:" in text:
                    val = text.replace("TD Avg.:", "").strip()
                    fighter["takedowns_avg_per_15min"] = float(val) if val != "--" else None
                elif "TD Acc.:" in text:
                    fighter["takedown_accuracy"] = self._parse_percentage(
                        text.replace("TD Acc.:", "")
                    )
                elif "TD Def.:" in text:
                    fighter["takedown_defense"] = self._parse_percentage(
                        text.replace("TD Def.:", "")
                    )
                elif "Sub. Avg.:" in text:
                    val = text.replace("Sub. Avg.:", "").strip()
                    fighter["submissions_avg_per_15min"] = float(val) if val != "--" else None

        return fighter

    # =========================================================================
    # EVENT SCRAPING
    # =========================================================================

    def scrape_all_events(self) -> List[Dict[str, Any]]:
        """
        Scrape all completed events.

        Returns:
            List of event dicts
        """
        logger.info("Scraping all completed events...")
        events = []

        try:
            soup = self._get_page(f"{self.EVENTS_COMPLETED_URL}?page=all")
            if soup:
                events = self._parse_events_list(soup, completed=True)
        except Exception as e:
            logger.error(f"Failed to scrape events: {e}")

        logger.info(f"Scraped {len(events)} events")
        return events

    def scrape_upcoming_events(self) -> List[Dict[str, Any]]:
        """
        Scrape upcoming events.

        Returns:
            List of upcoming event dicts
        """
        logger.info("Scraping upcoming events...")
        events = []

        try:
            soup = self._get_page(self.EVENTS_UPCOMING_URL)
            if soup:
                events = self._parse_events_list(soup, completed=False)
        except Exception as e:
            logger.error(f"Failed to scrape upcoming events: {e}")

        logger.info(f"Scraped {len(events)} upcoming events")
        return events

    def _parse_events_list(
        self,
        soup: BeautifulSoup,
        completed: bool = True
    ) -> List[Dict[str, Any]]:
        """Parse the events list page."""
        events = []

        table = soup.find("table", class_="b-statistics__table-events")
        if not table:
            return events

        rows = table.find_all("tr", class_="b-statistics__table-row")
        for row in rows:
            link = row.find("a", class_="b-link")
            if not link:
                continue

            date_span = row.find("span", class_="b-statistics__date")
            location_td = row.find_all("td")

            event = {
                "name": link.text.strip(),
                "date": self._parse_date(date_span.text.strip()) if date_span else None,
                "location": location_td[1].text.strip() if len(location_td) > 1 else None,
                "ufc_stats_url": link.get("href", ""),
                "is_completed": completed,
            }
            events.append(event)

        return events

    def scrape_event_details(self, event_url: str) -> Optional[Dict[str, Any]]:
        """
        Scrape detailed information for an event including all fights.

        Args:
            event_url: URL to the event page

        Returns:
            Event details with fights or None
        """
        try:
            soup = self._get_page(event_url)
            if not soup:
                return None

            return self._parse_event_page(soup)
        except Exception as e:
            logger.error(f"Failed to scrape event details: {e}")
            return None

    def _parse_event_page(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Parse an event detail page."""
        event = {"fights": []}

        # Event name
        name_elem = soup.find("span", class_="b-content__title-highlight")
        if name_elem:
            event["name"] = name_elem.text.strip()

        # Event info
        info_list = soup.find("ul", class_="b-list__box-list")
        if info_list:
            items = info_list.find_all("li")
            for item in items:
                text = item.text.strip()
                if "Date:" in text:
                    event["date"] = self._parse_date(text.replace("Date:", ""))
                elif "Location:" in text:
                    event["location"] = text.replace("Location:", "").strip()

        # Parse fights
        fight_table = soup.find("table", class_="b-fight-details__table")
        if fight_table:
            rows = fight_table.find_all("tr", class_="b-fight-details__table-row")
            for row in rows[1:]:  # Skip header
                fight = self._parse_fight_row(row)
                if fight:
                    event["fights"].append(fight)

        return event

    def _parse_fight_row(self, row) -> Optional[Dict[str, Any]]:
        """Parse a fight row from an event page."""
        cols = row.find_all("td")
        if len(cols) < 8:
            return None

        # Fighter names
        fighter_links = cols[1].find_all("a")
        if len(fighter_links) < 2:
            return None

        fight = {
            "fighter_red_name": fighter_links[0].text.strip(),
            "fighter_blue_name": fighter_links[1].text.strip(),
            "weight_class": cols[6].text.strip(),
            "method": cols[7].text.strip() if len(cols) > 7 else None,
            "round": cols[8].text.strip() if len(cols) > 8 else None,
            "time": cols[9].text.strip() if len(cols) > 9 else None,
        }

        # Get fight detail URL from the row (first link in the row)
        row_link = row.get("data-link")
        if row_link:
            fight["fight_url"] = row_link

        # Determine winner (first name is typically the winner)
        result_col = cols[0]
        if "win" in result_col.text.lower():
            fight["winner_name"] = fight["fighter_red_name"]
        elif "loss" in result_col.text.lower():
            fight["winner_name"] = fight["fighter_blue_name"]

        return fight

    def scrape_fight_stats(self, fight_url: str) -> Optional[Dict[str, Any]]:
        """
        Scrape detailed per-fight statistics from a fight detail page.

        Args:
            fight_url: URL to the fight detail page on UFCStats.com

        Returns:
            Dict with stats for both fighters or None if failed
        """
        try:
            soup = self._get_page(fight_url)
            if not soup:
                return None

            result = {
                "red": {},
                "blue": {}
            }

            # Get fighter names from the page
            fighter_names = soup.find_all("a", class_="b-link b-fight-details__person-link")
            if len(fighter_names) >= 2:
                result["red"]["name"] = fighter_names[0].text.strip()
                result["blue"]["name"] = fighter_names[1].text.strip()

            # Find the Totals table - it's typically the first table on the page
            # The Totals table doesn't have a specific class, but it follows a "Totals" header
            # and has exactly 1 row in tbody with 10 columns
            tables = soup.find_all("table")

            totals_table = None
            for table in tables:
                tbody = table.find("tbody")
                if tbody:
                    rows = tbody.find_all("tr")
                    # Totals table has exactly 1 row with 10 columns
                    if len(rows) == 1:
                        cols = rows[0].find_all("td")
                        if len(cols) == 10:
                            # Verify it's the Totals table by checking the first column has fighter names
                            first_col_p = cols[0].find_all("p")
                            if len(first_col_p) >= 2:
                                totals_table = table
                                break

            if totals_table:
                tbody = totals_table.find("tbody")
                row = tbody.find("tr")
                cols = row.find_all("td")

                # Parse the stats:
                # Col 0: Fighter names
                # Col 1: KD (Knockdowns)
                # Col 2: Sig. str. (Significant strikes)
                # Col 3: Sig. str. %
                # Col 4: Total str. (Total strikes)
                # Col 5: Td (Takedowns)
                # Col 6: Td %
                # Col 7: Sub. att (Submission attempts)
                # Col 8: Rev. (Reversals)
                # Col 9: Ctrl (Control time)

                # Knockdowns
                kd_col = cols[1].find_all("p")
                if len(kd_col) >= 2:
                    result["red"]["knockdowns"] = self._parse_int(kd_col[0].text)
                    result["blue"]["knockdowns"] = self._parse_int(kd_col[1].text)

                # Significant strikes (format: "144 of 254")
                sig_str_col = cols[2].find_all("p")
                if len(sig_str_col) >= 2:
                    red_sig = self._parse_strike_stat(sig_str_col[0].text)
                    blue_sig = self._parse_strike_stat(sig_str_col[1].text)
                    result["red"]["sig_strikes_landed"] = red_sig[0]
                    result["red"]["sig_strikes_attempted"] = red_sig[1]
                    result["blue"]["sig_strikes_landed"] = blue_sig[0]
                    result["blue"]["sig_strikes_attempted"] = blue_sig[1]

                # Total strikes
                total_str_col = cols[4].find_all("p")
                if len(total_str_col) >= 2:
                    red_total = self._parse_strike_stat(total_str_col[0].text)
                    blue_total = self._parse_strike_stat(total_str_col[1].text)
                    result["red"]["total_strikes_landed"] = red_total[0]
                    result["red"]["total_strikes_attempted"] = red_total[1]
                    result["blue"]["total_strikes_landed"] = blue_total[0]
                    result["blue"]["total_strikes_attempted"] = blue_total[1]

                # Takedowns
                td_col = cols[5].find_all("p")
                if len(td_col) >= 2:
                    red_td = self._parse_strike_stat(td_col[0].text)
                    blue_td = self._parse_strike_stat(td_col[1].text)
                    result["red"]["takedowns_landed"] = red_td[0]
                    result["red"]["takedowns_attempted"] = red_td[1]
                    result["blue"]["takedowns_landed"] = blue_td[0]
                    result["blue"]["takedowns_attempted"] = blue_td[1]

                # Submission attempts
                sub_col = cols[7].find_all("p")
                if len(sub_col) >= 2:
                    result["red"]["submission_attempts"] = self._parse_int(sub_col[0].text)
                    result["blue"]["submission_attempts"] = self._parse_int(sub_col[1].text)

                # Control time
                ctrl_col = cols[9].find_all("p")
                if len(ctrl_col) >= 2:
                    result["red"]["control_time"] = self._parse_control_time(ctrl_col[0].text)
                    result["blue"]["control_time"] = self._parse_control_time(ctrl_col[1].text)

            return result

        except Exception as e:
            logger.error(f"Failed to scrape fight stats from {fight_url}: {e}")
            return None

    def _parse_strike_stat(self, text: str) -> tuple:
        """Parse a strike stat like '50 of 100' into (landed, attempted)."""
        try:
            text = text.strip()
            if " of " in text:
                parts = text.split(" of ")
                return (int(parts[0].strip()), int(parts[1].strip()))
            return (0, 0)
        except:
            return (0, 0)

    def _parse_int(self, text: str) -> int:
        """Parse an integer from text."""
        try:
            return int(text.strip())
        except:
            return 0

    def _parse_control_time(self, text: str) -> int:
        """Parse control time like '4:30' into total seconds."""
        try:
            text = text.strip()
            if ":" in text:
                parts = text.split(":")
                return int(parts[0]) * 60 + int(parts[1])
            elif text == "--" or not text:
                return 0
            return int(text)
        except:
            return 0

    # =========================================================================
    # INCREMENTAL UPDATES
    # =========================================================================

    def scrape_new_events(self, since_date: Optional[date] = None) -> List[Dict[str, Any]]:
        """
        Scrape only events that occurred after the given date.

        Args:
            since_date: Only return events after this date

        Returns:
            List of new events
        """
        all_events = self.scrape_all_events()

        if since_date is None:
            return all_events

        new_events = []
        for event in all_events:
            event_date = event.get("date")
            if event_date:
                try:
                    ed = datetime.strptime(event_date, "%Y-%m-%d").date()
                    if ed > since_date:
                        new_events.append(event)
                except ValueError:
                    pass

        return new_events

    def scrape_new_fighters(self) -> List[Dict[str, Any]]:
        """
        Scrape and return all fighters (for incremental updates).

        Returns:
            List of fighters
        """
        return self.scrape_all_fighters()

    def get_completed_events_since(self, since_date: Optional[datetime]) -> List[Dict[str, Any]]:
        """
        Get completed events since a specific date.

        Args:
            since_date: Datetime to check from

        Returns:
            List of new completed events
        """
        if since_date is None:
            return self.scrape_all_events()

        return self.scrape_new_events(since_date.date())

    def scrape_upcoming_event_fights(self, event_url: str) -> List[Dict[str, Any]]:
        """
        Scrape the fight card for an upcoming event.

        Args:
            event_url: URL to the event page on UFCStats

        Returns:
            List of fight dicts with fighter names and weight class
        """
        logger.info(f"Scraping upcoming event fights from: {event_url}")
        fights = []

        try:
            soup = self._get_page(event_url)
            if not soup:
                return fights

            # Try to find fight table
            fight_table = soup.find("table", class_="b-fight-details__table")
            if fight_table:
                rows = fight_table.find_all("tr", class_="b-fight-details__table-row")
                bout_order = len(rows)

                for row in rows[1:]:  # Skip header
                    fight = self._parse_upcoming_fight_row(row, bout_order)
                    if fight:
                        fights.append(fight)
                        bout_order -= 1

            logger.info(f"Found {len(fights)} upcoming fights")
        except Exception as e:
            logger.error(f"Failed to scrape upcoming event fights: {e}")

        return fights

    def _parse_upcoming_fight_row(self, row, bout_order: int) -> Optional[Dict[str, Any]]:
        """Parse a fight row for an upcoming event (no results yet)."""
        cols = row.find_all("td")
        if len(cols) < 7:
            return None

        # Fighter names
        fighter_links = cols[1].find_all("a")
        if len(fighter_links) < 2:
            # Sometimes fighters are in spans instead
            fighter_spans = cols[1].find_all("span")
            if len(fighter_spans) < 2:
                return None
            fighter_red = fighter_spans[0].text.strip()
            fighter_blue = fighter_spans[1].text.strip()
        else:
            fighter_red = fighter_links[0].text.strip()
            fighter_blue = fighter_links[1].text.strip()

        # Weight class
        weight_class = cols[6].text.strip() if len(cols) > 6 else "Unknown"

        # Check for title fight indicator
        title_indicator = row.find(string=re.compile(r"title", re.I))
        is_title = title_indicator is not None

        # Determine card position based on bout order
        if bout_order >= len(cols):
            card_position = "main_card"
        elif bout_order <= 2:
            card_position = "early_prelims"
        elif bout_order <= 5:
            card_position = "prelims"
        else:
            card_position = "main_card"

        fight = {
            "fighter_red_name": fighter_red,
            "fighter_blue_name": fighter_blue,
            "weight_class": weight_class,
            "is_title_fight": is_title,
            "is_main_event": bout_order == 1,
            "card_position": card_position,
            "bout_order": bout_order,
        }

        return fight

    def scrape_and_save_upcoming_fights(self, data_service) -> int:
        """
        Scrape all upcoming events and save their fight cards.

        Args:
            data_service: DataService instance to save fights

        Returns:
            Total number of fights saved
        """
        total_saved = 0

        # Get upcoming events
        upcoming_events = self.scrape_upcoming_events()
        logger.info(f"Found {len(upcoming_events)} upcoming events")

        for event in upcoming_events:
            event_url = event.get("ufc_stats_url")
            event_name = event.get("name")

            if not event_url:
                logger.warning(f"No URL for event: {event_name}")
                continue

            # Get event from database
            db_event = data_service.get_event_by_name(event_name)
            if not db_event:
                # Try to find by date
                event_date = event.get("date")
                if event_date:
                    db_events = data_service.get_events_by_date(event_date)
                    if db_events:
                        db_event = db_events[0]

            if not db_event:
                logger.warning(f"Event not found in database: {event_name}")
                continue

            event_id = db_event.get("event_id")

            # Scrape fights for this event
            fights = self.scrape_upcoming_event_fights(event_url)

            if not fights:
                logger.info(f"No fights found for: {event_name}")
                continue

            # Match fighter names to IDs
            for fight in fights:
                red_fighter = data_service.search_fighters(fight["fighter_red_name"], limit=1)
                blue_fighter = data_service.search_fighters(fight["fighter_blue_name"], limit=1)

                if red_fighter and blue_fighter:
                    fight["fighter_red_id"] = red_fighter[0]["fighter_id"]
                    fight["fighter_blue_id"] = blue_fighter[0]["fighter_id"]
                else:
                    logger.warning(
                        f"Could not find fighters: {fight['fighter_red_name']} vs {fight['fighter_blue_name']}"
                    )
                    continue

            # Save fights with matched IDs
            fights_with_ids = [f for f in fights if "fighter_red_id" in f and "fighter_blue_id" in f]
            saved = data_service.save_upcoming_fights_batch(event_id, fights_with_ids)
            total_saved += saved
            logger.info(f"Saved {saved} fights for: {event_name}")

        return total_saved

    # =========================================================================
    # DATA QUALITY VALIDATION
    # =========================================================================

    def validate_fight_result(self, fight: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a scraped fight result for data quality.

        Args:
            fight: Scraped fight data

        Returns:
            Dict with validation results and any issues found
        """
        issues = []
        warnings = []
        is_valid = True

        # Required fields check
        required_fields = ["fighter_red_name", "fighter_blue_name"]
        for field in required_fields:
            if not fight.get(field):
                issues.append(f"Missing required field: {field}")
                is_valid = False

        # Method validation
        method = (fight.get("method") or "").upper()
        valid_methods = [
            "KO", "TKO", "SUB", "SUBMISSION", "DEC", "DECISION",
            "U-DEC", "S-DEC", "M-DEC", "UNANIMOUS", "SPLIT", "MAJORITY",
            "NC", "NO CONTEST", "DQ", "DISQUALIFICATION", "DRAW",
            "OVERTURNED", "DOCTOR", "TECHNICAL"
        ]

        if method and not any(vm in method for vm in valid_methods):
            warnings.append(f"Unknown method format: {method}")

        # Round validation
        round_num = fight.get("round")
        if round_num:
            try:
                rnd = int(round_num)
                if rnd < 1 or rnd > 5:
                    warnings.append(f"Unusual round number: {rnd}")
            except (ValueError, TypeError):
                warnings.append(f"Invalid round format: {round_num}")

        # Winner validation for completed fights
        if method and "NC" not in method and "DRAW" not in method:
            if not fight.get("winner_name"):
                issues.append("Missing winner for completed fight")
                is_valid = False

        # Weight class validation
        weight_class = fight.get("weight_class", "")
        valid_classes = [
            "Strawweight", "Flyweight", "Bantamweight", "Featherweight",
            "Lightweight", "Welterweight", "Middleweight", "Light Heavyweight",
            "Heavyweight", "Women's Strawweight", "Women's Flyweight",
            "Women's Bantamweight", "Women's Featherweight", "Catchweight",
            "Open Weight"
        ]
        if weight_class and not any(vc.lower() in weight_class.lower() for vc in valid_classes):
            warnings.append(f"Unknown weight class: {weight_class}")

        return {
            "is_valid": is_valid,
            "issues": issues,
            "warnings": warnings,
            "data_quality_score": 1.0 - (len(issues) * 0.3 + len(warnings) * 0.1),
        }

    def validate_fighter_data(self, fighter: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate scraped fighter data for data quality.

        Args:
            fighter: Scraped fighter data

        Returns:
            Dict with validation results
        """
        issues = []
        warnings = []
        is_valid = True

        # Name is required
        if not fighter.get("name"):
            issues.append("Missing fighter name")
            is_valid = False

        # Validate numeric ranges
        height = fighter.get("height_cm")
        if height is not None:
            if height < 140 or height > 220:
                warnings.append(f"Unusual height: {height}cm")

        reach = fighter.get("reach_cm")
        if reach is not None:
            if reach < 140 or reach > 230:
                warnings.append(f"Unusual reach: {reach}cm")

        weight = fighter.get("weight_kg")
        if weight is not None:
            if weight < 40 or weight > 160:
                warnings.append(f"Unusual weight: {weight}kg")

        # Validate percentages
        for pct_field in ["sig_strike_accuracy", "sig_strike_defense",
                          "takedown_accuracy", "takedown_defense"]:
            value = fighter.get(pct_field)
            if value is not None:
                if value < 0 or value > 1:
                    issues.append(f"Invalid percentage for {pct_field}: {value}")
                    is_valid = False

        # Validate record consistency
        wins = fighter.get("wins", 0) or 0
        losses = fighter.get("losses", 0) or 0
        draws = fighter.get("draws", 0) or 0

        if wins < 0 or losses < 0 or draws < 0:
            issues.append("Negative value in fight record")
            is_valid = False

        return {
            "is_valid": is_valid,
            "issues": issues,
            "warnings": warnings,
            "completeness_score": self._calculate_completeness(fighter),
        }

    def _calculate_completeness(self, fighter: Dict[str, Any]) -> float:
        """
        Calculate data completeness score for a fighter.

        Args:
            fighter: Fighter data dict

        Returns:
            Completeness score 0.0-1.0
        """
        important_fields = [
            "name", "height_cm", "reach_cm", "wins", "losses",
            "sig_strikes_landed_per_min", "sig_strike_accuracy",
            "sig_strike_defense", "takedown_accuracy", "takedown_defense"
        ]

        present = sum(1 for f in important_fields if fighter.get(f) is not None)
        return present / len(important_fields)

    def validate_event_data(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate scraped event data for data quality.

        Args:
            event: Scraped event data

        Returns:
            Dict with validation results
        """
        issues = []
        warnings = []
        is_valid = True

        # Required fields
        if not event.get("name"):
            issues.append("Missing event name")
            is_valid = False

        if not event.get("date"):
            issues.append("Missing event date")
            is_valid = False
        else:
            # Validate date format
            try:
                event_date = datetime.strptime(event["date"], "%Y-%m-%d")
                # Check if date is reasonable (UFC started in 1993)
                if event_date.year < 1993 or event_date.year > datetime.now().year + 1:
                    warnings.append(f"Unusual event date: {event['date']}")
            except ValueError:
                issues.append(f"Invalid date format: {event['date']}")
                is_valid = False

        # Validate fights if present
        fights = event.get("fights", [])
        fight_issues = []
        for i, fight in enumerate(fights):
            validation = self.validate_fight_result(fight)
            if not validation["is_valid"]:
                fight_issues.append(f"Fight {i+1}: {validation['issues']}")

        if fight_issues:
            warnings.extend(fight_issues)

        return {
            "is_valid": is_valid,
            "issues": issues,
            "warnings": warnings,
            "fights_validated": len(fights),
            "fights_with_issues": len(fight_issues),
        }

    def scrape_and_validate_event(self, event_url: str) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        """
        Scrape an event and validate the data quality.

        Args:
            event_url: URL to scrape

        Returns:
            Tuple of (event_data, validation_results)
        """
        event = self.scrape_event_details(event_url)
        if not event:
            return None, {"is_valid": False, "issues": ["Failed to scrape event"]}

        validation = self.validate_event_data(event)
        return event, validation
