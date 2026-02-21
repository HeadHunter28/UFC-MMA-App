"""
UFC News Service Module.

Handles web scraping of UFC news from various MMA news sources.
Implements daily caching to avoid excessive requests.
"""

import logging
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from xml.etree import ElementTree

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CACHE_DIR, SCRAPER_USER_AGENT, NEWS_CACHE_TTL_HOURS, NEWS_MAX_ITEMS

logger = logging.getLogger(__name__)

# News cache settings
NEWS_CACHE_FILE = CACHE_DIR / "ufc_news_cache.json"


class UFCNewsService:
    """
    Service for fetching and caching UFC news from various sources.

    Uses RSS feeds for reliable, structured data access.
    """

    # RSS feed sources for UFC/MMA news
    NEWS_SOURCES = [
        {
            "name": "MMAJunkie",
            "url": "https://mmajunkie.usatoday.com/feed",
            "type": "rss",
        },
        {
            "name": "MMA Fighting",
            "url": "https://www.mmafighting.com/rss/current",
            "type": "rss",
        },
        {
            "name": "ESPN MMA",
            "url": "https://www.espn.com/espn/rss/mma/news",
            "type": "rss",
        },
    ]

    def __init__(self):
        """Initialize the news service."""
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": SCRAPER_USER_AGENT,
            "Accept": "application/rss+xml, application/xml, text/xml, */*",
        })
        # Ensure cache directory exists
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _load_cache(self) -> Optional[Dict[str, Any]]:
        """Load cached news if available and not expired."""
        try:
            if NEWS_CACHE_FILE.exists():
                with open(NEWS_CACHE_FILE, 'r', encoding='utf-8') as f:
                    cache = json.load(f)

                # Check if cache is still valid
                cached_time = datetime.fromisoformat(cache.get('timestamp', '2000-01-01'))
                if datetime.now() - cached_time < timedelta(hours=NEWS_CACHE_TTL_HOURS):
                    logger.debug("Using cached news data")
                    return cache

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Failed to load news cache: {e}")

        return None

    def _save_cache(self, news_items: List[Dict[str, Any]]) -> None:
        """Save news items to cache."""
        try:
            cache = {
                'timestamp': datetime.now().isoformat(),
                'items': news_items,
            }
            with open(NEWS_CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
            logger.debug(f"Saved {len(news_items)} news items to cache")
        except Exception as e:
            logger.error(f"Failed to save news cache: {e}")

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=5)
    )
    def _fetch_rss_feed(self, url: str) -> Optional[str]:
        """Fetch RSS feed content."""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            logger.warning(f"Failed to fetch RSS feed {url}: {e}")
            return None

    def _parse_rss_feed(self, xml_content: str, source_name: str) -> List[Dict[str, Any]]:
        """Parse RSS feed XML into news items."""
        items = []

        try:
            root = ElementTree.fromstring(xml_content)

            # Handle different RSS formats
            # Standard RSS 2.0
            channel = root.find('channel')
            if channel is not None:
                for item in channel.findall('item'):
                    news_item = self._parse_rss_item(item, source_name)
                    if news_item:
                        items.append(news_item)

            # Atom format
            else:
                # Try Atom namespace
                ns = {'atom': 'http://www.w3.org/2005/Atom'}
                for entry in root.findall('.//atom:entry', ns) or root.findall('.//entry'):
                    news_item = self._parse_atom_entry(entry, source_name, ns)
                    if news_item:
                        items.append(news_item)

        except ElementTree.ParseError as e:
            logger.error(f"Failed to parse RSS feed from {source_name}: {e}")

        return items

    def _parse_rss_item(self, item: ElementTree.Element, source_name: str) -> Optional[Dict[str, Any]]:
        """Parse a single RSS item element."""
        try:
            title_elem = item.find('title')
            link_elem = item.find('link')
            desc_elem = item.find('description')
            pub_date_elem = item.find('pubDate')

            if title_elem is None or title_elem.text is None:
                return None

            title = title_elem.text.strip()

            # Filter for UFC-related news
            if not self._is_ufc_related(title):
                return None

            link = link_elem.text.strip() if link_elem is not None and link_elem.text else ""
            description = self._clean_html(desc_elem.text) if desc_elem is not None and desc_elem.text else ""

            # Parse publication date
            pub_date = None
            if pub_date_elem is not None and pub_date_elem.text:
                pub_date = self._parse_date(pub_date_elem.text)

            return {
                'title': title,
                'link': link,
                'description': description[:300] + "..." if len(description) > 300 else description,
                'source': source_name,
                'published': pub_date.isoformat() if pub_date else None,
                'published_display': self._format_relative_time(pub_date) if pub_date else "Recently",
            }

        except Exception as e:
            logger.debug(f"Failed to parse RSS item: {e}")
            return None

    def _parse_atom_entry(self, entry: ElementTree.Element, source_name: str, ns: dict) -> Optional[Dict[str, Any]]:
        """Parse a single Atom entry element."""
        try:
            # Try with and without namespace
            title_elem = entry.find('atom:title', ns) or entry.find('title')
            link_elem = entry.find('atom:link', ns) or entry.find('link')
            summary_elem = entry.find('atom:summary', ns) or entry.find('summary')
            updated_elem = entry.find('atom:updated', ns) or entry.find('updated')

            if title_elem is None or title_elem.text is None:
                return None

            title = title_elem.text.strip()

            if not self._is_ufc_related(title):
                return None

            # Handle link (Atom uses href attribute)
            link = ""
            if link_elem is not None:
                link = link_elem.get('href', '') or (link_elem.text or "")

            description = self._clean_html(summary_elem.text) if summary_elem is not None and summary_elem.text else ""

            pub_date = None
            if updated_elem is not None and updated_elem.text:
                pub_date = self._parse_date(updated_elem.text)

            return {
                'title': title,
                'link': link,
                'description': description[:300] + "..." if len(description) > 300 else description,
                'source': source_name,
                'published': pub_date.isoformat() if pub_date else None,
                'published_display': self._format_relative_time(pub_date) if pub_date else "Recently",
            }

        except Exception as e:
            logger.debug(f"Failed to parse Atom entry: {e}")
            return None

    def _is_ufc_related(self, text: str) -> bool:
        """Check if text is UFC-related."""
        ufc_keywords = [
            'ufc', 'mma', 'bellator', 'pfl', 'one championship',
            'knockout', 'ko', 'tko', 'submission', 'decision',
            'octagon', 'cage', 'fight', 'fighter', 'fighting',
            'title', 'champion', 'championship', 'bout', 'main event',
            'prelim', 'weigh-in', 'weight', 'weight class',
        ]
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in ufc_keywords)

    def _clean_html(self, text: str) -> str:
        """Remove HTML tags from text."""
        if not text:
            return ""
        # Remove HTML tags
        clean = re.sub(r'<[^>]+>', '', text)
        # Remove extra whitespace
        clean = re.sub(r'\s+', ' ', clean).strip()
        # Decode common HTML entities
        clean = clean.replace('&amp;', '&')
        clean = clean.replace('&lt;', '<')
        clean = clean.replace('&gt;', '>')
        clean = clean.replace('&quot;', '"')
        clean = clean.replace('&#39;', "'")
        clean = clean.replace('&nbsp;', ' ')
        return clean

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse various date formats."""
        formats = [
            "%a, %d %b %Y %H:%M:%S %z",  # RFC 2822: Mon, 15 Jan 2025 12:00:00 +0000
            "%a, %d %b %Y %H:%M:%S %Z",  # With timezone name
            "%Y-%m-%dT%H:%M:%S%z",        # ISO 8601
            "%Y-%m-%dT%H:%M:%SZ",         # ISO 8601 UTC
            "%Y-%m-%d %H:%M:%S",          # Simple datetime
            "%Y-%m-%d",                    # Just date
        ]

        date_str = date_str.strip()

        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue

        # Try parsing ISO format with fromisoformat (handles microseconds)
        try:
            return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        except ValueError:
            pass

        return None

    def _format_relative_time(self, dt: datetime) -> str:
        """Format datetime as relative time string."""
        if dt is None:
            return "Recently"

        # Handle timezone-aware datetime
        now = datetime.now()
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)

        diff = now - dt

        if diff.days > 7:
            return dt.strftime("%b %d, %Y")
        elif diff.days > 1:
            return f"{diff.days} days ago"
        elif diff.days == 1:
            return "Yesterday"
        elif diff.seconds > 3600:
            hours = diff.seconds // 3600
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        elif diff.seconds > 60:
            minutes = diff.seconds // 60
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        else:
            return "Just now"

    def get_latest_news(self, limit: int = 10, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Get the latest UFC news from all sources.

        Args:
            limit: Maximum number of news items to return
            force_refresh: If True, bypass cache and fetch fresh data

        Returns:
            List of news items sorted by publication date
        """
        # Check cache first unless force refresh
        if not force_refresh:
            cache = self._load_cache()
            if cache:
                return cache['items'][:limit]

        # Fetch from all sources
        all_news = []

        for source in self.NEWS_SOURCES:
            logger.info(f"Fetching news from {source['name']}...")

            xml_content = self._fetch_rss_feed(source['url'])
            if xml_content:
                items = self._parse_rss_feed(xml_content, source['name'])
                all_news.extend(items)
                logger.info(f"Got {len(items)} UFC-related items from {source['name']}")

        # Sort by publication date (newest first)
        all_news.sort(
            key=lambda x: x.get('published') or '1970-01-01',
            reverse=True
        )

        # Remove duplicates (same title from different sources)
        seen_titles = set()
        unique_news = []
        for item in all_news:
            # Normalize title for comparison
            normalized = item['title'].lower().strip()
            if normalized not in seen_titles:
                seen_titles.add(normalized)
                unique_news.append(item)

        # Cache the results
        if unique_news:
            self._save_cache(unique_news)

        return unique_news[:limit]

    def get_news_by_source(self, source_name: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get news filtered by source."""
        all_news = self.get_latest_news(limit=50)
        return [n for n in all_news if n['source'].lower() == source_name.lower()][:limit]

    def search_news(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search news by keyword."""
        all_news = self.get_latest_news(limit=50)
        query_lower = query.lower()

        matching = [
            n for n in all_news
            if query_lower in n['title'].lower() or query_lower in n.get('description', '').lower()
        ]

        return matching[:limit]

    def get_cache_status(self) -> Dict[str, Any]:
        """Get information about the news cache."""
        if NEWS_CACHE_FILE.exists():
            try:
                with open(NEWS_CACHE_FILE, 'r', encoding='utf-8') as f:
                    cache = json.load(f)

                cached_time = datetime.fromisoformat(cache.get('timestamp', '2000-01-01'))
                age_hours = (datetime.now() - cached_time).total_seconds() / 3600

                return {
                    'exists': True,
                    'timestamp': cache.get('timestamp'),
                    'age_hours': round(age_hours, 1),
                    'item_count': len(cache.get('items', [])),
                    'is_stale': age_hours >= NEWS_CACHE_TTL_HOURS,
                }
            except Exception:
                pass

        return {
            'exists': False,
            'timestamp': None,
            'age_hours': None,
            'item_count': 0,
            'is_stale': True,
        }
