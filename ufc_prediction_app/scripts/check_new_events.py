#!/usr/bin/env python3
"""
Check for newly completed UFC events since last update.

Writes a flag file for use with GitHub Actions.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.data_service import DataService
from services.scraper_service import UFCStatsScraper


def main():
    data_service = DataService()
    scraper = UFCStatsScraper()

    # Get last update timestamp
    last_update = data_service.get_last_update_timestamp()

    # Check for new completed events
    if last_update:
        new_events = scraper.get_completed_events_since(last_update)
    else:
        new_events = scraper.scrape_all_events()[:5]  # Limit for first run

    has_new_events = len(new_events) > 0

    # Write flag for GitHub Actions
    flag_file = Path(".new_events_flag")
    with open(flag_file, "w") as f:
        f.write("true" if has_new_events else "false")

    if has_new_events:
        print(f"Found {len(new_events)} new completed events:")
        for event in new_events[:5]:
            print(f"  - {event.get('name', 'Unknown')} ({event.get('date', 'N/A')})")
        if len(new_events) > 5:
            print(f"  ... and {len(new_events) - 5} more")
    else:
        print("No new completed events found.")

    return 0 if has_new_events else 1


if __name__ == "__main__":
    sys.exit(main())
