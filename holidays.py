HOLIDAYS = [
    "2026-01-26",  # Republic Day
    "2026-08-15",  # Independence Day
    "2026-10-02",  # Gandhi Jayanti
    "2026-12-25",  # Christmas
]
from datetime import datetime

def is_invalid_leave_date(date_str: str):
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")

        # Weekend
        if date_obj.weekday() >= 5:
            return True, "Selected date falls on a weekend."

        # Holiday
        if date_str in HOLIDAYS:
            return True, "Selected date is a company holiday."

        return False, None

    except:
        return True, "Invalid date format."