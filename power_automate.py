import os
import requests
from dotenv import load_dotenv

load_dotenv()

POWER_AUTOMATE_URL = os.getenv("POWER_AUTOMATE_URL")
MANAGER_EMAIL = os.getenv("MANAGER_EMAIL")


def send_leave_email(payload: dict):
    if not POWER_AUTOMATE_URL:
        print("POWER_AUTOMATE_URL missing")
        return

    try:
        res = requests.post(POWER_AUTOMATE_URL, json=payload, timeout=10)
        print("Power Automate status:", res.status_code)
        print("Power Automate response:", res.text)
    except Exception as e:
        print("Power Automate error:", e)