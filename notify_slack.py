#!/usr/bin/env python3
"""Send a Slack notification via incoming webhook.

Usage:
    python notify_slack.py "Your message here"

Reads SLACK_WEBHOOK_URL from .env in the current directory.
"""

import json
import os
import sys
import urllib.request
from pathlib import Path


def load_webhook() -> str:
    webhook = os.environ.get("SLACK_WEBHOOK_URL", "")
    if webhook:
        return webhook
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("SLACK_WEBHOOK_URL="):
                return line.split("=", 1)[1].strip()
    return ""


def send(message: str):
    webhook = load_webhook()
    if not webhook:
        print("ERROR: SLACK_WEBHOOK_URL not set in .env or environment", file=sys.stderr)
        sys.exit(1)
    payload = json.dumps({"text": message}).encode()
    req = urllib.request.Request(
        webhook, data=payload, headers={"Content-Type": "application/json"}
    )
    urllib.request.urlopen(req, timeout=10)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <message>", file=sys.stderr)
        sys.exit(1)
    send(" ".join(sys.argv[1:]))
    print("Sent.")
