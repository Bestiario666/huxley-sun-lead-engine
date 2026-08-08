#!/usr/bin/env python3
"""Huxley Sun automation gate.

Reads the live Google Sheet and decides whether paid lead discovery should run.

This gate mirrors the Gmail sender's actual send-safety rules so the READY count
represents unique contacts the sender can really use.

A READY contact counts only when:
- Status == READY
- Email Quality is DIRECT or REPRESENTATIVE
- Email is syntactically usable and not on the sender blocklist
- Email Source URL exists
- Outreach Subject exists
- Outreach Body exists
- Follow-Up Body exists
- Suggested Song is currently ACTIVE in the Songs tab
- The email has not already been contacted by another lead
- The same READY email is counted only once

This script uses ZERO OpenAI calls.

Safety behavior:
- Reads each entire tab by sheet name instead of requesting a fixed wide range.
- Retries Google Sheets reads.
- Fails closed if required structure is missing.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import Dict, List, Set, Tuple

from google.oauth2 import service_account
from googleapiclient.discovery import build

VERSION = "HS-PIPELINE-GATE-V3-20260808"

LEADS_TAB = "Leads"
SONGS_TAB = "Songs"
DEFAULT_THRESHOLD = 30

SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]

REQUIRED_LEAD_HEADERS = {
    "Status",
    "Email",
    "Email Source URL",
    "Email Quality",
    "Outreach Subject",
    "Outreach Body",
    "Follow-Up Body",
    "Suggested Song",
}

REQUIRED_SONG_HEADERS = {"Song", "Active"}
ALLOWED_EMAIL_QUALITY = {"DIRECT", "REPRESENTATIVE"}

CONTACTED_STATUSES = {
    "SENT",
    "FOLLOWED_UP",
    "REPLIED",
    "CLOSED_NO_REPLY",
    "CONTACTED",
    "SENDING_INITIAL",
    "SENDING_FOLLOWUP",
}

BLOCKED_EMAIL_DOMAINS = {
    "example.com",
    "email.com",
    "domain.com",
    "test.com",
    "company.com",
    "sentry.io",
}

BLOCKED_LOCAL_PARTS = {
    "noreply",
    "no-reply",
    "do-not-reply",
    "donotreply",
    "privacy",
    "legal",
    "abuse",
    "security",
    "postmaster",
    "mailer-daemon",
}

EMAIL_RE = re.compile(
    r"^[A-Z0-9.!#$%&'*+/=?^_`{|}~-]+@"
    r"[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?"
    r"(?:\.[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?)+$",
    re.IGNORECASE,
)


def env_required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def normalize(value) -> str:
    return str(value or "").strip()


def normalize_email(value) -> str:
    return normalize(value).lower()


def build_sheets_service():
    raw = env_required("GOOGLE_SERVICE_ACCOUNT_JSON")
    try:
        info = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_JSON is not valid JSON") from exc

    credentials = service_account.Credentials.from_service_account_info(
        info,
        scopes=SCOPES,
    )

    return build(
        "sheets",
        "v4",
        credentials=credentials,
        cache_discovery=False,
    )


def read_tab_values(service, spreadsheet_id: str, tab_name: str) -> List[List[str]]:
    """Read all existing cells from one tab using the tab name itself."""
    last_error = None

    for attempt in range(1, 5):
        try:
            result = (
                service
                .spreadsheets()
                .values()
                .get(
                    spreadsheetId=spreadsheet_id,
                    range=f"'{tab_name}'",
                    majorDimension="ROWS",
                )
                .execute()
            )
            return result.get("values", [])
        except Exception as exc:
            last_error = exc
            if attempt == 4:
                break
            wait = 2 ** (attempt - 1)
            print(
                f"Google Sheets read for '{tab_name}' failed; "
                f"retry {attempt}/4 in {wait}s: {type(exc).__name__}"
            )
            time.sleep(wait)

    raise RuntimeError(f"Could not read '{tab_name}' after retries: {last_error}")


def row_dict(headers: List[str], row: List[str]) -> Dict[str, str]:
    padded = list(row) + [""] * max(0, len(headers) - len(row))
    return {headers[i]: normalize(padded[i]) for i in range(len(headers))}


def validate_headers(actual_headers: List[str], required: Set[str], tab_name: str) -> None:
    missing = sorted(required - set(actual_headers))
    if missing:
        raise RuntimeError(
            f"'{tab_name}' is missing required columns: " + ", ".join(missing)
        )


def load_active_songs(song_rows: List[List[str]]) -> Set[str]:
    if not song_rows:
        raise RuntimeError("Songs sheet is empty or unreadable")

    headers = [normalize(x) for x in song_rows[0]]
    validate_headers(headers, REQUIRED_SONG_HEADERS, SONGS_TAB)

    active: Set[str] = set()
    for row in song_rows[1:]:
        record = row_dict(headers, row)
        flag = record.get("Active", "").upper()
        if flag in {"YES", "Y", "TRUE", "1"}:
            song = record.get("Song", "")
            if song:
                active.add(song)

    if not active:
        raise RuntimeError("Songs sheet contains no active songs")

    return active


def valid_sender_email(email: str) -> bool:
    email = normalize_email(email)
    if not email or not EMAIL_RE.match(email):
        return False

    local, domain = email.rsplit("@", 1)
    if domain in BLOCKED_EMAIL_DOMAINS:
        return False
    if local in BLOCKED_LOCAL_PARTS:
        return False
    return True


def analyze_ready_queue(
    lead_rows: List[List[str]],
    active_songs: Set[str],
) -> Tuple[int, int, int]:
    """Return unique_sendable, raw_ready_rows, duplicate_or_used_ready_rows."""
    if not lead_rows:
        raise RuntimeError("Leads sheet is empty or unreadable")

    headers = [normalize(x) for x in lead_rows[0]]
    validate_headers(headers, REQUIRED_LEAD_HEADERS, LEADS_TAB)
    records = [row_dict(headers, row) for row in lead_rows[1:]]

    contacted_emails: Set[str] = set()
    for record in records:
        status = record.get("Status", "").upper()
        if status in CONTACTED_STATUSES:
            email = normalize_email(record.get("Email", ""))
            if email:
                contacted_emails.add(email)

    raw_ready_rows = 0
    duplicate_or_used = 0
    unique_ready: Set[str] = set()

    for record in records:
        if record.get("Status", "").upper() != "READY":
            continue

        raw_ready_rows += 1
        quality = record.get("Email Quality", "").upper()
        email = normalize_email(record.get("Email", ""))
        song = record.get("Suggested Song", "")

        sendable = (
            quality in ALLOWED_EMAIL_QUALITY
            and valid_sender_email(email)
            and bool(record.get("Email Source URL"))
            and bool(record.get("Outreach Subject"))
            and bool(record.get("Outreach Body"))
            and bool(record.get("Follow-Up Body"))
            and song in active_songs
        )

        if not sendable:
            continue

        if email in contacted_emails or email in unique_ready:
            duplicate_or_used += 1
            continue

        unique_ready.add(email)

    return len(unique_ready), raw_ready_rows, duplicate_or_used


def set_github_output(name: str, value: str) -> None:
    output_file = os.environ.get("GITHUB_OUTPUT", "").strip()
    if not output_file:
        return
    with open(output_file, "a", encoding="utf-8") as fh:
        fh.write(f"{name}={value}\n")


def main() -> int:
    print(f"ENGINE VERSION: {VERSION}")

    spreadsheet_id = env_required("GOOGLE_SHEET_ID")
    threshold_raw = os.environ.get(
        "READY_REFILL_THRESHOLD",
        str(DEFAULT_THRESHOLD),
    ).strip()

    try:
        threshold = int(threshold_raw)
    except ValueError as exc:
        raise RuntimeError("READY_REFILL_THRESHOLD must be an integer") from exc

    if threshold < 1:
        raise RuntimeError("READY_REFILL_THRESHOLD must be at least 1")

    service = build_sheets_service()
    song_rows = read_tab_values(service, spreadsheet_id, SONGS_TAB)
    active_songs = load_active_songs(song_rows)
    lead_rows = read_tab_values(service, spreadsheet_id, LEADS_TAB)

    ready_count, raw_ready_rows, duplicate_or_used = analyze_ready_queue(
        lead_rows,
        active_songs,
    )

    run_discovery = ready_count < threshold

    print("======================================")
    print("HUXLEY SUN AUTOMATION GATE")
    print("======================================")
    print(f"Active songs: {len(active_songs)}")
    print(f"Rows currently marked READY: {raw_ready_rows}")
    print(f"Unique actually sendable READY contacts: {ready_count}")
    if duplicate_or_used:
        print(
            "READY rows excluded as duplicate/already-used emails: "
            f"{duplicate_or_used}"
        )
    print(f"Refill threshold: {threshold}")

    if run_discovery:
        print("Decision: queue needs work.")
    else:
        print("Decision: queue healthy; skip prep and paid discovery.")

    print("OpenAI cost for this gate: $0")

    set_github_output("ready_count", str(ready_count))
    set_github_output("run_discovery", "true" if run_discovery else "false")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"GATE ERROR: {exc}", file=sys.stderr)
        raise
