#!/usr/bin/env python3
"""Huxley Sun automation gate.

Purpose:
- Read the existing Leads sheet using the service account.
- Count genuinely send-ready rows.
- Tell GitHub Actions whether discovery should run.
- Spend zero OpenAI credits.

A row counts as READY only when:
- Status == READY
- Email Quality is DIRECT or REPRESENTATIVE
- Email is present
- Outreach Subject is present
- Outreach Body is present

The script fails closed if the sheet/schema cannot be read, so a broken
spreadsheet does not accidentally trigger paid discovery.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List

from google.oauth2 import service_account
from googleapiclient.discovery import build

VERSION = "HS-PIPELINE-GATE-V1-20260808"

LEADS_TAB = "Leads"
DEFAULT_THRESHOLD = 30

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets.readonly"
]

REQUIRED_HEADERS = {
    "Status",
    "Email",
    "Email Quality",
    "Outreach Subject",
    "Outreach Body",
}


def env_required(name: str) -> str:
    value = os.environ.get(name, "").strip()

    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}"
        )

    return value


def normalize(value) -> str:
    return str(value or "").strip()


def build_sheets_service():
    raw = env_required("GOOGLE_SERVICE_ACCOUNT_JSON")

    try:
        info = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "GOOGLE_SERVICE_ACCOUNT_JSON is not valid JSON"
        ) from exc

    credentials = (
        service_account.Credentials.from_service_account_info(
            info,
            scopes=SCOPES,
        )
    )

    return build(
        "sheets",
        "v4",
        credentials=credentials,
        cache_discovery=False,
    )


def read_leads_values(
    service,
    spreadsheet_id: str,
) -> List[List[str]]:

    last_error = None

    for attempt in range(1, 5):

        try:
            result = (
                service
                .spreadsheets()
                .values()
                .get(
                    spreadsheetId=spreadsheet_id,
                    range=f"'{LEADS_TAB}'!A:AZ",
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
                f"Google Sheets read failed; "
                f"retry {attempt}/4 in {wait}s: "
                f"{type(exc).__name__}"
            )

            time.sleep(wait)

    raise RuntimeError(
        f"Could not read Leads sheet after retries: "
        f"{last_error}"
    )


def row_dict(
    headers: List[str],
    row: List[str],
) -> Dict[str, str]:

    padded = list(row) + [""] * max(
        0,
        len(headers) - len(row)
    )

    return {
        headers[i]: normalize(padded[i])
        for i in range(len(headers))
    }


def is_send_ready(
    record: Dict[str, str]
) -> bool:

    status = record.get(
        "Status",
        "",
    ).upper()

    quality = record.get(
        "Email Quality",
        "",
    ).upper()

    return (
        status == "READY"
        and quality in {
            "DIRECT",
            "REPRESENTATIVE",
        }
        and bool(record.get("Email"))
        and bool(record.get("Outreach Subject"))
        and bool(record.get("Outreach Body"))
    )


def set_github_output(
    name: str,
    value: str,
) -> None:

    output_file = os.environ.get(
        "GITHUB_OUTPUT",
        "",
    ).strip()

    if not output_file:
        return

    with open(
        output_file,
        "a",
        encoding="utf-8",
    ) as fh:
        fh.write(
            f"{name}={value}\n"
        )


def main() -> int:

    print(
        f"ENGINE VERSION: {VERSION}"
    )

    spreadsheet_id = env_required(
        "GOOGLE_SHEET_ID"
    )

    threshold_raw = os.environ.get(
        "READY_REFILL_THRESHOLD",
        str(DEFAULT_THRESHOLD),
    ).strip()

    try:
        threshold = int(
            threshold_raw
        )
    except ValueError as exc:
        raise RuntimeError(
            "READY_REFILL_THRESHOLD "
            "must be an integer"
        ) from exc

    if threshold < 1:
        raise RuntimeError(
            "READY_REFILL_THRESHOLD "
            "must be at least 1"
        )

    service = build_sheets_service()

    values = read_leads_values(
        service,
        spreadsheet_id,
    )

    if not values:
        raise RuntimeError(
            "Leads sheet is empty "
            "or unreadable"
        )

    headers = [
        normalize(x)
        for x in values[0]
    ]

    missing = sorted(
        REQUIRED_HEADERS - set(headers)
    )

    if missing:
        raise RuntimeError(
            "Leads sheet is missing "
            "required outreach columns: "
            + ", ".join(missing)
        )

    ready_count = 0

    for row in values[1:]:

        record = row_dict(
            headers,
            row,
        )

        if is_send_ready(record):
            ready_count += 1

    run_discovery = (
        ready_count < threshold
    )

    print(
        "======================================"
    )

    print(
        "HUXLEY SUN AUTOMATION GATE"
    )

    print(
        "======================================"
    )

    print(
        f"Verified READY queue: "
        f"{ready_count}"
    )

    print(
        f"Refill threshold: "
        f"{threshold}"
    )

    if run_discovery:

        print(
            "Decision: RUN discovery once, "
            "then prepare the new leads."
        )

    else:

        print(
            "Decision: SKIP paid discovery; "
            "the READY queue is healthy."
        )

    print(
        "OpenAI cost for this gate: $0"
    )

    set_github_output(
        "ready_count",
        str(ready_count),
    )

    set_github_output(
        "run_discovery",
        (
            "true"
            if run_discovery
            else "false"
        ),
    )

    return 0


if __name__ == "__main__":

    try:
        raise SystemExit(
            main()
        )

    except Exception as exc:

        print(
            f"GATE ERROR: {exc}",
            file=sys.stderr,
        )

        # Fail closed.
        # A sheet problem must NOT trigger
        # paid discovery by accident.
        raise
