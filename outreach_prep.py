#!/usr/bin/env python3
"""
Huxley Sun Outreach Prep Engine

Purpose
-------
Turn existing NEW leads into safe, personalized outreach drafts WITHOUT sending email.

Pipeline
--------
1. Read existing Leads + active Songs from Google Sheets.
2. Inspect ONLY rows with Status=NEW and a non-empty Email.
3. Validate the email programmatically and inspect the public Email Source URL directly.
4. Use GPT-5 nano (NO web-search tool) to classify the email as:
      DIRECT / REPRESENTATIVE / REVIEW / REJECT
5. Only DIRECT/REPRESENTATIVE rows continue.
6. Re-match every accepted lead against the CURRENT ACTIVE Songs catalogue.
   This means songs with Active=NO can never be selected, even if an older row used them.
7. Generate Outreach Priority, subject, first email and one follow-up.
8. If multiple leads share the same email, keep only the highest-priority one READY.
9. Update the Leads sheet. Nothing is sent.

Status values written by this script
------------------------------------
READY              safe draft prepared for later Gmail sending
REVIEW_EMAIL       email is plausible but not sufficiently verified
REJECTED_EMAIL     invalid / placeholder / technical / unrelated address
DUPLICATE_CONTACT  same inbox already has a higher-priority READY lead

This script intentionally uses no OpenAI web-search calls.
"""

from __future__ import annotations

import html
import json
import os
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any

from google.oauth2 import service_account
from googleapiclient.discovery import build
from openai import OpenAI


# ============================================================
# VERSION / CONFIG
# ============================================================

ENGINE_VERSION = "HS-OUTREACH-PREP-V1-20260808"

LEADS_TAB = "Leads"
SONGS_TAB = "Songs"
MODEL = "gpt-5-nano"

CONTACT_BATCH_SIZE = 12
DRAFT_BATCH_SIZE = 6
MODEL_ATTEMPTS = 2

# We only prepare leads whose existing overall match score is at least this.
MIN_MATCH_SCORE = 80

# Keep drafts short. This is outreach, not a press release.
DRAFT_MAX_OUTPUT_TOKENS = 6500
CONTACT_MAX_OUTPUT_TOKENS = 3500

# Approximate current nano pricing used only for the terminal estimate.
MODEL_INPUT_USD_PER_M = 0.05
MODEL_OUTPUT_USD_PER_M = 0.40

# Direct source-page inspection is free of OpenAI cost.
HTTP_TIMEOUT_SECONDS = 8
SOURCE_CRAWL_WORKERS = 10
MAX_SOURCE_BYTES = 1_000_000
USER_AGENT = (
    "Mozilla/5.0 (compatible; HuxleySunOutreachPrep/1.0; "
    "+https://openai.com/)"
)

# Hard safety filters. These override any model classification.
PLACEHOLDER_DOMAINS = {
    "example.com",
    "example.org",
    "company.com",
    "domain.com",
    "test.com",
}
TECHNICAL_DOMAINS = {
    "sentry.io",
}
PLACEHOLDER_LOCALPARTS = {
    "brand",
    "yourname",
    "your.name",
    "email",
    "example",
    "test",
    "name",
}

# A contact found on these sites is often the publisher/directory's own inbox,
# not the creator's. These are NOT automatically rejected if the source page
# clearly identifies a representation relationship, but they default to REVIEW.
THIRD_PARTY_EDITORIAL_OR_DIRECTORY_DOMAINS = {
    "newyorker.com",
    "feedspot.com",
    "infldb.com",
    "socialveins.com",
    "collabstr.com",
    "creatordb.app",
    "gondola.cc",
    "thesocialshepherd.com",
    "trope.com",
    "visualsofearth.com",
    "yespress.io",
}

REPRESENTATION_WORDS = {
    "agent",
    "agency",
    "booking",
    "bookings",
    "management",
    "manager",
    "managed",
    "representation",
    "represented",
    "representative",
    "press inquiries",
    "business inquiries",
    "commercial inquiries",
}

EMAIL_RE = re.compile(
    r"^[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}$",
    re.IGNORECASE,
)

# Initialized in init_clients().
openai_client: OpenAI | None = None
sheets = None
GOOGLE_SHEET_ID = ""

usage_totals = {"input_tokens": 0, "output_tokens": 0}


# ============================================================
# SHEET SCHEMAS
# ============================================================

REQUIRED_SONG_HEADERS = [
    "Song",
    "Stream URL",
    "Playlist URL",
    "Album URL",
    "Vocal Type",
    "Main Instrument",
    "Mood Tags",
    "Visual Tags",
    "Energy",
    "Song Meaning / Theme",
    "Best For",
    "Description",
    "Active",
]

OUTREACH_HEADERS = [
    "Email Quality",
    "Email Verification Reason",
    "Outreach Priority",
    "Greeting Name",
    "Outreach Subject",
    "Outreach Body",
    "Follow-Up Body",
    "Prepared At",
    "Prep Version",
]

# Existing headers that this stage needs to read or update.
CORE_LEAD_HEADERS = [
    "Creator",
    "Platform",
    "Profile URL",
    "Followers",
    "Country",
    "Content Type",
    "Recent Content",
    "Suggested Song",
    "Song Match Score",
    "Song Link",
    "More Music Link",
    "More Music Type",
    "Alternative Song",
    "Alternative Song Match",
    "Alternative Song Link",
    "Match Score",
    "Reason",
    "Contact URL",
    "Email",
    "Email Source URL",
    "Status",
    "Notes",
]


# ============================================================
# GENERIC HELPERS
# ============================================================


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def clean(value: Any) -> str:
    return str(value or "").strip()


def normalize_email(value: str) -> str:
    return clean(value).replace("\\@", "@").lower()


def valid_email(value: str) -> bool:
    return bool(EMAIL_RE.match(normalize_email(value)))


def email_parts(value: str) -> tuple[str, str]:
    email = normalize_email(value)
    if "@" not in email:
        return "", ""
    local, domain = email.rsplit("@", 1)
    return local, domain.lower().strip(".")


def normalized_host(url: str) -> str:
    value = clean(url)
    if not value:
        return ""
    if "://" not in value:
        value = "https://" + value
    try:
        host = urllib.parse.urlparse(value).hostname or ""
    except ValueError:
        return ""
    host = host.lower().strip(".")
    if host.startswith("www."):
        host = host[4:]
    return host


def registrableish_domain(host: str) -> str:
    """Good-enough domain comparison without adding a public-suffix dependency."""
    host = clean(host).lower().strip(".")
    if not host:
        return ""
    parts = host.split(".")
    if len(parts) <= 2:
        return host
    # Preserve common ccTLD structures such as co.uk / com.au.
    if parts[-2] in {"co", "com", "org", "net", "ac"} and len(parts[-1]) == 2:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def same_site(domain_or_host_a: str, domain_or_host_b: str) -> bool:
    a = registrableish_domain(domain_or_host_a)
    b = registrableish_domain(domain_or_host_b)
    return bool(a and b and a == b)


def parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except Exception:
        return default


def parse_followers(value: str) -> int | None:
    """Parse common follower formats. Returns None when not meaningfully numeric."""
    text = clean(value).lower().replace(",", "")
    if not text or "not verified" in text or "not publicly" in text:
        return None
    # Use the first numeric quantity in strings such as '1M+ across platforms'.
    m = re.search(r"(\d+(?:\.\d+)?)\s*([kmb])?", text)
    if not m:
        return None
    number = float(m.group(1))
    suffix = m.group(2)
    multiplier = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}.get(suffix, 1)
    return int(number * multiplier)


def strip_html(raw_html: str) -> str:
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", raw_html)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def creator_name_tokens(creator: str) -> list[str]:
    tokens = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+", clean(creator).lower())
    stop = {
        "films", "film", "media", "studios", "studio", "official", "travel",
        "photography", "photographer", "cinematic", "journey", "production",
        "productions", "visuals", "visual", "the", "and", "of", "by",
    }
    return [t for t in tokens if len(t) >= 3 and t not in stop][:8]


def source_page_evidence(url: str, email: str, creator: str) -> dict[str, Any]:
    """Fetch public source URL directly and capture evidence around the email.

    This is NOT a web-search API call and has no OpenAI tool charge.
    Failure to fetch does not reject a lead by itself.
    """
    evidence = {
        "fetched": False,
        "http_status": "",
        "email_seen": False,
        "creator_seen": False,
        "representation_words_seen": False,
        "snippet": "",
        "source_host": normalized_host(url),
    }

    url = clean(url)
    if not url or not url.lower().startswith(("http://", "https://")):
        return evidence

    # PDFs and large binary assets are not parsed here.
    if urllib.parse.urlparse(url).path.lower().endswith(".pdf"):
        evidence["http_status"] = "PDF_SKIPPED"
        return evidence

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml",
        },
    )

    try:
        context = ssl.create_default_context()
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS, context=context) as response:
            evidence["http_status"] = str(getattr(response, "status", "") or "")
            content_type = clean(response.headers.get("Content-Type", "")).lower()
            if "html" not in content_type and "text" not in content_type:
                return evidence
            payload = response.read(MAX_SOURCE_BYTES)
            charset = response.headers.get_content_charset() or "utf-8"
            raw = payload.decode(charset, errors="replace")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ssl.SSLError, ValueError) as exc:
        evidence["http_status"] = f"FETCH_FAILED:{type(exc).__name__}"
        return evidence
    except Exception as exc:
        evidence["http_status"] = f"FETCH_FAILED:{type(exc).__name__}"
        return evidence

    evidence["fetched"] = True
    text = strip_html(raw)
    lower = text.lower()
    target_email = normalize_email(email)
    evidence["email_seen"] = target_email in lower

    tokens = creator_name_tokens(creator)
    evidence["creator_seen"] = any(token in lower for token in tokens)
    evidence["representation_words_seen"] = any(word in lower for word in REPRESENTATION_WORDS)

    # Capture context around the email when possible, otherwise around creator token.
    anchor = lower.find(target_email) if target_email else -1
    if anchor < 0:
        for token in tokens:
            anchor = lower.find(token)
            if anchor >= 0:
                break
    if anchor >= 0:
        start = max(0, anchor - 420)
        end = min(len(text), anchor + 620)
        evidence["snippet"] = text[start:end]
    else:
        evidence["snippet"] = text[:900]

    return evidence


def hard_email_problem(email: str) -> tuple[str, str] | None:
    """Return forced classification/reason for undeniable bad addresses."""
    normalized = normalize_email(email)
    if not valid_email(normalized):
        return "REJECT", "Email is missing or syntactically invalid."

    local, domain = email_parts(normalized)
    if domain in PLACEHOLDER_DOMAINS or local in PLACEHOLDER_LOCALPARTS:
        return "REJECT", "Placeholder/example email address."

    if domain in TECHNICAL_DOMAINS or domain.endswith(".sentry.io"):
        return "REJECT", "Technical telemetry/service address, not a creator contact."

    # Sentry-style generated UUID inboxes can occur on custom subdomains too.
    if re.fullmatch(r"[0-9a-f]{24,}", local):
        return "REJECT", "Machine-generated technical address, not a business contact."

    return None


def first_name_guess(creator: str) -> str:
    text = clean(creator)
    # Prefer text before separators.
    text = re.split(r"\s*[—/|]\s*", text)[0].strip()
    # Avoid greeting a company as a person's first name.
    company_words = {"media", "films", "studios", "studio", "production", "productions", "official", "cinema"}
    first = re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ'’-]", "", text.split()[0]) if text else ""
    if first.lower() in company_words or not first:
        return "there"
    return first


# ============================================================
# GOOGLE SHEETS
# ============================================================


def init_clients() -> None:
    global openai_client, sheets, GOOGLE_SHEET_ID

    openai_client = OpenAI(api_key=require_env("OPENAI_API_KEY"))
    GOOGLE_SHEET_ID = require_env("GOOGLE_SHEET_ID")
    service_account_info = json.loads(require_env("GOOGLE_SERVICE_ACCOUNT_JSON"))

    credentials = service_account.Credentials.from_service_account_info(
        service_account_info,
        scopes=["https://www.googleapis.com/auth/spreadsheets"],
    )
    sheets = build(
        "sheets",
        "v4",
        credentials=credentials,
        cache_discovery=False,
    )


def rebuild_sheets_client() -> None:
    global sheets
    service_account_info = json.loads(require_env("GOOGLE_SERVICE_ACCOUNT_JSON"))
    credentials = service_account.Credentials.from_service_account_info(
        service_account_info,
        scopes=["https://www.googleapis.com/auth/spreadsheets"],
    )
    sheets = build("sheets", "v4", credentials=credentials, cache_discovery=False)


def sheets_execute(request_builder, attempts: int = 5):
    global sheets
    delay = 1
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            return request_builder(sheets).execute()
        except Exception as exc:
            last_error = exc
            if attempt == attempts:
                break
            print(
                f"Google Sheets request failed ({type(exc).__name__}). "
                f"Retry {attempt}/{attempts - 1} in {delay}s..."
            )
            time.sleep(delay)
            delay = min(delay * 2, 12)
            rebuild_sheets_client()
    raise last_error


def get_tab_values(tab_name: str, range_suffix: str = "A1:ZZ") -> list[list[str]]:
    result = sheets_execute(
        lambda svc: svc.spreadsheets().values().get(
            spreadsheetId=GOOGLE_SHEET_ID,
            range=f"{tab_name}!{range_suffix}",
        )
    )
    return result.get("values", [])


def column_letter(number: int) -> str:
    result = ""
    while number:
        number, remainder = divmod(number - 1, 26)
        result = chr(65 + remainder) + result
    return result


def get_sheet_properties(tab_name: str) -> dict[str, int]:
    result = sheets_execute(
        lambda svc: svc.spreadsheets().get(
            spreadsheetId=GOOGLE_SHEET_ID,
            includeGridData=False,
            fields="sheets(properties(sheetId,title,gridProperties(rowCount,columnCount)))",
        )
    )
    for item in result.get("sheets", []):
        props = item.get("properties", {})
        if props.get("title") == tab_name:
            grid = props.get("gridProperties", {})
            return {
                "sheet_id": int(props.get("sheetId")),
                "row_count": int(grid.get("rowCount", 0) or 0),
                "column_count": int(grid.get("columnCount", 0) or 0),
            }
    raise RuntimeError(f"Google Sheet tab '{tab_name}' was not found.")


def ensure_tab_capacity(tab_name: str, min_columns: int | None = None, min_rows: int | None = None) -> None:
    props = get_sheet_properties(tab_name)
    current_cols = props["column_count"]
    current_rows = props["row_count"]
    target_cols = max(current_cols, int(min_columns or current_cols or 1))
    target_rows = max(current_rows, int(min_rows or current_rows or 1))

    if target_cols == current_cols and target_rows == current_rows:
        return

    grid_properties: dict[str, int] = {}
    fields: list[str] = []
    if target_cols != current_cols:
        grid_properties["columnCount"] = target_cols
        fields.append("gridProperties.columnCount")
    if target_rows != current_rows:
        grid_properties["rowCount"] = target_rows
        fields.append("gridProperties.rowCount")

    sheets_execute(
        lambda svc: svc.spreadsheets().batchUpdate(
            spreadsheetId=GOOGLE_SHEET_ID,
            body={
                "requests": [
                    {
                        "updateSheetProperties": {
                            "properties": {
                                "sheetId": props["sheet_id"],
                                "gridProperties": grid_properties,
                            },
                            "fields": ",".join(fields),
                        }
                    }
                ]
            },
        )
    )
    print(
        f"Expanded '{tab_name}' grid: columns {current_cols}->{target_cols}, "
        f"rows {current_rows}->{target_rows}."
    )


def ensure_outreach_headers() -> list[str]:
    rows = get_tab_values(LEADS_TAB, "1:1")
    headers = [clean(x) for x in (rows[0] if rows else [])]
    if not headers:
        raise RuntimeError("Leads tab has no header row.")

    missing_core = [h for h in CORE_LEAD_HEADERS if h not in headers]
    if missing_core:
        raise RuntimeError(
            "Leads sheet is missing required existing headers: " + ", ".join(missing_core)
        )

    missing = [h for h in OUTREACH_HEADERS if h not in headers]
    if missing:
        start_col = len(headers) + 1
        final_count = len(headers) + len(missing)
        ensure_tab_capacity(LEADS_TAB, min_columns=final_count)
        sheets_execute(
            lambda svc: svc.spreadsheets().values().update(
                spreadsheetId=GOOGLE_SHEET_ID,
                range=f"{LEADS_TAB}!{column_letter(start_col)}1",
                valueInputOption="RAW",
                body={"values": [missing]},
            )
        )
        headers.extend(missing)
        print("Added Leads headers: " + ", ".join(missing))

    # Verify after writing.
    check = get_tab_values(LEADS_TAB, "1:1")
    actual = [clean(x) for x in (check[0] if check else [])]
    still_missing = [h for h in OUTREACH_HEADERS if h not in actual]
    if still_missing:
        raise RuntimeError("Outreach header verification failed: " + ", ".join(still_missing))
    return actual


def batch_update_rows(headers: list[str], row_updates: dict[int, dict[str, Any]]) -> None:
    """Batch-update individual cells. row number is the actual 1-based Sheet row."""
    if not row_updates:
        return

    hmap = {header: idx + 1 for idx, header in enumerate(headers)}
    data = []
    for row_number, updates in row_updates.items():
        for header, value in updates.items():
            if header not in hmap:
                raise RuntimeError(f"Cannot update missing header: {header}")
            col = column_letter(hmap[header])
            data.append({
                "range": f"{LEADS_TAB}!{col}{row_number}",
                "values": [[value]],
            })

    # Google accepts many ranges at once, but chunk conservatively.
    for start in range(0, len(data), 400):
        chunk = data[start:start + 400]
        sheets_execute(
            lambda svc, chunk=chunk: svc.spreadsheets().values().batchUpdate(
                spreadsheetId=GOOGLE_SHEET_ID,
                body={
                    "valueInputOption": "RAW",
                    "data": chunk,
                },
            )
        )


# ============================================================
# SONGS
# ============================================================


def load_active_songs() -> list[dict[str, str]]:
    rows = get_tab_values(SONGS_TAB)
    if not rows:
        raise RuntimeError(f"The '{SONGS_TAB}' tab is empty.")

    headers = [clean(h) for h in rows[0]]
    missing = [h for h in REQUIRED_SONG_HEADERS if h not in headers]
    if missing:
        raise RuntimeError("Songs sheet is missing exact headers: " + ", ".join(missing))

    hmap = {h: i for i, h in enumerate(headers)}

    def cell(row: list[str], header: str) -> str:
        i = hmap[header]
        return clean(row[i]) if i < len(row) else ""

    songs: list[dict[str, str]] = []
    for row in rows[1:]:
        name = cell(row, "Song")
        if not name:
            continue
        active = cell(row, "Active").upper()
        if active not in {"YES", "Y", "TRUE", "1"}:
            continue

        playlist = cell(row, "Playlist URL")
        album = cell(row, "Album URL")
        playlist = "" if playlist.upper() == "NO" else playlist
        album = "" if album.upper() == "NO" else album
        more_url = playlist or album
        more_type = "Playlist" if playlist else "Album" if album else ""

        songs.append({
            "song": name,
            "stream_url": cell(row, "Stream URL"),
            "playlist_url": playlist,
            "album_url": album,
            "more_music_url": more_url,
            "more_music_type": more_type,
            "vocal_type": cell(row, "Vocal Type"),
            "main_instrument": cell(row, "Main Instrument"),
            "mood_tags": cell(row, "Mood Tags"),
            "visual_tags": cell(row, "Visual Tags"),
            "energy": cell(row, "Energy"),
            "meaning": cell(row, "Song Meaning / Theme"),
            "best_for": cell(row, "Best For"),
            "description": cell(row, "Description"),
        })

    if not songs:
        raise RuntimeError("No active songs found in Songs tab.")

    print("Active songs loaded: " + ", ".join(x["song"] for x in songs))
    return songs


def song_prompt_payload(songs: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "song": s["song"],
            "vocal_type": s["vocal_type"],
            "main_instrument": s["main_instrument"],
            "mood_tags": s["mood_tags"],
            "visual_tags": s["visual_tags"],
            "energy": s["energy"],
            "meaning": s["meaning"],
            "best_for": s["best_for"],
            "description": s["description"],
        }
        for s in songs
    ]


# ============================================================
# LEADS
# ============================================================


def load_new_email_leads(headers: list[str]) -> list[dict[str, Any]]:
    rows = get_tab_values(LEADS_TAB)
    if not rows:
        return []

    actual_headers = [clean(h) for h in rows[0]]
    hmap = {h: i for i, h in enumerate(actual_headers)}

    def cell(row: list[str], header: str) -> str:
        i = hmap.get(header)
        return clean(row[i]) if i is not None and i < len(row) else ""

    leads: list[dict[str, Any]] = []
    for sheet_row_number, row in enumerate(rows[1:], start=2):
        status = cell(row, "Status").upper()
        email = normalize_email(cell(row, "Email"))
        score = parse_int(cell(row, "Match Score"), 0)

        if status != "NEW":
            continue
        if not email:
            continue
        if score < MIN_MATCH_SCORE:
            continue

        leads.append({
            "row_number": sheet_row_number,
            "creator": cell(row, "Creator"),
            "platform": cell(row, "Platform"),
            "profile_url": cell(row, "Profile URL"),
            "followers": cell(row, "Followers"),
            "country": cell(row, "Country"),
            "content_type": cell(row, "Content Type"),
            "recent_content": cell(row, "Recent Content"),
            "existing_suggested_song": cell(row, "Suggested Song"),
            "existing_match_score": score,
            "reason": cell(row, "Reason"),
            "contact_url": cell(row, "Contact URL"),
            "email": email,
            "email_source_url": cell(row, "Email Source URL"),
            "notes": cell(row, "Notes"),
        })

    return leads


# ============================================================
# OPENAI SAFE STRUCTURED OUTPUT
# ============================================================


def record_usage(response) -> None:
    usage = getattr(response, "usage", None)
    if usage:
        usage_totals["input_tokens"] += int(getattr(usage, "input_tokens", 0) or 0)
        usage_totals["output_tokens"] += int(getattr(usage, "output_tokens", 0) or 0)


def response_summary(response) -> str:
    status = clean(getattr(response, "status", ""))
    incomplete = getattr(response, "incomplete_details", None)
    return f"status={status!r}, incomplete={incomplete!r}"


def parse_structured_json(response, label: str) -> dict[str, Any]:
    status = clean(getattr(response, "status", ""))
    text = clean(getattr(response, "output_text", ""))
    if status and status != "completed":
        raise RuntimeError(f"{label}: response incomplete ({response_summary(response)})")
    if not text:
        raise RuntimeError(f"{label}: empty output ({response_summary(response)})")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{label}: invalid structured JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"{label}: expected object, got {type(parsed).__name__}")
    return parsed


def call_structured(prompt: str, schema_name: str, schema: dict[str, Any], max_output_tokens: int) -> dict[str, Any]:
    if openai_client is None:
        raise RuntimeError("OpenAI client not initialized")

    last_error = None
    for attempt in range(1, MODEL_ATTEMPTS + 1):
        try:
            response = openai_client.responses.create(
                model=MODEL,
                reasoning={"effort": "minimal"},
                input=prompt,
                max_output_tokens=max_output_tokens,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": schema_name,
                        "schema": schema,
                        "strict": True,
                    }
                },
                store=False,
            )
            record_usage(response)
            return parse_structured_json(response, schema_name)
        except Exception as exc:
            last_error = exc
            if attempt == MODEL_ATTEMPTS:
                break
            print(f"{schema_name} response problem: {exc}. Retrying once...")
            time.sleep(1)
    raise last_error


# ============================================================
# EMAIL CLASSIFICATION
# ============================================================


def contact_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "row_number": {"type": "integer"},
                        "email_class": {
                            "type": "string",
                            "enum": ["DIRECT", "REPRESENTATIVE", "REVIEW", "REJECT"],
                        },
                        "reason": {"type": "string"},
                    },
                    "required": ["row_number", "email_class", "reason"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["results"],
        "additionalProperties": False,
    }


def classify_contact_batch(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    payload = []
    for lead in batch:
        payload.append({
            "row_number": lead["row_number"],
            "creator": lead["creator"],
            "platform": lead["platform"],
            "profile_url": lead["profile_url"],
            "contact_url": lead["contact_url"],
            "email": lead["email"],
            "email_source_url": lead["email_source_url"],
            "email_repeat_count": lead["email_repeat_count"],
            "source_evidence": lead["source_evidence"],
        })

    prompt = f"""
You are validating public business contact information for music-placement outreach.
Use ONLY the supplied data and source-page evidence. Do not invent facts.

Classify every row:
DIRECT = the public source reasonably demonstrates this is the creator/project's own business contact.
REPRESENTATIVE = the public source reasonably demonstrates an agent/manager/agency/press/representation relationship to this creator.
REVIEW = plausible, but the relationship cannot be verified strongly enough for automated outreach.
REJECT = placeholder, technical, publisher/directory contact unrelated to the creator, or clearly wrong person/company.

Important rules:
- A public Gmail/Outlook address is acceptable as DIRECT when it is shown on the creator's own website/source.
- An email domain different from the creator's website is NOT automatically a representative.
- A news publication, directory, influencer database, or article's generic help/info email is NOT the creator's email.
- If the same generic email appears for several unrelated creators, be highly skeptical.
- A representative is allowed only when the supplied evidence indicates representation/management/booking/press relationship.
- If unsure, choose REVIEW, never guess.
- Keep reason under 24 words.

ROWS:
{json.dumps(payload, ensure_ascii=False)}
"""

    data = call_structured(
        prompt,
        "email_contact_validation",
        contact_schema(),
        CONTACT_MAX_OUTPUT_TOKENS,
    )
    results = data.get("results", [])
    return results if isinstance(results, list) else []


def classify_all_contacts(leads: list[dict[str, Any]]) -> None:
    by_row = {lead["row_number"]: lead for lead in leads}

    # Hard programmatic decisions first.
    ai_needed = []
    for lead in leads:
        hard = hard_email_problem(lead["email"])
        if hard:
            lead["email_class"], lead["email_reason"] = hard
            continue

        email_domain = email_parts(lead["email"])[1]
        source_host = normalized_host(lead["email_source_url"])
        contact_host = normalized_host(lead["contact_url"])
        evidence = lead["source_evidence"]

        # If there is no public source URL, it cannot be auto-ready.
        if not clean(lead["email_source_url"]):
            lead["email_class"] = "REVIEW"
            lead["email_reason"] = "No Email Source URL is recorded."
            continue

        # Known third-party editorial/directory inbox on that same third-party site:
        # default to REVIEW and let AI decide only if source evidence suggests representation.
        source_root = registrableish_domain(source_host)
        email_root = registrableish_domain(email_domain)
        is_third_party = source_root in THIRD_PARTY_EDITORIAL_OR_DIRECTORY_DOMAINS
        if is_third_party and same_site(source_root, email_root) and not evidence.get("representation_words_seen"):
            lead["email_class"] = "REVIEW"
            lead["email_reason"] = "Email belongs to a third-party publisher/directory; creator relationship is not established."
            continue

        # Strong direct case: email is actually visible on creator/contact source and creator is visible too.
        if evidence.get("email_seen") and evidence.get("creator_seen"):
            if same_site(email_domain, source_host) or same_site(email_domain, contact_host) or email_domain in {"gmail.com", "outlook.com", "hotmail.com", "icloud.com", "yahoo.com"}:
                lead["email_class"] = "DIRECT"
                lead["email_reason"] = "Public source shows this email together with the creator/project."
                continue

        ai_needed.append(lead)

    print(f"Contact validation: {len(leads) - len(ai_needed)} programmatic decisions, {len(ai_needed)} need cheap AI review.")

    for start in range(0, len(ai_needed), CONTACT_BATCH_SIZE):
        batch = ai_needed[start:start + CONTACT_BATCH_SIZE]
        print(f"Contact AI batch {start // CONTACT_BATCH_SIZE + 1}: {len(batch)} leads...")
        try:
            results = classify_contact_batch(batch)
        except Exception as exc:
            print(f"Contact batch failed safely: {exc}. Marking batch REVIEW instead of crashing.")
            results = []

        returned = set()
        for item in results:
            row_number = parse_int(item.get("row_number"), 0)
            if row_number not in by_row:
                continue
            returned.add(row_number)
            label = clean(item.get("email_class")).upper()
            if label not in {"DIRECT", "REPRESENTATIVE", "REVIEW", "REJECT"}:
                label = "REVIEW"
            by_row[row_number]["email_class"] = label
            by_row[row_number]["email_reason"] = clean(item.get("reason"))[:400]

        for lead in batch:
            if lead["row_number"] not in returned:
                lead["email_class"] = "REVIEW"
                lead["email_reason"] = "Contact validation did not return a reliable decision."


# ============================================================
# OUTREACH DRAFT + ACTIVE SONG MATCHING
# ============================================================


def draft_schema(song_names: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "row_number": {"type": "integer"},
                        "outreach_priority": {"type": "integer", "minimum": 0, "maximum": 100},
                        "greeting_name": {"type": "string"},
                        "primary_song": {"type": "string", "enum": song_names},
                        "primary_song_match": {"type": "integer", "minimum": 0, "maximum": 100},
                        "alternative_song": {"type": "string", "enum": song_names},
                        "alternative_song_match": {"type": "integer", "minimum": 0, "maximum": 100},
                        "subject": {"type": "string"},
                        "body": {"type": "string"},
                        "follow_up": {"type": "string"},
                    },
                    "required": [
                        "row_number",
                        "outreach_priority",
                        "greeting_name",
                        "primary_song",
                        "primary_song_match",
                        "alternative_song",
                        "alternative_song_match",
                        "subject",
                        "body",
                        "follow_up",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["results"],
        "additionalProperties": False,
    }


def draft_batch(batch: list[dict[str, Any]], songs: list[dict[str, str]]) -> list[dict[str, Any]]:
    song_names = [s["song"] for s in songs]
    payload = []
    for lead in batch:
        payload.append({
            "row_number": lead["row_number"],
            "creator": lead["creator"],
            "platform": lead["platform"],
            "followers": lead["followers"],
            "country": lead["country"],
            "content_type": lead["content_type"],
            "recent_content": lead["recent_content"],
            "overall_match_score": lead["existing_match_score"],
            "existing_reason": lead["reason"],
            "email_class": lead["email_class"],
            "email": lead["email"],
        })

    prompt = f"""
You are preparing concise, respectful cold outreach for Huxley Sun, an independent music project.
The goal is NOT to sell aggressively. The recipient is a filmmaker/photographer/visual creator who may use music in future work.

Use ONLY the supplied creator data and ACTIVE SONG catalogue. Do not invent projects, awards, locations, clients, or things you supposedly watched.

For every lead:
1. Pick the best ACTIVE primary song and a DIFFERENT ACTIVE alternative song.
2. Give each song a 0-100 fit score.
3. Give an Outreach Priority 0-100 considering:
   - existing overall match score,
   - realistic likelihood of response,
   - creator scale (independent/mid-size can outrank huge creators),
   - clear professional music-use opportunity,
   - DIRECT contact slightly preferred to REPRESENTATIVE.
   Do not simply copy the existing match score.
4. Choose a natural greeting name. For organizations use "there" if no person is obvious.
5. Write a short subject, ideally 3-7 words, not clickbait and not salesy.
6. Write a plain-text first email, about 65-105 words BEFORE links/signature are appended by Python.
   - Start with "Hi NAME,".
   - Mention ONE concrete supplied aspect of their work.
   - Say you make music as Huxley Sun and the chosen song felt compatible.
   - Do NOT say you watched/saw a specific piece unless Recent Content actually names it.
   - Do NOT claim the music is royalty-free or free to use.
   - Invite them to listen and say you'd be happy to discuss usage/licensing if it ever fits.
   - Do NOT include any URLs; Python adds verified song links afterward.
   - Do NOT include a signature; Python adds it.
7. Write one gentle 30-55 word follow-up with no guilt, no urgency, no URLs and no signature.

All active Huxley Sun songs may be considered for short films and documentaries; Best For is additional guidance, not an exhaustive restriction.

ACTIVE SONGS:
{json.dumps(song_prompt_payload(songs), ensure_ascii=False)}

LEADS:
{json.dumps(payload, ensure_ascii=False)}
"""

    data = call_structured(
        prompt,
        "outreach_drafts",
        draft_schema(song_names),
        DRAFT_MAX_OUTPUT_TOKENS,
    )
    results = data.get("results", [])
    return results if isinstance(results, list) else []


def prepare_drafts(ready_leads: list[dict[str, Any]], songs: list[dict[str, str]]) -> None:
    by_row = {lead["row_number"]: lead for lead in ready_leads}
    for start in range(0, len(ready_leads), DRAFT_BATCH_SIZE):
        batch = ready_leads[start:start + DRAFT_BATCH_SIZE]
        print(f"Draft batch {start // DRAFT_BATCH_SIZE + 1}: {len(batch)} leads...")
        try:
            results = draft_batch(batch, songs)
        except Exception as exc:
            print(f"Draft batch failed safely: {exc}. These leads will remain REVIEW_EMAIL.")
            results = []

        returned = set()
        for item in results:
            row_number = parse_int(item.get("row_number"), 0)
            lead = by_row.get(row_number)
            if not lead:
                continue
            returned.add(row_number)
            lead["outreach_priority"] = max(0, min(100, parse_int(item.get("outreach_priority"), 0)))
            lead["greeting_name"] = clean(item.get("greeting_name")) or first_name_guess(lead["creator"])
            lead["primary_song"] = clean(item.get("primary_song"))
            lead["primary_song_match"] = max(0, min(100, parse_int(item.get("primary_song_match"), 0)))
            lead["alternative_song"] = clean(item.get("alternative_song"))
            lead["alternative_song_match"] = max(0, min(100, parse_int(item.get("alternative_song_match"), 0)))
            lead["subject"] = clean(item.get("subject"))[:180]
            lead["body_core"] = clean(item.get("body"))
            lead["follow_up_core"] = clean(item.get("follow_up"))

        for lead in batch:
            if lead["row_number"] not in returned:
                lead["draft_failed"] = True


def apply_song_links_and_final_text(leads: list[dict[str, Any]], songs: list[dict[str, str]]) -> None:
    song_map = {s["song"]: s for s in songs}
    active_names = set(song_map)

    for lead in leads:
        if lead.get("draft_failed"):
            continue

        primary = lead.get("primary_song", "")
        alternative = lead.get("alternative_song", "")
        if primary not in active_names or alternative not in active_names or primary == alternative:
            lead["draft_failed"] = True
            continue

        song = song_map[primary]
        alt = song_map[alternative]

        body = clean(lead.get("body_core"))
        # Ensure greeting exists even if model omitted it somehow.
        if not body.lower().startswith("hi "):
            body = f"Hi {lead.get('greeting_name') or first_name_guess(lead['creator'])},\n\n" + body

        body = body.rstrip()
        body += f"\n\n{primary}: {song['stream_url']}"
        if song.get("more_music_url"):
            body += f"\nMore music: {song['more_music_url']}"
        body += "\n\nBest,\nHuxley Sun"

        follow = clean(lead.get("follow_up_core")).rstrip()
        if follow and not follow.lower().startswith("hi "):
            follow = f"Hi {lead.get('greeting_name') or first_name_guess(lead['creator'])},\n\n" + follow
        follow += "\n\nBest,\nHuxley Sun"

        lead["song_link"] = song["stream_url"]
        lead["more_music_link"] = song.get("more_music_url", "")
        lead["more_music_type"] = song.get("more_music_type", "")
        lead["alternative_song_link"] = alt["stream_url"]
        lead["final_body"] = body
        lead["final_follow_up"] = follow


# ============================================================
# FINAL DEDUP / SHEET UPDATES
# ============================================================


def choose_one_lead_per_email(leads: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for lead in leads:
        groups[normalize_email(lead["email"])].append(lead)

    winners = []
    duplicates = []
    for _, group in groups.items():
        group.sort(
            key=lambda x: (
                parse_int(x.get("outreach_priority"), 0),
                parse_int(x.get("existing_match_score"), 0),
            ),
            reverse=True,
        )
        winners.append(group[0])
        duplicates.extend(group[1:])
    return winners, duplicates


def build_sheet_updates(
    all_leads: list[dict[str, Any]],
    final_ready: list[dict[str, Any]],
    duplicates: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    updates: dict[int, dict[str, Any]] = {}
    ready_rows = {x["row_number"] for x in final_ready}
    duplicate_rows = {x["row_number"] for x in duplicates}
    prepared_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    for lead in all_leads:
        row = lead["row_number"]
        classification = lead.get("email_class", "REVIEW")
        reason = clean(lead.get("email_reason"))

        base = {
            "Email Quality": classification,
            "Email Verification Reason": reason,
            "Prepared At": prepared_at,
            "Prep Version": ENGINE_VERSION,
        }

        if row in duplicate_rows:
            base.update({
                "Status": "DUPLICATE_CONTACT",
                "Notes": (clean(lead.get("notes")) + " | Shared email; a higher-priority lead owns this inbox for outreach.").strip(" |"),
            })
            updates[row] = base
            continue

        if row in ready_rows:
            base.update({
                "Status": "READY",
                "Outreach Priority": lead.get("outreach_priority", 0),
                "Greeting Name": lead.get("greeting_name", ""),
                "Outreach Subject": lead.get("subject", ""),
                "Outreach Body": lead.get("final_body", ""),
                "Follow-Up Body": lead.get("final_follow_up", ""),
                # Refresh song fields against ACTIVE catalogue.
                "Suggested Song": lead.get("primary_song", ""),
                "Song Match Score": lead.get("primary_song_match", 0),
                "Song Link": lead.get("song_link", ""),
                "More Music Link": lead.get("more_music_link", ""),
                "More Music Type": lead.get("more_music_type", ""),
                "Alternative Song": lead.get("alternative_song", ""),
                "Alternative Song Match": lead.get("alternative_song_match", 0),
                "Alternative Song Link": lead.get("alternative_song_link", ""),
            })
            updates[row] = base
            continue

        # Any accepted contact whose draft failed is not safe to auto-send.
        if classification in {"DIRECT", "REPRESENTATIVE"} and lead.get("draft_failed"):
            classification = "REVIEW"
            reason = (reason + " Draft generation failed safely; manual review required.").strip()
            base["Email Quality"] = classification
            base["Email Verification Reason"] = reason

        if classification == "REJECT":
            base["Status"] = "REJECTED_EMAIL"
        else:
            base["Status"] = "REVIEW_EMAIL"
        updates[row] = base

    return updates


# ============================================================
# SELF TESTS / COST
# ============================================================


def self_check() -> None:
    assert normalize_email("Test\\@Example.com") == "test@example.com"
    assert hard_email_problem("brand@company.com")[0] == "REJECT"
    assert hard_email_problem("abcdef1234567890abcdef1234567890@sentry.io")[0] == "REJECT"
    assert parse_followers("90.9K") == 90_900
    assert parse_followers("1M+ across platforms") == 1_000_000
    assert same_site("www.creator.co.uk", "mail.creator.co.uk")
    print("Startup self-check: PASS")


def estimated_cost() -> float:
    return (
        usage_totals["input_tokens"] / 1_000_000 * MODEL_INPUT_USD_PER_M
        + usage_totals["output_tokens"] / 1_000_000 * MODEL_OUTPUT_USD_PER_M
    )


def print_summary(leads: list[dict[str, Any]], final_ready: list[dict[str, Any]], duplicates: list[dict[str, Any]]) -> None:
    classes = Counter(clean(x.get("email_class", "REVIEW")) for x in leads)
    print("\n======================================")
    print("OUTREACH PREP RESULTS")
    print("======================================")
    print(f"Input NEW leads with email + match >= {MIN_MATCH_SCORE}: {len(leads)}")
    print(f"DIRECT: {classes.get('DIRECT', 0)}")
    print(f"REPRESENTATIVE: {classes.get('REPRESENTATIVE', 0)}")
    print(f"REVIEW: {classes.get('REVIEW', 0)}")
    print(f"REJECT: {classes.get('REJECT', 0)}")
    print(f"Duplicate shared inboxes suppressed: {len(duplicates)}")
    print(f"READY drafts written: {len(final_ready)}")

    print("\nTop READY leads:")
    for lead in sorted(final_ready, key=lambda x: parse_int(x.get("outreach_priority"), 0), reverse=True)[:15]:
        print(
            f" {parse_int(lead.get('outreach_priority'), 0):3d} | "
            f"{lead.get('platform', ''):9s} | {lead.get('creator', '')[:35]:35s} | "
            f"{lead.get('primary_song', '')} | {lead.get('email', '')}"
        )

    print("\n======================================")
    print("OPENAI USAGE / COST ESTIMATE")
    print("======================================")
    print(f"{MODEL}: input={usage_totals['input_tokens']:,} output={usage_totals['output_tokens']:,}")
    print("Web-search tool calls: 0")
    print(f"Estimated OpenAI cost this run: ${estimated_cost():.4f}")
    print("Nothing was sent. READY means draft prepared only.")


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    print(f"ENGINE VERSION: {ENGINE_VERSION}")
    self_check()
    print("======================================")
    print("HUXLEY SUN OUTREACH PREP")
    print("======================================")
    print("Mode: clean existing emails -> active-song rematch -> personalized drafts")
    print("OpenAI web-search calls: 0")
    print("This script NEVER sends email.")

    init_clients()
    headers = ensure_outreach_headers()
    songs = load_active_songs()

    leads = load_new_email_leads(headers)
    print(f"Eligible NEW rows with email and match >= {MIN_MATCH_SCORE}: {len(leads)}")
    if not leads:
        print("Nothing to prepare.")
        return

    # Count shared inboxes before classification.
    counts = Counter(normalize_email(x["email"]) for x in leads)
    for lead in leads:
        lead["email_repeat_count"] = counts[normalize_email(lead["email"])]

    # Directly inspect existing public source URLs. Run concurrently so one slow
    # website cannot make the whole preparation stage crawl.
    print("Inspecting public Email Source URLs directly (no OpenAI web-search cost)...")
    completed = 0
    with ThreadPoolExecutor(max_workers=SOURCE_CRAWL_WORKERS) as pool:
        future_map = {
            pool.submit(
                source_page_evidence,
                lead["email_source_url"],
                lead["email"],
                lead["creator"],
            ): lead
            for lead in leads
        }
        for future in as_completed(future_map):
            lead = future_map[future]
            try:
                lead["source_evidence"] = future.result()
            except Exception as exc:
                lead["source_evidence"] = {
                    "fetched": False,
                    "http_status": f"FETCH_FAILED:{type(exc).__name__}",
                    "email_seen": False,
                    "creator_seen": False,
                    "representation_words_seen": False,
                    "snippet": "",
                    "source_host": normalized_host(lead["email_source_url"]),
                }
            completed += 1
            if completed % 10 == 0 or completed == len(leads):
                print(f" Source inspection: {completed}/{len(leads)}")

    classify_all_contacts(leads)

    accepted = [x for x in leads if x.get("email_class") in {"DIRECT", "REPRESENTATIVE"}]
    print(f"Verified contacts continuing to draft stage: {len(accepted)}")

    if accepted:
        prepare_drafts(accepted, songs)
        apply_song_links_and_final_text(accepted, songs)

    draftable = [
        x for x in accepted
        if not x.get("draft_failed")
        and x.get("primary_song")
        and x.get("final_body")
        and parse_int(x.get("outreach_priority"), 0) > 0
    ]

    final_ready, duplicates = choose_one_lead_per_email(draftable)
    updates = build_sheet_updates(leads, final_ready, duplicates)
    batch_update_rows(headers, updates)

    print_summary(leads, final_ready, duplicates)


if __name__ == "__main__":
    main()
