import os
import json
import re
import time
import html as html_lib
import urllib.request
import urllib.error
from urllib.parse import urljoin, urlparse
from html.parser import HTMLParser
from datetime import datetime, timezone, timedelta

from openai import OpenAI
from google.oauth2 import service_account
from googleapiclient.discovery import build


# ============================================================
# CONFIG
# ============================================================

LEADS_TAB = "Leads"
SONGS_TAB = "Songs"
MIN_SCORE = 80
ENGINE_VERSION = "HS-EMAIL-CONTACT-FIRST-V7-20260808"

PLATFORM_TARGETS = {
    "Instagram": 30,
    "YouTube": 20,
    "TikTok": 10,
}

# COST-CONTROLLED CONTACT-FIRST DISCOVERY.
# This version targets 60 NEW rows per run: 30 Instagram / 20 YouTube / 10 TikTok.
# Every NEW discovery lead must already have a public email + source URL BEFORE scoring.
# At most 6 paid web Responses are allowed (initial + one refill per platform),
# and each response is capped at 3 built-in web-search calls via max_tool_calls.
SCORE_BATCH_SIZE = 25
MAX_WEB_RESPONSES = 6
MAX_TOOL_CALLS_PER_WEB_RESPONSE = 3
INITIAL_DISCOVERY_MULTIPLIER = 2
REFILL_DISCOVERY_MULTIPLIER = 3
MAX_DISCOVERY_RESULTS_PER_RESPONSE = 60
MAX_PAGES_PER_SITE = 4
HTTP_TIMEOUT_SECONDS = 6
MAX_EXISTING_FREE_ENRICH = 120

# Cost-conscious model split:
# - GPT-5 nano: cheap classification/ranking/song matching
# - GPT-5.6 Luna: web discovery + public email lookup
SCORE_MODEL = "gpt-5-nano"
WEB_MODEL = "gpt-5.6-luna"

# Approximate prices used only for a run-cost estimate in the log.
MODEL_PRICES = {
    "gpt-5-nano": {"input": 0.05, "output": 0.40},
    "gpt-5.6-luna": {"input": 0.20, "output": 1.20},
}
WEB_SEARCH_USD_PER_CALL = 0.01

HUXLEY_BRIEF = """
Huxley Sun is an independent music project.

Core fit:
- atmospheric, melancholic, cinematic
- introspective indie / alternative music
- night drives, roads, travel, nostalgia, memory
- understated rather than flashy
- mature audience, especially roughly 25-44
- suitable for cinematic reels, travel films, short films,
  documentaries, photography videos, road films, slow visual storytelling

We are NOT primarily looking for music reviewers or musicians.
We want visual creators who could naturally USE Huxley Sun music
inside their own content.

All active Huxley Sun songs may be considered for short films and
documentaries. A song's "Best For" field describes especially strong
scene/use matches, not the only allowed uses.
"""

YOUTUBE_QUERIES = [
    "cinematic travel film",
    "slow cinema travel",
    "cinematic street photography film",
    "urban solitude cinematic film",
    "analog travel film",
    "quiet documentary visual storytelling",
    "moody landscape filmmaker",
    "cinematic automotive road film",
    "rainy night cinematic film",
    "nostalgic short film",
]

INSTAGRAM_FOCUSES = [
    "cinematic filmmakers, travel/road creators, documentary storytellers, analog and night-city filmmakers, automotive creators, short-film directors, landscape filmmakers and atmospheric editorial creators",
]

TIKTOK_FOCUSES = [
    "cinematic travel, atmospheric filmmaking, night-city visuals, documentary, landscape, road films, analog/nostalgic imagery, fashion-film and reflective visual storytelling",
]





# ============================================================
# GOOGLE SHEET HEADERS
# ============================================================

REQUIRED_LEAD_HEADERS = [
    "Date Found",
    "Creator",
    "Platform",
    "Profile URL",
    "Followers",
    "Country",
    "Content Type",
    "Recent Content",
    "Aesthetic Match",
    "Music Match",
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
    "Date Contacted",
    "Reply",
    "Result",
    "Notes",
]

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


# ============================================================
# ENVIRONMENT / CLIENTS
# ============================================================


def require_env(name):
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


OPENAI_API_KEY = require_env("OPENAI_API_KEY")
YOUTUBE_API_KEY = require_env("YOUTUBE_API_KEY")
GOOGLE_SHEET_ID = require_env("GOOGLE_SHEET_ID")
GOOGLE_SERVICE_ACCOUNT_JSON = require_env("GOOGLE_SERVICE_ACCOUNT_JSON")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

service_account_info = json.loads(GOOGLE_SERVICE_ACCOUNT_JSON)
credentials = service_account.Credentials.from_service_account_info(
    service_account_info,
    scopes=["https://www.googleapis.com/auth/spreadsheets"],
)


def build_sheets_client():
    return build(
        "sheets",
        "v4",
        credentials=credentials,
        cache_discovery=False,
    )


sheets = build_sheets_client()
youtube = build(
    "youtube",
    "v3",
    developerKey=YOUTUBE_API_KEY,
    cache_discovery=False,
)


# ============================================================
# USAGE / COST LOGGING
# ============================================================

usage_totals = {
    "gpt-5-nano": {"input_tokens": 0, "output_tokens": 0},
    "gpt-5.6-luna": {"input_tokens": 0, "output_tokens": 0},
    "web_search_calls": 0,
    "web_responses": 0,
}


def record_openai_usage(response, model):
    usage = getattr(response, "usage", None)
    if usage and model in usage_totals:
        usage_totals[model]["input_tokens"] += int(
            getattr(usage, "input_tokens", 0) or 0
        )
        usage_totals[model]["output_tokens"] += int(
            getattr(usage, "output_tokens", 0) or 0
        )

    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", "") == "web_search_call":
            usage_totals["web_search_calls"] += 1


def estimated_openai_cost():
    total = usage_totals["web_search_calls"] * WEB_SEARCH_USD_PER_CALL
    for model, prices in MODEL_PRICES.items():
        model_usage = usage_totals.get(model, {})
        total += (
            model_usage.get("input_tokens", 0) / 1_000_000
        ) * prices["input"]
        total += (
            model_usage.get("output_tokens", 0) / 1_000_000
        ) * prices["output"]
    return total


def print_usage():
    print("\n======================================")
    print("OPENAI USAGE / COST ESTIMATE")
    print("======================================")
    for model in [SCORE_MODEL, WEB_MODEL]:
        data = usage_totals.get(model, {})
        print(
            f"{model}: input={data.get('input_tokens', 0):,} "
            f"output={data.get('output_tokens', 0):,}"
        )
    print(f"Paid web Responses used: {usage_totals['web_responses']}/{MAX_WEB_RESPONSES}")
    print(f"Web-search tool calls: {usage_totals['web_search_calls']} (hard ceiling {MAX_WEB_RESPONSES * MAX_TOOL_CALLS_PER_WEB_RESPONSE})")
    print(f"Estimated OpenAI cost this run: ${estimated_openai_cost():.4f}")


# ============================================================
# GENERIC HELPERS
# ============================================================

EMAIL_RE = re.compile(
    r"^[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}$",
    re.IGNORECASE,
)


def normalize_url(url):
    if not url:
        return ""
    value = str(url).strip()
    if value.upper() == "NO":
        return ""
    value = value.split("?")[0].rstrip("/")
    if value.startswith("http://"):
        value = "https://" + value[7:]
    return value.lower()


def display_url(url):
    if not url:
        return ""
    value = str(url).strip()
    return "" if value.upper() == "NO" else value


def normalize_name(name):
    if not name:
        return ""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def safe_int(value):
    try:
        return int(str(value).replace(",", "").strip())
    except (ValueError, TypeError):
        return 0


def chunks(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def valid_email(email):
    if not email:
        return False
    value = str(email).strip()
    if not EMAIL_RE.match(value):
        return False
    # Common junk/placeholder values the model should never pass through.
    bad = {
        "example.com",
        "email.com",
        "domain.com",
        "test.com",
    }
    domain = value.rsplit("@", 1)[-1].lower()
    return domain not in bad


def valid_http_url(url):
    if not url:
        return False
    value = str(url).strip().lower()
    return value.startswith("https://") or value.startswith("http://")


def email_ready(lead):
    return valid_email(lead.get("email")) and valid_http_url(
        lead.get("email_source_url")
    )


def candidate_key(lead):
    return (
        str(lead.get("platform", "")).lower(),
        normalize_url(lead.get("profile_url", "")),
        normalize_name(lead.get("creator", "")),
    )


EMAIL_FIND_RE = re.compile(
    r"[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}",
    re.IGNORECASE,
)
URL_FIND_RE = re.compile(r"https?://[^\s<>'\"\])}]+", re.IGNORECASE)


class LinkCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() != "a":
            return
        attrs = dict(attrs)
        href = attrs.get("href")
        text = attrs.get("title", "") or ""
        if href:
            self.links.append((href, text))


def extract_urls(text):
    if not text:
        return []
    seen = set()
    out = []
    for url in URL_FIND_RE.findall(str(text)):
        url = html_lib.unescape(url).rstrip(".,;:!?)]}")
        if url not in seen:
            seen.add(url)
            out.append(url)
    return out


def is_social_or_search_url(url):
    try:
        host = urlparse(url).netloc.lower().split(":")[0]
    except Exception:
        return True
    blocked = (
        "instagram.com", "tiktok.com", "youtube.com", "youtu.be",
        "facebook.com", "x.com", "twitter.com", "google.com", "bing.com",
    )
    return any(host == d or host.endswith("." + d) for d in blocked)


def fetch_html(url):
    if not valid_http_url(url):
        return ""
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.8",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_SECONDS) as resp:
            ctype = (resp.headers.get("Content-Type") or "").lower()
            if "text/html" not in ctype and "text/plain" not in ctype:
                return ""
            raw = resp.read(1_500_000)
            charset = resp.headers.get_content_charset() or "utf-8"
            return raw.decode(charset, errors="ignore")
    except Exception:
        return ""


def email_quality(email):
    """Higher is better; generic business inboxes are fine, junk/system inboxes are not."""
    e = email.lower().strip().strip(".,;:()[]<>\"'")
    if not valid_email(e):
        return -100
    local = e.split("@", 1)[0]
    bad_locals = {
        "noreply", "no-reply", "donotreply", "do-not-reply", "abuse",
        "privacy", "legal", "security", "webmaster", "postmaster",
    }
    if local in bad_locals:
        return -100
    preferred = ("hello", "hi", "contact", "info", "booking", "book", "work", "business", "studio", "film")
    score = 10
    if any(local.startswith(x) for x in preferred):
        score += 5
    if local in {"gmail", "outlook", "hotmail"}:
        score -= 2
    return score


def emails_from_html(page_html):
    if not page_html:
        return []
    text = html_lib.unescape(page_html)
    candidates = set(EMAIL_FIND_RE.findall(text))
    # mailto can contain URL encoding or query params; regex above catches the address.
    cleaned = []
    for email in candidates:
        e = email.strip().strip(".,;:()[]<>\"'")
        if email_quality(e) > -100:
            cleaned.append(e)
    cleaned.sort(key=email_quality, reverse=True)
    return cleaned


def contact_links_from_html(base_url, page_html):
    if not page_html:
        return []
    parser = LinkCollector()
    try:
        parser.feed(page_html)
    except Exception:
        return []
    base_host = urlparse(base_url).netloc.lower()
    keywords = ("contact", "about", "work", "booking", "book", "hello", "info", "connect")
    links = []
    seen = set()
    for href, title in parser.links:
        if href.lower().startswith("mailto:"):
            continue
        absolute = urljoin(base_url, href)
        try:
            parsed = urlparse(absolute)
        except Exception:
            continue
        if parsed.scheme not in {"http", "https"}:
            continue
        if parsed.netloc.lower() != base_host:
            continue
        hay = (parsed.path + " " + title).lower()
        if not any(k in hay for k in keywords):
            continue
        clean = absolute.split("#", 1)[0]
        if clean not in seen:
            seen.add(clean)
            links.append(clean)
    return links[:8]


def seed_websites_for_lead(lead):
    seeds = []
    for value in [lead.get("contact_url", "")]:
        if valid_http_url(value) and not is_social_or_search_url(value):
            seeds.append(value)
    for value in lead.get("website_candidates", []) or []:
        if valid_http_url(value) and not is_social_or_search_url(value):
            seeds.append(value)
    for field in ["recent_content", "content_type"]:
        for value in extract_urls(lead.get(field, "")):
            if not is_social_or_search_url(value):
                seeds.append(value)
    out, seen = [], set()
    for url in seeds:
        key = normalize_url(url)
        if key and key not in seen:
            seen.add(key)
            out.append(url)
    return out[:5]


def find_public_email_free(lead):
    """Directly crawl public websites. No OpenAI request is made here."""
    if email_ready(lead):
        return lead.get("email", ""), lead.get("email_source_url", ""), lead.get("contact_url", "")

    for seed in seed_websites_for_lead(lead):
        queue = [seed]
        visited = set()
        pages_checked = 0
        while queue and pages_checked < MAX_PAGES_PER_SITE:
            url = queue.pop(0)
            key = normalize_url(url)
            if not key or key in visited:
                continue
            visited.add(key)
            page_html = fetch_html(url)
            pages_checked += 1
            if not page_html:
                continue
            emails = emails_from_html(page_html)
            if emails:
                return emails[0], url, seed
            for link in contact_links_from_html(url, page_html):
                if normalize_url(link) not in visited:
                    queue.append(link)
    return "", "", lead.get("contact_url", "")


# ============================================================
# GOOGLE SHEETS WITH RETRIES
# ============================================================


def sheets_execute(request_builder, attempts=5):
    """Retry transient Google/SSL failures and rebuild the Sheets client."""
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
            sheets = build_sheets_client()

    raise last_error


def get_tab_values(tab_name, range_suffix="A1:ZZ"):
    result = sheets_execute(
        lambda svc: svc.spreadsheets().values().get(
            spreadsheetId=GOOGLE_SHEET_ID,
            range=f"{tab_name}!{range_suffix}",
        )
    )
    return result.get("values", [])


def column_letter(number):
    result = ""
    while number:
        number, remainder = divmod(number - 1, 26)
        result = chr(65 + remainder) + result
    return result


def get_sheet_properties(tab_name):
    """Return sheetId and grid size for a tab, raising if the tab is missing."""
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
                "sheet_id": props.get("sheetId"),
                "row_count": int(grid.get("rowCount", 0) or 0),
                "column_count": int(grid.get("columnCount", 0) or 0),
            }
    raise RuntimeError(f"Google Sheet tab '{tab_name}' was not found.")


def ensure_tab_capacity(tab_name, min_columns=None, min_rows=None):
    """Expand a tab's grid before values.update/append touches cells outside it."""
    props = get_sheet_properties(tab_name)
    current_cols = props["column_count"]
    current_rows = props["row_count"]
    target_cols = max(current_cols, int(min_columns or current_cols or 1))
    target_rows = max(current_rows, int(min_rows or current_rows or 1))

    if target_cols == current_cols and target_rows == current_rows:
        return props

    fields = []
    grid_properties = {}
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
        f"Expanded '{tab_name}' grid: "
        f"columns {current_cols}->{target_cols}, rows {current_rows}->{target_rows}."
    )
    props["column_count"] = target_cols
    props["row_count"] = target_rows
    return props


def ensure_lead_headers():
    rows = get_tab_values(LEADS_TAB, "1:1")
    existing = rows[0] if rows else []

    if not existing:
        # A brand-new/empty tab may have fewer columns than the required schema.
        ensure_tab_capacity(LEADS_TAB, min_columns=len(REQUIRED_LEAD_HEADERS))
        sheets_execute(
            lambda svc: svc.spreadsheets().values().update(
                spreadsheetId=GOOGLE_SHEET_ID,
                range=f"{LEADS_TAB}!A1",
                valueInputOption="RAW",
                body={"values": [REQUIRED_LEAD_HEADERS]},
            )
        )
        return REQUIRED_LEAD_HEADERS[:]

    missing = [h for h in REQUIRED_LEAD_HEADERS if h not in existing]
    if missing:
        start_col = len(existing) + 1
        final_col_count = len(existing) + len(missing)
        # Critical: values.update does NOT automatically expand a Sheets grid.
        # Expand first, e.g. 27 columns (AA) -> 28 columns (AB).
        ensure_tab_capacity(LEADS_TAB, min_columns=final_col_count)
        sheets_execute(
            lambda svc: svc.spreadsheets().values().update(
                spreadsheetId=GOOGLE_SHEET_ID,
                range=f"{LEADS_TAB}!{column_letter(start_col)}1",
                valueInputOption="RAW",
                body={"values": [missing]},
            )
        )
        existing = existing + missing
        print("Added Leads headers: " + ", ".join(missing))

    # Verify the schema now, before any paid discovery can begin.
    check = get_tab_values(LEADS_TAB, "1:1")
    actual = check[0] if check else []
    still_missing = [h for h in REQUIRED_LEAD_HEADERS if h not in actual]
    if still_missing:
        raise RuntimeError(
            "Leads header verification failed after update. Missing: "
            + ", ".join(still_missing)
        )
    return actual


def verify_sheet_before_spending():
    # Cheap early check so we do not pay for discovery and then discover the
    # spreadsheet cannot be read/written.
    ensure_lead_headers()
    get_tab_values(SONGS_TAB, "1:2")
    print("Google Sheets connection check: OK")



def batch_update_sheet_cells(data):
    if not data:
        return
    sheets_execute(
        lambda svc: svc.spreadsheets().values().batchUpdate(
            spreadsheetId=GOOGLE_SHEET_ID,
            body={
                "valueInputOption": "RAW",
                "data": data,
            },
        )
    )


def enrich_existing_pipeline_free():
    """Try to convert already-paid-for leads into email-ready inventory for $0 OpenAI."""
    headers = ensure_lead_headers()
    rows = get_tab_values(LEADS_TAB)
    counts = {p: 0 for p in PLATFORM_TARGETS}
    if len(rows) < 2:
        return counts

    h = {name: i for i, name in enumerate(headers)}

    def val(row, name):
        idx = h.get(name)
        return str(row[idx]).strip() if idx is not None and idx < len(row) else ""

    # First count already-ready NEW/uncontacted leads.
    candidates = []
    for rownum, row in enumerate(rows[1:], start=2):
        platform = val(row, "Platform")
        if platform not in PLATFORM_TARGETS:
            continue
        if safe_int(val(row, "Match Score")) < MIN_SCORE:
            continue
        status = val(row, "Status").upper()
        if status not in {"", "NEW"}:
            continue
        email = val(row, "Email")
        source = val(row, "Email Source URL")
        if valid_email(email) and valid_http_url(source):
            counts[platform] += 1
            continue

        contact_url = val(row, "Contact URL")
        if not valid_http_url(contact_url) or is_social_or_search_url(contact_url):
            continue

        candidates.append((rownum, {
            "creator": val(row, "Creator"),
            "platform": platform,
            "profile_url": val(row, "Profile URL"),
            "contact_url": contact_url,
            "email": email,
            "email_source_url": source,
            "content_type": val(row, "Content Type"),
            "recent_content": val(row, "Recent Content"),
        }))

    if candidates:
        print(
            f"Trying free website email enrichment on up to "
            f"{min(len(candidates), MAX_EXISTING_FREE_ENRICH)} existing leads..."
        )

    updates = []
    for idx, (rownum, lead) in enumerate(candidates[:MAX_EXISTING_FREE_ENRICH], 1):
        if idx == 1 or idx % 10 == 0:
            print(f"Existing-lead free crawl: {idx}/{min(len(candidates), MAX_EXISTING_FREE_ENRICH)}")
        email, source, contact = find_public_email_free(lead)
        if not (valid_email(email) and valid_http_url(source)):
            continue

        platform = lead["platform"]
        counts[platform] += 1
        email_col = column_letter(h["Email"] + 1)
        source_col = column_letter(h["Email Source URL"] + 1)
        contact_col = column_letter(h["Contact URL"] + 1)
        updates.extend([
            {"range": f"{LEADS_TAB}!{email_col}{rownum}", "values": [[email]]},
            {"range": f"{LEADS_TAB}!{source_col}{rownum}", "values": [[source]]},
            {"range": f"{LEADS_TAB}!{contact_col}{rownum}", "values": [[contact or lead.get('contact_url', '')]]},
        ])

    if updates:
        batch_update_sheet_cells(updates)
        print(f"Updated {len(updates)//3} existing leads with free-crawled email/source.")

    print(
        "Existing email-ready NEW inventory: "
        + " | ".join(f"{p} {counts[p]}" for p in ["Instagram", "YouTube", "TikTok"])
    )
    return counts


# ============================================================
# SONG CATALOGUE
# ============================================================


def load_songs():
    rows = get_tab_values(SONGS_TAB)
    if not rows:
        raise RuntimeError(f"The '{SONGS_TAB}' tab is empty.")

    headers = [str(h).strip() for h in rows[0]]
    missing = [h for h in REQUIRED_SONG_HEADERS if h not in headers]
    if missing:
        raise RuntimeError(
            "Songs sheet is missing these exact headers: " + ", ".join(missing)
        )

    index = {header: headers.index(header) for header in headers}

    def cell(row, header):
        pos = index[header]
        return str(row[pos]).strip() if pos < len(row) else ""

    songs = []
    for row in rows[1:]:
        song_name = cell(row, "Song")
        if not song_name:
            continue

        active = cell(row, "Active").upper()
        if active not in {"YES", "Y", "TRUE", "1"}:
            continue

        playlist_url = display_url(cell(row, "Playlist URL"))
        album_url = display_url(cell(row, "Album URL"))
        more_music_url = playlist_url or album_url
        more_music_type = (
            "Playlist" if playlist_url else "Album" if album_url else ""
        )

        songs.append({
            "song": song_name,
            "stream_url": display_url(cell(row, "Stream URL")),
            "playlist_url": playlist_url,
            "album_url": album_url,
            "more_music_url": more_music_url,
            "more_music_type": more_music_type,
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
        raise RuntimeError("No active songs found. Set Active to YES for at least one song.")

    print("Active songs loaded: " + ", ".join(s["song"] for s in songs))
    return songs


def songs_for_prompt(songs):
    return [{
        "song": s["song"],
        "vocal_type": s["vocal_type"],
        "main_instrument": s["main_instrument"],
        "mood_tags": s["mood_tags"],
        "visual_tags": s["visual_tags"],
        "energy": s["energy"],
        "meaning": s["meaning"],
        "best_for": s["best_for"],
        "description": s["description"],
    } for s in songs]


# ============================================================
# EXISTING LEADS / DEDUPLICATION
# ============================================================


def get_existing_leads():
    rows = get_tab_values(LEADS_TAB)
    if not rows:
        return set(), set()

    headers = rows[0]
    hmap = {name: i for i, name in enumerate(headers)}
    creator_col = hmap.get("Creator")
    url_col = hmap.get("Profile URL")

    urls, names = set(), set()
    for row in rows[1:]:
        if creator_col is not None and creator_col < len(row):
            name = normalize_name(row[creator_col])
            if name:
                names.add(name)
        if url_col is not None and url_col < len(row):
            url = normalize_url(row[url_col])
            if url:
                urls.add(url)
    return urls, names


def dedupe_candidates(candidates, existing_urls, existing_names):
    output = []
    seen_urls = set()
    seen_names = set()

    for lead in candidates:
        platform = str(lead.get("platform", "")).strip()
        url = normalize_url(lead.get("profile_url"))
        name = normalize_name(lead.get("creator"))
        if platform not in PLATFORM_TARGETS or not url or not name:
            continue
        if url in existing_urls or name in existing_names:
            continue
        if url in seen_urls or (platform.lower(), name) in seen_names:
            continue

        copy = dict(lead)
        copy["candidate_id"] = f"C{len(output) + 1:04d}"
        output.append(copy)
        seen_urls.add(url)
        seen_names.add((platform.lower(), name))

    return output


# ============================================================
# YOUTUBE DISCOVERY (NO OPENAI WEB COST)
# ============================================================


def search_youtube():
    print("Searching YouTube through YouTube Data API...")
    channels_found = {}
    published_after = (
        datetime.now(timezone.utc) - timedelta(days=540)
    ).isoformat().replace("+00:00", "Z")

    for query in YOUTUBE_QUERIES:
        try:
            response = youtube.search().list(
                part="snippet",
                q=query,
                type="video",
                maxResults=YOUTUBE_RESULTS_PER_QUERY,
                order="date",
                safeSearch="moderate",
                publishedAfter=published_after,
            ).execute()
        except Exception as exc:
            print(f"YouTube query failed for '{query}': {exc}")
            continue

        for item in response.get("items", []):
            snippet = item.get("snippet", {})
            channel_id = snippet.get("channelId")
            if not channel_id:
                continue

            record = channels_found.setdefault(channel_id, {
                "channel_id": channel_id,
                "creator": snippet.get("channelTitle", ""),
                "recent_titles": [],
                "queries": set(),
            })
            title = snippet.get("title", "")
            if title and title not in record["recent_titles"] and len(record["recent_titles"]) < 3:
                record["recent_titles"].append(title)
            record["queries"].add(query)

    ids = list(channels_found)
    for batch in chunks(ids, 50):
        response = youtube.channels().list(
            part="snippet,statistics",
            id=",".join(batch),
        ).execute()
        for item in response.get("items", []):
            cid = item["id"]
            if cid not in channels_found:
                continue
            snippet = item.get("snippet", {})
            stats = item.get("statistics", {})
            description = snippet.get("description", "") or ""
            external_urls = [u for u in extract_urls(description) if not is_social_or_search_url(u)]
            channels_found[cid].update({
                "platform": "YouTube",
                "profile_url": f"https://www.youtube.com/channel/{cid}",
                "followers": stats.get("subscriberCount", ""),
                "country": snippet.get("country", ""),
                "content_type": "; ".join(sorted(channels_found[cid]["queries"])),
                "recent_content": " | ".join(channels_found[cid]["recent_titles"]),
                "contact_url": external_urls[0] if external_urls else "",
                "website_candidates": external_urls[:5],
                "email": "",
                "email_source_url": "",
            })

    results = []
    for lead in channels_found.values():
        for key in ["channel_id", "recent_titles", "queries"]:
            lead.pop(key, None)
        results.append(lead)

    print(f"YouTube raw candidates found: {len(results)}")
    return results


# ============================================================
# CONTACT-FIRST WEB DISCOVERY (ALL THREE PLATFORMS)
# ============================================================

CONTACT_SCHEMA = {
    "type": "object",
    "properties": {
        "leads": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "creator": {"type": "string"},
                    "platform": {"type": "string", "enum": ["Instagram", "YouTube", "TikTok"]},
                    "profile_url": {"type": "string"},
                    "followers": {"type": "string"},
                    "country": {"type": "string"},
                    "content_type": {"type": "string"},
                    "recent_content": {"type": "string"},
                    "contact_url": {"type": "string"},
                    "email": {"type": "string"},
                    "email_source_url": {"type": "string"},
                },
                "required": [
                    "creator", "platform", "profile_url", "followers", "country",
                    "content_type", "recent_content", "contact_url", "email",
                    "email_source_url",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["leads"],
    "additionalProperties": False,
}


def profile_matches_platform(url, platform):
    if not valid_http_url(url):
        return False
    try:
        host = urlparse(str(url).strip()).netloc.lower().split(":")[0]
    except Exception:
        return False
    allowed = {
        "Instagram": ("instagram.com",),
        "YouTube": ("youtube.com", "youtu.be"),
        "TikTok": ("tiktok.com",),
    }
    return any(host == d or host.endswith("." + d) for d in allowed.get(platform, ()))


def verified_contact_lead(lead):
    """Hard requirement for NEW discovery leads."""
    return (
        lead.get("platform") in PLATFORM_TARGETS
        and profile_matches_platform(lead.get("profile_url", ""), lead.get("platform", ""))
        and valid_email(lead.get("email"))
        and valid_http_url(lead.get("email_source_url"))
    )


def platform_focus(platform):
    common = (
        "cinematic travel films, road films, short films, documentary and human-interest work, "
        "analog/nostalgic visual storytelling, night-city or street films, landscape films, "
        "cinematic automotive work, restrained fashion/editorial films, slow visual storytelling"
    )
    if platform == "YouTube":
        return common + "; prioritize independent channels/filmmakers who publish a business email on an official site or public contact page"
    if platform == "TikTok":
        return common + "; prioritize creators who also have an official portfolio/site with a public business email"
    return common + "; prioritize creators with an official portfolio/site and public business email"


def contact_first_discovery(platform, requested_count, exclude_names=None, pass_label="initial"):
    if requested_count <= 0:
        return []
    if usage_totals["web_responses"] >= MAX_WEB_RESPONSES:
        print(f"Skipping {platform} {pass_label} discovery: paid web-response ceiling reached.")
        return []

    count = max(12, min(MAX_DISCOVERY_RESULTS_PER_RESPONSE, int(requested_count)))
    exclusions = sorted(list(exclude_names or []))[:80]
    exclusion_text = ", ".join(exclusions) if exclusions else "none"

    print(
        f"Contact-first {platform} discovery ({pass_label}): "
        f"requesting up to {count} verified-email candidates..."
    )

    prompt = f"""
{HUXLEY_BRIEF}

Find up to {count} DISTINCT, real, currently active {platform} creators who are realistic
music-placement outreach prospects.

Creative focus:
{platform_focus(platform)}

CRITICAL CONTACT-FIRST RULE:
- ONLY return a creator if you can verify a PUBLIC PROFESSIONAL/BUSINESS EMAIL ADDRESS.
- Every returned lead MUST have a valid email AND email_source_url.
- email_source_url must be the exact public page where that email is shown (official website,
  official contact/about page, portfolio, management/representation page, or the creator's
  own public profile page if the email is actually visible there).
- Never guess an email from a domain or naming pattern.
- Never invent a source URL.
- If you cannot verify the email, OMIT the creator entirely. Do not return blank-email rows.
- profile_url must be the direct real {platform} profile/channel URL.
- contact_url should be the creator's official website/contact/portfolio page when available.
- Prefer reachable independent and small-to-mid-size creators, but include larger creators
  when the fit and public contact route are unusually strong.
- Exclude musicians-only accounts, music reviewers, repost farms, fan pages, generic companies,
  and creators whose work is not genuinely useful for music placement.
- Keep recent_content and content_type concise; no biographies.
- Avoid these already-known creator normalized names where possible: {exclusion_text}

Use web search efficiently. This response is capped programmatically at
{MAX_TOOL_CALLS_PER_WEB_RESPONSE} total web-search calls, so search broadly and return only verified contacts.
"""

    usage_totals["web_responses"] += 1
    try:
        response = openai_client.responses.create(
            model=WEB_MODEL,
            reasoning={"effort": "none"},
            tools=[{"type": "web_search", "search_context_size": "low"}],
            tool_choice="required",
            max_tool_calls=MAX_TOOL_CALLS_PER_WEB_RESPONSE,
            max_output_tokens=12000,
            input=prompt,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "contact_first_creator_discovery",
                    "schema": CONTACT_SCHEMA,
                    "strict": True,
                }
            },
            store=False,
        )
        record_openai_usage(response, WEB_MODEL)
        data = json.loads(response.output_text)
    except Exception as exc:
        print(f"{platform} {pass_label} web discovery failed safely: {type(exc).__name__}: {exc}")
        return []

    good = []
    rejected = 0
    for raw in data.get("leads", []):
        lead = {k: (str(v).strip() if v is not None else "") for k, v in raw.items()}
        if lead.get("platform") != platform or not verified_contact_lead(lead):
            rejected += 1
            continue
        good.append(lead)

    print(
        f"{platform} {pass_label}: {len(good)} verified-email candidates accepted"
        + (f"; {rejected} rejected by hard contact validation." if rejected else ".")
    )
    return good


# ============================================================
# CHEAP MATCH + SONG SCORING
# ============================================================


def build_score_schema(song_names):
    return {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "candidate_id": {"type": "string"},
                        "aesthetic_match": {"type": "integer", "minimum": 0, "maximum": 30},
                        "music_match": {"type": "integer", "minimum": 0, "maximum": 25},
                        "audience_fit": {"type": "integer", "minimum": 0, "maximum": 15},
                        "creator_usefulness": {"type": "integer", "minimum": 0, "maximum": 20},
                        "activity_relevance": {"type": "integer", "minimum": 0, "maximum": 10},
                        "match_score": {"type": "integer", "minimum": 0, "maximum": 100},
                        "primary_song": {"type": "string", "enum": song_names},
                        "primary_song_match": {"type": "integer", "minimum": 0, "maximum": 100},
                        "alternative_song": {"type": "string", "enum": song_names},
                        "alternative_song_match": {"type": "integer", "minimum": 0, "maximum": 100},
                        "reason": {"type": "string"},
                    },
                    "required": [
                        "candidate_id", "aesthetic_match", "music_match", "audience_fit",
                        "creator_usefulness", "activity_relevance", "match_score",
                        "primary_song", "primary_song_match", "alternative_song",
                        "alternative_song_match", "reason",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["results"],
        "additionalProperties": False,
    }


def score_batch(batch, songs):
    song_names = [s["song"] for s in songs]
    compact = [{
        "candidate_id": lead["candidate_id"],
        "creator": lead.get("creator", ""),
        "platform": lead.get("platform", ""),
        "followers": lead.get("followers", ""),
        "country": lead.get("country", ""),
        "content_type": lead.get("content_type", ""),
        "recent_content": lead.get("recent_content", ""),
    } for lead in batch]

    prompt = f"""
{HUXLEY_BRIEF}

Strictly score every candidate. Do NOT use contactability in the score; email
verification is a separate hard filter after scoring.

Score components:
- Aesthetic fit: 0-30
- Music-use opportunity: 0-25
- Audience fit: 0-15
- Creator usefulness / realistic collaboration value: 0-20
- Activity/relevance: 0-10
match_score MUST equal the five-component sum.

80+ means a genuinely strong outreach target. Do not inflate to hit quotas.

For song matching compare every creator with the ACTIVE catalogue using mood,
visual tags, energy, vocal/instrumental character, instrument, meaning, Best For
and description. Pick one primary and one DIFFERENT alternative song.

ACTIVE SONGS:
{json.dumps(songs_for_prompt(songs), ensure_ascii=False)}

CANDIDATES:
{json.dumps(compact, ensure_ascii=False)}
"""

    response = openai_client.responses.create(
        model=SCORE_MODEL,
        input=prompt,
        max_output_tokens=9000,
        text={
            "format": {
                "type": "json_schema",
                "name": "creator_scores",
                "schema": build_score_schema(song_names),
                "strict": True,
            }
        },
        store=False,
    )
    record_openai_usage(response, SCORE_MODEL)
    return json.loads(response.output_text).get("results", [])


def score_candidates(candidates, songs):
    if not candidates:
        return []

    print(f"Cheap AI scoring {len(candidates)} candidates...")
    all_scores = []
    for i, batch in enumerate(chunks(candidates, SCORE_BATCH_SIZE), 1):
        print(f"Scoring batch {i} ({len(batch)} candidates)...")
        all_scores.extend(score_batch(batch, songs))

    score_by_id = {x["candidate_id"]: x for x in all_scores}
    song_by_name = {s["song"]: s for s in songs}
    results = []

    for candidate in candidates:
        score = score_by_id.get(candidate["candidate_id"])
        if not score:
            continue

        total = (
            safe_int(score.get("aesthetic_match"))
            + safe_int(score.get("music_match"))
            + safe_int(score.get("audience_fit"))
            + safe_int(score.get("creator_usefulness"))
            + safe_int(score.get("activity_relevance"))
        )
        total = max(0, min(100, total))

        primary_name = score.get("primary_song")
        alternative_name = score.get("alternative_song")
        if primary_name not in song_by_name:
            continue
        if alternative_name not in song_by_name or alternative_name == primary_name:
            alternative_name = next(
                (s["song"] for s in songs if s["song"] != primary_name),
                primary_name,
            )

        primary = song_by_name[primary_name]
        alternative = song_by_name[alternative_name]

        lead = dict(candidate)
        lead.update({
            "aesthetic_match": max(0, min(30, safe_int(score.get("aesthetic_match")))),
            "music_match": max(0, min(25, safe_int(score.get("music_match")))),
            "match_score": total,
            "suggested_song": primary_name,
            "song_match_score": max(0, min(100, safe_int(score.get("primary_song_match")))),
            "song_link": primary["stream_url"],
            "more_music_link": primary["more_music_url"],
            "more_music_type": primary["more_music_type"],
            "alternative_song": alternative_name,
            "alternative_song_match": max(0, min(100, safe_int(score.get("alternative_song_match")))),
            "alternative_song_link": alternative["stream_url"],
            "reason": str(score.get("reason", "")).strip(),
        })
        results.append(lead)

    results.sort(
        key=lambda x: (x.get("match_score", 0), x.get("song_match_score", 0)),
        reverse=True,
    )
    return results


# ============================================================
# FINAL PLATFORM SELECTION
# ============================================================


def select_platform_targets(scored):
    final = []
    counts = {p: 0 for p in PLATFORM_TARGETS}
    for platform in ["Instagram", "YouTube", "TikTok"]:
        pool = [
            x for x in scored
            if x.get("platform") == platform
            and x.get("match_score", 0) >= MIN_SCORE
            and verified_contact_lead(x)
        ]
        pool.sort(
            key=lambda x: (x.get("match_score", 0), x.get("song_match_score", 0)),
            reverse=True,
        )
        chosen = pool[:PLATFORM_TARGETS[platform]]
        counts[platform] = len(chosen)
        final.extend(chosen)
    return final, counts


def shortages_from_counts(counts):
    return {
        p: max(0, PLATFORM_TARGETS[p] - int(counts.get(p, 0)))
        for p in PLATFORM_TARGETS
    }


# ============================================================
# WRITE TO GOOGLE SHEETS - FINAL FAIL-SAFE
# ============================================================


def append_to_sheet(leads):
    headers = ensure_lead_headers()

    # FINAL HARD GATE: every newly written row must have score >=80, a real email,
    # a source URL, and a valid platform profile.
    clean = [
        lead for lead in leads
        if lead.get("match_score", 0) >= MIN_SCORE and verified_contact_lead(lead)
    ]
    rejected = len(leads) - len(clean)
    if rejected:
        print(
            f"FINAL WRITE GUARD rejected {rejected} lead(s) missing score/email/source/platform validation."
        )

    if not clean:
        print("No email-ready qualified leads to write.")
        return 0

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    records = []
    for lead in clean:
        records.append({
            "Date Found": today,
            "Creator": lead.get("creator", ""),
            "Platform": lead.get("platform", ""),
            "Profile URL": lead.get("profile_url", ""),
            "Followers": lead.get("followers", ""),
            "Country": lead.get("country", ""),
            "Content Type": lead.get("content_type", ""),
            "Recent Content": lead.get("recent_content", ""),
            "Aesthetic Match": lead.get("aesthetic_match", ""),
            "Music Match": lead.get("music_match", ""),
            "Suggested Song": lead.get("suggested_song", ""),
            "Song Match Score": lead.get("song_match_score", ""),
            "Song Link": lead.get("song_link", ""),
            "More Music Link": lead.get("more_music_link", ""),
            "More Music Type": lead.get("more_music_type", ""),
            "Alternative Song": lead.get("alternative_song", ""),
            "Alternative Song Match": lead.get("alternative_song_match", ""),
            "Alternative Song Link": lead.get("alternative_song_link", ""),
            "Match Score": lead.get("match_score", ""),
            "Reason": lead.get("reason", ""),
            "Contact URL": lead.get("contact_url", ""),
            "Email": lead.get("email", ""),
            "Email Source URL": lead.get("email_source_url", ""),
            "Status": "NEW",
            "Date Contacted": "",
            "Reply": "",
            "Result": "",
            "Notes": "",
        })

    rows = [[record.get(header, "") for header in headers] for record in records]
    sheets_execute(
        lambda svc: svc.spreadsheets().values().append(
            spreadsheetId=GOOGLE_SHEET_ID,
            range=f"{LEADS_TAB}!A:{column_letter(len(headers))}",
            valueInputOption="RAW",
            insertDataOption="INSERT_ROWS",
            body={"values": rows},
        )
    )
    print(f"Added {len(rows)} verified-email leads to Google Sheets.")
    return len(rows)


# ============================================================
# MAIN
# ============================================================


def startup_self_check():
    expected_targets = {"Instagram": 30, "YouTube": 20, "TikTok": 10}
    if PLATFORM_TARGETS != expected_targets:
        raise RuntimeError(f"Bad platform targets in code: {PLATFORM_TARGETS}")
    if MIN_SCORE != 80:
        raise RuntimeError(f"Bad MIN_SCORE in code: {MIN_SCORE}")
    if "Email Source URL" not in REQUIRED_LEAD_HEADERS:
        raise RuntimeError("Email Source URL header missing from code.")
    if len(REQUIRED_LEAD_HEADERS) != 28:
        raise RuntimeError(
            f"Unexpected Leads schema width: {len(REQUIRED_LEAD_HEADERS)} columns; expected 28."
        )
    if MAX_WEB_RESPONSES != 6 or MAX_TOOL_CALLS_PER_WEB_RESPONSE != 3:
        raise RuntimeError("Cost-ceiling constants changed unexpectedly.")
    print(f"ENGINE VERSION: {ENGINE_VERSION}")
    print("Startup self-check: PASS")


def main():
    startup_self_check()
    print("======================================")
    print("HUXLEY SUN LEAD ENGINE - CONTACT-FIRST EMAIL MODE")
    print("======================================")
    print("Per-run NEW lead targets: Instagram 30 | YouTube 20 | TikTok 10")
    print("Hard requirements: match score >= 80 + public email + exact email source URL")
    print(
        f"Hard paid-search ceiling: {MAX_WEB_RESPONSES} web Responses x "
        f"{MAX_TOOL_CALLS_PER_WEB_RESPONSE} tool calls = "
        f"{MAX_WEB_RESPONSES * MAX_TOOL_CALLS_PER_WEB_RESPONSE} max web-search calls."
    )
    print("Existing rows are used ONLY for deduplication; they do NOT satisfy this run's 60-new-lead target.")

    # Fail before any paid OpenAI request if Sheets or Songs are broken.
    verify_sheet_before_spending()
    songs = load_songs()
    print("Active songs loaded: " + ", ".join(s["song"] for s in songs))

    existing_urls, existing_names = get_existing_leads()
    print(f"Existing lead URLs in sheet (dedupe only): {len(existing_urls)}")

    # PASS 1: contact-first discovery. No email-less candidate is sent to scoring.
    discovered = []
    for platform in ["Instagram", "YouTube", "TikTok"]:
        request_count = min(
            MAX_DISCOVERY_RESULTS_PER_RESPONSE,
            PLATFORM_TARGETS[platform] * INITIAL_DISCOVERY_MULTIPLIER,
        )
        discovered.extend(
            contact_first_discovery(
                platform,
                request_count,
                exclude_names=existing_names,
                pass_label="initial",
            )
        )

    candidates = dedupe_candidates(discovered, existing_urls, existing_names)
    candidates = [x for x in candidates if verified_contact_lead(x)]
    print(f"Verified-email NEW candidates before scoring: {len(candidates)}")

    scored = score_candidates(candidates, songs)
    final, counts = select_platform_targets(scored)
    shortages = shortages_from_counts(counts)

    print(
        "After initial scoring: "
        + " | ".join(
            f"{p} {counts[p]}/{PLATFORM_TARGETS[p]}"
            for p in ["Instagram", "YouTube", "TikTok"]
        )
    )

    # PASS 2: one bounded refill per short platform, still contact-first.
    if any(shortages.values()) and usage_totals["web_responses"] < MAX_WEB_RESPONSES:
        seen_urls = existing_urls | {normalize_url(x.get("profile_url", "")) for x in candidates}
        seen_names = existing_names | {normalize_name(x.get("creator", "")) for x in candidates}
        refill_raw = []
        for platform in ["Instagram", "YouTube", "TikTok"]:
            need = shortages[platform]
            if need <= 0:
                continue
            request_count = min(
                MAX_DISCOVERY_RESULTS_PER_RESPONSE,
                max(12, need * REFILL_DISCOVERY_MULTIPLIER),
            )
            refill_raw.extend(
                contact_first_discovery(
                    platform,
                    request_count,
                    exclude_names=seen_names,
                    pass_label="refill",
                )
            )

        refill = dedupe_candidates(refill_raw, seen_urls, seen_names)
        refill = [x for x in refill if verified_contact_lead(x)]
        print(f"Verified-email refill candidates before scoring: {len(refill)}")
        refill_scored = score_candidates(refill, songs)
        scored.extend(refill_scored)
        final, counts = select_platform_targets(scored)
        shortages = shortages_from_counts(counts)

    # Last, independent safety assertion before writing.
    invalid = [x for x in final if not verified_contact_lead(x) or x.get("match_score", 0) < MIN_SCORE]
    if invalid:
        raise RuntimeError(
            f"Internal safety check failed: {len(invalid)} selected rows violate email/source/score rules."
        )

    print("\n======================================")
    print("FINAL NEW EMAIL-READY RESULTS")
    print("======================================")
    for platform in ["Instagram", "YouTube", "TikTok"]:
        print(f"{platform}: {counts[platform]}/{PLATFORM_TARGETS[platform]}")
    print(f"Total NEW rows ready to write: {len(final)}/60")

    for lead in final:
        print(
            f"{lead.get('match_score', 0):>3} | "
            f"{lead.get('platform', ''):<9} | "
            f"{lead.get('creator', '')[:32]:<32} | "
            f"{lead.get('email', '')}"
        )

    # IMPORTANT: write whatever valid results were actually found; do not require all 60.
    written = append_to_sheet(final)
    print(f"Rows actually written: {written}")
    print_usage()

    if any(shortages.values()):
        print("\nNOTE: The run wrote every verified score-80+ email lead it found.")
        print(
            "Remaining shortage: "
            + " | ".join(f"{p} {shortages[p]}" for p in ["Instagram", "YouTube", "TikTok"])
        )
        print("It did not lower quality or exceed the hard web-search ceiling to force 60.")


if __name__ == "__main__":
    main()
