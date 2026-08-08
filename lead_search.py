import os
import json
import re
from datetime import datetime, timezone, timedelta

from openai import OpenAI
from google.oauth2 import service_account
from googleapiclient.discovery import build


# ============================================================
# CONFIG
# ============================================================

LEADS_TAB = "Leads"
SONGS_TAB = "Songs"

# We TARGET 60 qualified leads, but we never lower the quality threshold
# just to fill the sheet.
MIN_QUALIFIED_LEADS = 60
MAX_NEW_LEADS = 100
MIN_SCORE = 80

# Discovery volume. We deliberately search far more raw candidates than
# the final target because most candidates should be rejected.
MAX_RAW_CANDIDATES = 360
YOUTUBE_RESULTS_PER_QUERY = 20
SCORE_BATCH_SIZE = 35

MODEL = "gpt-5.6-luna"

# Approximate standard API prices used only for logging estimated cost.
# Check current OpenAI pricing if you want exact billing calculations.
MODEL_INPUT_USD_PER_1M = 0.20
MODEL_OUTPUT_USD_PER_1M = 1.20
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
    "cinematic night drive film",
    "melancholic travel film",
    "moody road trip cinematic",
    "atmospheric short film",
    "analog travel film",
    "rainy night cinematic",
    "desert road cinematic film",
    "slow cinema travel",
    "cinematic street photography film",
    "quiet documentary visual storytelling",
    "cinematic countryside film",
    "nostalgic photography film",
    "urban solitude cinematic film",
    "pre dawn city cinematic",
    "moody landscape filmmaker",
    "cinematic automotive road film",
]


# Each task asks OpenAI web search for a different creator niche.
# The separation improves diversity and reduces duplicate/generic leads.
SOCIAL_SEARCH_TASKS = [
    ("Instagram", "cinematic travel, road trips, landscape filmmaking"),
    ("Instagram", "night city, street photography, rain, urban solitude"),
    ("Instagram", "analog photography, nostalgic filmmaking, slow cinema"),
    ("Instagram", "documentary, human-interest, observational visual storytelling"),
    ("Instagram", "fashion/editorial, nocturnal, cinematic visual reels"),
    ("Instagram", "automotive, driving, roads, atmospheric car filmmaking"),
    ("TikTok", "cinematic travel, road trips, landscape filmmaking"),
    ("TikTok", "night city, street photography, rain, urban solitude"),
    ("TikTok", "analog, nostalgic, slow cinematic visual storytelling"),
    ("TikTok", "documentary, human-interest, observational filmmaking"),
    ("TikTok", "fashion/editorial, nocturnal, cinematic visual creators"),
    ("TikTok", "automotive, driving, roads, atmospheric filmmaking"),
]

# Used only if the first scoring pass produces fewer than 60 score-80+ leads.
EXTRA_SOCIAL_SEARCH_TASKS = [
    ("Instagram", "quiet nature films, countryside, dawn, weather, open landscapes"),
    ("Instagram", "experimental art film, introspective reels, minimal visual stories"),
    ("TikTok", "quiet nature films, countryside, dawn, weather, open landscapes"),
    ("TikTok", "experimental art film, introspective visual stories, minimal cinema"),
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
# ENVIRONMENT
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


# ============================================================
# CLIENTS
# ============================================================

openai_client = OpenAI(api_key=OPENAI_API_KEY)

service_account_info = json.loads(GOOGLE_SERVICE_ACCOUNT_JSON)

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
    "input_tokens": 0,
    "output_tokens": 0,
    "web_search_calls": 0,
}


def record_openai_usage(response):
    usage = getattr(response, "usage", None)
    if usage:
        usage_totals["input_tokens"] += int(
            getattr(usage, "input_tokens", 0) or 0
        )
        usage_totals["output_tokens"] += int(
            getattr(usage, "output_tokens", 0) or 0
        )

    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", "") == "web_search_call":
            usage_totals["web_search_calls"] += 1


def estimated_openai_cost():
    input_cost = (
        usage_totals["input_tokens"] / 1_000_000
    ) * MODEL_INPUT_USD_PER_1M

    output_cost = (
        usage_totals["output_tokens"] / 1_000_000
    ) * MODEL_OUTPUT_USD_PER_1M

    search_cost = (
        usage_totals["web_search_calls"] * WEB_SEARCH_USD_PER_CALL
    )

    return input_cost + output_cost + search_cost


# ============================================================
# HELPERS
# ============================================================

def normalize_url(url):
    if not url:
        return ""

    url = str(url).strip().lower()

    if url.upper() == "NO":
        return ""

    # Remove common tracking/query parameters for deduplication.
    url = url.split("?")[0]
    url = url.rstrip("/")

    if url.startswith("http://"):
        url = "https://" + url[7:]

    return url


def display_url(url):
    """Keep the original usable URL, but convert NO to blank."""
    if not url:
        return ""

    value = str(url).strip()
    if value.upper() == "NO":
        return ""

    return value


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


# ============================================================
# GOOGLE SHEETS - GENERIC
# ============================================================

def get_tab_values(tab_name, range_suffix="A1:ZZ"):
    result = sheets.spreadsheets().values().get(
        spreadsheetId=GOOGLE_SHEET_ID,
        range=f"{tab_name}!{range_suffix}",
    ).execute()

    return result.get("values", [])


def ensure_lead_headers():
    rows = get_tab_values(LEADS_TAB, "1:1")
    existing = rows[0] if rows else []

    # If the sheet has no header row at all, create the full one.
    if not existing:
        sheets.spreadsheets().values().update(
            spreadsheetId=GOOGLE_SHEET_ID,
            range=f"{LEADS_TAB}!A1",
            valueInputOption="RAW",
            body={"values": [REQUIRED_LEAD_HEADERS]},
        ).execute()
        return REQUIRED_LEAD_HEADERS[:]

    missing = [
        header
        for header in REQUIRED_LEAD_HEADERS
        if header not in existing
    ]

    if missing:
        start_col = len(existing) + 1

        # Sheets API accepts open-ended row range for appending values.
        sheets.spreadsheets().values().update(
            spreadsheetId=GOOGLE_SHEET_ID,
            range=f"{LEADS_TAB}!{column_letter(start_col)}1",
            valueInputOption="RAW",
            body={"values": [missing]},
        ).execute()

        existing = existing + missing

    return existing


def column_letter(number):
    result = ""
    while number:
        number, remainder = divmod(number - 1, 26)
        result = chr(65 + remainder) + result
    return result


# ============================================================
# SONG CATALOGUE
# ============================================================

def load_songs():
    rows = get_tab_values(SONGS_TAB)

    if not rows:
        raise RuntimeError(
            f"The '{SONGS_TAB}' tab is empty. Add the song catalogue first."
        )

    headers = [str(h).strip() for h in rows[0]]

    missing = [
        header
        for header in REQUIRED_SONG_HEADERS
        if header not in headers
    ]

    if missing:
        raise RuntimeError(
            "Songs sheet is missing these exact headers: "
            + ", ".join(missing)
        )

    index = {header: headers.index(header) for header in headers}

    def cell(row, header):
        pos = index[header]
        return row[pos].strip() if pos < len(row) else ""

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

        # Playlist is preferred; album is the fallback.
        more_music_url = playlist_url or album_url
        more_music_type = (
            "Playlist" if playlist_url
            else "Album" if album_url
            else ""
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
        raise RuntimeError(
            "No active songs were found. Set Active to YES for at least one song."
        )

    print(f"Active songs loaded: {len(songs)}")
    return songs


def songs_for_prompt(songs):
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
# EXISTING LEAD DATA
# ============================================================

def get_existing_leads():
    rows = get_tab_values(LEADS_TAB)

    if not rows:
        return set(), set()

    headers = rows[0]
    header_map = {name: i for i, name in enumerate(headers)}

    creator_col = header_map.get("Creator")
    url_col = header_map.get("Profile URL")

    urls = set()
    names = set()

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


# ============================================================
# YOUTUBE DISCOVERY
# ============================================================

def search_youtube():
    print("Searching YouTube...")

    channels_found = {}

    published_after = (
        datetime.now(timezone.utc) - timedelta(days=365)
    ).isoformat().replace("+00:00", "Z")

    for query in YOUTUBE_QUERIES:
        response = youtube.search().list(
            part="snippet",
            q=query,
            type="video",
            maxResults=YOUTUBE_RESULTS_PER_QUERY,
            order="date",
            safeSearch="moderate",
            publishedAfter=published_after,
        ).execute()

        for item in response.get("items", []):
            snippet = item.get("snippet", {})
            channel_id = snippet.get("channelId")

            if not channel_id:
                continue

            # Keep up to three recent discovered titles as evidence.
            if channel_id not in channels_found:
                channels_found[channel_id] = {
                    "channel_id": channel_id,
                    "creator": snippet.get("channelTitle", ""),
                    "recent_titles": [],
                    "search_queries": set(),
                }

            title = snippet.get("title", "")
            if (
                title
                and title not in channels_found[channel_id]["recent_titles"]
                and len(channels_found[channel_id]["recent_titles"]) < 3
            ):
                channels_found[channel_id]["recent_titles"].append(title)

            channels_found[channel_id]["search_queries"].add(query)

    if not channels_found:
        return []

    ids = list(channels_found.keys())

    for batch in chunks(ids, 50):
        response = youtube.channels().list(
            part="snippet,statistics",
            id=",".join(batch),
        ).execute()

        for item in response.get("items", []):
            channel_id = item["id"]

            if channel_id not in channels_found:
                continue

            snippet = item.get("snippet", {})
            stats = item.get("statistics", {})

            channels_found[channel_id].update({
                "platform": "YouTube",
                "profile_url":
                    f"https://www.youtube.com/channel/{channel_id}",
                "followers":
                    stats.get("subscriberCount", ""),
                "country":
                    snippet.get("country", ""),
                "content_type":
                    "; ".join(
                        sorted(channels_found[channel_id]["search_queries"])
                    ),
                "recent_content":
                    " | ".join(
                        channels_found[channel_id]["recent_titles"]
                    ),
                "contact_url": "",
                "email": "",
            })

    results = []

    for lead in channels_found.values():
        lead.pop("search_queries", None)
        lead.pop("recent_titles", None)
        lead.pop("channel_id", None)
        results.append(lead)

    return results


# ============================================================
# INSTAGRAM + TIKTOK DISCOVERY
# ============================================================

SOCIAL_SCHEMA = {
    "type": "object",
    "properties": {
        "leads": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "creator": {"type": "string"},
                    "platform": {
                        "type": "string",
                        "enum": ["Instagram", "TikTok"],
                    },
                    "profile_url": {"type": "string"},
                    "followers": {"type": "string"},
                    "country": {"type": "string"},
                    "content_type": {"type": "string"},
                    "recent_content": {"type": "string"},
                    "contact_url": {"type": "string"},
                    "email": {"type": "string"},
                },
                "required": [
                    "creator",
                    "platform",
                    "profile_url",
                    "followers",
                    "country",
                    "content_type",
                    "recent_content",
                    "contact_url",
                    "email",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["leads"],
    "additionalProperties": False,
}


def social_search_task(platform, focus, count=18):
    print(f"Searching {platform}: {focus}")

    prompt = f"""
{HUXLEY_BRIEF}

Search the public web for CURRENT, ACTIVE {platform} creators.

FOCUS:
{focus}

Find up to {count} strong, DISTINCT candidates.

We specifically want public creator/professional accounts such as:
- filmmakers
- cinematic travel creators
- atmospheric photographers who make video/reels
- visual storytellers
- short-form directors
- small documentary creators
- art-film creators
- cinematic automotive creators when appropriate

Prefer people whose work naturally uses music underneath visuals.

Generally favor small and medium creators over celebrities. A useful
rough range is hundreds to roughly 100,000 followers, but do not reject
an excellent match solely because the exact follower count is unavailable.

IMPORTANT:
- Use web search.
- Return only REAL accounts you actually found.
- platform must be exactly {platform}.
- profile_url must be the real direct {platform} profile URL.
- Never invent a username, follower count, country, contact URL or email.
- If a value is unavailable, return an empty string.
- Only include public/professional creator activity.
- Exclude spam, repost farms, generic meme pages, fan pages,
  corporations with no identifiable creator work, and huge celebrities.
- Do not include music reviewers simply because they discuss music.
"""

    response = openai_client.responses.create(
        model=MODEL,
        tools=[{"type": "web_search"}],
        tool_choice="required",
        input=prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": "social_creator_search",
                "schema": SOCIAL_SCHEMA,
                "strict": True,
            }
        },
        store=False,
    )

    record_openai_usage(response)

    data = json.loads(response.output_text)

    leads = []

    for lead in data.get("leads", []):
        # Guard against model output crossing platforms.
        if lead.get("platform") == platform:
            leads.append(lead)

    return leads


def search_social(tasks):
    leads = []

    for platform, focus in tasks:
        leads.extend(
            social_search_task(
                platform=platform,
                focus=focus,
                count=18,
            )
        )

    return leads


# ============================================================
# MERGE / DEDUPE
# ============================================================

def combine_candidates(*lead_groups):
    combined = []

    seen_urls = set()
    seen_platform_names = set()

    for group in lead_groups:
        for lead in group:
            url = normalize_url(lead.get("profile_url", ""))
            name = normalize_name(lead.get("creator", ""))
            platform = str(lead.get("platform", "")).strip()

            key = f"{platform.lower()}:{name}"

            if not url or not name or not platform:
                continue

            if url in seen_urls:
                continue

            if key in seen_platform_names:
                continue

            seen_urls.add(url)
            seen_platform_names.add(key)

            lead = dict(lead)
            lead["candidate_id"] = f"C{len(combined) + 1:04d}"
            combined.append(lead)

            if len(combined) >= MAX_RAW_CANDIDATES:
                return combined

    return combined


def remove_existing(candidates, existing_urls, existing_names):
    new_candidates = []

    for lead in candidates:
        url = normalize_url(lead.get("profile_url", ""))
        name = normalize_name(lead.get("creator", ""))

        if url in existing_urls:
            continue

        # Name deduplication is deliberately conservative. A creator may have
        # different names on different platforms, but exact normalized matches
        # already in the sheet are skipped.
        if name in existing_names:
            continue

        new_candidates.append(lead)

    return new_candidates


# ============================================================
# AI SCORING + SONG MATCHING
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
                        "aesthetic_match": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 30,
                        },
                        "music_match": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 25,
                        },
                        "audience_fit": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 15,
                        },
                        "creator_usefulness": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 15,
                        },
                        "activity_relevance": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 10,
                        },
                        "contactability": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 5,
                        },
                        "match_score": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 100,
                        },
                        "primary_song": {
                            "type": "string",
                            "enum": song_names,
                        },
                        "primary_song_match": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 100,
                        },
                        "alternative_song": {
                            "type": "string",
                            "enum": song_names,
                        },
                        "alternative_song_match": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 100,
                        },
                        "reason": {"type": "string"},
                    },
                    "required": [
                        "candidate_id",
                        "aesthetic_match",
                        "music_match",
                        "audience_fit",
                        "creator_usefulness",
                        "activity_relevance",
                        "contactability",
                        "match_score",
                        "primary_song",
                        "primary_song_match",
                        "alternative_song",
                        "alternative_song_match",
                        "reason",
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
    score_schema = build_score_schema(song_names)

    compact_candidates = []

    for lead in batch:
        compact_candidates.append({
            "candidate_id": lead["candidate_id"],
            "creator": lead.get("creator", ""),
            "platform": lead.get("platform", ""),
            "followers": lead.get("followers", ""),
            "country": lead.get("country", ""),
            "content_type": lead.get("content_type", ""),
            "recent_content": lead.get("recent_content", ""),
            "profile_url": lead.get("profile_url", ""),
            "has_email": bool(lead.get("email")),
            "has_contact_url": bool(lead.get("contact_url")),
        })

    prompt = f"""
{HUXLEY_BRIEF}

You are the strict lead-qualification AND song-matching system for
Huxley Sun.

Score EVERY candidate below.

LEAD SCORE:
- Aesthetic fit: 0-30
- Music-use opportunity: 0-25
- Audience fit: 0-15
- Creator usefulness/realistic size: 0-15
- Recent activity/relevance: 0-10
- Contactability: 0-5

match_score MUST equal the sum of those six components.

Be strict. 80+ means a genuinely strong outreach target, not merely
"somewhat relevant." Do not inflate scores to satisfy a quota.

SONG MATCHING:
Compare each creator against the full active song catalogue below.
Use visual tags, mood, energy, vocal/instrumental character, instrument,
song meaning, best-for use cases, and sonic description.

Choose:
1. one PRIMARY song
2. one DIFFERENT ALTERNATIVE song

primary_song_match and alternative_song_match are 0-100 compatibility
scores between that creator's work and the individual song.

All songs can work for short films and documentaries. "Best For" means
especially strong uses, not the only possible uses.

ACTIVE SONG CATALOGUE:
{json.dumps(songs_for_prompt(songs), ensure_ascii=False)}

CANDIDATES:
{json.dumps(compact_candidates, ensure_ascii=False)}
"""

    response = openai_client.responses.create(
        model=MODEL,
        input=prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": "creator_scores",
                "schema": score_schema,
                "strict": True,
            }
        },
        store=False,
    )

    record_openai_usage(response)
    return json.loads(response.output_text)["results"]


def score_candidates(candidates, songs):
    if not candidates:
        return []

    print(f"AI scoring {len(candidates)} candidates...")

    all_scores = []

    for index, batch in enumerate(chunks(candidates, SCORE_BATCH_SIZE), 1):
        print(f"Scoring batch {index} ({len(batch)} candidates)...")
        all_scores.extend(score_batch(batch, songs))

    score_by_id = {
        item["candidate_id"]: item
        for item in all_scores
    }

    song_by_name = {
        s["song"]: s
        for s in songs
    }

    results = []

    for candidate in candidates:
        score = score_by_id.get(candidate["candidate_id"])
        if not score:
            continue

        components_total = (
            safe_int(score["aesthetic_match"])
            + safe_int(score["music_match"])
            + safe_int(score["audience_fit"])
            + safe_int(score["creator_usefulness"])
            + safe_int(score["activity_relevance"])
            + safe_int(score["contactability"])
        )

        # Trust the explicit component sum, not an inconsistent total.
        match_score = max(0, min(100, components_total))

        primary_name = score["primary_song"]
        alternative_name = score["alternative_song"]

        # Defensive fallback if the model somehow returns the same song twice.
        if (
            alternative_name == primary_name
            and len(songs) > 1
        ):
            alternative_name = next(
                s["song"]
                for s in songs
                if s["song"] != primary_name
            )

        primary = song_by_name[primary_name]
        alternative = song_by_name[alternative_name]

        enriched = dict(candidate)
        enriched.update({
            "aesthetic_match":
                max(0, min(30, safe_int(score["aesthetic_match"]))),
            "music_match":
                max(0, min(25, safe_int(score["music_match"]))),
            "match_score": match_score,
            "suggested_song": primary_name,
            "song_match_score":
                max(0, min(100, safe_int(score["primary_song_match"]))),
            "song_link": primary["stream_url"],
            "more_music_link": primary["more_music_url"],
            "more_music_type": primary["more_music_type"],
            "alternative_song": alternative_name,
            "alternative_song_match":
                max(0, min(100, safe_int(score["alternative_song_match"]))),
            "alternative_song_link": alternative["stream_url"],
            "reason": score["reason"],
        })

        results.append(enriched)

    results.sort(
        key=lambda x: (
            x["match_score"],
            x["song_match_score"],
        ),
        reverse=True,
    )

    return results


# ============================================================
# WRITE TO GOOGLE SHEETS
# ============================================================

def append_to_sheet(leads):
    if not leads:
        print("No new qualified leads to write.")
        return

    headers = ensure_lead_headers()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    records = []

    for lead in leads:
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
            "Alternative Song Match":
                lead.get("alternative_song_match", ""),
            "Alternative Song Link":
                lead.get("alternative_song_link", ""),
            "Match Score": lead.get("match_score", ""),
            "Reason": lead.get("reason", ""),
            "Contact URL": lead.get("contact_url", ""),
            "Email": lead.get("email", ""),
            "Status": "NEW",
            "Date Contacted": "",
            "Reply": "",
            "Result": "",
            "Notes": "",
        })

    rows = [
        [record.get(header, "") for header in headers]
        for record in records
    ]

    sheets.spreadsheets().values().append(
        spreadsheetId=GOOGLE_SHEET_ID,
        range=f"{LEADS_TAB}!A:{column_letter(len(headers))}",
        valueInputOption="RAW",
        insertDataOption="INSERT_ROWS",
        body={"values": rows},
    ).execute()

    print(f"Added {len(rows)} new score-{MIN_SCORE}+ leads to Google Sheets.")


# ============================================================
# MAIN
# ============================================================

def main():
    print("======================================")
    print("HUXLEY SUN LEAD ENGINE")
    print("======================================")
    print(
        f"Target: at least {MIN_QUALIFIED_LEADS} leads "
        f"with match score >= {MIN_SCORE} "
        f"(up to {MAX_NEW_LEADS} saved per run)"
    )

    # Automatically add any new required columns to Leads.
    ensure_lead_headers()

    songs = load_songs()

    existing_urls, existing_names = get_existing_leads()
    print(f"Existing lead URLs in sheet: {len(existing_urls)}")

    youtube_leads = search_youtube()
    print(f"YouTube raw candidates found: {len(youtube_leads)}")

    social_leads = search_social(SOCIAL_SEARCH_TASKS)
    print(
        "Instagram/TikTok raw candidates found: "
        f"{len(social_leads)}"
    )

    candidates = combine_candidates(
        youtube_leads,
        social_leads,
    )

    new_candidates = remove_existing(
        candidates,
        existing_urls,
        existing_names,
    )

    print(
        "New raw candidates after deduplication: "
        f"{len(new_candidates)}"
    )

    scored = score_candidates(new_candidates, songs)

    qualified = [
        lead
        for lead in scored
        if lead["match_score"] >= MIN_SCORE
    ]

    # If we are short of the 60-lead target, do one additional discovery pass.
    if len(qualified) < MIN_QUALIFIED_LEADS:
        shortage = MIN_QUALIFIED_LEADS - len(qualified)

        print(
            f"Only {len(qualified)} score-{MIN_SCORE}+ leads found. "
            f"Need {shortage} more. Running extra discovery pass..."
        )

        extra_social = search_social(EXTRA_SOCIAL_SEARCH_TASKS)

        # Deduplicate against everything already discovered/scored.
        all_candidates = combine_candidates(
            new_candidates,
            extra_social,
        )

        # Find only candidates not already scored.
        already_scored_ids = {
            normalize_url(x.get("profile_url", ""))
            for x in scored
        }

        extra_candidates = [
            lead
            for lead in all_candidates
            if normalize_url(lead.get("profile_url", ""))
            not in already_scored_ids
        ]

        extra_candidates = remove_existing(
            extra_candidates,
            existing_urls,
            existing_names,
        )

        if extra_candidates:
            # Reassign candidate IDs so they are unique for the extra pass.
            for i, lead in enumerate(extra_candidates, 1):
                lead["candidate_id"] = f"X{i:04d}"

            extra_scored = score_candidates(
                extra_candidates,
                songs,
            )

            qualified.extend(
                lead
                for lead in extra_scored
                if lead["match_score"] >= MIN_SCORE
            )

    # Final dedupe and ranking.
    final_by_url = {}

    for lead in qualified:
        url = normalize_url(lead.get("profile_url", ""))
        if not url:
            continue

        current = final_by_url.get(url)

        if (
            current is None
            or lead["match_score"] > current["match_score"]
        ):
            final_by_url[url] = lead

    qualified = sorted(
        final_by_url.values(),
        key=lambda x: (
            x["match_score"],
            x["song_match_score"],
        ),
        reverse=True,
    )

    # Never lower the 80 threshold to force the minimum. If more than 60
    # honest matches exist, keep them too, up to the safety cap.
    qualified = qualified[:MAX_NEW_LEADS]

    print(
        f"Final qualified leads (score >= {MIN_SCORE}): "
        f"{len(qualified)}"
    )

    if len(qualified) < MIN_QUALIFIED_LEADS:
        print(
            "NOTE: The engine did NOT pad the result with weaker leads. "
            f"It found {len(qualified)} genuine score-{MIN_SCORE}+ "
            f"matches this run."
        )

    for lead in qualified:
        print(
            f'{lead["match_score"]:3} | '
            f'{lead["platform"]:9} | '
            f'{lead["creator"][:35]:35} | '
            f'{lead["suggested_song"]} '
            f'({lead["song_match_score"]})'
        )

    append_to_sheet(qualified)

    print("--------------------------------------")
    print("OPENAI USAGE ESTIMATE")
    print("--------------------------------------")
    print(f'Input tokens: {usage_totals["input_tokens"]:,}')
    print(f'Output tokens: {usage_totals["output_tokens"]:,}')
    print(f'Web-search calls: {usage_totals["web_search_calls"]}')
    print(
        "Estimated OpenAI cost this run: "
        f"${estimated_openai_cost():.4f}"
    )
    print("--------------------------------------")
    print("Lead search complete.")


if __name__ == "__main__":
    main()
