import os
import json
import re
from datetime import datetime, timezone

from openai import OpenAI
from google.oauth2 import service_account
from googleapiclient.discovery import build


# ============================================================
# CONFIG
# ============================================================

SHEET_TAB = "Leads"
MIN_SCORE = 70
MAX_NEW_LEADS = 20

HUXLEY_BRIEF = """
Huxley Sun is an independent music project.

Core fit:
- atmospheric, melancholic, cinematic
- introspective indie / alternative music
- night drives, roads, travel, nostalgia, memory
- understated rather than flashy
- mature audience, especially roughly 25-44
- suitable for cinematic reels, travel films, short films,
  photography videos, road films, slow visual storytelling

We are NOT looking primarily for music reviewers or musicians.
We want visual creators who could naturally USE Huxley Sun music
inside their own content.

Available songs:
- Clouds
- Always
- Had You Still
"""


YOUTUBE_QUERIES = [
    "cinematic night drive",
    "melancholic travel film",
    "moody road trip cinematic",
    "atmospheric short film",
    "analog travel film",
    "rainy night cinematic",
    "desert road cinematic film",
    "slow cinema travel",
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
# HELPERS
# ============================================================

def normalize_url(url):
    if not url:
        return ""

    url = url.strip().lower()
    url = url.split("?")[0]
    url = url.rstrip("/")

    if url.startswith("http://"):
        url = "https://" + url[7:]

    return url


def normalize_name(name):
    if not name:
        return ""

    return re.sub(r"[^a-z0-9]", "", name.lower())


def safe_int(value):
    try:
        return int(value)
    except (ValueError, TypeError):
        return 0


# ============================================================
# EXISTING SHEET DATA
# ============================================================

def get_existing_leads():
    result = sheets.spreadsheets().values().get(
        spreadsheetId=GOOGLE_SHEET_ID,
        range=f"{SHEET_TAB}!B2:D",
    ).execute()

    rows = result.get("values", [])

    urls = set()
    names = set()

    for row in rows:
        if len(row) >= 1:
            names.add(normalize_name(row[0]))

        if len(row) >= 3:
            urls.add(normalize_url(row[2]))

    return urls, names


# ============================================================
# YOUTUBE DISCOVERY
# ============================================================

def search_youtube():
    print("Searching YouTube...")

    channels_found = {}

    for query in YOUTUBE_QUERIES:
        response = youtube.search().list(
            part="snippet",
            q=query,
            type="video",
            maxResults=5,
            order="date",
            safeSearch="moderate",
        ).execute()

        for item in response.get("items", []):
            snippet = item["snippet"]
            channel_id = snippet["channelId"]

            if channel_id not in channels_found:
                channels_found[channel_id] = {
                    "channel_id": channel_id,
                    "creator": snippet.get("channelTitle", ""),
                    "recent_content": snippet.get("title", ""),
                    "search_query": query,
                }

    if not channels_found:
        return []

    ids = list(channels_found.keys())

    # YouTube channels.list accepts batches of IDs.
    for start in range(0, len(ids), 50):
        batch = ids[start:start + 50]

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
                    channels_found[channel_id]["search_query"],
                "contact_url": "",
                "email": "",
            })

    return list(channels_found.values())


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


def search_instagram_tiktok():
    print("Searching Instagram and TikTok with OpenAI web search...")

    prompt = f"""
{HUXLEY_BRIEF}

Search the public web for CURRENT, ACTIVE creators on:

1. Instagram
2. TikTok

Find approximately 10 strong candidates from each platform.

We specifically want creators such as:
- cinematic travel creators
- short-film makers
- atmospheric photographers who make video/reels
- night-drive creators
- road-trip filmmakers
- analog-film creators
- visual storytellers
- moody landscape creators
- small documentary creators
- cinematic automotive creators
- slow-living / slow-cinema visual creators

Prefer creators who appear realistically contactable and who use
music underneath visual content.

Generally prefer small or medium creators rather than celebrities.

IMPORTANT:

- Use web search.
- Only return REAL accounts you actually found.
- profile_url must be the real direct Instagram or TikTok profile URL
  found during research.
- Never invent a username.
- Never invent an email address.
- If follower count is unavailable, return an empty string.
- If country is unavailable, return an empty string.
- If no public email is found, return an empty string.
- If a website/contact page exists, put it in contact_url.
- Do not include private individuals whose content is unrelated to
  professional/public creator activity.
- Do not include obvious spam, repost farms, meme pages, or huge celebrities.
"""

    response = openai_client.responses.create(
        model="gpt-5.6-luna",
        tools=[
            {
                "type": "web_search",
            }
        ],
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

    data = json.loads(response.output_text)

    return data.get("leads", [])


# ============================================================
# MERGE / DEDUPE
# ============================================================

def combine_candidates(youtube_leads, social_leads):
    combined = []

    seen_urls = set()
    seen_platform_names = set()

    for lead in youtube_leads + social_leads:
        url = normalize_url(lead.get("profile_url", ""))
        name = normalize_name(lead.get("creator", ""))
        platform = lead.get("platform", "")

        key = f"{platform}:{name}"

        if not url or not name:
            continue

        if url in seen_urls:
            continue

        if key in seen_platform_names:
            continue

        seen_urls.add(url)
        seen_platform_names.add(key)

        lead["candidate_id"] = f"C{len(combined) + 1:03d}"
        combined.append(lead)

    return combined


# ============================================================
# AI SCORING
# ============================================================

SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "candidate_id": {"type": "string"},
                    "aesthetic_match": {"type": "integer"},
                    "music_match": {"type": "integer"},
                    "suggested_song": {
                        "type": "string",
                        "enum": [
                            "Clouds",
                            "Always",
                            "Had You Still",
                        ],
                    },
                    "match_score": {"type": "integer"},
                    "reason": {"type": "string"},
                },
                "required": [
                    "candidate_id",
                    "aesthetic_match",
                    "music_match",
                    "suggested_song",
                    "match_score",
                    "reason",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["results"],
    "additionalProperties": False,
}


def score_candidates(candidates):
    if not candidates:
        return []

    print(f"AI scoring {len(candidates)} candidates...")

    compact_candidates = []

    for lead in candidates:
        compact_candidates.append({
            "candidate_id": lead["candidate_id"],
            "creator": lead.get("creator", ""),
            "platform": lead.get("platform", ""),
            "followers": lead.get("followers", ""),
            "country": lead.get("country", ""),
            "content_type": lead.get("content_type", ""),
            "recent_content": lead.get("recent_content", ""),
            "profile_url": lead.get("profile_url", ""),
        })

    prompt = f"""
{HUXLEY_BRIEF}

You are the lead qualification system for Huxley Sun.

Score every candidate below.

SCORING:

Aesthetic fit: 0-30
Does their visual world naturally fit Huxley Sun?

Music-use opportunity: 0-25
Would Huxley Sun music make sense underneath their content?

Audience fit: 0-15
Does the creator appear to attract an audience that may realistically
connect with this kind of music?

Creator usefulness/size: 0-15
Favor creators large enough to matter but small enough that collaboration
or music use is realistic.

Activity/relevance: 0-10
Does recent content indicate they are actively creating?

Contactability: 0-5
Does this appear to be a genuine public creator/business account that
could realistically be contacted?

Total match_score must be 0-100.

Be STRICT.
Do not give everybody high scores.
A mediocre generic influencer should score low.

Choose exactly one suggested song:
Clouds, Always, or Had You Still.

Candidates:

{json.dumps(compact_candidates, ensure_ascii=False)}
"""

    response = openai_client.responses.create(
        model="gpt-5.6-luna",
        input=prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": "creator_scores",
                "schema": SCORE_SCHEMA,
                "strict": True,
            }
        },
        store=False,
    )

    scores = json.loads(response.output_text)["results"]

    score_by_id = {
        item["candidate_id"]: item
        for item in scores
    }

    results = []

    for candidate in candidates:
        score = score_by_id.get(candidate["candidate_id"])

        if not score:
            continue

        candidate["aesthetic_match"] = max(
            0, min(30, safe_int(score["aesthetic_match"]))
        )
        candidate["music_match"] = max(
            0, min(25, safe_int(score["music_match"]))
        )
        candidate["match_score"] = max(
            0, min(100, safe_int(score["match_score"]))
        )
        candidate["suggested_song"] = score["suggested_song"]
        candidate["reason"] = score["reason"]

        results.append(candidate)

    results.sort(
        key=lambda x: x["match_score"],
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

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    rows = []

    for lead in leads:
        rows.append([
            today,                              # Date Found
            lead.get("creator", ""),           # Creator
            lead.get("platform", ""),          # Platform
            lead.get("profile_url", ""),       # Profile URL
            lead.get("followers", ""),         # Followers
            lead.get("country", ""),           # Country
            lead.get("content_type", ""),      # Content Type
            lead.get("recent_content", ""),    # Recent Content
            lead.get("aesthetic_match", ""),   # Aesthetic Match
            lead.get("music_match", ""),       # Music Match
            lead.get("suggested_song", ""),    # Suggested Song
            lead.get("match_score", ""),       # Match Score
            lead.get("reason", ""),            # Reason
            lead.get("contact_url", ""),       # Contact URL
            lead.get("email", ""),             # Email
            "NEW",                              # Status
            "",                                 # Date Contacted
            "",                                 # Reply
            "",                                 # Result
            "",                                 # Notes
        ])

    sheets.spreadsheets().values().append(
        spreadsheetId=GOOGLE_SHEET_ID,
        range=f"{SHEET_TAB}!A:T",
        valueInputOption="RAW",
        insertDataOption="INSERT_ROWS",
        body={"values": rows},
    ).execute()

    print(f"Added {len(rows)} new leads to Google Sheets.")


# ============================================================
# MAIN
# ============================================================

def main():
    print("======================================")
    print("HUXLEY SUN LEAD ENGINE")
    print("======================================")

    existing_urls, existing_names = get_existing_leads()

    print(f"Existing leads in sheet: {len(existing_urls)}")

    youtube_leads = search_youtube()
    print(f"YouTube candidates found: {len(youtube_leads)}")

    social_leads = search_instagram_tiktok()
    print(f"Instagram/TikTok candidates found: {len(social_leads)}")

    candidates = combine_candidates(
        youtube_leads,
        social_leads,
    )

    # Remove anything already in the Google Sheet.
    new_candidates = []

    for lead in candidates:
        url = normalize_url(lead.get("profile_url", ""))
        name = normalize_name(lead.get("creator", ""))

        if url in existing_urls:
            continue

        if name in existing_names:
            continue

        new_candidates.append(lead)

    print(f"New candidates after deduplication: {len(new_candidates)}")

    scored = score_candidates(new_candidates)

    qualified = [
        lead
        for lead in scored
        if lead["match_score"] >= MIN_SCORE
    ]

    qualified = qualified[:MAX_NEW_LEADS]

    print(f"Qualified leads (score >= {MIN_SCORE}): {len(qualified)}")

    for lead in qualified:
        print(
            f'{lead["match_score"]:3} | '
            f'{lead["platform"]:9} | '
            f'{lead["creator"]} | '
            f'{lead["suggested_song"]}'
        )

    append_to_sheet(qualified)

    print("Lead search complete.")


if __name__ == "__main__":
    main()
