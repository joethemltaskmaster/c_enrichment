"""extractor_service.py: turn search results and scraped pages into field values.

Reads creator_cache.json (search results, snippets, social profiles) and
scraped_data.json (page content) and extracts, per creator:

    emails, phones, social profile URLs, website, link-in-bio page,
    location, category, audience size

Every value is a CANDIDATE with a confidence (0-1) and the sources that
support it (URL, page type, an evidence snippet, date). Values seen in
several independent sources gain confidence. Nothing is written to the sheet
here; validator_service.py and sheets_writer.py are the next stages.

Rules only, no LLM. Both input files are optional, but at least one must exist.

Setup:
    pip install python-dotenv            (required)
    pip install phonenumbers             (optional, better phone validation)

Optional .env settings:
    CREATOR_CACHE_FILE=creator_cache.json
    SCRAPE_OUTPUT_FILE=scraped_data.json
    EXTRACT_OUTPUT_FILE=extracted_data.json
    EXTRACT_DEFAULT_REGION=NG            (region for phone numbers without +country code)

Examples:
    python extractor_service.py --show
    python extractor_service.py --only-id CT_0001 --show
    python extractor_service.py --export-csv extracted.csv
"""

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlsplit

from dotenv import load_dotenv

load_dotenv(r"C:\Users\Joseph\Desktop\Creator_enrichment_pipeline\.env")

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root

from source_ranker import canonical_key, classify, handle_similarity, name_match_level, name_tokens, parse_social

try:
    import phonenumbers  # optional
except ImportError:  # pragma: no cover
    phonenumbers = None

CACHE_FILE = os.getenv("CREATOR_CACHE_FILE") or "creator_cache.json"
SCRAPED_FILE = os.getenv("SCRAPE_OUTPUT_FILE") or "scraped_data.json"
OUTPUT_FILE = os.getenv("EXTRACT_OUTPUT_FILE") or "extracted_data.json"
DEFAULT_REGION = (os.getenv("EXTRACT_DEFAULT_REGION") or "NG").upper()

SOCIAL_PLATFORMS = ("instagram", "tiktok", "youtube", "twitter", "facebook", "linkedin", "twitch", "threads")

# How much a page type is trusted for contact details (emails, phones).
TYPE_BASE = {
    "contact_page": 0.80, "booking_page": 0.80, "press_page": 0.70, "official_site": 0.65,
    "about_page": 0.60, "directory": 0.45, "link_in_bio": 0.40, "interview": 0.35, "other": 0.30,
}
SNIPPET_FACTOR = 0.6  # a search snippet is only a fragment of a page
NAME_FACTOR = {3: 1.0, 2: 0.85, 1: 0.5, 0: 0.15}  # how sure the text is about THIS creator
CAP = 0.95


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def host_of(url: str) -> str:
    host = (urlsplit(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def clip(text: str, n: int = 160) -> str:
    return " ".join((text or "").split())[:n]


def around(text: str, needle: str, width: int = 70) -> str:
    i = (text or "").lower().find(needle.lower())
    if i < 0:
        return clip(text, 140)
    return clip(text[max(0, i - width): i + len(needle) + width], 180)


def date_of(iso: str) -> str:
    return (iso or "")[:10]


@dataclass
class Doc:
    """One piece of text about a creator: a search result or a scraped page."""
    kind: str                 # "search_result" | "scraped_page"
    url: str
    page_type: str
    text: str
    level: int                # 0-3: how sure we are it is about THIS creator
    official: bool            # domain looks like the creator's own site
    as_of: str
    emails: List[str] = field(default_factory=list)
    phones: List[str] = field(default_factory=list)
    social_urls: List[str] = field(default_factory=list)
    profile: Optional[Dict[str, Any]] = None  # set when the URL is itself a social profile


class Field:
    """Candidates for one field. The same source counts once; separate sources combine."""

    def __init__(self) -> None:
        self.items: Dict[Any, Dict[str, Any]] = {}

    def add(self, value: Any, conf: float, source: Dict[str, Any], **extra: Any) -> None:
        if value in (None, ""):
            return
        conf = max(0.0, min(conf, 0.9))
        item = self.items.setdefault(value, {"sources": {}, "extra": {}})
        key = source.get("url") or source.get("kind")
        prev = item["sources"].get(key)
        if prev is None or conf > prev[0]:
            item["sources"][key] = (conf, source)
        item["extra"].update(extra)

    def candidates(self) -> List[Dict[str, Any]]:
        out = []
        for value, item in self.items.items():
            miss = 1.0
            for conf, _ in item["sources"].values():
                miss *= 1.0 - conf
            ordered = sorted(item["sources"].values(), key=lambda cs: -cs[0])
            out.append({
                "value": value,
                "confidence": round(min(CAP, 1.0 - miss), 2),
                "sources": [dict(s, confidence=round(c, 2)) for c, s in ordered[:5]],
                **item["extra"],
            })
        out.sort(key=lambda c: -c["confidence"])
        return out


def source(doc: Doc, evidence: str) -> Dict[str, Any]:
    return {"url": doc.url, "kind": doc.kind, "page_type": doc.page_type, "evidence": clip(evidence), "as_of": doc.as_of}


def base_conf(doc: Doc) -> float:
    base = TYPE_BASE.get(doc.page_type, 0.30)
    return base * (SNIPPET_FACTOR if doc.kind == "search_result" else 1.0) * NAME_FACTOR[doc.level]


# --------------------------------------------------------------------------
# Emails
# --------------------------------------------------------------------------
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}")
OBF_AT = re.compile(r"\s*[\[(]\s*at\s*[\])]\s*", re.I)
OBF_DOT = re.compile(r"\s*[\[(]\s*dot\s*[\])]\s*", re.I)
JUNK_SUFFIX = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".css", ".js")
JUNK_DOMAINS = ("example.com", "domain.com", "yourdomain.com", "sentry.io", "wixpress.com", "godaddy.com")
JUNK_LOCAL = re.compile(r"^(no-?reply|donotreply|do-not-reply|postmaster|webmaster|abuse|privacy|dpo|unsubscribe|mailer-daemon)")
ROLE_LOCALS = {
    "info", "contact", "hello", "hi", "booking", "bookings", "book", "management", "manager", "mgmt",
    "press", "pr", "media", "collab", "collabs", "collaborations", "partnerships", "business",
    "enquiries", "inquiries", "enquiry", "inquiry", "team", "sales", "admin", "office", "talent", "agent",
}
FREE_MAIL = {"gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "icloud.com", "proton.me", "protonmail.com", "live.com", "aol.com", "yahoo.co.uk"}


def clean_email(raw: str) -> Optional[str]:
    e = raw.strip().strip(".,;:<>()[]\"'").lower()
    if not EMAIL_RE.fullmatch(e):
        return None
    local, domain = e.split("@", 1)
    if e.endswith(JUNK_SUFFIX) or re.search(r"@\d+x\.", e) or JUNK_LOCAL.match(local):
        return None
    if any(domain == d or domain.endswith("." + d) for d in JUNK_DOMAINS):
        return None
    return e


def find_emails(text: str) -> List[str]:
    text = OBF_DOT.sub(".", OBF_AT.sub("@", text or ""))
    out: List[str] = []
    for match in EMAIL_RE.findall(text):
        e = clean_email(match)
        if e and e not in out:
            out.append(e)
    return out


def domain_matches_creator(domain: str, name: str, creator_domains: Iterable[str]) -> bool:
    if domain in creator_domains:
        return True
    toks = name_tokens(name)
    core = re.sub(r"[^a-z0-9]", "", domain)
    joined, reverse = "".join(toks), "".join(reversed(toks))
    return len(joined) >= 5 and (joined in core or reverse in core)


# --------------------------------------------------------------------------
# Phones
# --------------------------------------------------------------------------
PHONE_RE = re.compile(r"(?<![\w@./:-])(\+?\d[\d\s().\-]{6,18}\d)(?![\w@-]|\.\d)")
PHONE_CONTEXT = re.compile(r"(call|phone|tel\b|mobile|whatsapp|wa\.me|text us|contact|booking|enquir|inquir|reach)", re.I)
DATE_LIKE = re.compile(r"^(?:19|20)\d{2}[-/. ]\d{1,2}[-/. ]\d{1,2}$|^\d{1,2}[-/. ]\d{1,2}[-/. ](?:19|20)\d{2}$")
NG_MOBILE = re.compile(r"^0[789][01]\d{8}$")


def normalize_phone(raw: str, region: str) -> Optional[str]:
    """Return an E.164 number ('+2348031234567') or None."""
    digits = re.sub(r"\D", "", raw)
    plus = raw.strip().startswith("+")
    if not 8 <= len(digits) <= 15 or len(set(digits)) <= 2:
        return None
    if phonenumbers:
        try:
            num = phonenumbers.parse(raw.strip(), None if plus else region)
        except Exception:
            return None
        if not phonenumbers.is_valid_number(num):
            return None
        return phonenumbers.format_number(num, phonenumbers.PhoneNumberFormat.E164)
    if plus:
        return "+" + digits
    if region == "NG":
        if digits.startswith("234") and len(digits) == 13:
            return "+" + digits
        if digits.startswith("0") and len(digits) == 11:
            return "+234" + digits[1:]
    return None


def find_phones(text: str, region: str) -> List[Tuple[str, str]]:
    """[(e164, raw)] for numbers that have a '+', a Nigerian mobile format, or a nearby 'call/phone/whatsapp' word."""
    found: List[Tuple[str, str]] = []
    for m in PHONE_RE.finditer(text or ""):
        raw = m.group(1).strip()
        if DATE_LIKE.match(raw):
            continue
        e164 = normalize_phone(raw, region)
        if not e164:
            continue
        digits = re.sub(r"\D", "", raw)
        strong = raw.startswith("+") or NG_MOBILE.match(digits) is not None
        context = PHONE_CONTEXT.search(text[max(0, m.start() - 40): m.start()]) is not None
        if (strong or context) and e164 not in [f[0] for f in found]:
            found.append((e164, raw))
    return found


# --------------------------------------------------------------------------
# Social URLs
# --------------------------------------------------------------------------
def canonical_social(platform: str, handle: str, url: str) -> str:
    if platform == "instagram":
        return f"https://www.instagram.com/{handle}"
    if platform == "tiktok":
        return f"https://www.tiktok.com/@{handle}"
    if platform == "twitter":
        return f"https://x.com/{handle}"
    if platform == "facebook":
        return f"https://www.facebook.com/{handle}"
    if platform == "twitch":
        return f"https://www.twitch.tv/{handle}"
    if platform == "threads":
        return f"https://www.threads.net/@{handle}"
    if platform == "linkedin":
        segs = [s for s in urlsplit(url).path.split("/") if s]
        return f"https://www.linkedin.com/{segs[0]}/{handle}" if segs else url
    if platform == "youtube":
        segs = [s for s in urlsplit(url).path.split("/") if s]
        if segs and segs[0].startswith("@"):
            return f"https://www.youtube.com/{segs[0]}"
        if len(segs) >= 2:
            return f"https://www.youtube.com/{segs[0]}/{segs[1]}"
    return url.split("?")[0].rstrip("/")


def social_profile(url: str, name: str) -> Optional[Dict[str, Any]]:
    """{platform, handle, value, similarity} if the URL is a creator profile, else None."""
    info = parse_social(url)
    if not info or info["kind"] != "profile":
        return None
    return {
        "platform": info["platform"],
        "handle": info["handle"],
        "value": canonical_social(info["platform"], info["handle"], url),
        "similarity": handle_similarity(info["handle"], name),
    }


# --------------------------------------------------------------------------
# Location
# --------------------------------------------------------------------------
CITY_COUNTRY = {
    "lagos": "Nigeria", "abuja": "Nigeria", "port harcourt": "Nigeria", "ibadan": "Nigeria", "kano": "Nigeria",
    "enugu": "Nigeria", "benin city": "Nigeria", "kaduna": "Nigeria", "warri": "Nigeria", "owerri": "Nigeria",
    "abeokuta": "Nigeria", "uyo": "Nigeria", "calabar": "Nigeria", "jos": "Nigeria", "asaba": "Nigeria",
    "akure": "Nigeria", "ilorin": "Nigeria", "onitsha": "Nigeria", "aba": "Nigeria",
    "accra": "Ghana", "kumasi": "Ghana", "nairobi": "Kenya", "mombasa": "Kenya",
    "johannesburg": "South Africa", "cape town": "South Africa", "durban": "South Africa", "pretoria": "South Africa",
    "london": "United Kingdom", "manchester": "United Kingdom", "birmingham": "United Kingdom",
    "new york": "United States", "los angeles": "United States", "atlanta": "United States", "houston": "United States",
    "chicago": "United States", "miami": "United States", "dallas": "United States", "washington dc": "United States",
    "toronto": "Canada", "vancouver": "Canada", "dubai": "United Arab Emirates", "abu dhabi": "United Arab Emirates",
    "paris": "France", "berlin": "Germany", "amsterdam": "Netherlands", "mumbai": "India", "delhi": "India",
    "sydney": "Australia",
}
AMBIGUOUS_CITIES = {"jos", "aba"}  # also ordinary words/names: only accepted after "based in" style phrases
COUNTRIES = {
    "nigeria": "Nigeria", "ghana": "Ghana", "kenya": "Kenya", "south africa": "South Africa",
    "united kingdom": "United Kingdom", "uk": "United Kingdom", "united states": "United States",
    "united states of america": "United States", "usa": "United States", "canada": "Canada",
    "united arab emirates": "United Arab Emirates", "uae": "United Arab Emirates", "uganda": "Uganda",
    "tanzania": "Tanzania", "rwanda": "Rwanda", "cameroon": "Cameroon", "senegal": "Senegal", "egypt": "Egypt",
    "ethiopia": "Ethiopia", "france": "France", "germany": "Germany", "netherlands": "Netherlands",
    "ireland": "Ireland", "india": "India", "australia": "Australia", "brazil": "Brazil", "jamaica": "Jamaica",
}
PLACE = r"([A-Z][\w'.\-]*(?:[ ,]+[A-Z][\w'.\-]*){0,3})"
PHRASE_STRONG = re.compile(r"\b(?:based in|based out of|located in|lives in|living in|resides in|residing in)\s+" + PLACE)
PHRASE_WEAK = re.compile(r"\b(?:hails from|born in|raised in|from)\s+" + PLACE)
HYPHEN_BASED = re.compile(r"\b([A-Z][a-z]+(?: [A-Z][a-z]+)?)-based\b")
PLACE_STOP = {"the", "a", "an", "this", "my", "our", "and", "instagram", "youtube", "tiktok", "twitter", "facebook", "linkedin"}


def normalize_place(text: str, accept_unknown: bool = False) -> Optional[str]:
    """'Lagos, Nigeria' -> 'Lagos, Nigeria'; 'Lagos' -> 'Lagos, Nigeria'; unknown -> None (or cleaned text)."""
    low = re.sub(r"\s+", " ", text.lower()).strip(" ,.")
    for city in sorted(CITY_COUNTRY, key=len, reverse=True):
        if re.search(rf"\b{re.escape(city)}\b", low):
            return f"{city.title()}, {CITY_COUNTRY[city]}"
    for key in sorted(COUNTRIES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(key)}\b", low):
            return COUNTRIES[key]
    if accept_unknown:
        words = [w for w in re.split(r"[ ,]+", text.strip(" ,.")) if w and w.lower() not in PLACE_STOP]
        if 1 <= len(words) <= 3:
            return " ".join(words)
    return None


def find_locations(text: str) -> List[Tuple[str, str, float]]:
    """[(place, matched text, strength 0-1)], strongest first."""
    found: Dict[str, Tuple[str, float]] = {}

    def keep(place: Optional[str], matched: str, strength: float) -> None:
        if place and strength > found.get(place, ("", 0.0))[1]:
            found[place] = (matched, strength)

    for m in PHRASE_STRONG.finditer(text):
        place = normalize_place(m.group(1))
        keep(place or normalize_place(m.group(1), True), m.group(0), 0.65 if place else 0.45)
    for m in PHRASE_WEAK.finditer(text):
        keep(normalize_place(m.group(1)), m.group(0), 0.50)
    for m in HYPHEN_BASED.finditer(text):
        keep(normalize_place(m.group(1)), m.group(0), 0.60)
    low = text.lower()
    for city in CITY_COUNTRY:
        if city in AMBIGUOUS_CITIES:
            continue
        m = re.search(rf"\b{re.escape(city)}\b", low)
        if m:
            keep(f"{city.title()}, {CITY_COUNTRY[city]}", text[m.start(): m.end()], 0.30)
    for key, country in COUNTRIES.items():
        if len(key) <= 3:
            continue
        m = re.search(rf"\b{re.escape(key)}\b", low)
        if m:
            keep(country, text[m.start(): m.end()], 0.25)
    return sorted(((p, mt, s) for p, (mt, s) in found.items()), key=lambda x: -x[2])


# --------------------------------------------------------------------------
# Category
# --------------------------------------------------------------------------
CATEGORY_KEYWORDS = {
    "Fitness & wellness": ["fitness", "workout", "gym", "personal trainer", "yoga", "weight loss", "bodybuilding", "wellness", "pilates", "nutrition", "nutritionist", "health"],
    "Fashion & style": ["fashion", "stylist", "outfit", "ootd", "streetwear", "modelling", "modeling", "clothing brand", "fashion designer"],
    "Beauty & makeup": ["makeup", "beauty", "skincare", "makeup artist", "hairstylist", "cosmetics", "lashes", "nails"],
    "Comedy & skits": ["comedy", "comedian", "skit", "skits", "funny", "humor", "humour", "stand-up", "standup"],
    "Food & cooking": ["food", "recipe", "recipes", "cooking", "chef", "baking", "foodie", "cuisine"],
    "Travel": ["travel", "travelling", "traveling", "wanderlust", "tourism"],
    "Lifestyle & vlogging": ["lifestyle", "vlog", "vlogger", "vlogging", "daily life"],
    "Tech": ["tech", "technology", "gadgets", "software", "developer", "coding", "programming"],
    "Gaming": ["gaming", "gamer", "esports", "streamer", "playstation", "fifa"],
    "Music": ["musician", "singer", "songwriter", "rapper", "dj", "afrobeats", "gospel music"],
    "Dance": ["dance", "dancer", "choreographer", "choreography"],
    "Education": ["educator", "tutor", "teacher", "tutorials", "study tips", "education"],
    "Business & finance": ["entrepreneur", "finance", "investing", "investment", "startup", "forex", "crypto", "real estate"],
    "Parenting & family": ["mom", "mum", "mother", "parenting", "motherhood", "dad", "family"],
    "Sports": ["footballer", "football", "athlete", "sports", "basketball", "athletics"],
    "Art & design": ["painter", "illustrator", "graphic designer", "visual artist", "crafts", "diy"],
    "Photography & film": ["photographer", "photography", "filmmaker", "videographer", "cinematographer"],
    "Faith & inspiration": ["pastor", "christian", "faith", "motivational speaker", "motivation", "inspirational"],
    "Environment & advocacy": ["environment", "sustainability", "climate", "conservation", "activist", "advocacy"],
    "Film & TV": ["actor", "actress", "nollywood", "tv host", "presenter", "media personality"],
}
CATEGORY_RE = {
    cat: re.compile(r"\b(" + "|".join(re.escape(k) for k in kws) + r")\b", re.I)
    for cat, kws in CATEGORY_KEYWORDS.items()
}
CONTEXT_WEIGHT = {
    "about_page": 0.9, "official_site": 0.9, "directory": 0.8, "interview": 0.8, "press_page": 0.8,
    "booking_page": 0.7, "contact_page": 0.6, "link_in_bio": 0.7,
}


def context_weight(doc: Doc) -> float:
    weight = CONTEXT_WEIGHT.get(doc.page_type, 0.6)
    if doc.kind == "search_result":
        weight *= 0.8
    return weight * NAME_FACTOR[doc.level]


# --------------------------------------------------------------------------
# Audience size
# --------------------------------------------------------------------------
NUM = r"(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)"
MULT = r"(k|m|b|thousand|million|billion)?"
AUD_BEFORE = re.compile(rf"(?<![\w.]){NUM}\s*{MULT}\+?\s*(?:followers?|subscribers?)\b", re.I)
AUD_AFTER = re.compile(rf"\b(?:followers?|subscribers?)\s*[:\-–]?\s*{NUM}\s*{MULT}\b", re.I)
MULTIPLIER = {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6, "b": 1e9, "billion": 1e9}
PLATFORM_WORDS = {
    "instagram": "instagram", "tiktok": "tiktok", "youtube": "youtube", "twitter": "twitter",
    "facebook": "facebook", "linkedin": "linkedin", "twitch": "twitch",
}


def to_count(number: str, mult: Optional[str]) -> Optional[int]:
    try:
        value = float(number.replace(",", "")) * MULTIPLIER.get((mult or "").lower(), 1)
    except ValueError:
        return None
    return int(value) if 1 <= value <= 5e9 else None


def find_audience(text: str) -> List[Tuple[int, str, int, int]]:
    """[(count, raw text, start, end)] for 'N followers' / 'N subscribers' style mentions."""
    out = []
    for regex in (AUD_BEFORE, AUD_AFTER):
        for m in regex.finditer(text or ""):
            count = to_count(m.group(1), m.group(2))
            if count:
                out.append((count, m.group(0), m.start(), m.end()))
    return out


def platform_near(text: str, start: int, end: int) -> str:
    window = text[max(0, start - 50): end + 50].lower()
    hits = [(window.find(w), p) for w, p in PLATFORM_WORDS.items() if w in window]
    return min(hits)[1] if hits else "unspecified"


# --------------------------------------------------------------------------
# Building documents for one creator
# --------------------------------------------------------------------------
def collect_results(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """All search results in a cache record (v1 and v2), merged by URL."""
    merged: Dict[str, Dict[str, Any]] = {}
    for key in ("selected", "candidates", "reserve", "urls", "legacy_urls"):
        for r in rec.get(key) or []:
            link = r.get("link")
            if not link:
                continue
            k = canonical_key(link)
            cur = merged.setdefault(k, {})
            longer = len(r.get("snippet") or "") > len(cur.get("snippet") or "")
            for field_name, value in r.items():
                if value in (None, ""):
                    continue
                if field_name == "snippet" and not longer:
                    continue
                cur[field_name] = value
    for p in rec.get("social_profiles") or []:  # social profiles may be missing from the top-20 candidates
        url = p.get("url")
        if not url:
            continue
        cur = merged.get(canonical_key(url))
        if cur is None:
            merged[canonical_key(url)] = {"link": url, "title": p.get("title", ""), "snippet": p.get("snippet", "")}
        elif len(p.get("snippet") or "") > len(cur.get("snippet") or ""):
            cur["snippet"] = p["snippet"]
    return list(merged.values())


def build_docs(name: str, rec: Dict[str, Any], entry: Dict[str, Any]) -> List[Doc]:
    docs: List[Doc] = []
    searched = date_of(rec.get("searched_at", ""))
    for r in collect_results(rec) if rec else []:
        link, title, snippet = r["link"], r.get("title", ""), r.get("snippet", "")
        level, official = name_match_level(name, title, snippet, link)
        docs.append(Doc(
            kind="search_result", url=link, page_type=r.get("type") or classify(link, title, name),
            text=f"{title}. {snippet}", level=level, official=official, as_of=searched,
            profile=social_profile(link, name),
        ))
    for url, page in ((entry or {}).get("pages") or {}).items():
        if page.get("status") != "ok":
            continue
        final = page.get("final_url") or url
        title = page.get("title", "")
        text = " ".join(filter(None, [
            title, page.get("og_title", ""), page.get("site_name", ""), page.get("description", ""),
            " ".join(page.get("headings") or []), page.get("text") or page.get("excerpt", ""),
        ]))
        _, official = name_match_level(name, title, "", final)
        level = 3 if page.get("mentions_name") else (2 if official else 0)
        docs.append(Doc(
            kind="scraped_page", url=final, page_type=classify(final, title, name), text=text,
            level=level, official=official, as_of=date_of(page.get("fetched_at", "")),
            emails=list(page.get("emails") or []), phones=list(page.get("phones") or []),
            social_urls=[s.get("url", "") for s in page.get("social_links") or [] if s.get("url")],
        ))
    return docs


# --------------------------------------------------------------------------
# Extraction for one creator
# --------------------------------------------------------------------------
def extract_creator(creator_id: str, rec: Dict[str, Any], entry: Dict[str, Any], region: str, min_conf: float) -> Dict[str, Any]:
    name = (rec or {}).get("name") or (entry or {}).get("name") or ""
    docs = build_docs(name, rec or {}, entry or {})
    creator_domains = {host_of(d.url) for d in docs if d.official}

    emails, phones, website, link_in_bio, location, category = Field(), Field(), Field(), Field(), Field(), Field()
    socials = {p: Field() for p in SOCIAL_PLATFORMS}
    audience: Dict[str, Field] = {}
    cat_scores: Dict[str, float] = {}
    cat_evidence: Dict[str, List[Tuple[float, Dict[str, Any]]]] = {}

    for doc in docs:
        # ---- emails ----------------------------------------------------
        seen_here = list(dict.fromkeys(doc.emails + find_emails(doc.text)))
        for e in (clean_email(x) for x in seen_here):
            if not e:
                continue
            local, domain = e.split("@", 1)
            conf = base_conf(doc)
            role = local in ROLE_LOCALS
            if role:
                conf += 0.05
            if domain_matches_creator(domain, name, creator_domains):
                conf += 0.10
                scope = "creator"
            else:
                scope = "agency" if doc.page_type == "booking_page" else "unknown"
                if domain in FREE_MAIL and doc.page_type not in ("contact_page", "booking_page", "press_page"):
                    conf -= 0.10
            emails.add(e, conf, source(doc, around(doc.text, e) if e in doc.text.lower() else "listed on page"),
                       scope=scope, kind="role" if role else "personal", free_provider=domain in FREE_MAIL)
            if scope == "creator" and domain not in FREE_MAIL:
                website.add(f"https://{domain}/", 0.30, source(doc, f"inferred from email {e}"), basis="inferred_from_email")

        # ---- phones ----------------------------------------------------
        found = [(e, r, False) for e, r in find_phones(doc.text, region)]
        for raw in doc.phones:  # tel: links, if the scraper stored them
            e164 = normalize_phone(raw, region)
            if e164:
                found.append((e164, raw, True))
        for e164, raw, from_tel_link in found:
            conf = base_conf(doc) + (0.10 if from_tel_link else 0.0)
            scope = "creator" if doc.official else ("agency" if doc.page_type == "booking_page" else "unknown")
            phones.add(e164, conf, source(doc, around(doc.text, raw)), scope=scope, raw=raw)

        # ---- social profiles ------------------------------------------
        if doc.profile:
            p = doc.profile
            conf = 0.55 * p["similarity"] + 0.45 * doc.level / 3
            if conf >= 0.30 and p["platform"] in socials:
                socials[p["platform"]].add(p["value"], min(conf, 0.80), dict(source(doc, doc.text), url=p["value"]), handle=p["handle"])
        for raw in doc.social_urls:
            p = social_profile(raw, name)
            if not p or p["platform"] not in socials:
                continue
            conf = (0.75 if (doc.official or doc.page_type in ("official_site", "about_page", "contact_page", "booking_page", "link_in_bio", "press_page")) else 0.50)
            conf = conf * NAME_FACTOR[doc.level] + 0.15 * p["similarity"]
            socials[p["platform"]].add(p["value"], conf, dict(source(doc, f"linked from {host_of(doc.url)}"), url=doc.url + " -> " + p["value"]), handle=p["handle"])

        # ---- website and link-in-bio -----------------------------------
        if doc.official and doc.level >= 2:
            conf = 0.80 if doc.page_type in ("official_site", "about_page", "contact_page", "booking_page", "press_page") else 0.65
            website.add(f"https://{host_of(doc.url)}/", conf * (0.9 if doc.kind == "search_result" else 1.0), source(doc, doc.text), basis="official domain")
        if doc.page_type == "link_in_bio":
            segs = [s for s in urlsplit(doc.url).path.split("/") if s]
            if segs and (handle_similarity(segs[0], name) >= 0.5 or doc.level == 3):
                link_in_bio.add(f"https://{host_of(doc.url)}/{segs[0]}", 0.70 * NAME_FACTOR[doc.level], source(doc, doc.text))

        # ---- location and category (only text that is about this creator) --
        if doc.level >= 1:
            weight = context_weight(doc)
            for place, matched, strength in find_locations(doc.text):
                location.add(place, strength * weight * 1.2, source(doc, around(doc.text, matched)))
            for cat, regex in CATEGORY_RE.items():
                hits = {h.lower() for h in regex.findall(doc.text)}
                if hits:
                    score = min(len(hits), 3) * weight
                    cat_scores[cat] = cat_scores.get(cat, 0.0) + score
                    cat_evidence.setdefault(cat, []).append((score, source(doc, "keywords: " + ", ".join(sorted(hits)))))

        # ---- audience size ---------------------------------------------
        if doc.level >= 1 or doc.profile:
            for count, raw, start, end in find_audience(doc.text):
                if doc.profile and doc.profile["platform"] in SOCIAL_PLATFORMS:
                    platform = doc.profile["platform"]
                    profile_conf = 0.55 * doc.profile["similarity"] + 0.45 * doc.level / 3
                    conf = 0.50 + 0.40 * profile_conf if profile_conf >= 0.30 else 0.25
                else:
                    platform = platform_near(doc.text, start, end)
                    conf = (0.45 if platform != "unspecified" else 0.35) * NAME_FACTOR[doc.level]
                src = source(doc, raw)
                if doc.profile:  # same profile found twice (result + social_profiles) counts once
                    src["url"] = doc.profile["value"]
                audience.setdefault(platform, Field()).add(count, conf, src)

    for cat, score in cat_scores.items():
        top_ev = max(cat_evidence[cat], key=lambda x: x[0])[1]
        category.add(cat, min(0.9, 0.25 + 0.12 * score), top_ev, evidence_pages=len(cat_evidence[cat]))

    cands = {
        "emails": emails.candidates(), "phones": phones.candidates(), "website": website.candidates(),
        "link_in_bio": link_in_bio.candidates(), "social": {p: f.candidates() for p, f in socials.items() if f.items},
        "location": location.candidates(), "category": category.candidates(),
        "audience": {p: f.candidates() for p, f in audience.items()},
    }
    best, review = pick_best(cands, min_conf)
    return {
        "name": name,
        "row": (rec or {}).get("row") or (entry or {}).get("row"),
        "context": {"location": (rec or {}).get("location", ""), "category": (rec or {}).get("category", "")},
        "inputs": {
            "search_results": sum(1 for d in docs if d.kind == "search_result"),
            "scraped_pages_ok": sum(1 for d in docs if d.kind == "scraped_page"),
        },
        "best": best,
        "needs_review": review,
        "candidates": cands,
        "extracted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def pick_best(cands: Dict[str, Any], min_conf: float) -> Tuple[Dict[str, Any], List[str]]:
    best: Dict[str, Any] = {}
    review: List[str] = []

    def top(items: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        return items[0] if items and items[0]["confidence"] >= min_conf else None

    def entry(label: str, item: Dict[str, Any], items: List[Dict[str, Any]], extra: Optional[Dict[str, Any]] = None) -> None:
        out = {"value": item["value"], "confidence": item["confidence"],
               "source": item["sources"][0]["url"] if item.get("sources") else ""}
        out.update(extra or {})
        best[label] = out
        close = len(items) > 1 and items[1]["confidence"] >= max(0.4, item["confidence"] - 0.10)
        if label in ("website", "location", "category") and close:
            out["conflict_with"] = items[1]["value"]
        if item["confidence"] < 0.6 or "conflict_with" in out:
            review.append(label)

    for label, key in (("email", "emails"), ("phone", "phones"), ("website", "website"), ("link_in_bio", "link_in_bio"),
                       ("location", "location"), ("category", "category")):
        item = top(cands[key])
        if item:
            entry(label, item, cands[key], {k: item[k] for k in ("scope", "kind") if k in item})
    for platform, items in cands["social"].items():
        item = top(items)
        if item:
            entry(platform, item, items)
    per_platform = {p: top(items) for p, items in cands["audience"].items()}
    per_platform = {p: i for p, i in per_platform.items() if i}
    if per_platform:
        plat, item = max(per_platform.items(), key=lambda kv: (kv[1]["value"] if kv[1]["confidence"] >= 0.5 else 0, kv[1]["confidence"]))
        entry("audience_size", item, cands["audience"][plat], {
            "platform": plat, "as_of": item["sources"][0].get("as_of", ""),
            "by_platform": {p: i["value"] for p, i in per_platform.items()},
        })
    return best, review


# --------------------------------------------------------------------------
# Files, export and CLI
# --------------------------------------------------------------------------
def load_json(path: str, label: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        print(f"Note: {label} not found at {path}; continuing without it.")
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        sys.exit(f"Could not read {path}: {e}")


def save_json(path: str, data: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def extract_all(cache: Dict[str, Any], scraped: Dict[str, Any], only_ids: Optional[Sequence[str]], region: str, min_conf: float) -> Dict[str, Any]:
    ids = [i for i in dict.fromkeys(list(cache) + list(scraped)) if not only_ids or i in only_ids]
    return {cid: extract_creator(cid, cache.get(cid) or {}, scraped.get(cid) or {}, region, min_conf) for cid in ids}


CSV_FIELDS = ["email", "phone", "website", "link_in_bio", "location", "category"]


def export_csv(results: Dict[str, Any], path: str) -> int:
    header = ["creator_id", "name", "row"]
    for f in CSV_FIELDS:
        header += [f, f"{f}_confidence"]
    header += list(SOCIAL_PLATFORMS) + ["audience_size", "audience_platform", "audience_as_of", "email_scope", "email_source", "needs_review"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for cid, r in results.items():
            b = r["best"]
            row = [cid, r["name"], r.get("row", "")]
            for fld in CSV_FIELDS:
                row += [b.get(fld, {}).get("value", ""), b.get(fld, {}).get("confidence", "")]
            row += [b.get(p, {}).get("value", "") for p in SOCIAL_PLATFORMS]
            aud = b.get("audience_size", {})
            row += [aud.get("value", ""), aud.get("platform", ""), aud.get("as_of", ""),
                    b.get("email", {}).get("scope", ""), b.get("email", {}).get("source", ""), "; ".join(r["needs_review"])]
            w.writerow(row)
    return len(results)


def show(results: Dict[str, Any]) -> None:
    for cid, r in results.items():
        print(f"\n{cid} | {r['name']}  (search results: {r['inputs']['search_results']}, pages: {r['inputs']['scraped_pages_ok']})")
        if not r["best"]:
            print("   nothing found above the confidence threshold")
        for label, item in r["best"].items():
            extra = f" [{item['platform']}]" if label == "audience_size" else ""
            flag = "  <- review" if label in r["needs_review"] else ""
            print(f"   {label:13} {item['value']}{extra}  ({item['confidence']}){flag}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract contact, social, location, category and audience data.")
    p.add_argument("--cache", default=CACHE_FILE, help="creator_cache.json")
    p.add_argument("--scraped", default=SCRAPED_FILE, help="scraped_data.json")
    p.add_argument("--output", default=OUTPUT_FILE)
    p.add_argument("--only-id", action="append", help="Only this creator ID (repeatable).")
    p.add_argument("--default-region", default=DEFAULT_REGION, help="Region for phone numbers without a + code (default NG).")
    p.add_argument("--min-confidence", type=float, default=0.35, help="Lowest confidence kept in 'best' (default 0.35).")
    p.add_argument("--export-csv", metavar="FILE", help="Also write one row per creator.")
    p.add_argument("--show", action="store_true", help="Print the best values per creator.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    cache = load_json(args.cache, "creator cache")
    scraped = load_json(args.scraped, "scraped data")
    if not cache and not scraped:
        sys.exit("Nothing to extract: neither creator_cache.json nor scraped_data.json has data.")

    results = extract_all(cache, scraped, args.only_id, args.default_region.upper(), args.min_confidence)
    save_json(args.output, results)

    found: Dict[str, int] = {}
    for r in results.values():
        for label in r["best"]:
            found[label] = found.get(label, 0) + 1
    print(f"Extracted {len(results)} creator(s) -> {args.output}")
    print("Fields found:", json.dumps(found, indent=2) if found else "none")
    if not phonenumbers:
        print("Tip: pip install phonenumbers for stricter phone validation.")
    if args.show:
        show(results)
    if args.export_csv:
        print(f"Exported {export_csv(results, args.export_csv)} creator row(s) to {args.export_csv}")


if __name__ == "__main__":
    main()
