"""source_ranker.py: turn raw search results into a ranked, diverse shortlist.

Pipeline for one creator:
    1. de-duplicate URLs across all queries
    2. classify each page (contact page, booking page, official site, ...)
    3. check that the page is really about this creator (name + context match)
    4. score pages by how likely they are to contain the MISSING fields
    5. pick up to N pages to scrape, keeping the set diverse
    6. keep every plausible social profile URL, even though socials are not scraped

Social profiles are destinations to discover and verify, not pages to scrape,
so they go into `social_profiles` and never use up a scrape slot.
"""

import difflib
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import parse_qsl, urlencode, urlsplit

from query_builder import TARGET_FIELDS, Query

# --------------------------------------------------------------------------
# Domain knowledge
# --------------------------------------------------------------------------
SOCIAL_HOSTS = {
    "instagram.com": "instagram",
    "tiktok.com": "tiktok",
    "youtube.com": "youtube",
    "youtu.be": "youtube",
    "twitter.com": "twitter",
    "x.com": "twitter",
    "facebook.com": "facebook",
    "linkedin.com": "linkedin",
    "threads.net": "threads",
    "twitch.tv": "twitch",
}

RESERVED_FIRST_SEGMENT = {
    "instagram": {"p", "reel", "reels", "explore", "stories", "tv", "accounts", "directory", "about", "legal", "web", "popular", "locations", "tags", "developer"},
    "twitter": {"home", "search", "explore", "i", "intent", "share", "hashtag", "settings", "login", "notifications", "messages", "compose", "tos", "privacy"},
    "facebook": {"sharer", "sharer.php", "share", "share.php", "groups", "watch", "events", "login", "marketplace", "photo", "photo.php", "permalink.php", "profile.php", "pages", "people", "public", "hashtag", "dialog", "plugins", "policies", "help", "gaming"},
    "twitch": {"directory", "videos", "p", "downloads", "jobs", "store", "turbo", "settings", "subscriptions"},
}

LINK_IN_BIO_DOMAINS = (
    "linktr.ee", "beacons.ai", "bio.link", "lnk.bio", "linkin.bio", "campsite.bio",
    "solo.to", "carrd.co", "stan.store", "allmylinks.com", "taplink.cc", "msha.ke",
)
DIRECTORY_DOMAINS = (
    "about.me", "behance.net", "imdb.com", "crunchbase.com", "muckrack.com",
    "sessionize.com", "speakerhub.com", "eventbrite.com", "dribbble.com", "contently.com",
)
# People-search sites and email-lead databases: unreliable and privacy-sensitive,
# so they are never selected for scraping. Edit to taste.
DATA_BROKER_DOMAINS = (
    "spokeo.com", "whitepages.com", "radaris.com", "mylife.com", "beenverified.com",
    "peoplefinders.com", "truthfinder.com", "intelius.com", "fastpeoplesearch.com",
    "truepeoplesearch.com", "thatsthem.com", "zabasearch.com", "peekyou.com", "pipl.com",
    "instantcheckmate.com", "nuwber.com", "ussearch.com", "anywho.com",
    "contactout.com", "rocketreach.co", "zoominfo.com", "apollo.io", "signalhire.com", "lusha.com",
)
AGENCY_HOST_WORDS = ("agency", "talent", "management", "mgmt", "booking", "speakers", "bureau", "represent", "roster")

CONTACT_WORDS = {"contact", "contacts", "inquiries", "enquiries", "inquiry", "enquiry", "hire", "reach"}
BOOKING_WORDS = {"booking", "bookings", "book", "speaker", "speakers", "management", "manager", "agent", "agency", "talent", "represent", "representation", "collab", "collaborate", "collaboration", "partnerships"}
PRESS_WORDS = {"press", "mediakit", "presskit"}
ABOUT_WORDS = {"about", "bio", "biography", "story", "meet"}
INTERVIEW_WORDS = {"interview", "podcast", "news", "article", "blog", "feature", "magazine", "spotlight", "review"}

TRACKING_PARAMS = {"fbclid", "gclid", "igshid", "igsh", "si", "ref", "ref_src", "feature", "mc_cid", "mc_eid"}
CONTEXT_STOPWORDS = {"creator", "content", "influencer", "the", "and", "for", "with", "blogger", "channel"}

# How useful each kind of page is for each missing field (0-10).
AFFINITY: Dict[str, Dict[str, float]] = {
    "email": {"contact_page": 10, "booking_page": 10, "press_page": 8, "about_page": 6, "official_site": 6, "directory": 4, "link_in_bio": 3, "interview": 1, "other": 0.5},
    "phone": {"contact_page": 10, "booking_page": 9, "press_page": 6, "official_site": 5, "about_page": 5, "directory": 3, "link_in_bio": 2, "interview": 1, "other": 0.5},
    "website": {"official_site": 10, "about_page": 8, "link_in_bio": 7, "contact_page": 6, "directory": 5, "booking_page": 5, "press_page": 5, "interview": 4, "other": 1},
    "instagram": {"link_in_bio": 8, "official_site": 7, "about_page": 6, "contact_page": 4, "press_page": 4, "directory": 4, "booking_page": 3, "interview": 2, "other": 1},
    "tiktok": {"link_in_bio": 8, "official_site": 7, "about_page": 6, "contact_page": 4, "press_page": 4, "directory": 4, "booking_page": 3, "interview": 2, "other": 1},
    "youtube": {"link_in_bio": 7, "official_site": 6, "about_page": 5, "press_page": 4, "directory": 3, "contact_page": 3, "booking_page": 3, "interview": 2, "other": 1},
    "location": {"about_page": 9, "interview": 8, "directory": 6, "official_site": 5, "press_page": 5, "booking_page": 4, "contact_page": 4, "link_in_bio": 2, "other": 1},
    "category": {"about_page": 8, "interview": 6, "official_site": 6, "directory": 5, "press_page": 5, "booking_page": 4, "link_in_bio": 3, "contact_page": 2, "other": 1},
}
DEFAULT_AFFINITY = 0.5
NAME_FACTOR = {3: 1.0, 2: 0.85, 1: 0.5, 0: 0.1}


# --------------------------------------------------------------------------
# Text and URL helpers
# --------------------------------------------------------------------------
def fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    return "".join(ch for ch in text if not unicodedata.combining(ch)).lower()


def words(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", fold(text))


def name_tokens(name: str) -> List[str]:
    return [t for t in words(name) if len(t) > 1]


def _host(url: str) -> str:
    host = (urlsplit(url).hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    for prefix in ("m.", "mobile."):
        if host.startswith(prefix) and host[len(prefix):] in SOCIAL_HOSTS:
            host = host[len(prefix):]
    return host


def _host_in(host: str, domains: Iterable[str]) -> bool:
    return any(host == d or host.endswith("." + d) for d in domains)


def canonical_key(url: str) -> str:
    """Key used to spot duplicates: no www, no fragment, no tracking params."""
    parts = urlsplit(url)
    host = _host(url)
    path = parts.path.rstrip("/")
    if host in SOCIAL_HOSTS:
        path = path.lower()
    query = sorted(
        (k, v) for k, v in parse_qsl(parts.query)
        if not k.lower().startswith("utm_") and k.lower() not in TRACKING_PARAMS
    )
    return host + path + ("?" + urlencode(query) if query else "")


# --------------------------------------------------------------------------
# Social profile parsing
# --------------------------------------------------------------------------
def parse_social(url: str) -> Optional[Dict[str, str]]:
    """Return {platform, kind, handle} for social URLs, else None.

    kind is 'profile' for a creator's own page (instagram.com/handle) and
    'content' for posts, videos, share links and similar.
    """
    host = _host(url)
    platform = SOCIAL_HOSTS.get(host) or next(
        (p for d, p in SOCIAL_HOSTS.items() if host.endswith("." + d)), None
    )
    if not platform:
        return None
    info = {"platform": platform, "kind": "content", "handle": ""}
    if host not in SOCIAL_HOSTS:  # subdomains such as music.youtube.com
        return info

    segs = [s for s in urlsplit(url).path.split("/") if s]
    first = segs[0] if segs else ""
    low = first.lower()
    handle = ""

    if platform == "instagram":
        if len(segs) == 1 and low not in RESERVED_FIRST_SEGMENT["instagram"]:
            handle = first
    elif platform == "tiktok":
        if first.startswith("@") and len(segs) == 1:
            handle = first[1:]
    elif platform == "youtube":
        if host != "youtu.be":
            sub_pages = {"videos", "about", "featured", "shorts", "streams", "playlists", "community"}
            if first.startswith("@") and (len(segs) == 1 or (len(segs) == 2 and segs[1].lower() in sub_pages)):
                handle = first[1:]
            elif low in ("channel", "c", "user") and len(segs) >= 2:
                handle = segs[1]
    elif platform in ("twitter", "facebook", "twitch"):
        if len(segs) == 1 and low not in RESERVED_FIRST_SEGMENT[platform]:
            handle = first
    elif platform == "linkedin":
        if low in ("in", "company", "school") and len(segs) >= 2:
            handle = segs[1]
    elif platform == "threads":
        if first.startswith("@") and len(segs) == 1:
            handle = first[1:]

    if handle:
        info.update(kind="profile", handle=handle)
    return info


def handle_similarity(handle: str, name: str) -> float:
    """0-1 similarity between a social handle and the creator's name."""
    h = re.sub(r"[^a-z0-9]", "", fold(handle))
    toks = name_tokens(name)
    joined = "".join(toks)
    if not h or not joined:
        return 0.0
    if len(h) >= 4 and (joined in h or h in joined):
        return 1.0
    reverse = "".join(reversed(toks))
    return max(
        difflib.SequenceMatcher(None, h, joined).ratio(),
        difflib.SequenceMatcher(None, h, reverse).ratio(),
    )


# --------------------------------------------------------------------------
# Classification and matching
# --------------------------------------------------------------------------
def _official_domain(host: str, name: str) -> bool:
    if host in SOCIAL_HOSTS or _host_in(host, LINK_IN_BIO_DOMAINS) or _host_in(host, DATA_BROKER_DOMAINS):
        return False
    toks = name_tokens(name)
    if not toks:
        return False
    core = re.sub(r"[^a-z0-9]", "", host)
    joined, reverse = "".join(toks), "".join(reversed(toks))
    return len(joined) >= 5 and (joined in core or reverse in core)


def classify(link: str, title: str, name: str) -> str:
    """Label a result with a source type used by the affinity table."""
    social = parse_social(link)
    if social:
        return f"social_{social['platform']}" if social["kind"] == "profile" else "social_content"

    host = _host(link)
    if _host_in(host, DATA_BROKER_DOMAINS):
        return "data_broker"
    if _host_in(host, LINK_IN_BIO_DOMAINS):
        return "link_in_bio"
    if _host_in(host, DIRECTORY_DOMAINS):
        return "directory"

    path = urlsplit(link).path.lower()
    path_words = set(re.findall(r"[a-z0-9]+", path))
    title_words = set(words(title))
    bag = path_words | title_words
    media_kit = {"media", "kit"} <= bag

    if bag & CONTACT_WORDS:
        return "contact_page"
    if bag & BOOKING_WORDS or any(w in host for w in AGENCY_HOST_WORDS):
        return "booking_page"
    if bag & PRESS_WORDS or media_kit:
        return "press_page"
    if path_words & ABOUT_WORDS or (title_words & ABOUT_WORDS and len(path_words) <= 2):
        return "about_page"
    if _official_domain(host, name):
        return "official_site"
    if bag & INTERVIEW_WORDS or re.search(r"/20\d\d/", path):
        return "interview"
    return "other"


def name_match_level(name: str, title: str, snippet: str, link: str) -> Tuple[int, bool]:
    """(0-3, official_domain). 3 = full name appears in the title/snippet."""
    toks = name_tokens(name)
    if not toks:
        return 0, False
    host = _host(link)
    official = _official_domain(host, name)

    text = " ".join(words(f"{title} {snippet}"))
    phrase, reverse = " ".join(toks), " ".join(reversed(toks))
    haystack = set(words(f"{title} {snippet} {urlsplit(link).path}"))

    if phrase in text or reverse in text:
        level = 3
    elif all(t in haystack for t in toks):
        level = 2
    elif official:
        level = 2
    elif sum(t in haystack for t in toks) / len(toks) >= 0.5:
        level = 1
    else:
        level = 0
    if len(toks) == 1:
        level = min(level, 2)  # a single-word name is easy to confuse
    return level, official


def context_level(creator: Dict[str, Any], title: str, snippet: str) -> int:
    """0-2: how many location/category words from the creator appear in the result."""
    ctx = words(f"{creator.get('location') or ''} {creator.get('category') or ''}")
    ctx = {w for w in ctx if len(w) > 2 and w not in CONTEXT_STOPWORDS}
    if not ctx:
        return 0
    present = set(words(f"{title} {snippet}"))
    return min(2, len(ctx & present))


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def _type_value(source_type: str, missing: Sequence[str]) -> Tuple[float, List[str]]:
    affs = sorted(
        ((AFFINITY[f].get(source_type, DEFAULT_AFFINITY), f) for f in missing if f in AFFINITY),
        reverse=True,
    )
    if not affs:
        return 1.0, []
    value = affs[0][0] + 0.15 * sum(a for a, _ in affs[1:])
    helps = [f for a, f in affs if a >= 6]
    return value, helps


def dedupe_results(raw_by_query: Sequence[Tuple[Query, List[Dict[str, Any]]]]) -> List[Dict[str, Any]]:
    """Merge results from every query into one record per canonical URL."""
    merged: Dict[str, Dict[str, Any]] = {}
    for query, results in raw_by_query:
        for item in results:
            link = item.get("link")
            if not link:
                continue
            key = canonical_key(link)
            pos = item.get("position") or 99
            entry = merged.get(key)
            if entry is None:
                merged[key] = {
                    "link": link,
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "domain": _host(link),
                    "position": pos,
                    "queries": [query.purpose],
                }
            else:
                entry["position"] = min(entry["position"], pos)
                if query.purpose not in entry["queries"]:
                    entry["queries"].append(query.purpose)
                if len(item.get("snippet", "")) > len(entry["snippet"]):
                    entry["snippet"] = item["snippet"]
    return list(merged.values())


def score_candidates(
    candidates: List[Dict[str, Any]],
    creator: Dict[str, Any],
    missing: Sequence[str],
    exclude_domains: Sequence[str] = (),
    min_score: float = 3.0,
) -> List[Dict[str, Any]]:
    """Add type, name/context match, score, reasons and eligibility to each candidate."""
    name = creator["name"]
    excluded = [d.lower().removeprefix("www.") for d in exclude_domains if d]
    for c in candidates:
        c["type"] = classify(c["link"], c["title"], name)
        level, official = name_match_level(name, c["title"], c["snippet"], c["link"])
        ctx = context_level(creator, c["title"], c["snippet"])
        c.update(name_match=level, context_match=ctx, official_domain=official)

        type_val, helps = _type_value(c["type"], missing)
        rank_pts = max(0, 10 - min(c["position"], 10)) * 0.2
        consensus = min(1.5, 0.5 * (len(c["queries"]) - 1))
        score = type_val * NAME_FACTOR[level] + 0.75 * ctx + rank_pts + consensus + (2.0 if official else 0.0)
        c["score"] = round(score, 2)
        c["likely_fields"] = helps if level >= 1 else []

        reasons = [f"type={c['type']}", f"name_match={level}/3"]
        if official:
            reasons.append("domain looks like the creator's own site")
        if ctx:
            reasons.append(f"context_match={ctx}")
        if len(c["queries"]) > 1:
            reasons.append(f"found by {len(c['queries'])} queries")
        if helps:
            reasons.append("may contain: " + ", ".join(helps))

        scrapable = not c["type"].startswith("social_") and c["type"] != "data_broker"
        if not scrapable:
            reasons.append("not scraped (social/data-broker)")
        if _host_in(c["domain"], excluded):
            scrapable = False
            reasons.append("excluded domain")
        if scrapable and level == 0:
            reasons.append("creator name not found in result")
        c["eligible"] = bool(scrapable and level >= 1 and c["score"] >= min_score)
        if scrapable and level >= 1 and c["score"] < min_score:
            reasons.append("score below threshold")
        c["reasons"] = reasons
    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates


def select_sources(
    candidates: List[Dict[str, Any]],
    max_urls: int = 5,
    max_per_domain: int = 2,
    reserve_size: int = 5,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Choose up to `max_urls` pages to scrape, preferring variety.

    Each extra page of an already-chosen type is discounted (x0.7 per repeat)
    and no domain may supply more than `max_per_domain` pages. Returns
    (selected, reserve); reserve is the next-best pages, used when a selected
    page turns out to be blocked or empty.
    """
    pool = [c for c in candidates if c["eligible"]]
    selected: List[Dict[str, Any]] = []
    type_counts: Dict[str, int] = {}
    domain_counts: Dict[str, int] = {}

    while pool and len(selected) < max_urls:
        best, best_adj = None, -1.0
        for c in pool:
            if domain_counts.get(c["domain"], 0) >= max_per_domain:
                continue
            adj = c["score"] * (0.7 ** type_counts.get(c["type"], 0))
            if adj > best_adj:
                best, best_adj = c, adj
        if best is None:
            break
        pool.remove(best)
        selected.append(best)
        type_counts[best["type"]] = type_counts.get(best["type"], 0) + 1
        domain_counts[best["domain"]] = domain_counts.get(best["domain"], 0) + 1

    reserve = sorted(pool, key=lambda c: c["score"], reverse=True)[:reserve_size]
    return selected, reserve


def collect_social_profiles(
    candidates: List[Dict[str, Any]],
    creator: Dict[str, Any],
    min_confidence: float = 0.35,
    per_platform: int = 2,
) -> List[Dict[str, Any]]:
    """Keep plausible social profile URLs (they are never scraped).

    Confidence blends how closely the handle resembles the name with whether
    the result text mentions the name. The best match per platform is marked
    primary when confidence is at least 0.5.
    """
    found: List[Dict[str, Any]] = []
    for c in candidates:
        info = parse_social(c["link"])
        if not info or info["kind"] != "profile":
            continue
        sim = handle_similarity(info["handle"], creator["name"])
        conf = 0.55 * sim + 0.45 * (c["name_match"] / 3)
        if conf < min_confidence:
            continue
        parts = urlsplit(c["link"])
        found.append(
            {
                "platform": info["platform"],
                "handle": info["handle"],
                "url": f"https://{_host(c['link'])}{parts.path}".rstrip("/"),
                "confidence": round(conf, 2),
                "match": "likely" if conf >= 0.7 else "possible",
                "title": c["title"],
                "queries": c["queries"],
                "primary": False,
            }
        )
    found.sort(key=lambda p: p["confidence"], reverse=True)

    result, counts = [], {}
    for p in found:
        n = counts.get(p["platform"], 0)
        if n >= per_platform:
            continue
        p["primary"] = n == 0 and p["confidence"] >= 0.5
        counts[p["platform"]] = n + 1
        result.append(p)
    return result


def _compact(c: Dict[str, Any]) -> Dict[str, Any]:
    keep = ("link", "title", "domain", "position", "type", "score", "name_match",
            "context_match", "official_domain", "likely_fields", "queries", "reasons", "eligible")
    out = {k: c[k] for k in keep if k in c}
    out["snippet"] = (c.get("snippet") or "")[:240]
    return out


def rank_sources(
    creator: Dict[str, Any],
    missing: Sequence[str],
    raw_by_query: Sequence[Tuple[Query, List[Dict[str, Any]]]],
    exclude_domains: Sequence[str] = (),
    max_urls: int = 5,
    max_per_domain: int = 2,
    pool_size: int = 20,
    min_score: float = 3.0,
) -> Dict[str, Any]:
    """Full ranking pass. Returns candidates, social_profiles, selected, reserve."""
    candidates = dedupe_results(raw_by_query)
    score_candidates(candidates, creator, missing, exclude_domains, min_score)
    socials = collect_social_profiles(candidates, creator)
    selected, reserve = select_sources(candidates, max_urls, max_per_domain)
    return {
        "total_unique": len(candidates),
        "candidates": [_compact(c) for c in candidates[:pool_size]],
        "social_profiles": socials,
        "selected": [_compact(c) for c in selected],
        "reserve": [_compact(c) for c in reserve],
    }
