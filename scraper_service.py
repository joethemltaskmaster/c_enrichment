"""Scraper service: visit the URLs saved in creator_cache.json and extract
useful public information with requests + BeautifulSoup.

For every page it records: title, meta description, Open Graph info, headings,
a short text excerpt, public email addresses, social profile links, likely
contact/about/booking links, and whether the page actually mentions the
creator's name (a cheap relevance check).

Setup:
    pip install requests beautifulsoup4 python-dotenv

Optional .env settings:
    CREATOR_CACHE_FILE=creator_cache.json     (input, written by creator_search.py)
    SCRAPE_OUTPUT_FILE=scraped_data.json      (output)
    SCRAPER_USER_AGENT=Mozilla/5.0 (compatible; CreatorResearchBot/1.0; you@example.com)

Examples:
    python scraper_service.py --dry-run
    python scraper_service.py --limit 5
    python scraper_service.py --only-id id1 --export-csv contacts.csv
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import urljoin, urlsplit
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

INPUT_FILE = os.getenv("CREATOR_CACHE_FILE") or "creator_cache.json"
OUTPUT_FILE = os.getenv("SCRAPE_OUTPUT_FILE") or "scraped_data.json"
USER_AGENT = os.getenv("SCRAPER_USER_AGENT") or "Mozilla/5.0 (compatible; CreatorResearchBot/1.0)"
ROBOT_NAME = "CreatorResearchBot"  # token matched against robots.txt rules
MAX_BYTES = 2_000_000  # never download more than ~2 MB per page

# Sites that require login / JavaScript and forbid scraping in their terms.
DEFAULT_SKIP_DOMAINS = (
    "instagram.com",
    "facebook.com",
    "x.com",
    "twitter.com",
    "linkedin.com",
    "tiktok.com",
)

SOCIAL_DOMAINS = {
    "instagram.com": "instagram",
    "twitter.com": "twitter",
    "x.com": "twitter",
    "facebook.com": "facebook",
    "linkedin.com": "linkedin",
    "tiktok.com": "tiktok",
    "youtube.com": "youtube",
    "youtu.be": "youtube",
    "twitch.tv": "twitch",
    "github.com": "github",
    "patreon.com": "patreon",
    "linktr.ee": "linktree",
}
SHARE_PATH_HINTS = ("/share", "/sharer", "/intent", "/dialog", "/plugins")
CONTACT_KEYWORDS = ("contact", "about", "business", "booking", "collab", "press", "media kit", "work with", "work-with")

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}")
JUNK_EMAIL_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".css", ".js")
JUNK_EMAIL_DOMAINS = ("example.com", "domain.com", "yourdomain.com", "sentry.io", "wixpress.com")

RETRYABLE_STATUSES = ("error",)  # everything else is final until --force


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
class ScrapeError(Exception):
    """Network-level failure that may succeed on a later run."""


def _domain(url: str) -> str:
    host = urlsplit(url).hostname or ""
    return host.lower().removeprefix("www.")


def _domain_matches(domain: str, candidates: Sequence[str]) -> bool:
    return any(domain == d or domain.endswith("." + d) for d in candidates if d)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _clean_email(raw: str) -> Optional[str]:
    email = raw.strip().strip(".,;:<>()[]\"'").lower()
    if not EMAIL_RE.fullmatch(email):
        return None
    if email.endswith(JUNK_EMAIL_SUFFIXES):
        return None
    if _domain_matches(email.split("@", 1)[1], JUNK_EMAIL_DOMAINS):
        return None
    return email


# --------------------------------------------------------------------------
# Robots.txt
# --------------------------------------------------------------------------
class RobotsCache:
    """Fetches and caches robots.txt once per site."""

    def __init__(self, session: requests.Session, timeout: int = 10):
        self.session = session
        self.timeout = timeout
        self._parsers: Dict[str, RobotFileParser] = {}

    def allowed(self, url: str) -> bool:
        parts = urlsplit(url)
        base = f"{parts.scheme}://{parts.netloc}"
        rp = self._parsers.get(base)
        if rp is None:
            rp = RobotFileParser()
            try:
                resp = self.session.get(base + "/robots.txt", timeout=self.timeout)
                if resp.status_code == 200:
                    rp.parse(resp.text.splitlines())
                elif resp.status_code in (401, 403):
                    rp.disallow_all = True
                else:
                    rp.allow_all = True  # no robots.txt => allowed
            except requests.RequestException:
                rp.allow_all = True
            self._parsers[base] = rp
        return rp.can_fetch(ROBOT_NAME, url)


# --------------------------------------------------------------------------
# Fetching
# --------------------------------------------------------------------------
def fetch_page(
    session: requests.Session,
    url: str,
    timeout: int = 20,
    max_bytes: int = MAX_BYTES,
    retries: int = 2,
) -> Dict[str, Any]:
    """GET a page. Returns {status_code, final_url, content_type, content}.

    `content` is None for HTTP errors and non-HTML responses. Raises
    ScrapeError for network failures / repeated 429 or 5xx responses.
    """
    for attempt in range(retries + 1):
        try:
            resp = session.get(url, timeout=timeout, stream=True, allow_redirects=True)
        except requests.RequestException as e:
            if attempt == retries:
                raise ScrapeError(f"network error: {e}")
            time.sleep(2 ** attempt)
            continue

        try:
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt == retries:
                    raise ScrapeError(f"HTTP {resp.status_code} after {retries} retries")
                try:
                    wait = float(resp.headers.get("Retry-After", 2 ** attempt))
                except ValueError:
                    wait = 2 ** attempt
                time.sleep(min(wait, 30))
                continue

            result = {
                "status_code": resp.status_code,
                "final_url": resp.url,
                "content_type": resp.headers.get("Content-Type", ""),
                "content": None,
            }
            if resp.status_code >= 400:
                return result
            ctype = result["content_type"].lower()
            if ctype and "html" not in ctype:
                return result

            chunks, size = [], 0
            for chunk in resp.iter_content(65536):
                chunks.append(chunk)
                size += len(chunk)
                if size >= max_bytes:
                    break
            result["content"] = b"".join(chunks)[:max_bytes]
            return result
        finally:
            resp.close()
    raise ScrapeError("unreachable")  # pragma: no cover


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------
def parse_page(content: bytes, base_url: str, creator_name: str = "") -> Dict[str, Any]:
    """Extract structured info from raw HTML bytes."""
    soup = BeautifulSoup(content, "html.parser")

    def meta(*names: str) -> str:
        for n in names:
            tag = soup.find("meta", attrs={"name": n}) or soup.find("meta", attrs={"property": n})
            if tag and tag.get("content"):
                return tag["content"].strip()
        return ""

    title = soup.title.get_text(" ", strip=True) if soup.title else ""

    # Links and mailto: addresses (collected before the tree is stripped).
    emails, links = set(), []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        low = href.lower()
        if low.startswith("mailto:"):
            for part in href[7:].split("?")[0].split(","):
                cleaned = _clean_email(part)
                if cleaned:
                    emails.add(cleaned)
        elif low.startswith(("javascript:", "#", "tel:")) or not href:
            continue
        else:
            absolute = urljoin(base_url, href)
            if urlsplit(absolute).scheme in ("http", "https"):
                links.append((a.get_text(" ", strip=True), absolute))

    script_count = len(soup.find_all("script"))
    for tag in soup(["script", "style", "noscript", "svg", "template"]):
        tag.decompose()
    text = " ".join(soup.get_text(" ", strip=True).split())
    # Very little visible text plus several scripts usually means the content is
    # built by JavaScript, which requests + BeautifulSoup cannot see.
    thin_content = len(text) < 250 and script_count >= 3

    for match in EMAIL_RE.findall(text[:300_000]):
        cleaned = _clean_email(match)
        if cleaned:
            emails.add(cleaned)

    headings = []
    for h in soup.find_all(["h1", "h2"]):
        t = " ".join(h.get_text(" ", strip=True).split())
        if t:
            headings.append(t[:120])
        if len(headings) >= 6:
            break

    social, seen_social = [], set()
    contact_links, seen_contact = [], set()
    base_host = _domain(base_url)
    for link_text, link in links:
        parts = urlsplit(link)
        host = _domain(link)
        platform = next((p for d, p in SOCIAL_DOMAINS.items() if _domain_matches(host, [d])), None)
        if platform:
            path = parts.path.lower()
            if parts.path.strip("/") and not any(h in path for h in SHARE_PATH_HINTS):
                clean = f"{parts.scheme}://{parts.netloc}{parts.path}".rstrip("/")
                if clean not in seen_social:
                    seen_social.add(clean)
                    social.append({"platform": platform, "url": clean})
            continue
        if host == base_host and len(contact_links) < 5:
            blob = f"{link_text} {parts.path}".lower()
            if any(k in blob for k in CONTACT_KEYWORDS) and link not in seen_contact:
                seen_contact.add(link)
                contact_links.append(link)

    lowered = text.lower()
    name = creator_name.strip().lower()
    return {
        "title": title,
        "description": meta("description", "og:description"),
        "og_title": meta("og:title"),
        "site_name": meta("og:site_name"),
        "headings": headings,
        "excerpt": text[:600],
        "emails": sorted(emails),
        "social_links": social,
        "contact_links": contact_links,
        "mentions_name": bool(name) and (name in lowered or name in title.lower()),
        "thin_content": thin_content,
    }


def scrape_url(
    url: str,
    creator_name: str,
    session: requests.Session,
    robots: RobotsCache,
    skip_domains: Sequence[str] = DEFAULT_SKIP_DOMAINS,
    timeout: int = 20,
) -> Dict[str, Any]:
    """Scrape one URL and return a record with a `status` field:
    ok | skipped_domain | robots_blocked | http_error | non_html | error
    """
    record: Dict[str, Any] = {"url": url, "fetched_at": _now()}
    if _domain_matches(_domain(url), skip_domains):
        record["status"] = "skipped_domain"
        return record
    if not robots.allowed(url):
        record["status"] = "robots_blocked"
        return record
    try:
        page = fetch_page(session, url, timeout=timeout)
    except ScrapeError as e:
        record.update(status="error", error=str(e))
        return record

    record.update(http_status=page["status_code"], final_url=page["final_url"])
    if page["status_code"] >= 400:
        record["status"] = "http_error"
    elif page["content"] is None:
        record["status"] = "non_html"
    else:
        record.update(parse_page(page["content"], page["final_url"], creator_name))
        record["status"] = "ok"
    return record


# --------------------------------------------------------------------------
# Storage and targets
# --------------------------------------------------------------------------
class ScrapeStore:
    """{creator_id: {name, row, pages: {url: record}}}, saved after every page."""

    def __init__(self, path: str = OUTPUT_FILE):
        self.path = path
        self.data: Dict[str, Dict[str, Any]] = {}
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                print(f"Warning: could not read {path} ({e}); starting empty.")

    def get_page(self, creator_id: str, url: str) -> Optional[Dict[str, Any]]:
        return self.data.get(creator_id, {}).get("pages", {}).get(url)

    def set_page(self, creator_id: str, name: str, row: Any, url: str, record: Dict[str, Any]) -> None:
        entry = self.data.setdefault(creator_id, {"name": name, "row": row, "pages": {}})
        entry["name"], entry["row"] = name, row
        entry["pages"][url] = record
        self.save()

    def save(self) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, self.path)


def load_targets(path: str = INPUT_FILE, only_ids: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """Read creator_cache.json and return creators that have pages to scrape.

    `urls` holds the pages selected by creator_search.py (old single-query
    records work too). `reserve` holds next-best pages used as fallbacks when a
    selected page is blocked or empty.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run creator_search.py first.")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    targets = []
    for creator_id, rec in raw.items():
        if only_ids and creator_id not in only_ids:
            continue
        urls = [u["link"] for u in rec.get("urls", []) if u.get("link")]
        reserve = [u["link"] for u in rec.get("reserve", []) if u.get("link")]
        if rec.get("status") == "done" and urls:
            targets.append(
                {"creator_id": creator_id, "name": rec.get("name", ""), "row": rec.get("row"),
                 "urls": urls, "reserve": [u for u in reserve if u not in urls]}
            )
    return targets


def _needs_scrape(store: ScrapeStore, creator_id: str, url: str, force: bool) -> bool:
    if force:
        return True
    rec = store.get_page(creator_id, url)
    return rec is None or rec.get("status") in RETRYABLE_STATUSES


def _failed(record: Optional[Dict[str, Any]]) -> bool:
    """True when a page gave us nothing usable (blocked, missing, empty or JS-only)."""
    return bool(record) and (record.get("status") != "ok" or bool(record.get("thin_content")))


def _reason(record: Dict[str, Any]) -> str:
    if record.get("status") == "ok":
        return "thin_content_js_suspected"
    if record.get("status") == "http_error":
        return f"http_error_{record.get('http_status')}"
    return record.get("status", "unknown")


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------
def run_scrape(
    targets: List[Dict[str, Any]],
    store: ScrapeStore,
    skip_domains: Sequence[str] = DEFAULT_SKIP_DOMAINS,
    delay: float = 1.5,
    timeout: int = 20,
    limit: Optional[int] = None,
    force: bool = False,
    session: Optional[requests.Session] = None,
    max_fallbacks: int = 2,
) -> Dict[str, Any]:
    """Scrape every pending URL. `limit` caps how many pages are fetched this run.

    When a page fails (blocked, robots, 4xx, non-HTML, JS-only), the reason is
    recorded on that page and the next reserve page for the creator is tried,
    up to `max_fallbacks` replacements per creator.
    """
    session = session or requests.Session()
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.5",
            "Accept-Language": "en-US,en;q=0.8",
        }
    )
    robots = RobotsCache(session)
    summary: Dict[str, Any] = {"scraped": 0, "already_done": 0, "fallbacks_used": 0, "by_status": {}}
    last_hit: Dict[str, float] = {}

    for t in targets:
        queue = list(t["urls"])
        reserve = [u for u in t.get("reserve", []) if u not in queue]
        fallback_of: Dict[str, str] = {}
        fallbacks = 0

        def add_fallback(failed_url: str) -> None:
            nonlocal fallbacks
            if reserve and fallbacks < max_fallbacks:
                nxt = reserve.pop(0)
                fallback_of[nxt] = failed_url
                queue.append(nxt)
                fallbacks += 1

        while queue:
            url = queue.pop(0)
            if not _needs_scrape(store, t["creator_id"], url, force):
                summary["already_done"] += 1
                if _failed(store.get_page(t["creator_id"], url)):
                    add_fallback(url)
                continue
            if limit is not None and summary["scraped"] >= limit:
                return summary

            domain = _domain(url)
            wait = delay - (time.monotonic() - last_hit.get(domain, -1e9))
            if wait > 0:
                time.sleep(wait)
            record = scrape_url(url, t["name"], session, robots, skip_domains, timeout)
            last_hit[domain] = time.monotonic()

            if url in fallback_of:
                record["fallback_for"] = fallback_of[url]
                summary["fallbacks_used"] += 1
            if _failed(record):
                record["blocked_reason"] = _reason(record)
                add_fallback(url)

            store.set_page(t["creator_id"], t["name"], t["row"], url, record)
            summary["scraped"] += 1
            status = record["status"]
            summary["by_status"][status] = summary["by_status"].get(status, 0) + 1
            if record.get("blocked_reason"):
                extra = f" | reason: {record['blocked_reason']}"
            else:
                extra = f" | emails: {len(record.get('emails', []))}"
            tag = " (fallback)" if url in fallback_of else ""
            print(f"[{t['creator_id']}] {status:15} {url}{tag}{extra}")
    return summary


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------
def export_csv(store: ScrapeStore, path: str) -> int:
    """One row per creator, combining all successfully scraped pages."""
    rows = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["creator_id", "name", "row", "pages_ok", "emails", "social_links", "contact_links", "pages_mentioning_name"])
        for cid, entry in store.data.items():
            ok = [p for p in entry.get("pages", {}).values() if p.get("status") == "ok"]
            emails = sorted({e for p in ok for e in p.get("emails", [])})
            social = sorted({s["url"] for p in ok for s in p.get("social_links", [])})
            contact = sorted({c for p in ok for c in p.get("contact_links", [])})
            mentions = sum(1 for p in ok if p.get("mentions_name"))
            w.writerow([cid, entry.get("name", ""), entry.get("row", ""), len(ok),
                        "; ".join(emails), "; ".join(social), "; ".join(contact), mentions])
            rows += 1
    return rows


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Scrape the URLs saved in creator_cache.json.")
    p.add_argument("--input", default=INPUT_FILE, help="creator_cache.json from creator_search.py")
    p.add_argument("--output", default=OUTPUT_FILE)
    p.add_argument("--only-id", action="append", help="Only this creator ID (repeatable).")
    p.add_argument("--limit", type=int, help="Max pages to fetch this run (for testing).")
    p.add_argument("--delay", type=float, default=1.5, help="Seconds between requests to the same site.")
    p.add_argument("--timeout", type=int, default=20)
    p.add_argument(
        "--skip-domains",
        default=",".join(DEFAULT_SKIP_DOMAINS),
        help="Comma-separated domains never fetched. Pass '' to skip nothing.",
    )
    p.add_argument("--max-fallbacks", type=int, default=2, help="Reserve pages to try per creator when pages fail.")
    p.add_argument("--force", action="store_true", help="Re-scrape pages already done.")
    p.add_argument("--dry-run", action="store_true", help="List what would be scraped; fetch nothing.")
    p.add_argument("--export-csv", metavar="FILE", help="Also write a one-row-per-creator CSV.")
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    skip = [d.strip().lower() for d in args.skip_domains.split(",") if d.strip()]

    try:
        targets = load_targets(args.input, args.only_id)
    except FileNotFoundError as e:
        sys.exit(str(e))
    store = ScrapeStore(args.output)
    total = sum(len(t["urls"]) for t in targets)
    print(f"{len(targets)} creator(s) with {total} URL(s) in {args.input}.")

    if args.dry_run:
        for t in targets:
            print(f"{t['creator_id']} | {t['name']} ({len(t['reserve'])} reserve page(s))")
            for url in t["urls"]:
                if _domain_matches(_domain(url), skip):
                    state = "skip (blocked domain)"
                elif _needs_scrape(store, t["creator_id"], url, args.force):
                    state = "would scrape"
                else:
                    state = "already done"
                print(f"   {state:24} {url}")
        return

    summary = run_scrape(
        targets, store, skip_domains=skip, delay=args.delay,
        timeout=args.timeout, limit=args.limit, force=args.force,
        max_fallbacks=args.max_fallbacks,
    )
    print("Summary:", json.dumps(summary, indent=2))
    print(f"Saved to {args.output}")
    if args.export_csv:
        print(f"Exported {export_csv(store, args.export_csv)} creator row(s) to {args.export_csv}")


if __name__ == "__main__":
    main()