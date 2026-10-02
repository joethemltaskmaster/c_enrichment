"""Creator search pipeline.

Reads creator IDs (default A2:A500) and names (default B2:B500) from the
sheet, searches the web for each name through SearchAPI.io (Google engine),
keeps the best few URLs per creator, remembers them keyed by creator ID, and
can optionally write the URLs back to the sheet.

Setup:
    pip install requests python-dotenv google-api-python-client google-auth
    Create a .env file next to this script (see .env.example):
        SEARCHAPI_API_KEY=...
        SHEETS_SPREADSHEET_ID=...
        SHEETS_KEY_FILE=your-service-account.json

Examples:
    python creator_search.py --dry-run                 # list creators it would search
    python creator_search.py --limit 3                 # search the first 3 only
    python creator_search.py --extra "youtube" --write-back --out-col C
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import urlsplit

import requests
from dotenv import load_dotenv
from googleapiclient.errors import HttpError

load_dotenv(r"C:\Users\Joseph\Desktop\Creator_enrichment_pipeline\.env")  # must run before os.environ / os.getenv are read below

from services.sheets_client import CREATOR_SHEET, SERVICE_ACCOUNT_FILE, SPREADSHEET_ID, SheetsClient

SEARCHAPI_URL = "https://www.searchapi.io/api/v1/search"
CACHE_FILE = "creator_cache.json"
FATAL_STATUSES = (401, 402, 403)  # bad key / no credits: stop the whole run
api_key = os.environ.get("SEARCHAPI_API_KEY")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
class SearchError(Exception):
    def __init__(self, message: str, fatal: bool = False):
        super().__init__(message)
        self.fatal = fatal


@dataclass
class Creator:
    creator_id: str
    name: str
    row: int
    id_generated: bool = False  # True when the ID cell was blank


def col_to_index(col: str) -> int:
    n = 0
    for ch in col.upper():
        n = n * 26 + (ord(ch) - 64)
    return n


def index_to_col(n: int) -> str:
    s = ""
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def _a1(sheet: str, rng: str) -> str:
    """Build a safely quoted A1 reference, e.g. 'Creators'!A2:A500."""
    return "'" + sheet.replace("'", "''") + "'!" + rng


def _cell(rows: List[List[Any]], i: int) -> str:
    """First cell of row i, or '' if the row/cell is missing or empty."""
    if i < len(rows) and rows[i]:
        return str(rows[i][0]).strip()
    return ""


def _normalize_url(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.netloc.lower()}{parts.path.rstrip('/')}?{parts.query}"


# --------------------------------------------------------------------------
# 1. Read creators from the sheet
# --------------------------------------------------------------------------
def load_creators(
    client: SheetsClient,
    sheet: str = CREATOR_SHEET,
    first_row: int = 2,
    last_row: int = 500,
    id_col: str = "A",
    name_col: str = "B",
):
    """Return (creators, skipped).

    Works with whatever is populated: rows shorter than the range, blank
    rows, and gaps are tolerated. A row with a name but no ID gets a
    fallback ID ("row<N>"). A row with an ID but no name is skipped because
    there is nothing to search.
    """
    id_rows = client.read_range(_a1(sheet, f"{id_col}{first_row}:{id_col}{last_row}"))
    name_rows = client.read_range(_a1(sheet, f"{name_col}{first_row}:{name_col}{last_row}"))

    creators: List[Creator] = []
    skipped: List[Dict[str, Any]] = []
    for i in range(max(len(id_rows), len(name_rows))):
        row = first_row + i
        cid, name = _cell(id_rows, i), _cell(name_rows, i)
        if not cid and not name:
            continue
        if not name:
            skipped.append({"row": row, "creator_id": cid, "reason": "no name"})
            continue
        generated = not cid
        creators.append(Creator(cid or f"row{row}", name, row, generated))
    return creators, skipped


# --------------------------------------------------------------------------
# 2. Memory: in-process dict backed by a JSON file
# --------------------------------------------------------------------------
class CreatorMemory:
    """Results keyed by creator_id. Lives in RAM and is saved after each update."""

    def __init__(self, path: str = CACHE_FILE):
        self.path = path
        self.data: Dict[str, Dict[str, Any]] = {}
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                print(f"Warning: could not read {path} ({e}); starting with empty memory.")

    def get(self, creator_id: str) -> Optional[Dict[str, Any]]:
        return self.data.get(creator_id)

    def set(self, creator_id: str, record: Dict[str, Any]) -> None:
        self.data[creator_id] = record
        self.save()

    def save(self) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, self.path)


# --------------------------------------------------------------------------
# 3. Search via SearchAPI.io
# --------------------------------------------------------------------------
def build_query(name: str, extra: str = "") -> str:
    """Exact-match the name; `extra` (platform, niche...) helps disambiguate."""
    return f'"{name}" {extra}'.strip()


def search_urls(
    query: str,
    api_key: str,
    max_urls: int = 5,
    max_per_domain: int = 2,
    exclude_domains: Sequence[str] = (),
    retries: int = 3,
    timeout: int = 30,
) -> List[Dict[str, Any]]:
    """Run one Google search and return up to `max_urls` filtered results.

    SearchAPI.io returns 10 organic results per page; we filter duplicates,
    excluded domains and over-represented domains, then keep the top few.
    """
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {"engine": "google", "q": query}

    payload = None
    for attempt in range(retries + 1):
        try:
            resp = requests.get(SEARCHAPI_URL, params=params, headers=headers, timeout=timeout)
        except requests.RequestException as e:
            if attempt == retries:
                raise SearchError(f"network error: {e}")
            time.sleep(2 ** attempt)
            continue

        if resp.status_code in FATAL_STATUSES:
            raise SearchError(f"HTTP {resp.status_code}: check API key / credits. {resp.text[:200]}", fatal=True)
        if resp.status_code == 429 or resp.status_code >= 500:
            if attempt == retries:
                raise SearchError(f"HTTP {resp.status_code} after {retries} retries")
            try:
                wait = float(resp.headers.get("Retry-After", 2 ** attempt))
            except ValueError:
                wait = 2 ** attempt
            time.sleep(wait)
            continue
        if resp.status_code != 200:
            raise SearchError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        payload = resp.json()
        break

    if not payload or payload.get("error"):
        raise SearchError(f"API error: {(payload or {}).get('error', 'empty response')}")

    exclude = [d.lower().removeprefix("www.") for d in exclude_domains if d]
    seen, per_domain, results = set(), {}, []
    for item in payload.get("organic_results", []):
        link = item.get("link")
        if not link:
            continue
        key = _normalize_url(link)
        if key in seen:
            continue
        domain = (item.get("domain") or urlsplit(link).netloc).lower().removeprefix("www.")
        if any(domain == d or domain.endswith("." + d) for d in exclude):
            continue
        if per_domain.get(domain, 0) >= max_per_domain:
            continue
        seen.add(key)
        per_domain[domain] = per_domain.get(domain, 0) + 1
        results.append(
            {
                "position": item.get("position"),
                "title": item.get("title", ""),
                "link": link,
                "domain": domain,
                "snippet": item.get("snippet", ""),
            }
        )
        if len(results) >= max_urls:
            break
    return results


# --------------------------------------------------------------------------
# 4. Pipeline
# --------------------------------------------------------------------------
def run_pipeline(
    creators: List[Creator],
    api_key: str,
    memory: CreatorMemory,
    extra: str = "",
    max_urls: int = 5,
    max_per_domain: int = 2,
    exclude_domains: Sequence[str] = (),
    delay: float = 1.0,
    force: bool = False,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Search each creator not already done. Returns {'results', 'summary'}.

    `results` maps creator_id -> record for every creator handled this run
    (fresh or remembered). Re-running is safe: finished creators are skipped
    unless `force` is set or the query changed.
    """
    results: Dict[str, Dict[str, Any]] = {}
    summary = {"searched": 0, "remembered": 0, "no_results": 0, "errors": []}

    for c in creators:
        query = build_query(c.name, extra)
        rec = memory.get(c.creator_id)
        if (
            rec
            and not force
            and rec.get("query") == query
            and rec.get("status") in ("done", "no_results")
        ):
            results[c.creator_id] = rec
            summary["remembered"] += 1
            continue

        if limit is not None and summary["searched"] >= limit:
            break

        record = {
            "creator_id": c.creator_id,
            "name": c.name,
            "row": c.row,
            "query": query,
            "searched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        try:
            urls = search_urls(query, api_key, max_urls, max_per_domain, exclude_domains)
            record.update(status="done" if urls else "no_results", urls=urls)
            if not urls:
                summary["no_results"] += 1
            print(f"[row {c.row}] {c.creator_id} | {c.name}: {len(urls)} URL(s)")
        except SearchError as e:
            if e.fatal:
                raise
            record.update(status="error", urls=[], error=str(e))
            summary["errors"].append({"row": c.row, "creator_id": c.creator_id, "error": str(e)})
            print(f"[row {c.row}] {c.creator_id} | {c.name}: ERROR {e}")

        memory.set(c.creator_id, record)
        results[c.creator_id] = record
        summary["searched"] += 1
        time.sleep(delay)

    return {"results": results, "summary": summary}


# --------------------------------------------------------------------------
# 5. Optional write-back to the sheet
# --------------------------------------------------------------------------
def write_back(
    client: SheetsClient,
    creators: List[Creator],
    results: Dict[str, Dict[str, Any]],
    sheet: str = CREATOR_SHEET,
    out_col: str = "C",
    max_urls: int = 5,
    overwrite: bool = False,
) -> Dict[str, int]:
    """Write each creator's URLs into `max_urls` cells starting at out_col.

    Rows whose target cells already contain data are left alone unless
    `overwrite` is True.
    """
    end_col = index_to_col(col_to_index(out_col) + max_urls - 1)
    todo = [
        (c.row, results[c.creator_id]["urls"])
        for c in creators
        if results.get(c.creator_id, {}).get("status") == "done"
    ]
    if not todo:
        return {"written": 0, "skipped_existing": 0}

    first, last = min(r for r, _ in todo), max(r for r, _ in todo)
    existing = client.read_range(_a1(sheet, f"{out_col}{first}:{end_col}{last}"))

    updates, skipped = {}, 0
    for row, urls in todo:
        idx = row - first
        has_data = idx < len(existing) and any(str(v).strip() for v in existing[idx])
        if has_data and not overwrite:
            skipped += 1
            continue
        cells = [u["link"] for u in urls] + [""] * (max_urls - len(urls))
        updates[_a1(sheet, f"{out_col}{row}:{end_col}{row}")] = [cells]

    if updates:
        client.batch_update(updates)
    return {"written": len(updates), "skipped_existing": skipped}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Search the web for each creator in the sheet.")
    p.add_argument("--spreadsheet-id", default=os.environ.get("SHEETS_SPREADSHEET_ID", SPREADSHEET_ID))
    p.add_argument("--key-file", default=os.environ.get("SHEETS_KEY_FILE", SERVICE_ACCOUNT_FILE))
    p.add_argument("--sheet", default=CREATOR_SHEET)
    p.add_argument("--first-row", type=int, default=2)
    p.add_argument("--last-row", type=int, default=500)
    p.add_argument("--id-col", default="A")
    p.add_argument("--name-col", default="B")
    p.add_argument("--extra", default="", help='Extra query words, e.g. "youtube" or "fitness".')
    p.add_argument("--max-urls", type=int, default=5)
    p.add_argument("--max-per-domain", type=int, default=2)
    p.add_argument("--exclude-domains", default="", help="Comma-separated, e.g. pinterest.com,quora.com")
    p.add_argument("--delay", type=float, default=1.0, help="Seconds between searches.")
    p.add_argument("--limit", type=int, help="Max number of NEW searches this run (for testing).")
    p.add_argument("--force", action="store_true", help="Re-search creators already done.")
    p.add_argument("--dry-run", action="store_true", help="Only list creators; no searches.")
    p.add_argument("--cache-file", default=CACHE_FILE)
    p.add_argument("--write-back", action="store_true", help="Write URLs to the sheet.")
    p.add_argument("--out-col", default="C", help="First column for URLs when writing back.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite non-empty output cells.")
    p.add_argument("--json", action="store_true", help="Print full results as JSON at the end.")
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    if args.spreadsheet_id == "YOUR_SPREADSHEET_ID_HERE":
        sys.exit("Error: set a real spreadsheet ID (--spreadsheet-id or SHEETS_SPREADSHEET_ID).")
    api_key = os.environ.get("SEARCHAPI_API_KEY", "")
    if not api_key and not args.dry_run:
        sys.exit("Error: set the SEARCHAPI_API_KEY environment variable.")

    try:
        client = SheetsClient(spreadsheet_id=args.spreadsheet_id, service_account_file=args.key_file)
        creators, skipped = load_creators(
            client, args.sheet, args.first_row, args.last_row, args.id_col, args.name_col
        )
        print(f"Found {len(creators)} creator(s); skipped {len(skipped)} row(s) without a name.")
        for s in skipped:
            print(f"  skipped row {s['row']} (ID {s['creator_id']}): {s['reason']}")
        generated = [c for c in creators if c.id_generated]
        if generated:
            print(f"  {len(generated)} row(s) had no ID; using fallback IDs like '{generated[0].creator_id}'.")

        if args.dry_run:
            for c in creators:
                print(f"  row {c.row}: {c.creator_id} | {c.name} -> {build_query(c.name, args.extra)}")
            return

        memory = CreatorMemory(args.cache_file)
        out = run_pipeline(
            creators,
            api_key,
            memory,
            extra=args.extra,
            max_urls=args.max_urls,
            max_per_domain=args.max_per_domain,
            exclude_domains=[d.strip() for d in args.exclude_domains.split(",")],
            delay=args.delay,
            force=args.force,
            limit=args.limit,
        )
        print("Summary:", json.dumps(out["summary"], indent=2))

        if args.write_back:
            res = write_back(
                client, creators, out["results"], args.sheet, args.out_col, args.max_urls, args.overwrite
            )
            print("Write-back:", res)

        if args.json:
            print(json.dumps(out["results"], indent=2, ensure_ascii=False))

    except SearchError as e:
        sys.exit(f"Search stopped: {e}")
    except HttpError as e:
        sys.exit(
            f"Google Sheets API error {e.resp.status}: {e._get_reason()}\n"
            "If 403/404, share the sheet with the service account's client_email and check the ID/tab name."
        )


if __name__ == "__main__":
    main()
