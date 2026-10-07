"""creator_search.py: field-driven source discovery (stages 1 and 2).

For each creator in the sheet it:
    1. works out which fields are still missing (email, phone, website,
       instagram, tiktok, youtube, location, category),
    2. plans a few targeted queries for those fields (query_builder.py),
    3. runs them through SearchAPI.io and collects MORE than 5 candidates,
    4. de-duplicates, scores and ranks them (source_ranker.py),
    5. selects up to 5 diverse pages to scrape, keeps a reserve list for
       fallbacks, and preserves every plausible social profile URL,
    6. remembers everything in creator_cache.json, keyed by creator ID.

Records keep the old `urls` list (now the selected pages), so
scraper_service.py and the old write-back keep working.

Setup:
    pip install requests python-dotenv google-api-python-client google-auth
    .env: SEARCHAPI_API_KEY, SHEETS_SPREADSHEET_ID, SHEETS_KEY_FILE
    Optional .env: CREATOR_COLUMNS=location=C,category=D,email=E,phone=F,website=G,instagram=H,tiktok=I,youtube=J

Examples:
    python creator_search.py --dry-run                  # show missing fields + planned queries
    python creator_search.py --limit 3                  # search 3 creators
    python creator_search.py --limit 10 --review-csv review.csv
    python creator_search.py --upgrade                  # re-run old (v1) cache records
"""

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests
from dotenv import load_dotenv
from googleapiclient.errors import HttpError

load_dotenv()  # must run before os.environ / os.getenv are read below

from query_builder import TARGET_FIELDS, Query, plan_queries, queries_signature
from sheets_client import CREATOR_SHEET, SERVICE_ACCOUNT_FILE, SPREADSHEET_ID, SheetsClient
from source_ranker import rank_sources

SEARCHAPI_URL = "https://www.searchapi.io/api/v1/search"
CACHE_FILE = os.getenv("CREATOR_CACHE_FILE") or "creator_cache.json"
PIPELINE_VERSION = "v2"
FATAL_STATUSES = (401, 402, 403)  # bad key / no credits: stop the whole run


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
    known: Dict[str, str] = field(default_factory=dict)  # field -> value already in the sheet

    @property
    def missing(self) -> List[str]:
        """Fields with no value in the sheet (unmapped fields count as unknown)."""
        return [f for f in TARGET_FIELDS if not self.known.get(f)]

    def as_dict(self, default_location: str = "", default_category: str = "") -> Dict[str, Any]:
        return {
            "creator_id": self.creator_id,
            "name": self.name,
            "location": self.known.get("location") or default_location,
            "category": self.known.get("category") or default_category,
        }


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
    if i < len(rows) and rows[i]:
        return str(rows[i][0]).strip()
    return ""


def parse_columns(text: str) -> Dict[str, str]:
    """'location=C, email=E' -> {'location': 'C', 'email': 'E'}."""
    mapping: Dict[str, str] = {}
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        key, _, col = part.partition("=")
        key, col = key.strip().lower(), col.strip().upper()
        if key not in TARGET_FIELDS or not col.isalpha():
            raise ValueError(f"Bad column mapping '{part}'. Use field=COLUMN with field in {', '.join(TARGET_FIELDS)}.")
        mapping[key] = col
    return mapping


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
    field_cols: Optional[Dict[str, str]] = None,
):
    """Return (creators, skipped).

    Works with whatever is populated: blank rows, short rows and gaps are
    tolerated. A name with no ID gets a fallback ID ("row<N>"); an ID with
    no name is skipped. `field_cols` maps fields to the sheet columns that
    already hold them, so only genuinely empty fields are treated as missing.
    """
    def column(col: str) -> List[List[Any]]:
        return client.read_range(_a1(sheet, f"{col}{first_row}:{col}{last_row}"))

    id_rows, name_rows = column(id_col), column(name_col)
    field_rows = {f: column(col) for f, col in (field_cols or {}).items()}

    total = max([len(id_rows), len(name_rows)] + [len(r) for r in field_rows.values()])
    creators: List[Creator] = []
    skipped: List[Dict[str, Any]] = []
    for i in range(total):
        row = first_row + i
        cid, name = _cell(id_rows, i), _cell(name_rows, i)
        if not cid and not name:
            continue
        if not name:
            skipped.append({"row": row, "creator_id": cid, "reason": "no name"})
            continue
        known = {f: _cell(rows, i) for f, rows in field_rows.items()}
        known = {f: v for f, v in known.items() if v}
        creators.append(Creator(cid or f"row{row}", name, row, not cid, known))
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
def search_raw(query: str, api_key: str, retries: int = 3, timeout: int = 30) -> List[Dict[str, Any]]:
    """Run one Google search and return its organic results (about 10)."""
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
            time.sleep(min(wait, 30))
            continue
        if resp.status_code != 200:
            raise SearchError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        payload = resp.json()
        break

    if not payload or payload.get("error"):
        raise SearchError(f"API error: {(payload or {}).get('error', 'empty response')}")
    return [
        {
            "position": item.get("position"),
            "title": item.get("title", ""),
            "link": item.get("link", ""),
            "snippet": item.get("snippet", ""),
        }
        for item in payload.get("organic_results", [])
        if item.get("link")
    ]


# --------------------------------------------------------------------------
# 4. Pipeline
# --------------------------------------------------------------------------
def _url_entry(c: Dict[str, Any]) -> Dict[str, Any]:
    """Selected page in the same shape the v1 `urls` list used (plus type/score)."""
    return {k: c.get(k) for k in ("position", "title", "link", "domain", "snippet", "type", "score")}


def run_pipeline(
    creators: List[Creator],
    api_key: str,
    memory: CreatorMemory,
    max_queries: int = 4,
    max_urls: int = 5,
    max_per_domain: int = 2,
    exclude_domains: Sequence[str] = (),
    delay: float = 1.0,
    force: bool = False,
    upgrade: bool = False,
    limit: Optional[int] = None,
    default_location: str = "",
    default_category: str = "",
    pool_size: int = 20,
) -> Dict[str, Any]:
    """Discover and rank sources for each creator. Returns {'results', 'summary'}.

    Safe to re-run: finished creators are skipped unless the query plan
    changed or `force` is set. Old v1 records are left alone unless `upgrade`.
    """
    results: Dict[str, Dict[str, Any]] = {}
    summary: Dict[str, Any] = {
        "searched": 0, "requests": 0, "remembered": 0, "legacy_kept": 0,
        "nothing_missing": 0, "no_results": 0, "errors": [],
    }

    for c in creators:
        cdict = c.as_dict(default_location, default_category)
        missing = c.missing
        queries = plan_queries(cdict, missing, max_queries)
        if not queries:
            summary["nothing_missing"] += 1
            continue
        signature = queries_signature(queries)

        rec = memory.get(c.creator_id)
        finished = bool(rec) and rec.get("status") in ("done", "no_results")
        if finished and not force:
            if rec.get("pipeline") == PIPELINE_VERSION and rec.get("plan") == signature:
                results[c.creator_id] = rec
                summary["remembered"] += 1
                continue
            if rec.get("pipeline") != PIPELINE_VERSION and not upgrade:
                results[c.creator_id] = rec
                summary["legacy_kept"] += 1
                continue

        if limit is not None and summary["searched"] >= limit:
            break

        raw_by_query: List[Tuple[Query, List[Dict[str, Any]]]] = []
        query_log: List[Dict[str, Any]] = []
        for q in queries:
            try:
                found = search_raw(q.text, api_key)
                raw_by_query.append((q, found))
                query_log.append({"text": q.text, "purpose": q.purpose, "results": len(found)})
            except SearchError as e:
                if e.fatal:
                    raise
                query_log.append({"text": q.text, "purpose": q.purpose, "error": str(e)})
            summary["requests"] += 1
            time.sleep(delay)

        record: Dict[str, Any] = {
            "pipeline": PIPELINE_VERSION,
            "creator_id": c.creator_id,
            "name": c.name,
            "row": c.row,
            "location": cdict["location"],
            "category": cdict["category"],
            "missing_fields": missing,
            "plan": signature,
            "queries": query_log,
            "searched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        if rec and rec.get("pipeline") != PIPELINE_VERSION and rec.get("urls"):
            record["legacy_urls"] = rec["urls"]  # keep what the old run found

        if not raw_by_query:
            record.update(status="error", urls=[], selected=[], reserve=[], candidates=[], social_profiles=[],
                          error="every query failed")
            summary["errors"].append({"row": c.row, "creator_id": c.creator_id, "error": "every query failed"})
            print(f"[row {c.row}] {c.creator_id} | {c.name}: ERROR every query failed")
        else:
            ranked = rank_sources(cdict, missing, raw_by_query, exclude_domains, max_urls, max_per_domain, pool_size)
            record.update(
                status="done" if ranked["total_unique"] else "no_results",
                total_unique=ranked["total_unique"],
                candidates=ranked["candidates"],
                social_profiles=ranked["social_profiles"],
                selected=ranked["selected"],
                reserve=ranked["reserve"],
                urls=[_url_entry(s) for s in ranked["selected"]],
            )
            if not ranked["total_unique"]:
                summary["no_results"] += 1
            print(
                f"[row {c.row}] {c.creator_id} | {c.name}: {len(queries)} queries, "
                f"{ranked['total_unique']} unique -> {len(ranked['selected'])} to scrape, "
                f"{len(ranked['social_profiles'])} social profile(s)"
            )

        memory.set(c.creator_id, record)
        results[c.creator_id] = record
        summary["searched"] += 1

    return {"results": results, "summary": summary}


# --------------------------------------------------------------------------
# 5. Review export (for manually checking results before scaling up)
# --------------------------------------------------------------------------
def export_review_csv(results: Dict[str, Dict[str, Any]], path: str) -> int:
    """One row per selected page, reserve page and social profile.

    The last two columns are blank so you can mark each source as useful (y/n)
    and add notes while checking creators by hand.
    """
    rows = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["creator_id", "name", "kind", "url", "type", "score", "name_match",
                    "found_by", "reasons", "useful (y/n)", "notes"])
        for cid, rec in results.items():
            name = rec.get("name", "")
            for kind in ("selected", "reserve"):
                for c in rec.get(kind, []):
                    w.writerow([cid, name, kind, c["link"], c.get("type", ""), c.get("score", ""),
                                c.get("name_match", ""), "|".join(c.get("queries", [])),
                                "; ".join(c.get("reasons", [])), "", ""])
                    rows += 1
            for p in rec.get("social_profiles", []):
                detail = f"{p['match']}; handle={p['handle']}" + ("; primary" if p.get("primary") else "")
                w.writerow([cid, name, "social_profile", p["url"], p["platform"], p["confidence"], "",
                            "|".join(p.get("queries", [])), detail, "", ""])
                rows += 1
    return rows


# --------------------------------------------------------------------------
# 6. Optional write-back of the selected URLs (kept from v1)
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
    """Write each creator's selected URLs into `max_urls` cells starting at out_col.

    Rows whose target cells already contain data are left alone unless
    `overwrite` is True. (The proper field-by-field sheet writer comes later.)
    """
    end_col = index_to_col(col_to_index(out_col) + max_urls - 1)
    todo = [
        (c.row, results[c.creator_id]["urls"])
        for c in creators
        if results.get(c.creator_id, {}).get("status") == "done" and results[c.creator_id].get("urls")
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
        cells = [u["link"] for u in urls][:max_urls]
        cells += [""] * (max_urls - len(cells))
        updates[_a1(sheet, f"{out_col}{row}:{end_col}{row}")] = [cells]

    if updates:
        client.batch_update(updates)
    return {"written": len(updates), "skipped_existing": skipped}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Discover and rank public sources for each creator in the sheet.")
    p.add_argument("--spreadsheet-id", default=os.environ.get("SHEETS_SPREADSHEET_ID") or SPREADSHEET_ID)
    p.add_argument("--key-file", default=os.environ.get("SHEETS_KEY_FILE") or SERVICE_ACCOUNT_FILE)
    p.add_argument("--sheet", default=CREATOR_SHEET)
    p.add_argument("--first-row", type=int, default=2)
    p.add_argument("--last-row", type=int, default=500)
    p.add_argument("--id-col", default="A")
    p.add_argument("--name-col", default="B")
    p.add_argument(
        "--cols",
        default=os.environ.get("CREATOR_COLUMNS", ""),
        help="Columns that already hold fields, e.g. 'location=C,category=D,email=E,instagram=H'. "
             "Unmapped fields are treated as missing.",
    )
    p.add_argument("--default-location", default="", help="Context for creators with no location, e.g. Nigeria.")
    p.add_argument("--default-category", default="", help="Context for creators with no category, e.g. 'fitness creator'.")
    p.add_argument("--max-queries", type=int, default=4, help="Max searches per creator (each costs one credit).")
    p.add_argument("--max-urls", type=int, default=5, help="Pages selected for scraping per creator.")
    p.add_argument("--max-per-domain", type=int, default=2)
    p.add_argument("--exclude-domains", default="", help="Comma-separated, e.g. pinterest.com,quora.com")
    p.add_argument("--delay", type=float, default=1.0, help="Seconds between searches.")
    p.add_argument("--limit", type=int, help="Max creators to search this run (for testing).")
    p.add_argument("--force", action="store_true", help="Re-search creators already done.")
    p.add_argument("--upgrade", action="store_true", help="Re-run records made by the old single-query version.")
    p.add_argument("--dry-run", action="store_true", help="Show missing fields and planned queries; no searches.")
    p.add_argument("--cache-file", default=CACHE_FILE)
    p.add_argument("--review-csv", metavar="FILE", help="Write a CSV for manually checking this run's results.")
    p.add_argument("--write-back", action="store_true", help="Write selected URLs to the sheet.")
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
        sys.exit("Error: set SEARCHAPI_API_KEY in your .env file.")
    try:
        field_cols = parse_columns(args.cols)
    except ValueError as e:
        sys.exit(f"Error: {e}")

    try:
        client = SheetsClient(spreadsheet_id=args.spreadsheet_id, service_account_file=args.key_file)
        creators, skipped = load_creators(
            client, args.sheet, args.first_row, args.last_row, args.id_col, args.name_col, field_cols
        )
        print(f"Found {len(creators)} creator(s); skipped {len(skipped)} row(s) without a name.")
        for s in skipped:
            print(f"  skipped row {s['row']} (ID {s['creator_id']}): {s['reason']}")
        generated = [c for c in creators if c.id_generated]
        if generated:
            print(f"  {len(generated)} row(s) had no ID; using fallback IDs like '{generated[0].creator_id}'.")

        if args.dry_run:
            total = 0
            for c in creators:
                cdict = c.as_dict(args.default_location, args.default_category)
                queries = plan_queries(cdict, c.missing, args.max_queries)
                total += len(queries)
                print(f"row {c.row}: {c.creator_id} | {c.name} | missing: {', '.join(c.missing) or 'nothing'}")
                for q in queries:
                    print(f"     [{q.purpose:9}] {q.text}")
            print(f"Up to {total} search request(s) for {len(creators)} creator(s) if every one is searched.")
            return

        memory = CreatorMemory(args.cache_file)
        out = run_pipeline(
            creators,
            api_key,
            memory,
            max_queries=args.max_queries,
            max_urls=args.max_urls,
            max_per_domain=args.max_per_domain,
            exclude_domains=[d.strip() for d in args.exclude_domains.split(",")],
            delay=args.delay,
            force=args.force,
            upgrade=args.upgrade,
            limit=args.limit,
            default_location=args.default_location,
            default_category=args.default_category,
        )
        print("Summary:", json.dumps(out["summary"], indent=2))
        if out["summary"]["legacy_kept"]:
            print(f"{out['summary']['legacy_kept']} creator(s) have old single-query results; add --upgrade to redo them.")

        if args.review_csv:
            touched = {cid: rec for cid, rec in out["results"].items() if rec.get("pipeline") == PIPELINE_VERSION}
            print(f"Review CSV: {export_review_csv(touched, args.review_csv)} row(s) -> {args.review_csv}")

        if args.write_back:
            res = write_back(client, creators, out["results"], args.sheet, args.out_col, args.max_urls, args.overwrite)
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