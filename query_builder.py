"""query_builder.py: field-driven query planning.

Given a creator and the fields we are still missing, choose a small set of
targeted search queries that together cover as many missing fields as
possible. Queries aimed at high-value fields (email, phone) are favoured, so
when the email is missing the contact and booking queries come first.

    creator = {
        "creator_id": "CR001",
        "name": "Jane Doe",
        "location": "Lagos, Nigeria",   # optional context
        "category": "Fitness creator",  # optional context
    }
    queries = plan_queries(creator, missing=["email", "youtube"], max_queries=4)
"""

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

TARGET_FIELDS = ("website", "email", "phone", "instagram", "tiktok", "youtube", "location", "category")

# How much we care about each missing field when choosing queries.
FIELD_WEIGHTS = {
    "email": 3.0,
    "phone": 2.0,
    "website": 1.5,
    "instagram": 1.5,
    "tiktok": 1.5,
    "youtube": 1.5,
    "location": 1.0,
    "category": 1.0,
}


@dataclass(frozen=True)
class Query:
    text: str
    purpose: str
    targets: Tuple[str, ...]


# (purpose, template, fields the query helps with). Catalog order breaks ties.
QUERY_CATALOG: List[Tuple[str, str, Tuple[str, ...]]] = [
    ("identity", '"{name}" {category}', ("website", "category", "location")),
    ("contact", '"{name}" official website contact', ("website", "email", "phone")),
    ("booking", '"{name}" booking agent OR management', ("email", "phone")),
    ("press", '"{name}" media kit OR press', ("email", "phone")),
    ("links", '"{name}" links OR portfolio OR speaker', ("instagram", "tiktok", "youtube", "website")),
    ("youtube", '"{name}" youtube channel', ("youtube",)),
    ("instagram", '"{name}" instagram', ("instagram",)),
    ("tiktok", '"{name}" tiktok', ("tiktok",)),
    ("bio", '"{name}" interview biography {country}', ("location", "category", "website")),
]


def _country(location: str) -> str:
    """'Lagos, Nigeria' -> 'Nigeria' (last comma-separated part)."""
    parts = [p.strip() for p in (location or "").split(",") if p.strip()]
    return parts[-1] if parts else ""


def _render(template: str, creator: Dict[str, Any]) -> str:
    text = template.format(
        name=creator["name"].strip(),
        category=(creator.get("category") or "").strip(),
        country=_country(creator.get("location") or ""),
    )
    return " ".join(text.split())


def plan_queries(
    creator: Dict[str, Any],
    missing: Sequence[str],
    max_queries: int = 4,
) -> List[Query]:
    """Pick up to `max_queries` queries that best cover the missing fields.

    Greedy set cover: each round takes the query with the highest weighted
    gain, where a field already covered by a chosen query counts for less.
    Queries that would help no missing field are never chosen.
    """
    missing_set = {f for f in missing if f in TARGET_FIELDS}
    covered = {f: 0 for f in missing_set}
    remaining = list(QUERY_CATALOG)
    chosen: List[Query] = []

    while remaining and len(chosen) < max_queries:
        best, best_gain = None, 0.0
        for entry in remaining:  # catalog order wins ties
            purpose, template, targets = entry
            gain = sum(FIELD_WEIGHTS[f] / (1 + covered[f]) for f in targets if f in missing_set)
            if gain > best_gain + 1e-9:
                best, best_gain = entry, gain
        if best is None:
            break
        remaining.remove(best)
        purpose, template, targets = best
        for f in targets:
            if f in covered:
                covered[f] += 1
        chosen.append(Query(_render(template, creator), purpose, tuple(targets)))
    return chosen


def queries_signature(queries: Sequence[Query]) -> str:
    """Short hash of the query texts; used to detect when a plan changed."""
    joined = "\n".join(q.text for q in queries)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:12]


if __name__ == "__main__":
    demo = {"creator_id": "CR001", "name": "Jane Doe", "location": "Lagos, Nigeria", "category": "Fitness creator"}
    for missing in (list(TARGET_FIELDS), ["email"], ["youtube"], ["location", "category"]):
        print(f"missing={missing}")
        for q in plan_queries(demo, missing):
            print(f"   [{q.purpose:9}] {q.text}")
