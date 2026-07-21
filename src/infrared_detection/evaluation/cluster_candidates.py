"""Classification and deterministic selection for cluster-pruning candidates."""

from collections.abc import Iterable, Mapping
from decimal import Decimal


def classify_candidate(baseline_map50_95: float, candidate_map50_95: float) -> str:
    """Classify a candidate by its absolute mAP50-95 drop from baseline."""
    drop = Decimal(str(baseline_map50_95)) - Decimal(str(candidate_map50_95))
    if drop <= Decimal("0.01"):
        return "primary_feasible"
    if drop <= Decimal("0.02"):
        return "exploratory_feasible"
    return "rejected_accuracy"


def _candidate_sort_key(row: Mapping[str, object]) -> tuple[object, int, float, float, str]:
    latency = row.get("latency_p50_ms")
    has_latency = latency is not None
    return (
        row["serialized_bytes"],
        0 if has_latency else 1,
        float(latency) if has_latency else 0.0,
        -float(row["map50_95"]),
        str(row.get("candidate_id", "")),
    )


def select_cluster_candidates(rows: Iterable[Mapping[str, object]]) -> dict[str, Mapping[str, object] | None]:
    """Select primary and exploratory winners from already-classified rows."""
    rows = list(rows)
    primary = [row for row in rows if row.get("status") == "primary_feasible"]
    exploratory = [row for row in rows if row.get("status") == "exploratory_feasible"]

    return {
        "primary": min(primary, key=_candidate_sort_key, default=None),
        "exploratory": min(exploratory, key=_candidate_sort_key, default=None),
    }
