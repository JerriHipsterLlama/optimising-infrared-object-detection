from infrared_detection.evaluation.frontier import pareto_optimal_rows, REQUIRED_REPORT_COLUMNS


def test_pareto_frontier_removes_models_that_are_worse_on_all_objectives():
    rows = [
        {"variant": "best", "map50_95": 0.8, "latency_p50_ms": 10.0, "peak_memory_mb": 100.0},
        {"variant": "dominated", "map50_95": 0.7, "latency_p50_ms": 12.0, "peak_memory_mb": 120.0},
    ]

    result = pareto_optimal_rows(rows)

    assert [row["variant"] for row in result] == ["best"]


def test_report_schema_contains_research_metrics():
    assert {"map50_95", "latency_p50_ms", "peak_memory_mb", "serialized_bytes"}.issubset(REQUIRED_REPORT_COLUMNS)
