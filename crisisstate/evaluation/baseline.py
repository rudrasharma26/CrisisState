"""Frozen Phase 1 Baseline Configuration and Metadata."""

import importlib.metadata
import platform
import sys
from typing import Any, Dict

from crisisstate.domain.vocabulary import (
    ALLOWED_CLAIM_VALUES,
    CONTRADICTION_PAIRS,
    ClaimType,
)
from crisisstate.engine.attention import SEVERITY_WEIGHTS


def get_package_version(pkg_name: str) -> str:
    """Safely get package version from importlib."""
    try:
        return importlib.metadata.version(pkg_name)
    except Exception:
        return "unknown"


def get_frozen_phase1_baseline() -> Dict[str, Any]:
    """
    Returns the complete machine-readable baseline definition for Phase 1.
    All future Phase 2 evaluations will compare against this baseline.
    """
    return {
        "baseline_tag": "v1.0.0-phase1-frozen",
        "system_name": "CrisisState",
        "phase": 1,
        "description": "Deterministic, zero-cost, CPU-first uncertainty-aware incident intelligence engine",
        "environment": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "os": platform.system(),
        },
        "dependencies": {
            "fastapi": get_package_version("fastapi"),
            "pydantic": get_package_version("pydantic"),
            "numpy": get_package_version("numpy"),
            "pytest": get_package_version("pytest"),
            "httpx": get_package_version("httpx"),
            "uvicorn": get_package_version("uvicorn"),
        },
        "vocabulary_version": "v1-urban-flooding",
        "claim_types": [ct.value for ct in ClaimType],
        "allowed_values": {
            ct.value: sorted(list(values))
            for ct, values in ALLOWED_CLAIM_VALUES.items()
        },
        "contradiction_pairs": {
            ct.value: [list(pair) for pair in sorted(list(pairs))]
            for ct, pairs in CONTRADICTION_PAIRS.items()
        },
        "engine_configuration": {
            "spatial_radius_km": 2.0,
            "incident_matching_time_window_hours": 8.0,
            "contradiction_window_hours": 2.0,
            "temporal_supersession_threshold_hours": 2.0,
            "scoring_factors": [
                "severity",
                "conflict",
                "recency",
                "new_evidence",
                "uncertainty",
            ],
            "severity_weights": {
                f"{k[0].value}:{k[1]}": v for k, v in SEVERITY_WEIGHTS.items()
            },
        },
    }
