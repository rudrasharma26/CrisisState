"""Incident matching and clustering engine."""

import math
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

from crisisstate.domain.models import Incident, Report, ensure_utc
from crisisstate.engine.extractor import extract_entity


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two points on the Earth in kilometers."""
    r = 6371.0  # Earth's radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r * c


class IncidentMatcher:
    """Matches an incoming report to an existing incident or triggers new incident creation."""

    def __init__(
        self,
        max_distance_km: float = 2.0,
        max_time_window_hours: float = 8.0,
    ):
        self.max_distance_km = max_distance_km
        self.max_time_window_hours = max_time_window_hours

    def match_or_create(
        self,
        report: Report,
        active_incidents: List[Incident],
    ) -> Tuple[Incident, bool]:
        """
        Determines whether report belongs to an existing incident or creates a new one.
        Returns: (Incident, is_new: bool)
        """
        entity = extract_entity(report)

        best_incident: Optional[Incident] = None
        best_score = 0.0

        for inc in active_incidents:
            # Check time window
            time_diff = abs((ensure_utc(report.timestamp) - ensure_utc(inc.updated_at)).total_seconds())
            if time_diff > self.max_time_window_hours * 3600:
                continue

            score = 0.0

            # 1. Location name / entity similarity
            inc_loc_lower = inc.location_name.strip().lower()
            rep_loc_lower = entity.strip().lower()

            if inc_loc_lower == rep_loc_lower:
                score += 0.7
            elif inc_loc_lower in rep_loc_lower or rep_loc_lower in inc_loc_lower:
                score += 0.5

            # 2. Coordinate proximity
            if (
                report.latitude is not None
                and report.longitude is not None
                and inc.latitude is not None
                and inc.longitude is not None
            ):
                dist = haversine_distance_km(
                    report.latitude, report.longitude, inc.latitude, inc.longitude
                )
                if dist <= self.max_distance_km:
                    coord_score = max(0.0, 0.4 * (1.0 - (dist / self.max_distance_km)))
                    score += coord_score
                else:
                    # Beyond distance threshold, penalize match
                    score -= 0.5

            if score > best_score and score >= 0.5:
                best_score = score
                best_incident = inc

        if best_incident is not None:
            return best_incident, False

        # Create new incident
        new_title = f"Urban Flooding at {entity}"
        new_incident = Incident(
            title=new_title,
            location_name=entity,
            latitude=report.latitude,
            longitude=report.longitude,
            created_at=report.timestamp,
            updated_at=report.timestamp,
        )
        return new_incident, True
