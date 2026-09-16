"""Engine package for CrisisState."""
from crisisstate.engine.extractor import extract_claims, extract_entity
from crisisstate.engine.matcher import IncidentMatcher
from crisisstate.engine.contradiction import ContradictionEngine
from crisisstate.engine.attention import AttentionScorer
from crisisstate.engine.pipeline import IncidentPipeline, PipelineResult

__all__ = [
    "extract_claims",
    "extract_entity",
    "IncidentMatcher",
    "ContradictionEngine",
    "AttentionScorer",
    "IncidentPipeline",
    "PipelineResult",
]
