"""Open Deep Research - Programmatic research agent with LangGraph."""

from open_deep_research.client import (
    DeepResearchClient,
    DeepResearchResult,
    ResearchLogHandler,
)
from open_deep_research.configuration import Configuration, MCPConfig, SearchAPI

__version__ = "0.0.16"

__all__ = [
    "DeepResearchClient",
    "DeepResearchResult",
    "ResearchLogHandler",
    "Configuration",
    "MCPConfig",
    "SearchAPI",
]
