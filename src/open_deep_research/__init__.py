"""Open Deep Research - Programmatic research agent with LangGraph."""

__version__ = "0.0.16"

# Lazy imports to avoid circular import issues
def __getattr__(name):
    if name == "DeepResearchClient":
        from open_deep_research.client import DeepResearchClient
        return DeepResearchClient
    elif name == "DeepResearchResult":
        from open_deep_research.client import DeepResearchResult
        return DeepResearchResult
    elif name == "ResearchLogHandler":
        from open_deep_research.client import ResearchLogHandler
        return ResearchLogHandler
    elif name == "Configuration":
        from open_deep_research.configuration import Configuration
        return Configuration
    elif name == "MCPConfig":
        from open_deep_research.configuration import MCPConfig
        return MCPConfig
    elif name == "SearchAPI":
        from open_deep_research.configuration import SearchAPI
        return SearchAPI
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    "DeepResearchClient",
    "DeepResearchResult",
    "ResearchLogHandler",
    "Configuration",
    "MCPConfig",
    "SearchAPI",
]
