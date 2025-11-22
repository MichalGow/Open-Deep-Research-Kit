"""High-level Python API for running the Deep Research graph.

This module provides a programmatic interface that mirrors the behavior of the
CLI runner, but is designed for direct use from Python code. It intentionally
avoids loading environment variables itself – callers are expected to pass API
keys and other settings via constructor arguments (or read them from their own
.env files).
"""

import asyncio
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage

from open_deep_research.configuration import Configuration, SearchAPI
from open_deep_research.deep_researcher import deep_researcher


class ResearchLogHandler(BaseCallbackHandler):
    """Callback handler that logs tool usage (search + think_tool) in real time."""

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> None:
        """Run when a tool starts running."""
        tool_name = serialized.get("name")

        if tool_name in ("web_search", "tavily_search"):
            # Try to parse input if it's JSON to get cleaner query/queries
            try:
                if isinstance(input_str, str):
                    input_data = json.loads(input_str)
                    if isinstance(input_data, dict):
                        queries = input_data.get("queries", [])
                        if isinstance(queries, list):
                            query_text = ", ".join(queries)
                        else:
                            query_text = str(queries)
                    else:
                        query_text = str(input_data)
                else:
                    query_text = str(input_str)
            except Exception:
                query_text = str(input_str)

            if len(query_text) > 100:
                query_text = query_text[:97] + "..."

            print(f"   🔎 Searching: {query_text}")

        elif tool_name == "think_tool":
            # Show a short preview of the reflection content
            try:
                if isinstance(input_str, str):
                    input_data = json.loads(input_str)
                    if isinstance(input_data, dict):
                        reflection = input_data.get("reflection", "")
                    else:
                        reflection = str(input_data)
                else:
                    reflection = str(input_str)
            except Exception:
                reflection = str(input_str)

            if len(reflection) > 160:
                reflection_preview = reflection[:157] + "..."
            else:
                reflection_preview = reflection

            if reflection_preview.strip():
                print(f"   🤔 Thinking: {reflection_preview}")
            else:
                print("   🤔 Thinking...")


@dataclass
class DeepResearchResult:
    """Result of a Deep Research run."""

    prompt: str
    final_report: str
    metadata: Dict[str, Any]
    state: Dict[str, Any]
    output_dir: Optional[Path]


class DeepResearchClient:
    """Programmatic client for running the Deep Research graph.

    This client is designed to be used directly from Python code. It supports
    optional filesystem output, OpenRouter integration, and explicit API key
    injection via constructor arguments instead of relying on dotenv inside
    the library.
    """

    def __init__(
        self,
        *,
        output_base_dir: Optional[Path] = None,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        google_api_key: Optional[str] = None,
        tavily_api_key: Optional[str] = None,
        openai_api_base: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
        openrouter_model: Optional[str] = None,
        model_overrides: Optional[Dict[str, Any]] = None,
        search_api: Optional[SearchAPI | str] = None,
        callbacks: Optional[List[Any]] = None,
        summarization_timeout_seconds: Optional[float] = None,
        summarization_max_attempts: Optional[int] = None,
        log_level: str = "tools",
    ) -> None:
        self.output_base_dir = output_base_dir
        self.callbacks = callbacks or []
        self.model_overrides: Dict[str, Any] = dict(model_overrides or {})
        self.search_api = search_api
        self.summarization_timeout_seconds = summarization_timeout_seconds
        self.summarization_max_attempts = summarization_max_attempts
        # log_level controls printing to stdout:
        # - "none": no logging
        # - "minimal": start/end summary only
        # - "tools": start/end + tool-level logging (default)
        # - "verbose": tools + per-node progress updates
        self.log_level = log_level

        # Collect API keys to pass via config (preferred over env)
        api_keys: Dict[str, str] = {}
        if openai_api_key is not None:
            api_keys["OPENAI_API_KEY"] = openai_api_key
        if anthropic_api_key is not None:
            api_keys["ANTHROPIC_API_KEY"] = anthropic_api_key
        if google_api_key is not None:
            api_keys["GOOGLE_API_KEY"] = google_api_key
        if tavily_api_key is not None:
            api_keys["TAVILY_API_KEY"] = tavily_api_key
        self.api_keys: Optional[Dict[str, str]] = api_keys or None

        # Optionally override OpenAI base URL
        if openai_api_base is not None:
            os.environ["OPENAI_API_BASE"] = openai_api_base

        # OpenRouter integration (mirrors the CLI --openrouter behavior)
        if openrouter_api_key is not None or openrouter_model is not None:
            if not openrouter_api_key or not openrouter_model:
                raise ValueError(
                    "Both openrouter_api_key and openrouter_model must be provided together."
                )

            # Set OpenAI-compatible environment variables for underlying clients
            os.environ["OPENAI_API_KEY"] = openrouter_api_key
            os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

            # Also pass the key via config, so it works even if env is ignored
            if self.api_keys is None:
                self.api_keys = {}
            self.api_keys.setdefault("OPENAI_API_KEY", openrouter_api_key)

            # Ensure model has 'openai:' prefix if not present
            full_model_name = (
                openrouter_model
                if str(openrouter_model).startswith("openai:")
                else f"openai:{openrouter_model}"
            )

            # Override all core models unless explicitly overridden
            for field in (
                "research_model",
                "summarization_model",
                "compression_model",
                "final_report_model",
            ):
                self.model_overrides.setdefault(field, full_model_name)

        if self.output_base_dir is not None:
            self.output_base_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, prompt: str) -> DeepResearchResult:
        """Synchronous wrapper around :meth:`arun`.

        This is convenient for scripts and simple usage. For async contexts,
        call :meth:`arun` directly.
        """

        return asyncio.run(self.arun(prompt))

    async def arun(self, prompt: str) -> DeepResearchResult:
        """Run the Deep Research graph asynchronously and return structured results."""
        if not prompt or not prompt.strip():
            raise ValueError("prompt cannot be empty")

        prompt = prompt.strip()
        config = self._build_config()

        # Optionally adjust summarization behavior via env for this run
        env_backup = self._apply_summarization_env_overrides()

        output_dir: Optional[Path] = None
        if self.output_base_dir is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = self.output_base_dir / timestamp
            output_dir.mkdir(parents=True, exist_ok=True)

        if self.log_level in ("minimal", "tools", "verbose"):
            print("🔬 Starting research...")
            print(f"📝 Prompt: {prompt}")
            if output_dir is not None:
                print(f"📁 Output: {output_dir}")
            print()

        start_time = time.time()
        final_state: Dict[str, Any] = {}

        try:
            async for event in deep_researcher.astream(
                {"messages": [HumanMessage(content=prompt)]},
                stream_mode="updates",
                config=config,
            ):
                for node_name, node_content in event.items():
                    final_state.update(node_content)
                    if self.log_level == "verbose":
                        self._log_node_progress(node_name, node_content)

            execution_time = time.time() - start_time

            if output_dir is not None:
                self._save_results(prompt, final_state, execution_time, output_dir, config)

            metadata = self._extract_metadata(final_state, execution_time, config)

            if self.log_level in ("minimal", "tools", "verbose"):
                print("\n✅ Research completed")
                print(f"⏱️  Execution time: {self._format_duration(execution_time)}")
                print(f"📊 Report length: {len(final_state.get('final_report', ''))} characters")
                if output_dir is not None:
                    print(f"📁 Results saved to: {output_dir}")

            return DeepResearchResult(
                prompt=prompt,
                final_report=final_state.get("final_report", ""),
                metadata=metadata,
                state=final_state,
                output_dir=output_dir,
            )
        except Exception as e:  # noqa: BLE001
            execution_time = time.time() - start_time
            if output_dir is not None:
                self._save_error(prompt, e, execution_time, output_dir)
            raise
        finally:
            self._restore_summarization_env(env_backup)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_config(self) -> Dict[str, Any]:
        """Build the RunnableConfig dict passed into the graph."""
        config: Dict[str, Any] = {}
        configurable: Dict[str, Any] = {}

        if self.model_overrides:
            configurable.update(self.model_overrides)

        if self.search_api is not None:
            if isinstance(self.search_api, SearchAPI):
                configurable["search_api"] = self.search_api.value
            else:
                configurable["search_api"] = self.search_api

        if self.api_keys:
            configurable["apiKeys"] = self.api_keys

        if configurable:
            config["configurable"] = configurable

        callbacks: List[Any] = []
        if self.log_level in ("tools", "verbose"):
            callbacks.append(ResearchLogHandler())

        callbacks.extend(self.callbacks)
        if callbacks:
            config["callbacks"] = callbacks

        return config

    def _apply_summarization_env_overrides(self) -> Dict[str, Optional[str]]:
        """Apply per-run summarization overrides via env, returning previous values."""
        keys = [
            "SUMMARIZATION_TIMEOUT_SECONDS",
            "SUMMARIZATION_MAX_ATTEMPTS",
        ]
        backup: Dict[str, Optional[str]] = {k: os.environ.get(k) for k in keys}

        if self.summarization_timeout_seconds is not None:
            os.environ["SUMMARIZATION_TIMEOUT_SECONDS"] = str(
                self.summarization_timeout_seconds
            )
        if self.summarization_max_attempts is not None:
            os.environ["SUMMARIZATION_MAX_ATTEMPTS"] = str(
                self.summarization_max_attempts
            )

        return backup

    @staticmethod
    def _restore_summarization_env(previous: Dict[str, Optional[str]]) -> None:
        """Restore environment variables that were overridden for summarization."""
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format duration in human-readable form."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """Format file size in human-readable format."""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        if size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        return f"{size_bytes / (1024 * 1024):.1f} MB"

    def _log_node_progress(self, node_name: str, node_content: Dict[str, Any]) -> None:
        """Log per-node progress updates when log_level == 'verbose'."""
        if node_name == "clarify_with_user":
            print("   ✓ User intent clarified")
        elif node_name == "write_research_brief":
            brief = node_content.get("research_brief", "")
            lines = str(brief).split("\n")
            print(f"   ✓ Research brief generated ({len(brief)} chars):")
            for line in lines[:3]:
                if line.strip():
                    print(f"       | {line.strip()}")
            if len(lines) > 3:
                print("       | ...")
        elif node_name == "research_supervisor":
            messages = node_content.get("supervisor_messages", [])
            if messages:
                last_msg = messages[-1]
                tool_calls = getattr(last_msg, "tool_calls", None)
                if tool_calls:
                    topics: List[str] = []
                    for tc in tool_calls:
                        try:
                            if tc.get("name") == "ConductResearch":
                                topics.append(tc.get("args", {}).get("research_topic"))
                        except Exception:
                            continue
                    if topics:
                        print(f"   ✓ Supervisor delegated {len(topics)} research tasks:")
                        for topic in topics:
                            print(f"       → {topic}")
                    else:
                        print("   ✓ Supervisor reflecting...")
                else:
                    print("   ✓ Supervisor step complete")
            else:
                print("   ✓ Supervisor step complete")
        elif node_name == "researcher":
            print("   ✓ Researcher received new instructions")
        elif node_name == "researcher_tools":
            print("   ✓ Tools executed")
        elif node_name == "compress_research":
            print("   ✓ Research findings compressed")
        elif node_name == "final_report_generation":
            print("   ✓ Final report generated")
        else:
            print(f"   ✓ Step completed: {node_name}")

    def _extract_metadata(
        self,
        state: Dict[str, Any],
        execution_time: float,
        config: Any,
    ) -> Dict[str, Any]:
        """Extract execution metadata, including effective model configuration."""
        messages = state.get("messages", [])
        message_types: Dict[str, int] = {}
        for msg in messages:
            msg_type = type(msg).__name__
            message_types[msg_type] = message_types.get(msg_type, 0) + 1

        research_brief = state.get("research_brief", "")
        notes = state.get("notes", [])
        raw_notes = state.get("raw_notes", [])

        metadata: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "execution_time_seconds": round(execution_time, 2),
            "execution_time_formatted": self._format_duration(execution_time),
            "success": "final_report" in state and bool(state["final_report"]),
            "research_brief": research_brief,
            "message_count": len(messages),
            "message_types": message_types,
            "notes_count": len(notes),
            "raw_notes_count": len(raw_notes),
            "has_final_report": "final_report" in state,
            "report_length_chars": len(state.get("final_report", "")),
            "report_length_words": len(state.get("final_report", "").split()),
        }

        if config is not None:
            try:
                effective_config = Configuration.from_runnable_config(config)
                metadata["models"] = {
                    "summarization_model": effective_config.summarization_model,
                    "research_model": effective_config.research_model,
                    "compression_model": effective_config.compression_model,
                    "final_report_model": effective_config.final_report_model,
                    "search_api": getattr(
                        effective_config.search_api,
                        "value",
                        str(effective_config.search_api),
                    ),
                }
            except Exception:  # noqa: BLE001
                # If anything goes wrong extracting config, skip model metadata
                pass

        return metadata

    def _serialize_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize state for JSON output, handling non-serializable objects."""
        serialized: Dict[str, Any] = {}
        for key, value in state.items():
            try:
                json.dumps(value)
                serialized[key] = value
            except (TypeError, ValueError):
                if key in {"messages", "supervisor_messages"}:
                    serialized[key] = [
                        {
                            "type": type(msg).__name__,
                            "content": getattr(msg, "content", str(msg)),
                        }
                        for msg in value
                    ]
                else:
                    serialized[key] = str(value)
        return serialized

    def _save_results(
        self,
        prompt: str,
        state: Dict[str, Any],
        execution_time: float,
        output_dir: Path,
        config: Any,
    ) -> None:
        """Persist prompt, metadata, report, and full state to disk."""
        metadata = self._extract_metadata(state, execution_time, config)

        (output_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

        with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        final_report = state.get("final_report", "No report generated")
        (output_dir / "report.md").write_text(final_report, encoding="utf-8")
        (output_dir / "report.txt").write_text(final_report, encoding="utf-8")

        serialized_state = self._serialize_state(state)
        with open(output_dir / "full_state.json", "w", encoding="utf-8") as f:
            json.dump(serialized_state, f, indent=2, ensure_ascii=False)

        if state.get("notes"):
            notes_text = "\n\n---\n\n".join(state["notes"])
            (output_dir / "notes.txt").write_text(notes_text, encoding="utf-8")

        if state.get("research_brief"):
            (output_dir / "research_brief.txt").write_text(
                state["research_brief"],
                encoding="utf-8",
            )

    def _save_error(
        self,
        prompt: str,
        error: Exception,
        execution_time: float,
        output_dir: Path,
    ) -> None:
        """Persist error details for failed runs."""
        error_info = {
            "timestamp": datetime.now().isoformat(),
            "execution_time_seconds": round(execution_time, 2),
            "success": False,
            "error": str(error),
            "error_type": type(error).__name__,
        }

        with open(output_dir / "error.json", "w", encoding="utf-8") as f:
            json.dump(error_info, f, indent=2, ensure_ascii=False)

        (output_dir / "prompt.txt").write_text(prompt, encoding="utf-8")


__all__ = [
    "DeepResearchClient",
    "DeepResearchResult",
    "ResearchLogHandler",
]
