"""Evaluator base classes and result containers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class EvaluatorInfo:
    key: str                      # registry key, e.g. "textstat"
    name: str                     # display name
    category: str                 # "safety" | "quality" | "text_stats" | "validation" | "ml_obs"
    inputs: list[str]             # canonical columns required
    optional_inputs: list[str] = field(default_factory=list)
    needs_llm: bool = False       # requires an LLM credential
    description: str = ""


class Evaluator(ABC):
    """Base evaluator. Subclasses score a dataframe row-wise and return
    a dict of {output_column: list_of_values} aligned with df.index."""

    info: EvaluatorInfo

    def available(self) -> tuple[bool, str]:
        """Whether runtime dependencies are importable. (ok, reason)"""
        return True, ""

    def applicable(self, df) -> tuple[bool, str]:
        """Whether the dataframe has the required inputs."""
        missing = [c for c in self.info.inputs if c not in df.columns]
        if missing:
            return False, f"missing columns: {', '.join(missing)}"
        return True, ""

    @abstractmethod
    def evaluate(self, df, config: dict | None = None) -> dict[str, list]:
        ...
