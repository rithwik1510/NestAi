from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    content: str
    version: str

    def render(self, **kwargs: str) -> str:
        return self.content.format(**kwargs)


class PromptTemplateRegistry:
    """Registry for prompt templates used during synthesis."""

    def __init__(self) -> None:
        self._templates: Dict[str, PromptTemplate] = {}

    def register(self, template: PromptTemplate) -> None:
        self._templates[template.name] = template

    def get(self, name: str) -> PromptTemplate:
        try:
            return self._templates[name]
        except KeyError as exc:
            raise ValueError(f"Prompt template '{name}' is not registered.") from exc
