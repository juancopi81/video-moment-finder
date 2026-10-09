"""Typed, source-grounded data for the native MCP workspace.

Generated views are conversation artifacts, not persisted learner records.
Media capabilities stay in tool-result metadata, never in generated documents.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit
from uuid import UUID

from mcp.types import CallToolResult, TextContent
from pydantic import BaseModel, ConfigDict, Field, model_validator

# Resource URIs are host cache keys, including the resource security policy.
# Refresh the revision when enabling the scoped storage-backed workspace.
UI_URI = "ui://vmf/workspace/0.4.1.html"
UI_MIME = "text/html;profile=mcp-app"
UI_PATH = Path(__file__).with_name("assets") / "workspace.html"
Seconds = Annotated[float, Field(ge=0, le=86400, allow_inf_nan=False)]
Text = Annotated[str, Field(min_length=1, max_length=12000)]
EvidenceId = Annotated[str, Field(pattern=r"^[a-z][a-z0-9-]{0,63}$")]


class ViewModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Coverage(ViewModel):
    kind: Literal["excerpt", "full"] = "excerpt"
    start_s: Seconds
    end_s: Seconds
    gaps: list[Text] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def ordered(self):
        if self.end_s < self.start_s:
            raise ValueError("Coverage end must follow its start")
        return self


class Evidence(ViewModel):
    id: EvidenceId
    kind: Literal["transcript", "frame"]
    start_s: Seconds
    end_s: Seconds
    summary: Text

    @model_validator(mode="after")
    def ordered(self):
        if self.end_s < self.start_s:
            raise ValueError("Evidence end must follow its start")
        return self


class Cited(ViewModel):
    citations: list[EvidenceId] = Field(min_length=1, max_length=20)


class Section(Cited):
    title: Text
    origin: Literal["lecture", "generated"]
    paragraphs: list[Text] = Field(default_factory=list, max_length=20)
    bullets: list[Text] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def content_required(self):
        if not self.paragraphs and not self.bullets:
            raise ValueError("A section needs paragraphs or bullets")
        return self


class Card(Cited):
    front: Text
    back: Text
    hint: str = Field(default="", max_length=12000)
    origin: Literal["lecture", "generated"] = "generated"
    rationale: str = Field(default="", max_length=12000)
    tags: list[Annotated[str, Field(pattern=r"^[a-z0-9_-]{1,64}$")]] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def reasoning_required(self):
        if self.origin == "generated" and not self.rationale.strip():
            raise ValueError("Generated card answers need reasoning")
        return self


class Slide(Cited):
    title: Text
    bullets: list[Text] = Field(min_length=1, max_length=8)
    notes: Text
    origin: Literal["lecture", "generated"]


class VectorInitial(ViewModel):
    v: list[Annotated[float, Field(ge=-5, le=5, allow_inf_nan=False)]] = Field(min_length=2, max_length=2)
    w: list[Annotated[float, Field(ge=-5, le=5, allow_inf_nan=False)]] = Field(min_length=2, max_length=2)


class VectorPlayground(Cited):
    model: Literal["dot-product-2d"]
    question: Text
    model_limit: Text
    extension_note: Text
    initial: VectorInitial


class LabOption(ViewModel):
    id: EvidenceId
    label: Text


class LabControl(ViewModel):
    id: EvidenceId
    label: Text
    options: list[LabOption] = Field(min_length=2, max_length=6)


class LabCase(Cited):
    when: dict[EvidenceId, EvidenceId]
    title: Text
    outcome: Text
    explanation: Text
    assumptions: list[Text] = Field(min_length=1, max_length=10)
    unknowns: list[Text] = Field(default_factory=list, max_length=10)


class FinitePlayground(ViewModel):
    model: Literal["reviewed-cases"]
    question: Text
    model_limit: Text
    extension_note: Text
    controls: list[LabControl] = Field(min_length=1, max_length=3)
    cases: list[LabCase] = Field(min_length=2, max_length=64)

    @model_validator(mode="after")
    def complete_cases(self):
        from itertools import product

        names = [c.id for c in self.controls]
        if len(set(names)) != len(names):
            raise ValueError("Control IDs must be unique")
        for c in self.controls:
            if len({o.id for o in c.options}) != len(c.options):
                raise ValueError("Option IDs must be unique")
        expected = set(product(*[[o.id for o in c.options] for c in self.controls]))
        actual = [tuple(case.when.get(n, "") for n in names) for case in self.cases]
        if any(set(case.when) != set(names) for case in self.cases) or len(set(actual)) != len(actual) or set(actual) != expected:
            raise ValueError("Provide one reviewed case for every control combination")
        return self


class LearningView(ViewModel):
    kind: Literal["study-guide", "flashcards", "playground", "presentation"]
    video_id: UUID
    title: Annotated[str, Field(min_length=1, max_length=200)]
    subtitle: str = Field(default="", max_length=1000)
    coverage: Coverage
    sources: list[Evidence] = Field(min_length=1, max_length=60)
    sections: list[Section] = Field(default_factory=list, max_length=40)
    cards: list[Card] = Field(default_factory=list, max_length=100)
    slides: list[Slide] = Field(default_factory=list, max_length=30)
    playground: Annotated[VectorPlayground | FinitePlayground, Field(discriminator="model")] | None = None

    @model_validator(mode="after")
    def validate_view(self):
        # A compact tool payload, not an arbitrary document or a signed media cache.
        if len(json.dumps(self.model_dump(mode="json"))) > 200000:
            raise ValueError("Learning view exceeds 200 KB")
        ids = {s.id for s in self.sources}
        if len(ids) != len(self.sources):
            raise ValueError("Evidence IDs must be unique")
        required = {"study-guide": self.sections, "flashcards": self.cards,
                    "presentation": self.slides, "playground": self.playground}
        if not required[self.kind]:
            raise ValueError(f"Missing content for {self.kind}")
        if any(value for kind, value in required.items() if kind != self.kind):
            raise ValueError("Each view contains exactly one workflow")
        cited = [*self.sections, *self.cards, *self.slides]
        if isinstance(self.playground, VectorPlayground):
            cited.append(self.playground)
        if isinstance(self.playground, FinitePlayground):
            cited.extend(self.playground.cases)
        if any(set(item.citations) - ids for item in cited):
            raise ValueError("Citations must reference provided evidence IDs")
        if self.cards and len({c.front.casefold() for c in self.cards}) != len(self.cards):
            raise ValueError("Card questions must be unique")
        # Reject capabilities rather than accidentally saving them as prose.
        encoded = json.dumps(self.model_dump(mode="json")).lower()
        if any(token in encoded for token in ("x-amz-signature=", "x-amz-credential=", "access_token=", "refresh_token=", "bearer ")):
            raise ValueError("Credentials and signed URLs do not belong in a learning view")
        return self


def origin(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Workspace media requires a configured HTTPS storage origin")
    return f"https://{parsed.netloc}"


def result(summary: dict, **private) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps({k: v for k, v in summary.items() if k not in {"view", "companion_views"}}))],
        structuredContent=summary,
        _meta={"vmf": private},
    )
