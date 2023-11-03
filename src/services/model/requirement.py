from dataclasses import dataclass
from enum import Enum


class RequirementKind(str, Enum):
    CONSTRAINT = "CONSTRAINT",
    CRITERION = "CRITERION",
    INFORMATION = "INFORMATION"


@dataclass
class Requirement:
    identifier: str
    name: str
    description: str
    kind: RequirementKind
    hasConcepts: list[str]
    source: int
    expression: str | None = None
