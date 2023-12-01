from dataclasses import dataclass


@dataclass
class InformationConcept:
    identifier: str
    name: str
    description: str
    unit: str
    source: int
    properties: dict[str] | None = None
