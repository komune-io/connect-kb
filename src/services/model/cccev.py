from dataclasses import dataclass

from .data_unit import DataUnit
from .information_concept import InformationConcept
from .requirement import Requirement


@dataclass
class Cccev:
    requirements: list[Requirement]
    information_concepts: list[InformationConcept]
    data_units: list[DataUnit]

    @staticmethod
    def empty():
        return Cccev(
            data_units=[],
            information_concepts=[],
            requirements=[]
        )
