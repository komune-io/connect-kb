from dataclasses import dataclass
from enum import Enum


class DataUnitType(str, Enum):
    BOOLEAN = "BOOLEAN",
    DATE = "DATE",
    NUMBER = "NUMBER",
    STRING = "STRING",
    ENUM = "ENUM"


@dataclass
class DataUnit:
    identifier: str
    name: str
    description: str
    type: DataUnitType
    notation: str | None = None
    options: list[str] | None = None


class PredefinedDataUnits:
    DATE: DataUnit = DataUnit(
        identifier="xsdDate",
        name="XSDDate",
        description="Date",
        type=DataUnitType.DATE
    )
    DOUBLE: DataUnit = DataUnit(
        identifier="xsdDouble",
        name="XSDDouble",
        description="Any real number",
        type=DataUnitType.NUMBER
    )
    INT: DataUnit = DataUnit(
        identifier="xsdInt",
        name="XSDInt",
        description="Any integer number",
        type=DataUnitType.NUMBER
    )
    STRING: DataUnit = DataUnit(
        identifier="xsdString",
        name="XSDString",
        description="Any arbitrary string of characters",
        type=DataUnitType.STRING
    )
    BOOLEAN: DataUnit = DataUnit(
        identifier="xsdBoolean",
        name="XSDBoolean",
        description="True or false",
        type=DataUnitType.BOOLEAN
    )
