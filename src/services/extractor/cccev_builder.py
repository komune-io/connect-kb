from ..model.cccev import Cccev
from ..model.data_unit import DataUnit, PredefinedDataUnits, DataUnitType
from ..model.information_concept import InformationConcept
from ..model.requirement import Requirement, RequirementKind


def build_cccev(fields: list[dict[str]], requirements: list[dict[str]]) -> Cccev:
    cccev = Cccev([], [], [])
    existing_unit_identifiers = set()
    for field in fields:
        field_cccev = _extract_field(field)
        cccev.requirements.extend(field_cccev.requirements)
        cccev.information_concepts.extend(field_cccev.information_concepts)
        for unit in field_cccev.data_units:
            if unit.identifier not in existing_unit_identifiers:
                cccev.data_units.append(unit)
                existing_unit_identifiers.add(unit.identifier)

    for requirement in requirements:
        cccev.requirements.append(Requirement(
            identifier=requirement["identifier"],
            name=requirement["name"],
            description=requirement["description"],
            kind=RequirementKind.CONSTRAINT,
            hasConcepts=requirement["fields"],
            source=requirement["page"],
            expression=requirement["expression"]
        ))

    return cccev


def _extract_field(field: dict) -> Cccev:
    unit = _extract_data_unit(field)
    concept = _extract_concept(field, unit.identifier)
    requirement = Requirement(
        identifier=f"{concept.identifier}_req",
        name=concept.name,
        description="",
        kind=RequirementKind.INFORMATION,
        hasConcepts=[concept.identifier],
        source=concept.source
    )
    return Cccev(
        requirements=[requirement],
        information_concepts=[concept],
        data_units=[unit]
    )


def _extract_data_unit(field: dict) -> DataUnit:
    field_type = field["type"]

    if len(field.get("options", [])) > 0:
        return DataUnit(
            identifier=f"""{field["identifier"]}_du""",
            name=field["label"],
            description="",
            type=DataUnitType.ENUM,
            options=field["options"]
        )

    if field_type == "text":
        return PredefinedDataUnits.STRING
    elif field_type == "checkbox":
        return PredefinedDataUnits.BOOLEAN
    elif field_type == "number":
        return PredefinedDataUnits.DOUBLE
    elif field_type == "date":
        return PredefinedDataUnits.DATE
    else:
        print(field)
        return PredefinedDataUnits.STRING


def _extract_concept(field: dict, unit_identifier: str) -> InformationConcept:
    concept = InformationConcept(
        identifier=field["identifier"],
        name=field["label"],
        description="",
        unit=unit_identifier,
        source=field["page"]
    )
    if _str_to_bool(field.get("multiple", "false")):
        concept.properties = {"multiple": True}

    return concept


def _str_to_bool(value: str) -> bool:
    return eval(str(value).lower().capitalize())
