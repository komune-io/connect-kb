import json
import uuid

from langchain.graphs import Neo4jGraph

from .schema import LABEL_DATA_UNIT, LABEL_INFORMATION_CONCEPT, LABEL_REQUIREMENT, LABEL_DOCUMENT
from ..model.cccev import Cccev


class CccevRepository:
    graph: Neo4jGraph = None

    def __init__(self, graph: Neo4jGraph, **data):
        super().__init__(**data)
        self.graph = graph

    def save(self, cccev: Cccev, document_path: str):
        doc = "doc"
        command = f"""MERGE ({doc}:{LABEL_DOCUMENT} {{path: $path}})
            ON CREATE
                SET doc.identifier = 'kb_{str(uuid.uuid4()).replace('-', '_')}'
                SET doc.name = $path
            WITH doc\n"""
        command_params = {"path": document_path}

        for unit in cccev.data_units:
            command += f"""MERGE ({unit.identifier}:{LABEL_DATA_UNIT} {{identifier: "{unit.identifier}"}})
                ON CREATE
                    SET {unit.identifier}.id = "{uuid.uuid4()}"
                    SET {unit.identifier}.name = $du_{unit.identifier}_name
                    SET {unit.identifier}.description = $du_{unit.identifier}_description
                    SET {unit.identifier}.type = "{unit.type.name}"
                    SET {unit.identifier}.status = "EXISTS"
                    {f"SET {unit.identifier}.notation = $du_{unit.identifier}_notation" if unit.notation is not None else ""}
                    {f"SET {unit.identifier}.options = $du_{unit.identifier}_options" if unit.options is not None else ""}
            """
            command_params[f"du_{unit.identifier}_name"] = unit.name
            command_params[f"du_{unit.identifier}_description"] = unit.description
            command_params[f"du_{unit.identifier}_notation"] = unit.notation
            command_params[f"du_{unit.identifier}_options"] = json.dumps(unit.options)

        for concept in cccev.information_concepts:
            command += f"""CREATE ({concept.identifier}:{LABEL_INFORMATION_CONCEPT} {{
                id: "{uuid.uuid4()}",
                identifier: "{concept.identifier}",
                name: $ic_{concept.identifier}_name,
                description: $ic_{concept.identifier}_description,
                status: "EXISTS",
                source: {concept.source}
                {f", properties: $ic_{concept.identifier}_properties" if concept.properties is not None else ""}
                
            }})
            CREATE ({concept.identifier})-[:HAS_UNIT]->({concept.unit})
            CREATE ({concept.identifier})-[:EXTRACTED_FROM]->({doc})
            """
            command_params[f"ic_{concept.identifier}_name"] = concept.name
            command_params[f"ic_{concept.identifier}_description"] = concept.description
            command_params[f"ic_{concept.identifier}_properties"] = json.dumps(concept.properties)

        for requirement in cccev.requirements:
            command += f"""CREATE ({requirement.identifier}:{LABEL_REQUIREMENT} {{
                id: "{uuid.uuid4()}",
                identifier: "{requirement.identifier}",
                name: $rq_{requirement.identifier}_name,
                description: $rq_{requirement.identifier}_description,
                kind: "{requirement.kind.name}",
                status: "CREATED",
                source: {requirement.source}
                {f", expression: $rq_{requirement.identifier}_expression" if requirement.expression is not None else ""}
            }})
            CREATE ({requirement.identifier})-[:EXTRACTED_FROM]->({doc})
            """
            command_params[f"rq_{requirement.identifier}_name"] = requirement.name
            command_params[f"rq_{requirement.identifier}_description"] = requirement.description
            command_params[f"rq_{requirement.identifier}_expression"] = requirement.expression

            for concept in requirement.hasConcepts:
                command += f"""CREATE ({requirement.identifier})-[:HAS_CONCEPT]->({concept})
                """

        self.graph.query(command, command_params)
