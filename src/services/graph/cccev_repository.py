import uuid

from services.model.cccev import CCCEV
from langchain.graphs import Neo4jGraph

from .schema import LABEL_DATA_UNIT, LABEL_INFORMATION_CONCEPT, LABEL_REQUIREMENT, LABEL_DOCUMENT


class CccevRepository:
    graph: Neo4jGraph = None

    def __init__(self, graph: Neo4jGraph, **data):
        super().__init__(**data)
        self.graph = graph

    def save(self, cccev: CCCEV):
        doc = "doc"
        command = f"""CREATE ({doc}:{LABEL_DOCUMENT} {{id: "{uuid.uuid4()}", name: "{cccev.document.name}"}})\n"""

        for unit in cccev.dataUnits:
            command += f"""CREATE ({unit.identifier}:{LABEL_DATA_UNIT} {{
                id: "{uuid.uuid4()}",
                identifier: "{unit.identifier}",
                name: "{unit.name}",
                description: "{unit.description}",
                type: "{unit.type}",
                status: "EXISTS"
            }})
            """

        for concept in cccev.informationConcepts:
            command += f"""CREATE ({concept.identifier}:{LABEL_INFORMATION_CONCEPT} {{
                id: "{uuid.uuid4()}",
                identifier: "{concept.identifier}",
                name: "{concept.name}",
                description: "{concept.description}",
                question: "{concept.question}",
                status: "EXISTS",
                source: {concept.source}
            }})
            CREATE ({concept.identifier})-[:HAS_UNIT]->({concept.unit})
            CREATE ({concept.identifier})-[:HAS_DOCUMENT]->({doc})
            """

        for requirement in cccev.requirements:
            command += f"""CREATE ({requirement.identifier}:{LABEL_REQUIREMENT} {{
                id: "{uuid.uuid4()}",
                identifier: "{requirement.identifier}",
                name: "{requirement.name}",
                description: "{requirement.description}",
                kind: "CRITERION",
                status: "CREATED",
                source: {requirement.source}
            }})
            CREATE ({requirement.identifier})-[:HAS_DOCUMENT]->({doc})
            """
            for concept in requirement.hasConcepts:
                command += f"""CREATE ({requirement.identifier})-[:HAS_CONCEPT]->({concept})
                """

        self.graph.query(command)
