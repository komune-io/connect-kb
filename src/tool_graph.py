from langchain import LLMChain
from langchain.chains import GraphCypherQAChain
from langchain.chains.graph_qa.prompts import CYPHER_GENERATION_PROMPT
from langchain.chat_models.base import BaseChatModel
from langchain.graphs import Neo4jGraph
from langchain.schema.embeddings import Embeddings
from langchain.tools import BaseTool
from langchain.document_loaders import PyPDFium2Loader as PdfReader
import aspose.words as aw


class Json2GraphTool(BaseTool):
    name = "Json2Graph"
    description = "useful for when you need to convert a json into a graph"

    llm: BaseChatModel = None
    graph: Neo4jGraph = None

    def __init__(self, llm: BaseChatModel, graph: Neo4jGraph, **data):
        super().__init__(**data)
        self.llm = llm
        self.graph = graph

    def _run(self, json_data: str):
        chain = GraphCypherQAChain.from_llm(
            llm=self.llm,
            graph=self.graph,
            verbose=True
        )

        chain.run(f"""
You are an expert in building knowledge graphs.
Your task is to transform the given json into a graph and save it.
The json contains Requirements, Information Concepts, Data Units and a Document. Each entity should be a node, and you should map the relationships between them
Json: {json_data}
            """)
        return "Graph saved in Neo4J"


class QuestionGraphTool(BaseTool):
    name = "QuestionGraph"
    description = "useful for when you need to retrieve context from a graph via vectorized search when a structured data extraction didn't work. The retrieved context might be incomplete"

    embedder: Embeddings = None
    graph: Neo4jGraph = None

    def __init__(self, embedder: Embeddings, graph: Neo4jGraph, **data):
        super().__init__(**data)
        self.embedder = embedder
        self.graph = graph

    def _run(self, question: str):
        question_embedding = self.embedder.embed_query(question)
        answer_context_parts = self.graph.query("""
            CALL db.index.vector.queryNodes('embeddingIndex', 50, $question_embedding) 
            YIELD node, score
            RETURN node.text AS text, score
        """, {"question_embedding": question_embedding})

        answer_context = '\n'.join(map(lambda context_part: context_part["text"], answer_context_parts))
        return f"Here are the most relevant data to your question. Beware as it might be incomplete:\n{answer_context}"


class QueryGraphTool(BaseTool):
    name = "QueryGraph"
    description = "useful for when you need to retrieve structured data from a graph. The resulting data is always an extraction of the graph that anwsers the question"

    llm: BaseChatModel = None
    graph: Neo4jGraph = None

    def __init__(self, llm: BaseChatModel, graph: Neo4jGraph, **data):
        super().__init__(**data)
        self.llm = llm
        self.graph = graph

    def _run(self, question: str):
        chain = LLMChain(llm=self.llm, prompt=CYPHER_GENERATION_PROMPT)
        prompt = f"""Retrieve as much details as you can to answer the following question, but do not ever fetch the 'Embedding' labeled nodes: {question}"""
        query = chain.run(
            {"question": prompt, "schema": self.graph.get_schema}
        )
        print(query)
        result = self.graph.query(query)
        return f"Here are the structured data that answer the question: {result}"
        # chain = GraphCypherQAChain.from_llm(
#             llm=self.llm,
#             graph=self.graph,
#             verbose=True,
#         )
#         return chain.run(f"""
# You are an expert in querying knowledge graphs.
# Your task is to fetch data that could answer the given question from the graph you are provided with.
# Retrieve as much details as you can.
# The final answer must contain the structured data retrieved from the graph.
# Question: {question}
#             """)


# INIT_GRAPH = """
# CREATE (d1:DataUnit {identifier: "boolean", name: "Boolean", description: "True or false", type: "BOOLEAN"}),
# (d2:DataUnit {identifier: "proportion", name: "Proportion", description: "A proportion of something in relation to another", type: "NUMBER"}),
# (d3:DataUnit {identifier: "forestManagementTechniques", name: "Forest Management Techniques", description: "The type of harvesting techniques used in forest management", type: "ENUM", enum: ["clearCuts","patchCuts","seedTrees","continuousThinning","groupSelection"]}),
# (d4:DataUnit {identifier: "projectAreaCertification", name: "Project Area Certification", description: "Whether the project area is certified by Forest Stewardship Council (FSC) or subject to an easement", type: "ENUM", enum: ["FSC","easement"]}),
# (d5:DataUnit {identifier: "projectLength", name: "Project Length", description: "The minimum project length defined in the project description", type: "NUMBER", notation: "years"}),
# (i1:InformationConcept {identifier: "forestManagementTechniques", name: "Forest Management Techniques", description: "The type of harvesting techniques used in forest management", unit: "forestManagementTechniques", question: "What type of harvesting techniques are used in forest management?"}),
# (i2:InformationConcept {identifier: "timberHarvestingObjective", name: "Timber Harvesting Objective", description: "Whether the forest is managed with an objective for earning revenue through timber harvesting", unit: "boolean", question: "Is the forest managed with an objective for earning revenue through timber harvesting?"}),
# (i3:InformationConcept {identifier: "projectAreaCertification", name: "Project Area Certification", description: "Whether the project area is certified by Forest Stewardship Council (FSC) or subject to an easement", unit: "projectAreaCertification", question: "Is the project area certified by FSC or subject to an easement?"}),
# (i4:InformationConcept {identifier: "projectLength", name: "Project Length", description: "The minimum project length defined in the project description", unit: "projectLength", question: "What is the minimum project length defined in the project description?"}),
# (i5:InformationConcept {identifier: "peatForestManagement", name: "Peat Forest Management", description: "Whether the project encompasses managed peat forests", unit: "boolean", question: "Does the project encompass managed peat forests?"}),
# (i6:InformationConcept {identifier: "wetlandProportion", name: "Wetland Proportion", description: "The proportion of wetlands in the project area", unit: "proportion", question: "What is the proportion of wetlands in the project area?"}),
# (i7:InformationConcept {identifier: "managementPracticesProjection", name: "Management Practices Projection", description: "Whether the project proponents have a projection of management practices in both with- and without-project scenarios", unit: "boolean", question: "Do the project proponents have a projection of management practices in both with- and without-project scenarios?"}),
# (i8:InformationConcept {identifier: "fireControlMeasures", name: "Fire Control Measures", description: "Whether fire control measures are taken to ensure fire does not spread outside the project area", unit: "boolean", question: "Are fire control measures taken to ensure fire does not spread outside the project area?"}),
# (i9:InformationConcept {identifier: "leakagePrevention", name: "Leakage Prevention", description: "Whether there is leakage through activity shifting to other lands owned or managed by project proponents outside the boundary of the project area", unit: "boolean", question: "Is there leakage through activity shifting to other lands owned or managed by project proponents outside the boundary of the project area?"}),
# (r1:Requirement {identifier: "forestManagementTechniquesRequirement", name: "Forest Management Techniques Requirement", description: "Forest management in both baseline and project scenarios involves harvesting techniques such as clear cuts, patch cuts, seed trees, continuous thinning, or group selection practices.", hasConcepts: ["forestManagementTechniques"]}),
# (r2:Requirement {identifier: "timberHarvestingObjectiveRequirement", name: "Timber Harvesting Objective Requirement", description: "Forests which are not subject to timber harvesting, or managed without an objective for earning revenue through timber harvesting in the baseline scenario are not eligible under this methodology.", hasConcepts: ["timberHarvestingObjective"]}),
# (r3:Requirement {identifier: "projectAreaCertificationRequirement", name: "Project Area Certification Requirement", description: "Prior to the first verification event, the project area must meet one of the following conditions: a) Certified by Forest Stewardship Council (FSC); or b) Subject to an easement, or equivalent instrument, recorded against the deed of property that prohibits commercial harvesting for the duration of the crediting period unless later certified by FSC.", hasConcepts: ["projectAreaCertification"]}),
# (r4:Requirement {identifier: "projectLengthRequirement", name: "Project Length Requirement", description: "Project proponents must define the minimum project length in the project description.", hasConcepts: ["projectLength"]}),
# (r5:Requirement {identifier: "peatForestManagementRequirement", name: "Peat Forest Management Requirement", description: "The project does not encompass managed peat forests, and the proportion of wetlands is not expected to change as part of the project.", hasConcepts: ["peatForestManagement","wetlandProportion"]}),
# (r6:Requirement {identifier: "managementPracticesProjectionRequirement", name: "Management Practices Projection Requirement", description: "Project proponents must have a projection of management practices in both with- and without-project scenarios.", hasConcepts: ["managementPracticesProjection"]}),
# (r7:Requirement {identifier: "fireControlMeasuresRequirement", name: "Fire Control Measures Requirement", description: "Where fire is used as part of forest management, fire control measures such as installation of firebreaks or back-burning must be taken to ensure fire does not spread outside the project area — that is, no biomass burning is permitted to occur beyond the project area due to forest management activities.", hasConcepts: ["fireControlMeasures"]}),
# (r8:Requirement {identifier: "leakagePreventionRequirement", name: "Leakage Prevention Requirement", description: "There must be no leakage through activity shifting to other lands owned or managed by project proponents outside the boundary of the project area.", hasConcepts: ["leakagePrevention"]}),
# (i1)-[:HAS_UNIT]->(d3),
# (i2)-[:HAS_UNIT]->(d1),
# (i3)-[:HAS_UNIT]->(d4),
# (i4)-[:HAS_UNIT]->(d5),
# (i5)-[:HAS_UNIT]->(d1),
# (i6)-[:HAS_UNIT]->(d2),
# (i7)-[:HAS_UNIT]->(d1),
# (i8)-[:HAS_UNIT]->(d1),
# (i9)-[:HAS_UNIT]->(d1),
# (r1)-[:HAS_CONCEPT]->(i1),
# (r2)-[:HAS_CONCEPT]->(i2),
# (r3)-[:HAS_CONCEPT]->(i3),
# (r4)-[:HAS_CONCEPT]->(i4),
# (r5)-[:HAS_CONCEPT]->(i5),
# (r5)-[:HAS_CONCEPT]->(i6),
# (r6)-[:HAS_CONCEPT]->(i7),
# (r7)-[:HAS_CONCEPT]->(i8),
# (r8)-[:HAS_CONCEPT]->(i9)
# """

INIT_GRAPH="""
CREATE (d:Document {name: "VM0003-IFM-Through-Extension-Of-Rotation-Age-v1.3.pdf"})

CREATE (du1:DataUnit {identifier: "xsdString", name: "XSDString", description: "Any string of characters", type: "STRING"})
CREATE (du2:DataUnit {identifier: "xsdBoolean", name: "XSDBoolean", description: "True or false", type: "BOOLEAN"})
CREATE (du3:DataUnit {identifier: "xsdDate", name: "XSDDate", description: "A date", type: "DATE"})

CREATE (ic1:InformationConcept {identifier: "forestManagementPractices", name: "Forest Management Practices", description: "The practices used to manage the forest, such as harvesting techniques", unit: "xsdString", question: "What are the forest management practices used?", source: "VM0003-IFM-Through-Extension-Of-Rotation-Age-v1.3.pdf page 5"})
CREATE (ic2:InformationConcept {identifier: "projectAreaCondition", name: "Project Area Condition", description: "The condition that the project area must meet prior to the first verification event", unit: "xsdString", question: "What condition does the project area meet?", source: "VM0003-IFM-Through-Extension-Of-Rotation-Age-v1.3.pdf page 5"})
CREATE (ic3:InformationConcept {identifier: "projectLength", name: "Project Length", description: "The minimum length of the project as defined in the project description", unit: "xsdDate", question: "What is the minimum project length?", source: "VM0003-IFM-Through-Extension-Of-Rotation-Age-v1.3.pdf page 5"})
CREATE (ic4:InformationConcept {identifier: "fireControlMeasures", name: "Fire Control Measures", description: "The measures taken to control fire in the project area", unit: "xsdString", question: "What fire control measures are taken?", source: "VM0003-IFM-Through-Extension-Of-Rotation-Age-v1.3.pdf page 6"})
CREATE (ic5:InformationConcept {identifier: "leakagePrevention", name: "Leakage Prevention", description: "The measures taken to prevent leakage through activity shifting to other lands", unit: "xsdBoolean", question: "Are there measures to prevent leakage?", source: "VM0003-IFM-Through-Extension-Of-Rotation-Age-v1.3.pdf page 6"})
CREATE (ic6:InformationConcept {identifier: "projectStartDate", name: "Project Start Date", description: "The start date of the project", unit: "xsdDate", question: "What is the start date of the project?", source: "VM0003-IFM-Through-Extension-Of-Rotation-Age-v1.3.pdf page 5"})
CREATE (ic7:InformationConcept {identifier: "methodology", name: "Methodology", description: "The methodology used for the project", unit: "xsdString", question: "What methodology is used for the project?", source: "VM0003-IFM-Through-Extension-Of-Rotation-Age-v1.3.pdf page 4"})
CREATE (ic8:InformationConcept {identifier: "timberHarvesting", name: "Timber Harvesting", description: "Whether the forest is subject to timber harvesting in the baseline scenario", unit: "xsdBoolean", question: "Is the forest subject to timber harvesting in the baseline scenario?", source: "VM0003-IFM-Through-Extension-Of-Rotation-Age-v1.3.pdf page 5"})

CREATE (r1:Requirement {identifier: "projectEligibility", name: "Project Eligibility", description: "The project must involve an extension in rotation age (ERA) and must not encompass managed peat forests. The project area must meet one of the specified conditions prior to the first verification event. The project must have a defined start date.", source: "VM0003-IFM-Through-Extension-Of-Rotation-Age-v1.3.pdf page 5"})
CREATE (r2:Requirement {identifier: "fireControl", name: "Fire Control", description: "If fire is used as part of forest management, fire control measures must be taken to ensure fire does not spread outside the project area.", source: "VM0003-IFM-Through-Extension-Of-Rotation-Age-v1.3.pdf page 6"})
CREATE (r3:Requirement {identifier: "leakagePrevention", name: "Leakage Prevention", description: "There must be no leakage through activity shifting to other lands owned or managed by project proponents outside the boundary of the project area.", source: "VM0003-IFM-Through-Extension-Of-Rotation-Age-v1.3.pdf page 6"})
CREATE (r4:Requirement {identifier: "projectStartDate", name: "Project Start Date", description: "The project must have a defined start date.", source: "VM0003-IFM-Through-Extension-Of-Rotation-Age-v1.3.pdf page 5"})

CREATE (r1)-[:HAS_CONCEPT]->(ic1)
CREATE (r1)-[:HAS_CONCEPT]->(ic2)
CREATE (r1)-[:HAS_CONCEPT]->(ic6)
CREATE (r1)-[:HAS_CONCEPT]->(ic7)
CREATE (r1)-[:HAS_CONCEPT]->(ic8)
CREATE (r2)-[:HAS_CONCEPT]->(ic4)
CREATE (r3)-[:HAS_CONCEPT]->(ic5)
CREATE (r4)-[:HAS_CONCEPT]->(ic6)

CREATE (ic1)-[:HAS_UNIT]->(du1)
CREATE (ic2)-[:HAS_UNIT]->(du1)
CREATE (ic3)-[:HAS_UNIT]->(du3)
CREATE (ic4)-[:HAS_UNIT]->(du1)
CREATE (ic5)-[:HAS_UNIT]->(du2)
CREATE (ic6)-[:HAS_UNIT]->(du3)
CREATE (ic7)-[:HAS_UNIT]->(du1)
CREATE (ic8)-[:HAS_UNIT]->(du2)

CREATE (r1)-[:IN_DOCUMENT]->(d)
CREATE (r2)-[:IN_DOCUMENT]->(d)
CREATE (r3)-[:IN_DOCUMENT]->(d)
CREATE (r4)-[:IN_DOCUMENT]->(d)
CREATE (ic1)-[:IN_DOCUMENT]->(d)
CREATE (ic2)-[:IN_DOCUMENT]->(d)
CREATE (ic3)-[:IN_DOCUMENT]->(d)
CREATE (ic4)-[:IN_DOCUMENT]->(d)
CREATE (ic5)-[:IN_DOCUMENT]->(d)
CREATE (ic6)-[:IN_DOCUMENT]->(d)
CREATE (ic7)-[:IN_DOCUMENT]->(d)
CREATE (ic8)-[:IN_DOCUMENT]->(d)
"""
