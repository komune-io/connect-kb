import json
import os
import uuid

from dotenv import load_dotenv
from langchain import PromptTemplate, LLMChain, ConversationChain
from langchain.agents import AgentType
from langchain.agents import initialize_agent
from langchain.agents.agent_toolkits import FileManagementToolkit
from langchain.chains import LLMMathChain, GraphCypherQAChain
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.document_loaders import PyPDFium2Loader as PdfReader
from langchain.embeddings import OpenAIEmbeddings
from langchain.graphs import Neo4jGraph
from langchain.memory import ConversationEntityMemory
from langchain.memory.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.tools import WriteFileTool

from tool_graph import Json2GraphTool, INIT_GRAPH, QuestionGraphTool, QueryGraphTool
from tool_cccev import CccevParser, CccevParserTool, CccevVerifierTool, CCCEV
from tool_pdf2txt import Pdf2TxtTool, Pdf2MdTool

DIR_DATA = "data"
DIR_OUTPUT = f"{DIR_DATA}/output"
DIR_INPUT = f"{DIR_DATA}/input"

FILE_CCP = "CCP-Book-R2-FINAL-26Jul23.pdf"
FILE_VM003 = "VM0003-IFM-Through-Extension-Of-Rotation-Age-v1.3.pdf"

load_dotenv()
GPT4 = ChatOpenAI(temperature=0, model_name="gpt-4")
GPT3 = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
ANTHROPIC = ChatAnthropic(temperature=0, model="claude-2")
LLM = GPT4

GRAPH = Neo4jGraph(url="neo4j://localhost:7687", username="neo4j", password="smartbsmartb")
EMBEDDER = OpenAIEmbeddings()


def extract_json_from_text(text: str):
    start_index = text.find("{")
    end_index = text.rfind("}") + 1
    extracted_json_str = text[start_index:end_index]
    extracted_json = json.loads(extracted_json_str)
    return extracted_json


def save_file(output_dir: str, output_file: str, content: str, file_extension="txt", use_dump=False):
    output_file_path = os.path.join(output_dir, output_file + '.' + file_extension)
    if file_extension == 'txt':
        with open(output_file_path, 'w') as file:
            file.write(str(content))
    else:
        with open(output_file_path, 'w') as file:
            if use_dump:
                content = extract_json_from_text(content)
                json.dump(content, file)
            else:
                file.write(str(content))
    print(f"Data saved to '{output_file_path}'")


def node_to_text(label: str, content: dict[str]) -> str:
    if label == "Requirement":
        return f"""{label}
         identifier: {content["identifier"]} 
         name: {content["name"]} 
         description: {content["description"]}"""
    elif label == "InformationConcept":
        return f"""{label}
        identifier: {content["identifier"]}
        name: {content["name"]}
        description: {content["description"]}"""
    elif label == "DataUnit":
        return f"""{label}
        identifier: {content["identifier"]}
        name: {content["name"]}
        description: {content["description"]}
        type: {content["type"]}"""
    elif label == "Document":
        return f"""{label}
        name: {content["name"]}
        """
    else:
        print(f"Unsupported label {label}")


def relationship_to_text(type: str, node_source: str, node_target: str) -> str:
    if type == 'HAS_CONCEPT':
        type_txt = "USES_INFORMATION"
    else:
        type_txt = type
    return f"{node_source} {type_txt} {node_target}"


def init_graph():
    # GRAPH.query(INIT_GRAPH)
    # GRAPH.query("CALL db.index.vector.createNodeIndex('embeddingIndex', 'Embedding', 'vector', 1536, 'cosine')")
    nodes = GRAPH.query("""
        MATCH (n)
        WHERE NOT (n)-[:HAS_EMBEDDING]->()
        AND NOT 'Embedding' in labels(n)
        RETURN n as content, labels(n) as label, id(n) as id
    """)
    for node in nodes:
        node_id: int = node["id"]
        node_label: str = node["label"][0]
        node_content: dict[str] = node["content"]

        relationships = GRAPH.query("""
            MATCH (n1)-[r]->(n2)
            WHERE id(n1) = $id
            AND type(r) <> 'HAS_EMBEDDING'
            RETURN n1 as source, id(n1) as source_id, type(r) as type, n2 as target, id(n2) as target_id
        """, {"id": node_id})

        node_embedded_text = node_to_text(node_label, node_content)
        node_embedding = EMBEDDER.embed_query(node_embedded_text)
        GRAPH.query("""
            CREATE (e: Embedding)
            SET e.text=$embedded_text

            WITH e
            CALL db.create.setVectorProperty(e, 'vector', $embedding)
            YIELD node

            WITH e
            MATCH (n)
            WHERE id(n) = $id
            CREATE (n)-[:HAS_EMBEDDING]->(e)
            SET n.id = $uuid
        """, {"embedded_text": node_embedded_text, "embedding": node_embedding, "id": node_id, "uuid": str(uuid.uuid4())})

        for relationship in relationships:
            r_type = relationship["type"]
            r_source = relationship["source"]
            r_source_id = relationship["source_id"]
            r_target: dict[str] = relationship["target"]
            r_target_id = relationship["target_id"]
            r_embedded_text = relationship_to_text(r_type, r_source.get("identifier", r_source["name"]), r_target.get("identifier", r_target["name"]))
            r_embedding = EMBEDDER.embed_query(r_embedded_text)
            GRAPH.query(f"""
                CREATE (e: Embedding)
                SET e.text=$embedded_text

                WITH e
                CALL db.create.setVectorProperty(e, 'vector', $embedding)
                YIELD node

                WITH e
                MATCH (source)
                WHERE id(source) = $source_id
                CREATE (source)-[:HAS_EMBEDDING]->(e)

                WITH e
                MATCH (target)
                WHERE id(target) = $target_id
                CREATE (target)-[:HAS_EMBEDDING]->(e)
            """, {"embedded_text": r_embedded_text, "embedding": r_embedding, "source_id": r_source_id, "target_id": r_target_id})


def question_graph(question: str) -> str:
    # question_embedding = EMBEDDER.embed_query(question)
    # answer_context_parts = GRAPH.query("""
    #     CALL db.index.vector.queryNodes('embeddingIndex', 20, $question_embedding)
    #     YIELD node, score
    #     RETURN node.text AS text, score
    # """, {"question_embedding": question_embedding})
    #
    # answer_context = '\n'.join(map(lambda context_part: context_part["text"], answer_context_parts))
    agent = initialize_agent(
        tools=[
            QuestionGraphTool(EMBEDDER, GRAPH),
            QueryGraphTool(LLM, GRAPH)
        ],
        llm=LLM,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        max_iterations=None,
        max_execution_time=None
    )

    prompt = f"""
    You are an intelligent chatbot. Please answer the question as truly as possible, and using only the context you are able to fetch from the knowledge graph. 
    You are allowed to question the graph multiple times.
    Even if the question seems vague or unclear, you must always question the graph at least once. In the question of a vague or unclear question, start with the QuestionGraph tool
    If you can’t find answer from the context, simplay say “Sorry I don’t have enough context for your question.”
    Question: {question}
    """
    # answer = GPT3.predict(prompt)
    print(prompt)
    answer = agent.run(prompt)
    print(answer)
    return answer


# cccev = """{"dataUnits": [{"identifier": "xsdString", "name": "XSDString", "description": "Any string of characters", "type": "STRING"}, {"identifier": "xsdBoolean", "name": "XSDBoolean", "description": "True or false", "type": "BOOLEAN"}, {"identifier": "xsdDate", "name": "XSDDate", "description": "A date", "type": "DATE"}], "informationConcepts": [{"identifier": "forestManagementPractices", "name": "Forest Management Practices", "description": "The practices used to manage the forest, such as harvesting techniques", "unit": "xsdString", "question": "What are the forest management practices used?", "source": "VM0003-IFM-Through-Extension-Of-Rotation-Age-v1.3.pdf page 5"}, {"identifier": "projectAreaCondition", "name": "Project Area Condition", "description": "The condition that the project area must meet prior to the first verification event", "unit": "xsdString", "question": "What condition does the project area meet?", "source": "VM0003-IFM-Through-Extension-Of-Rotation-Age-v1.3.pdf page 5"}, {"identifier": "projectLength", "name": "Project Length", "description": "The minimum length of the project as defined in the project description", "unit": "xsdDate", "question": "What is the minimum project length?", "source": "VM0003-IFM-Through-Extension-Of-Rotation-Age-v1.3.pdf page 5"}, {"identifier": "fireControlMeasures", "name": "Fire Control Measures", "description": "The measures taken to control fire in the project area", "unit": "xsdString", "question": "What fire control measures are taken?", "source": "VM0003-IFM-Through-Extension-Of-Rotation-Age-v1.3.pdf page 6"}, {"identifier": "leakagePrevention", "name": "Leakage Prevention", "description": "The measures taken to prevent leakage through activity shifting to other lands", "unit": "xsdBoolean", "question": "Are there measures to prevent leakage?", "source": "VM0003-IFM-Through-Extension-Of-Rotation-Age-v1.3.pdf page 6"}, {"identifier": "projectStartDate", "name": "Project Start Date", "description": "The start date of the project", "unit": "xsdDate", "question": "What is the start date of the project?", "source": "VM0003-IFM-Through-Extension-Of-Rotation-Age-v1.3.pdf page 5"}, {"identifier": "methodology", "name": "Methodology", "description": "The methodology used for the project", "unit": "xsdString", "question": "What methodology is used for the project?", "source": "VM0003-IFM-Through-Extension-Of-Rotation-Age-v1.3.pdf page 4"}, {"identifier": "timberHarvesting", "name": "Timber Harvesting", "description": "Whether the forest is subject to timber harvesting in the baseline scenario", "unit": "xsdBoolean", "question": "Is the forest subject to timber harvesting in the baseline scenario?", "source": "VM0003-IFM-Through-Extension-Of-Rotation-Age-v1.3.pdf page 5"}], "requirements": [{"identifier": "projectEligibility", "name": "Project Eligibility", "description": "The project must involve an extension in rotation age (ERA) and must not encompass managed peat forests. The project area must meet one of the specified conditions prior to the first verification event. The project must have a defined start date.", "hasRequirement": [], "hasConcepts": ["forestManagementPractices", "projectAreaCondition", "projectStartDate", "methodology", "timberHarvesting"], "source": "VM0003-IFM-Through-Extension-Of-Rotation-Age-v1.3.pdf page 5"}, {"identifier": "fireControl", "name": "Fire Control", "description": "If fire is used as part of forest management, fire control measures must be taken to ensure fire does not spread outside the project area.", "hasRequirement": [], "hasConcepts": ["fireControlMeasures"], "source": "VM0003-IFM-Through-Extension-Of-Rotation-Age-v1.3.pdf page 6"}, {"identifier": "leakagePrevention", "name": "Leakage Prevention", "description": "There must be no leakage through activity shifting to other lands owned or managed by project proponents outside the boundary of the project area.", "hasRequirement": [], "hasConcepts": ["leakagePrevention"], "source": "VM0003-IFM-Through-Extension-Of-Rotation-Age-v1.3.pdf page 6"}, {"identifier": "projectStartDate", "name": "Project Start Date", "description": "The project must have a defined start date.", "hasRequirement": [], "hasConcepts": ["projectStartDate"], "source": "VM0003-IFM-Through-Extension-Of-Rotation-Age-v1.3.pdf page 5"}], "document": "VM0003-IFM-Through-Extension-Of-Rotation-Age-v1.3.pdf"}"""


def extract_cccev(input_file_path: str, resultCount: int, verbose: bool):
    reader = PdfReader(input_file_path)
    pages = reader.load()
    unstructured_data = f"""
Document: {input_file_path.rpartition("/")[-1]}
Page 4: {pages[4 - 1].page_content}
---
Page 5: {pages[5 - 1].page_content}
---
Page 6: {pages[6 - 1].page_content}
"""

    reader_prompt = f"""You are an expert working to define methodologies to compute the impact of eligible projects.
Your task is to read the given text, and extract the requirements that must be fulfilled for a project to be eligible to this methodology.
Requirements that does not determine if a project is eligible to the methodology or not should be filtered out.
All requirements must be worded in a way that makes it clear that they are requirements.
You must keep the sources metadata (such as document name and pages) for every information you extract.
Text: {unstructured_data}"""
    summarized_text = LLM.predict(reader_prompt)

    detailed_task = "extract the CCCEV entities that must be fulfilled for a project to be eligible to this methodology"
    cccev = CccevParserTool(LLM).run({
        "detailed_task": detailed_task,
        "unstructured_data": summarized_text,
    })

    cccev_verified1 = CccevVerifierTool(LLM).run({
        "detailed_task": detailed_task,
        "unstructured_data": summarized_text,
        "generated_cccev": cccev,
        "verbose": verbose
    })

    cccev_verified = CccevVerifierTool(LLM).run({
        "detailed_task": detailed_task,
        "unstructured_data": summarized_text,
        "generated_cccev": cccev_verified1,
        "verbose": verbose
    })

    WriteFileTool().run({
        "file_path": f"{DIR_OUTPUT}/result-cccev-parser-{resultCount}.json",
        "text": cccev_verified
    })
    return cccev_verified


if __name__ == '__main__':
    if not os.path.exists(DIR_OUTPUT):
        os.makedirs(DIR_OUTPUT)

    # with open(f"{DIR_INPUT}/{FILE_CCP}") as file:
    #     file_content = file.read()

    # reader = PdfReader(f"{DIR_INPUT}/{FILE_CCP}")
    # pages = reader.load()
    # file_content = '\n'.join(map(lambda page: page.page_content, pages))

    # save_file(DIR_OUTPUT, FILE_CCP, file_content)

    llm_math_chain = LLMMathChain.from_llm(llm=LLM, verbose=True)
    tools = FileManagementToolkit().get_tools() + [
        Pdf2TxtTool(),
        # Pdf2MdTool(),
        Json2GraphTool(LLM, GRAPH),
        CccevParserTool(LLM)
        # CccevVerifierTool(LLM)
    ]
    agent = initialize_agent(
        tools=tools,
        llm=LLM,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        max_iterations=None,
        max_execution_time=None
    )

    # chain = GraphCypherQAChain.from_llm(llm=llm, graph=graph, verbose=True)
    #     chain.run("""
    # Generate a graph with 5 people. Each people has a human name, age and sex. They can be either friend, foe, or have no relation with one another.
    #     """)
    #     response = agent.run(f"""
    # You are an expert working to define requirements and criteria that have to be met by standards and programs.
    # Your task is to read the document at the given path, and to save a copy of it that contains only information about the requirements to be met
    # by carbon-credit programs and potential associated information needed to fully understand what is needed to create a valid carbon-credit program.
    # You must extract all related detailed information, not just the sections titles. Also, you must go through all the document. Do not stop in the middle
    # Do not decide if a section of the document is relevant or not just through its title. You must read its actual content.
    # When you find relevant information, you can save it in a temporary file and then forget it.
    # Then, at the end of the process, you will combine all those temporary files into one by keeping the right order.
    #
    # Path: {DIR_INPUT}"/{FILE_CCP}
    # Directory in which to save the generated document: {DIR_OUTPUT}
    #     """)
    #     response = agent.run(f"""
    # You are an expert working to define requirements and criteria that have to be met by standards and programs.
    # Your task is to read the document at the given path, and to select which section of the document contains the most relevant information about the
    # requirements to be met by carbon-credit programs and potential associated information needed to fully understand what is needed to create a
    # valid carbon-credit program.
    # Do not decide if a section of the document is relevant or not just through its title. You must at least read its introduction.
    # Also, you must go through all the sections before making a final decision. Do not under any circumstances choose a section without having read them all.
    #
    # Path: {DIR_INPUT}/{FILE_CCP}
    #     """)
    #     response = agent.run(f"""
    # You are an expert working to create carbon-crediting standards and programs.
    # Your task is to read the document at the given path, and make a todo list of what you have to do to create a valid carbon-crediting program.
    # Then, from this todo list, generate a json for each task.
    # Each task is a Requirement that has a name, a description (which can be extracted as-is from the document),
    # a list of the needed information to provide in order to meet this requirement,
    # and optionally a list of evidence documents that could prove those information.
    # An Information is defined by a name, a type (string, boolean, number, date or document).
    # Use only page 59.
    #
    # Path: {DIR_INPUT}/{FILE_CCP}
    #     """)
    #     response = agent.run(f"""
    # You are an expert working to create carbon-crediting standards and programs.
    # Your task is to read the document at the given path, and make a very detailed list of documents and information you need to provide in
    # order to create a valid carbon-crediting program.
    # Your results will be audited in order for you program to be validated, so you have to be thorough.
    # Use only pages 58 to 60.
    #
    # Path: {DIR_INPUT}/{FILE_CCP}
    #     """)
    #     response = agent.run(f"""
    # You are a system that automatically generates forms for creating projects according to given specifications.
    # Your task is to read the document at the given path, and make a very detailed list of fields to generate in the form in order to create a
    # valid project that matches the given specifications.
    # You are allowed to use any types of html input for your fields.
    # Use only pages 5 to 6.
    #
    # Path: {DIR_INPUT}/{FILE_VM003}
    #     """)

    prompt = f"""
You are an expert working to define methodologies to compute the impact of eligible projects.
Your task is to read the document at the given path, and extract the CCCEV entities that must be fulfilled for a project to be eligible to this methodology.
Don't forget to always keep information about the source of each and every data you have (document file name, page number)
Use only pages 3, 5 and 6.

For each Requirement and InformationConcept generated, add a `source` field that contains the document file name and page number of the passage that's been used to generate it.
You must then save a fusion of the CCCEV entities extracted from all relevant pages as a prettified json
into a file named `result-cccev-parser-X.json` where X is an incremental number based on existing files in the output directory.

Your final answer should be the path to the resulting file.

Path: {DIR_INPUT}/{FILE_VM003}
Output Directory: {DIR_OUTPUT}
    """

    #     prompt = f"""
    # You are an expert working to define methodologies to compute the impact of eligible projects.
    # Your task is to read the document at the given path, and generate a detailed knowledge graph from it.
    # Use only pages 3, 5 and 6.
    #
    # Path: {DIR_INPUT}/{FILE_VM003}
    #     """

    #     prompt = f"""
    # Your task is to split the given document in multiple files. Each file should contain the full text of one section of the document.
    #
    # Path: {DIR_INPUT}/{FILE_VM003}
    # Output Directory: {DIR_OUTPUT}/test
    #     """
    #     prompt = f"""
    # Write the formulas of page 14 with python operators.
    #
    # Path: {DIR_INPUT}/{FILE_VM003}
    #     """

    #     prompt = f"""
    # Your task is to convert the given json file into a graph.
    # Path: {DIR_OUTPUT}/result-cccev-parser-14.json
    #     """

    # print(prompt)
    # response = agent.run(prompt)
    # print(response)

    # detailed_task = "extract the CCCEV entities that must be fulfilled for a project to be eligible to this methodology"
    # unstructured_data = """This methodology applies to Improved Forest Management (IFM) project activities that involve an extension in rotation age (ERA). This methodology is applicable under the following conditions: 1) Forest management in both baseline and project scenarios involves harvesting techniques such as clear cuts, patch cuts, seed trees, continuous thinning, or group selection practices. 2) Forests which are not subject to timber harvesting, or managed without an objective for earning revenue through timber harvesting in the baseline scenario are not eligible under this methodology. 3) Prior to the first verification event, the project area must meet one of the following conditions: a) Certified by Forest Stewardship Council (FSC); or b) Subject to an easement, or equivalent instrument, recorded against the deed of property that prohibits commercial harvesting for the duration of the crediting period unless later certified by FSC. 4) Project proponents must define the minimum project length in the project description. 5) The project does not encompass managed peat forests, and the proportion of wetlands is not expected to change as part of the project. 6) Project proponents must have a projection of management practices in both with- and without-project scenarios. 7) Where fire is used as part of forest management, fire control measures such as installation of firebreaks or back-burning must be taken to ensure fire does not spread outside the project area — that is, no biomass burning is permitted to occur beyond the project area due to forest management activities. 8) There must be no leakage through activity shifting to other lands owned or managed by project proponents outside the boundary of the project area."""
    # generated_cccev = CccevParserTool(LLM)._run(detailed_task=detailed_task, unstructured_data=unstructured_data)
    # print(CccevVerifierTool(LLM)._run(detailed_task=detailed_task, unstructured_data=unstructured_data, generated_cccev=generated_cccev))

    # data="""VM0003, v1.3\r\nCONTENTS\r\n1 SOURCES ..............................................................................................................3\r\n2 SUMMARY DESCRIPTION OF THE METHODOLOGY ............................................3\r\n3 DEFINITIONS .........................................................................................................3\r\n4 APPLICABILITY CONDITIONS...............................................................................4\r\n5 PROJECT BOUNDARY ..........................................................................................5\r\n5.1 GHG Sources and Sinks.................................................................................................... 5\r\n5.2 Project Area and Eligibility of Land................................................................................. 6\r\n6 BASELINE SCENARIO ...........................................................................................7\r\n6.1 Selected Baseline Approach .......................................................................................... 7\r\n6.2 Preliminary Screening Based on the IFM Project Activity Start Date........................... 7\r\n6.3 Determination of Baseline Scenario ............................................................................... 7\r\n7 ADDITIONALITY..................................................................................................11\r\n8 QUANTIFICATION OF GHG EMISSION REDUCTIONS AND REMOVALS...........11\r\n8.1 Stratification..................................................................................................................... 11\r\n8.2 Baseline Net GHG Removals by Sinks........................................................................... 12\r\n8.3 Stock Changes in the Baseline...................................................................................... 14\r\n8.4 Baseline Emissions............................................................................................................ 15\r\n8.5 Project Net GHG Removals by Sinks............................................................................. 17\r\n8.6 Leakage........................................................................................................................... 34\r\n8.7 Summary of GHG Emission Reductions and/or Removals ......................................... 36\r\n9 MONITORING.....................................................................................................39\r\n9.1 Data and Parameters Available at Validation ........................................................... 39\r\n9.2 Data and Parameters Monitored ................................................................................. 49\r\n9.3 Description of the Monitoring Plan ............................................................................... 54\r\n10 REFERENCES .......................................................................................................56\r\nDOCUMENT HISTORY....................................................................................................59\r\nVM0003, v1.3\r\n4\r\nFirewood \r\nWood harvested and burned for personal use or limited sale as a heating fuel in the immediate \r\nvicinity\r\nGroup selection \r\nA variant of clear cut with groups of trees being left for wildlife habitat, wind firmness, soil \r\nretention or other silvicultural goals\r\nLogging slash \r\nBranches, other dead wood residues and foliage left on the forest floor after timber removal\r\nPatch cut \r\nA clear cut on a small area (less than one hectare) \r\nSanitation removal \r\nThe intentional removal of trees to prevent disease or correct a natural disturbance\r\nSeed tree \r\nA variant of clear cut with limited mature trees being left to provide seeds for regeneration\r\nTree \r\nA perennial woody plant with a diameter at breast height greater than 5 cm and a height \r\ngreater than 1.3 m\r\n4 APPLICABILITY CONDITIONS\r\nThis methodology applies to Improved Forest Management (IFM) project activities that involve \r\nan extension in rotation age (ERA).\r\nThis methodology is applicable under the following conditions:\r\n1) Forest management in both baseline and project scenarios involves harvesting \r\ntechniques such as clear cuts, patch cuts, seed trees, continuous thinning, or group \r\nselection practices.\r\n2) Forests which are not subject to timber harvesting, or managed without an objective for \r\nearning revenue through timber harvesting in the baseline scenario are not eligible \r\nunder this methodology.\r\n3) Prior to the first verification event, the project area must meet one of the following \r\nconditions: \r\na) Certified by Forest Stewardship Council (FSC); or \r\nb) Subject to an easement, or equivalent instrument, recorded against the deed of \r\nproperty that prohibits commercial harvesting for the duration of the crediting \r\nperiod unless later certified by FSC.\r\n4) Project proponents must define the minimum project length in the project description.\r\nVM0003, v1.3\r\n5\r\n5) The project does not encompass managed peat forests, and the proportion of wetlands \r\nis not expected to change as part of the project.\r\n6) Project proponents must have a projection of management practices in both with- and \r\nwithout-project scenarios.\r\n7) Where fire is used as part of forest management, fire control measures such as \r\ninstallation of firebreaks or back-burning must be taken to ensure fire does not spread \r\noutside the project area — that is, no biomass burning is permitted to occur beyond the \r\nproject area due to forest management activities.\r\n8) There must be no leakage through activity shifting to other lands owned or managed by \r\nproject proponents outside the boundary of the project area.\r\n5 PROJECT BOUNDARY\r\n5.1 GHG Sources and Sinks\r\nThe carbon pools included in or excluded from the project boundary are shown in Table 1. \r\nTable 1: Selected Carbon Pools\r\nCarbon Pools Selected? Justification/Explanation \r\nAbove-ground \r\nbiomass\r\nYes Major carbon pool subjected to the project activity\r\nBelow-ground \r\nbiomass\r\nYes Below-ground biomass stock is expected to increase due to \r\nimplementation of the IFM project activity. Below-ground \r\nbiomass subsequent to harvest is not assessed based on the \r\nconservative assumption of immediate emission.\r\nDead wood Conditional Dead wood stocks may be conservatively excluded unless the \r\nproject scenario produces greater levels of slash than the \r\nbaseline and slash is burned as part of forest management. \r\nWhere slash produced in the project scenario is left in the \r\nforest to become part of the dead wood pool, dead wood may \r\nbe conservatively excluded. Alternatively, project proponents \r\nmay elect to include the pool (where included, the pool must be \r\nestimated in both the baseline and with-project scenarios) as \r\nlong as the dead wood pool represents less than 50 percent of \r\ntotal carbon volume on the site in any given modeled year.\r\nLitter No Changes in the litter pool will be de minimis as a result of \r\nrotation extension."""
    # data="""VM0003, v1.3\r\n4\r\nFirewood \r\nWood harvested and burned for personal use or limited sale as a heating fuel in the immediate \r\nvicinity\r\nGroup selection \r\nA variant of clear cut with groups of trees being left for wildlife habitat, wind firmness, soil \r\nretention or other silvicultural goals\r\nLogging slash \r\nBranches, other dead wood residues and foliage left on the forest floor after timber removal\r\nPatch cut \r\nA clear cut on a small area (less than one hectare) \r\nSanitation removal \r\nThe intentional removal of trees to prevent disease or correct a natural disturbance\r\nSeed tree \r\nA variant of clear cut with limited mature trees being left to provide seeds for regeneration\r\nTree \r\nA perennial woody plant with a diameter at breast height greater than 5 cm and a height \r\ngreater than 1.3 m\r\n4 APPLICABILITY CONDITIONS\r\nThis methodology applies to Improved Forest Management (IFM) project activities that involve \r\nan extension in rotation age (ERA).\r\nThis methodology is applicable under the following conditions:\r\n1) Forest management in both baseline and project scenarios involves harvesting \r\ntechniques such as clear cuts, patch cuts, seed trees, continuous thinning, or group \r\nselection practices.\r\n2) Forests which are not subject to timber harvesting, or managed without an objective for \r\nearning revenue through timber harvesting in the baseline scenario are not eligible \r\nunder this methodology.\r\n3) Prior to the first verification event, the project area must meet one of the following \r\nconditions: \r\na) Certified by Forest Stewardship Council (FSC); or \r\nb) Subject to an easement, or equivalent instrument, recorded against the deed of \r\nproperty that prohibits commercial harvesting for the duration of the crediting \r\nperiod unless later certified by FSC.\r\n4) Project proponents must define the minimum project length in the project description.\r\nVM0003, v1.3\r\n5\r\n5) The project does not encompass managed peat forests, and the proportion of wetlands \r\nis not expected to change as part of the project.\r\n6) Project proponents must have a projection of management practices in both with- and \r\nwithout-project scenarios.\r\n7) Where fire is used as part of forest management, fire control measures such as \r\ninstallation of firebreaks or back-burning must be taken to ensure fire does not spread \r\noutside the project area — that is, no biomass burning is permitted to occur beyond the \r\nproject area due to forest management activities.\r\n8) There must be no leakage through activity shifting to other lands owned or managed by \r\nproject proponents outside the boundary of the project area.\r\n5 PROJECT BOUNDARY\r\n5.1 GHG Sources and Sinks\r\nThe carbon pools included in or excluded from the project boundary are shown in Table 1. \r\nTable 1: Selected Carbon Pools\r\nCarbon Pools Selected? Justification/Explanation \r\nAbove-ground \r\nbiomass\r\nYes Major carbon pool subjected to the project activity\r\nBelow-ground \r\nbiomass\r\nYes Below-ground biomass stock is expected to increase due to \r\nimplementation of the IFM project activity. Below-ground \r\nbiomass subsequent to harvest is not assessed based on the \r\nconservative assumption of immediate emission.\r\nDead wood Conditional Dead wood stocks may be conservatively excluded unless the \r\nproject scenario produces greater levels of slash than the \r\nbaseline and slash is burned as part of forest management. \r\nWhere slash produced in the project scenario is left in the \r\nforest to become part of the dead wood pool, dead wood may \r\nbe conservatively excluded. Alternatively, project proponents \r\nmay elect to include the pool (where included, the pool must be \r\nestimated in both the baseline and with-project scenarios) as \r\nlong as the dead wood pool represents less than 50 percent of \r\ntotal carbon volume on the site in any given modeled year.\r\nLitter No Changes in the litter pool will be de minimis as a result of \r\nrotation extension."""

    iteration = 30
    cccev = """{"dataUnits": [{"identifier": "xsdString", "name": "XSDString", "description": "Any string of characters", "type": "STRING", "status": "EXISTS"}, {"identifier": "xsdBoolean", "name": "XSDBoolean", "description": "True or false", "type": "BOOLEAN", "status": "EXISTS"}], "informationConcepts": [{"identifier": "forestManagementPractices", "name": "Forest Management Practices", "description": "The practices used to manage a forest", "unit": "xsdString", "question": "What are the forest management practices used?", "status": "EXISTS", "source": "VM0003-IFM-Through-Extension-Of-Rotation-Age-v1.3.pdf, page 4"}, {"identifier": "projectActivity", "name": "Project Activity", "description": "The activity that the project involves", "unit": "xsdString", "question": "What activity does the project involve?", "status": "EXISTS", "source": "VM0003-IFM-Through-Extension-Of-Rotation-Age-v1.3.pdf, page 5"}, {"identifier": "harvestingTechniques", "name": "Harvesting Techniques", "description": "The techniques used for harvesting in the forest", "unit": "xsdString", "question": "What harvesting techniques are used in the forest?", "status": "EXISTS", "source": "VM0003-IFM-Through-Extension-Of-Rotation-Age-v1.3.pdf, page 5"}, {"identifier": "projectAreaCertification", "name": "Project Area Certification", "description": "The certification of the project area", "unit": "xsdString", "question": "What is the certification of the project area?", "status": "EXISTS", "source": "VM0003-IFM-Through-Extension-Of-Rotation-Age-v1.3.pdf, page 5"}, {"identifier": "projectLength", "name": "Project Length", "description": "The minimum length of the project", "unit": "xsdString", "question": "What is the minimum length of the project?", "status": "EXISTS", "source": "VM0003-IFM-Through-Extension-Of-Rotation-Age-v1.3.pdf, page 5"}, {"identifier": "managementPracticesProjection", "name": "Management Practices Projection", "description": "The projection of management practices in both with- and without-project scenarios", "unit": "xsdString", "question": "What is the projection of management practices in both with- and without-project scenarios?", "status": "EXISTS", "source": "VM0003-IFM-Through-Extension-Of-Rotation-Age-v1.3.pdf, page 6"}, {"identifier": "fireControlMeasures", "name": "Fire Control Measures", "description": "The measures taken to control fire in the forest", "unit": "xsdString", "question": "What measures are taken to control fire in the forest?", "status": "EXISTS", "source": "VM0003-IFM-Through-Extension-Of-Rotation-Age-v1.3.pdf, page 6"}, {"identifier": "leakagePrevention", "name": "Leakage Prevention", "description": "The measures taken to prevent leakage to other lands", "unit": "xsdBoolean", "question": "Are there measures taken to prevent leakage to other lands?", "status": "EXISTS", "source": "VM0003-IFM-Through-Extension-Of-Rotation-Age-v1.3.pdf, page 6"}], "requirements": [{"identifier": "improvedForestManagement", "name": "Improved Forest Management", "kind": "CRITERION", "description": "The project must involve improving forest management practices to increase the carbon stock on land by extending the rotation age of a forest or patch of forest before harvesting.", "hasRequirement": [], "hasConcepts": ["forestManagementPractices"], "status": "CREATED", "source": "VM0003-IFM-Through-Extension-Of-Rotation-Age-v1.3.pdf, page 4"}, {"identifier": "projectActivityRequirement", "name": "Project Activity Requirement", "kind": "CRITERION", "description": "The project must be an Improved Forest Management (IFM) project activity that involves an extension in rotation age (ERA).", "hasRequirement": [], "hasConcepts": ["projectActivity"], "status": "CREATED", "source": "VM0003-IFM-Through-Extension-Of-Rotation-Age-v1.3.pdf, page 5"}, {"identifier": "harvestingTechniquesRequirement", "name": "Harvesting Techniques Requirement", "kind": "CRITERION", "description": "Forest management in both baseline and project scenarios must involve harvesting techniques such as clear cuts, patch cuts, seed trees, continuous thinning, or group selection practices.", "hasRequirement": [], "hasConcepts": ["harvestingTechniques"], "status": "CREATED", "source": "VM0003-IFM-Through-Extension-Of-Rotation-Age-v1.3.pdf, page 5"}, {"identifier": "projectAreaCertificationRequirement", "name": "Project Area Certification Requirement", "kind": "CRITERION", "description": "Prior to the first verification event, the project area must be either certified by Forest Stewardship Council (FSC); or subject to an easement, or equivalent instrument, recorded against the deed of property that prohibits commercial harvesting for the duration of the crediting period unless later certified by FSC.", "hasRequirement": [], "hasConcepts": ["projectAreaCertification"], "status": "CREATED", "source": "VM0003-IFM-Through-Extension-Of-Rotation-Age-v1.3.pdf, page 5"}, {"identifier": "projectLengthRequirement", "name": "Project Length Requirement", "kind": "CRITERION", "description": "Project proponents must define the minimum project length in the project description.", "hasRequirement": [], "hasConcepts": ["projectLength"], "status": "CREATED", "source": "VM0003-IFM-Through-Extension-Of-Rotation-Age-v1.3.pdf, page 5"}, {"identifier": "managementPracticesProjectionRequirement", "name": "Management Practices Projection Requirement", "kind": "CRITERION", "description": "Project proponents must have a projection of management practices in both with- and without-project scenarios.", "hasRequirement": [], "hasConcepts": ["managementPracticesProjection"], "status": "CREATED", "source": "VM0003-IFM-Through-Extension-Of-Rotation-Age-v1.3.pdf, page 6"}, {"identifier": "fireControlMeasuresRequirement", "name": "Fire Control Measures Requirement", "kind": "CRITERION", "description": "Where fire is used as part of forest management, fire control measures such as installation of firebreaks or back-burning must be taken to ensure fire does not spread outside the project area \u2014 that is, no biomass burning is permitted to occur beyond the project area due to forest management activities.", "hasRequirement": [], "hasConcepts": ["fireControlMeasures"], "status": "CREATED", "source": "VM0003-IFM-Through-Extension-Of-Rotation-Age-v1.3.pdf, page 6"}, {"identifier": "leakagePreventionRequirement", "name": "Leakage Prevention Requirement", "kind": "CRITERION", "description": "There must be no leakage through activity shifting to other lands owned or managed by project proponents outside the boundary of the project area.", "hasRequirement": [], "hasConcepts": ["leakagePrevention"], "status": "CREATED", "source": "VM0003-IFM-Through-Extension-Of-Rotation-Age-v1.3.pdf, page 6"}], "document": {"name": "VM0003-IFM-Through-Extension-Of-Rotation-Age-v1.3.pdf"}}"""
    # cccev = extract_cccev(f"{DIR_INPUT}/{FILE_VM003}", iteration, verbose=False)
    GRAPH.query("match (a) detach delete a")
    Json2GraphTool(LLM, GRAPH).run({'json_data': cccev})
    init_graph()

    # init_graph()
    # question = "What requirements must be met by a forest management project? Give me only short names"
    # question = "Detail the information I need to provide and their associated constraints"
    # question = "Give me the requirements that must be met, grouped by type of data"
    # question = "Which requirements use the same type of data as the Leakage Prevention Requirement?"
    # question = """Which requirements can I answer to with the following text? Every bit of information can be important and must be checked.
    # Text: This 3-year project does not use fire as part of its forest management.
    # """
    # question = """Find the questions of the information concept related to requirements on forest management projects, and find the answers in the following text.
    # Your final answer should contain every question that can be answered as a bullet list, along with their answer.
    # Text: This 3-year project does not use fire as part of its forest management.
    # """
    question = """What requirements have been extracted from the document VM0003-IFM-Through-Extension-Of-Rotation-Age-v1.3.pdf?"""
    question_graph(question)
