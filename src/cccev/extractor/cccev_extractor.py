import json
from abc import abstractproperty

from langchain import PromptTemplate, LLMChain
from langchain.chat_models.base import BaseChatModel
from langchain.output_parsers import PydanticOutputParser

from cccev.model.cccev import CCCEV

CccevParser = PydanticOutputParser(pydantic_object=CCCEV)

NO_REQUIREMENT_FOUND = "No requirement found."

PROMPT_REQUIREMENT_SUMMARIZER = f"""You are an expert working to define methodologies to compute the impact of eligible projects.
Your task is to read the given text, and extract the requirements that must be fulfilled for a project to be eligible to this methodology.
Requirements that does not determine if a project is eligible to the methodology or not should be filtered out.
All requirements must be worded in a way that makes it clear that they are requirements.
You must keep the sources metadata (such as document name and pages) for every information you extract. 
It should be easy to tell which extracted information come from where.
If you don't find any relevant requirement, just say: {NO_REQUIREMENT_FOUND}
Text: {{text}}"""

PROMPT_CCCEV_TASK = """read the given input and parse it.
The input data will contain a list of pre-filtered requirements, so you should extract every bit of information you can from it.
All requirements listed in the input should be present in the output.
Additional details: {detailed_task}

{format_instructions}

All identifiers must be prefixed by "{session_id}"
The final answer should be the resulting json, and only the json without further formatting (i.e. do NOT wrap with ```json ```)

Input: {unstructured_data}"""

PROMPT_CCCEV_PARSER = f"""
You are an expert in the specifications CCCEV (which will be described below).
Your task is to {PROMPT_CCCEV_TASK}
    """

PROMPT_CCCEV_VERIFIER_BASE = f"""
You are an expert in the specifications CCCEV (which will be described below).
Your task is to verify the work of one of your fellow experts.
Here was the task of the other expert:
{PROMPT_CCCEV_TASK}

And here is the produced result:
{{generated_cccev}}

If you identify some points of improvements, apply them and return a new version of the result. 
Points of improvements can include but are not limited to:
  - all requirements in the input are represented in the result
  - relevance of the data unit and information concepts against the requirements that use them
  - the description of a requirement makes it clear that it is a requirement (with words such as "must" for example)
  - there are no unused or duplicated information concepts or data units
  - the data units are all context-agnostic, whereas the information concepts should be put in context
  - each information concept uses only one data unit
  - all requirements must use at least one information concept that is relevant to it
  - any other improvements you can find
    """
PROMPT_CCCEV_VERIFIER = f"""{PROMPT_CCCEV_VERIFIER_BASE}
The final answer should be the resulting json, and only the json without further formatting (i.e. do NOT wrap with ```json ```)
"""
PROMPT_CCCEV_VERIFIER_DEBUG = f"""{PROMPT_CCCEV_VERIFIER_BASE}
The final answer should be a json, and only a json without further formatting (i.e. do NOT wrap with ```json ```) with:
  - `improvements` that is a list of the improvements points you found
  - `cccev` that is the final improved result
"""


class Object(object):
    def __init__(self, dict_):
        self.__dict__.update(dict_)


def json2object(jsonStr: str) -> Object:
    return json.loads(jsonStr, object_hook=Object)


class CccevExtractor:
    llm: BaseChatModel = None
    task_details: str

    def __init__(self, llm: BaseChatModel, task_details: str, **data):
        super().__init__(**data)
        self.llm = llm
        self.task_details = task_details

    def extract(self, text: str, session_id: str, debug: bool = False) -> CCCEV:
        summarized_text = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template=PROMPT_REQUIREMENT_SUMMARIZER,
                input_variables=["text"]
            ),
            verbose=True
        ).run({"text": text})

        if summarized_text == NO_REQUIREMENT_FOUND:
            print(summarized_text)
            return CCCEV.empty()

        chain_parser = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template=PROMPT_CCCEV_PARSER,
                input_variables=["detailed_task", "format_instructions", "unstructured_data", "session_id"]
            ),
            verbose=True
        )

        params = {
            "detailed_task": self.task_details,
            "format_instructions": CccevParser.get_format_instructions(),
            "unstructured_data": summarized_text,
            "session_id": session_id
        }

        cccev1 = chain_parser.run(params)

        print(cccev1)
        if len(json.loads(cccev1)["requirements"]) == 0:
            print(summarized_text)
            return CCCEV.empty()

        if debug:
            prompt = PROMPT_CCCEV_VERIFIER_DEBUG
        else:
            prompt = PROMPT_CCCEV_VERIFIER

        chain_verifier = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template=prompt,
                input_variables=["detailed_task", "format_instructions", "unstructured_data", "session_id", "generated_cccev"]
            ),
            verbose=True
        )

        params["generated_cccev"] = cccev1
        cccev2 = chain_verifier.run(params)
        if debug:
            print(cccev2)

        params["generated_cccev"] = cccev2
        result_str = chain_verifier.run(params)

        print(result_str)

        if debug:
            result = json.loads(result_str)
            return json2object(json.dumps(result["cccev"]))
        else:
            return json2object(result_str)


class MethodologyEligibilityCccevExtractor(CccevExtractor):
    def __init__(self, llm: BaseChatModel, **data):
        task_details = "extract the CCCEV entities that must be fulfilled for a project to be eligible to this methodology"
        super().__init__(llm, task_details, **data)
