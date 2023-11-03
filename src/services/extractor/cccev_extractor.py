import json
from abc import abstractproperty

from langchain import PromptTemplate, LLMChain
from langchain.chat_models.base import BaseChatModel
from langchain.output_parsers import PydanticOutputParser

from .cccev_builder import build_cccev
from ..model.cccev import Cccev

NO_REQUIREMENT_FOUND = "No requirement found."

PROMPT_REQUIREMENT_SUMMARIZER = f"""{{task_context}}.
All requirements must be worded in a way that makes it clear that they are requirements.
You must keep the sources metadata (such as document name and pages) for every information you extract. 
It should be easy to tell which extracted information come from where.
If you don't find any relevant requirement, just say: {NO_REQUIREMENT_FOUND}
Text: {{text}}"""

PROMPT_EXTRACT_FIELDS = """Your task is to read the given list of requirements and extract information from it.
From the input you must describe a form that will gather all the necessary data in order to evaluate those requirements.
For example, if a requirement says "There must be at least 3 bananas", then the form will need a field "Number of bananas" of type number.
The final result should be a json array of fields described by:
 - `label`: the displayed label of the field
 - `type`: the type of field (use one of the html input field types: text, checkbox, number, date, select)
 - `page`: the page number from which this field has been extracted
 - `options`: list of available options (add it only if relevant)
 - `multiple`: true if multiple options can be selected (add it only if relevant)
 
The resulting form should guide the user as much as possible and be very easy to fulfill.
Don't hesitate to use select fields for choice selection, checkbox for simple booleans, etc.
Don't hesitate to break down complex information into multiple simple fields if needed.

Requirements: {input}"""

PROMPT_EXTRACT_REQUIREMENTS = """Your task is to read the given list of requirements and associate each one of them with the relevant form fields.
You will receive two sets of data as input:
 - Requirements: a list of requirements that must be met
 - Form: the fields of a form that collects data relevant to the previous requirements
  
The final result should be a json array of requirements described by:
 - `name`: a short meaningful name for the requirement
 - `description`: a detailed description of the requirement
 - `fields`: an array of field identifiers affected by the requirement
 - `expression`: an SpEL (Spring Expression Language) expression that returns a boolean and where the identifiers of the fields are used as variables
 - `page`: the page number from which this requirement has been extracted

For example, if a requirement says "There must be at least 3 bananas", then it must be associated with the field "Number of bananas".

The final result should contain each and every one of the requirements given in the input.

Requirements: {input}

Form: {form}"""


class CccevExtractor:
    llm: BaseChatModel = None
    task_context: str

    def __init__(self, llm: BaseChatModel, task_context: str, **data):
        super().__init__(**data)
        self.llm = llm
        self.task_context = task_context

    def extract(self, text: str, session_id: str, debug: bool = False) -> Cccev:
        summarized_text = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template=PROMPT_REQUIREMENT_SUMMARIZER,
                input_variables=["task_context", "text"]
            ),
            verbose=True
        ).run({"task_context": self.task_context,"text": text})

        if summarized_text == NO_REQUIREMENT_FOUND:
            print(summarized_text)
            return Cccev.empty()

        fields = self._extract_fields(summarized_text, session_id, debug)
        requirements = self._extract_requirements(summarized_text, json.dumps(fields), session_id, debug)
        return build_cccev(fields, requirements)

    def _extract_fields(self, text: str, session_id: str, debug: bool) -> list[dict[str]]:
        fields_raw = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template=PROMPT_EXTRACT_FIELDS,
                input_variables=["input"]
            ),
            verbose=True
        ).run({"input": text})

        if debug:
            print(fields_raw)

        fields_raw_json = fields_raw[fields_raw.find("["):fields_raw.rfind("]") + 1]
        fields: list[dict[str]] = json.loads(fields_raw_json)
        counter = 0
        for field in fields:
            field["identifier"] = f"{session_id}_field_{counter}"
            counter += 1

        return fields

    def _extract_requirements(self, text: str, fields: str, session_id: str, debug: bool) -> list[dict[str]]:
        requirements_raw = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template=PROMPT_EXTRACT_REQUIREMENTS,
                input_variables=["input", "form"]
            ),
            verbose=True
        ).run({"input": text, "form": fields})

        if debug:
            print(requirements_raw)

        requirements_raw_json = requirements_raw[requirements_raw.find("["):requirements_raw.rfind("]") + 1]
        requirements: list[dict[str]] = json.loads(requirements_raw_json)
        counter = 0
        for requirement in requirements:
            requirement["identifier"] = f"{session_id}_requirement_{counter}"
            counter += 1

        return requirements


class MethodologyEligibilityCccevExtractor(CccevExtractor):
    def __init__(self, llm: BaseChatModel, **data):
        task_context = """You are an expert working to define methodologies to compute the impact of eligible projects.
Your task is to read the given text, and extract the requirements that must be fulfilled for a project to be eligible to this methodology.
Requirements that does not determine if a project is eligible to the methodology or not should be filtered out"""
        super().__init__(llm, task_context, **data)
