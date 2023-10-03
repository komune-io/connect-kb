import json
from enum import Enum
from typing import List

from langchain import PromptTemplate, LLMChain, LLMCheckerChain
from langchain.chat_models.base import BaseChatModel
from langchain.output_parsers import PydanticOutputParser
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool
from langchain_experimental.smart_llm import SmartLLMChain
from pydantic import validator

DEFINITION_REQUIREMENT = """
A Requirement is a condition or prerequisite that is to be proven by Evidence.
Requirement is a generic class representing any type of prerequisite that may be desired, needed or imposed as an obligation. 
To illustrate the notion: the European Directive on services in the internal market defines requirement as any obligation, prohibition, condition 
or limit provided for in the laws, regulations or administrative provisions of the Member States or in consequence of case-law, 
administrative practice, the rules of professional bodies, or the collective rules of professional associations or other professional organisations, 
adopted in the exercise of their legal autonomy.
"""

DEFINITION_INFORMATION_CONCEPT = """
Piece of information that an Evidence provides or the Requirement needs. This can be seen as one field of a form that a user will fill.
As the field of a form, the name of the concept would be the label of the form, and the DataUnit would define what type of field it is (number, date, ...) along with an optional suffix for the unit notation if relevant.
Unformatted example of the field: 
Label: Age of the person
DataUnit: 
  - name: age
  - description: number of years since the beginning of something
  - notation: years
  - type: number
"""

DEFINITION_DATA_UNIT = """
Represents a unit of data, such as a physical measurement, date, or number.
A DataUnit should only describe measurement units such as meter, ohms or hectare, or simple types such as boolean, string or date 
that will be used by information concepts.
Here are some already existing very basic data units that you should use:
[{
    identifier: "xsdDate",
    name: "XSDDate",
    description: "A date",
    type: "DATE"
}, {
    identifier: "xsdDouble",
    name: "XSDDouble",
    description: "Any real number",
    type: "NUMBER"
}, {
    identifier: "xsdInt",
    name: "XSDInt",
    description: "Any integer number",
    type: "NUMBER"
}, {
    identifier: "xsdString",
    name: "XSDString",
    description: "Any string of characters",
    type: "STRING"
}, {
    identifier: "xsdBoolean",
    name: "XSDBoolean",
    description: "True or false",
    type: "BOOLEAN"
}]
You can create also other units if the need arises, but only for units of measurement of type number. Here is an example of what you could define:
{
    identifier: "meter",
    name: "Meter",
    description: "A unit of length in the International System of Units (SI)",
    notation: "m"
    type: "NUMBER"
}
A DataUnit should be NEVER contain information about a context. It is always generic. For example, instead of:
{ "identifier": "wetlandProportion", "name": "Wetland Proportion", "description": "The proportion of a forest that is wetland", "type": "NUMBER" }
It should be:
{ "identifier": "proportion", "name": "Proportion", "description": "A proportion of something in relation to another", "type": "NUMBER" }
You can see that the context related to the forest is irrelevant to the measurement unit. Apply that principle for every data unit you want to defined.
"""


class Requirement(BaseModel):
    identifier: str = Field(
        default=...,
        title="identifier",
        description="Unique human-readable identifier of the requirement. It should very shortly summarize the description. Shall be used whenever this requirement is mentioned. Should use camelCase",
        example="highCloudsAltitude"
    )
    name: str = Field(
        default=...,
        title="name",
        description="Human-readable short and concise name of the requirement. It should summarize the description.",
        example="High Clouds Altitude"
    )
    description: str = Field(
        default=...,
        title="description",
        description="Detailed and explicit description of the requirement.",
        example="The high clouds should be below 5000m."
    )
    hasRequirement: List[str] = Field(
        default=[],
        title="hasRequirement",
        description="Identifiers of the sub-requirements that must be fulfilled for the requirement to be validated.",
    )
    hasConcepts: List[str] = Field(
        default=...,
        title="hasConcepts",
        description="Identifiers of the information concepts used by the requirement",
    )
    source: str = Field(
        default=...,
        title="source",
        description="The source from which information about this information concept has been generated. It should at least contain the name of the document and the page number."
    )


class InformationConcept(BaseModel):
    identifier: str = Field(
        default=...,
        title="identifier",
        description="Unique human-readable identifier of the information concept. It should be very short and use camelCase. Shall be used whenever this information concept is mentioned",
        example="altitudeOfClouds"
    )
    name: str = Field(
        default=...,
        title="name",
        description="Human-readable short and concise name of the information concept. It should concisely summarize the description.",
        example="Altitude of the clouds"
    )
    description: str = Field(
        default=...,
        title="description",
        description="Detailed and explicit description of the information concept.",
        example="Represents the height above sea level of the high clouds"
    )
    unit: str = Field(
        default=...,
        title="unit",
        description="The identifier of the data unit used for this information concept. This should always be refer to a DataUnit. You can either use an existing one or define one yourself, but try to avoid duplications by reusing already defined units when possible.",
        example="meter"
    )
    question: str = Field(
        default=...,
        title="question",
        description="Question to ask the user that will be tasked with fill in this information in a form",
        example="What is the altitude of the high clouds?"
    )
    source: str = Field(
        default=...,
        title="source",
        description="The source from which information about this information concept has been generated. It should at least contain the name of the document and the page number."
    )

    @validator("question")
    def question_ends_with_question_mark(cls, field):
        if field[-1] != "?":
            raise ValueError("Badly formed question!")
        return field


class DataUnitType(Enum):
    BOOLEAN = "BOOLEAN",
    DATE = "DATE",
    NUMBER = "NUMBER",
    STRING = "STRING"


class DataUnit(BaseModel):
    identifier: str = Field(
        default=...,
        title="identifier",
        description="Unique identifier of the data unit. It should be very short and use camelCase. Shall be used whenever this data unit is mentioned",
        example="meter"
    )
    name: str = Field(
        default=...,
        title="name",
        description="Human-readable short and concise name of the data unit",
        example="Meter"
    )
    description: str = Field(
        default=...,
        title="description",
        description="Detailed and explicit description of the data unit",
        example="A unit of length in the International System of Units (SI)"
    )
    notation: str = Field(
        default="",
        title="notation",
        description="The short notation for this data unit, if applicable",
        example="m"
    )
    type: DataUnitType = Field(
        default=...,
        title="type",
        description="The type of data used for this data unit",
        example="NUMBER"
    )


class Document(BaseModel):
    name: str = Field(
        default=...,
        title="name",
        description="Name of the document"
    )
    pages: List[str] = Field(
        default=...,
        title="pages",
        description="available information about each page of the document"
    )


class CCCEV(BaseModel):
    dataUnits: List[DataUnit] = Field(
        default=...,
        title="dataUnits",
        description=DEFINITION_DATA_UNIT
    )
    informationConcepts: List[InformationConcept] = Field(
        default=...,
        title="informationConcepts",
        description=DEFINITION_INFORMATION_CONCEPT
    )
    requirements: List[Requirement] = Field(
        default=...,
        title="requirements",
        description=DEFINITION_REQUIREMENT
    )
    document: str = Field(
        default=...,
        title="document",
        description="The source document from which the other entities have been extracted"
    )


CccevParser = PydanticOutputParser(pydantic_object=CCCEV)


PROMPT_CCCEV_TASK = """read the given unstructured data, and list all requirements that must be met as well as the information concepts they use.
Additional details: {detailed_task}

{format_instructions}

Unstructured data: {unstructured_data}"""

PROMPT_CCCEV_PARSER = f"""
You are an expert in the specifications CCCEV (which will be described below).
Your task is to {PROMPT_CCCEV_TASK}
    """

PROMPT_CCCEV_VERIFIER = f"""
You are an expert in the specifications CCCEV (which will be described below).
Your task is to verify the work of one of your fellow experts.
Here was the task of the other expert:
{PROMPT_CCCEV_TASK}

And here is the produced result:
{{generated_cccev}}

If you identify some points of improvements, apply them and return a new version of the result. 
Points of improvements can include but are not limited to:
  - relevance of the data unit and information concepts against the requirements that use them
  - the description of a requirement makes it clear that it is a requirement (with words such as "must" for example)
  - there are no unused or duplicated information concepts or data units
  - all requirements in the unstructured data are represented in the result
  - the data units are all context-agnostic, whereas the information concepts should be put in context
  - each information concept uses only one data unit
  - any other improvements you can find

The final answer should be a json, and only a json without further formatting (i.e. do NOT wrap with ```json ```) with:
  - `improvements` that is a list of the improvements points you found
  - `cccev` that is the final improved result
    """


class CccevParserTool(BaseTool):
    name = "CccevParser"
    description = "useful for when you need to transform unstructured data to CCCEV (Core Criterion and Core Evidence Vocabulary)"

    llm: BaseChatModel = None

    def __init__(self, llm: BaseChatModel, **data):
        super().__init__(**data)
        self.llm = llm

    def _run(self, unstructured_data: str, detailed_task: str) -> str:

        chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template=PROMPT_CCCEV_PARSER,
                input_variables=["detailed_task", "format_instructions", "unstructured_data"]
            ),
            verbose=True
        )
        generated_cccev = chain.run({
            "detailed_task": detailed_task,
            "format_instructions": CccevParser.get_format_instructions(),
            "unstructured_data": unstructured_data
        })

        return CccevVerifierTool(self.llm).run({
            "detailed_task": detailed_task,
            "unstructured_data": unstructured_data,
            "generated_cccev": generated_cccev
        })

        # chain = LLMChain(
        #     llm=self.llm,
        #     prompt=PromptTemplate(
        #         template=PROMPT_CCCEV_VERIFIER,
        #         input_variables=["detailed_task", "format_instructions", "unstructured_data", "generated_cccev"]
        #     ),
        #     verbose=True
        # )
        # return chain.run({
        #     "detailed_task": detailed_task,
        #     "format_instructions": CccevParser.get_format_instructions(),
        #     "unstructured_data": unstructured_data,
        #     "generated_cccev": generated_cccev
        # })


class CccevVerifierTool(BaseTool):
    name = "CccevVerifier"
    description = "useful for when you need to check the validity of a generated CCCEV structure and clean it"

    llm: BaseChatModel = None

    def __init__(self, llm: BaseChatModel, **data):
        super().__init__(**data)
        self.llm = llm

    def _run(self, generated_cccev: str, unstructured_data: str, detailed_task: str) -> str:
        chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template=PROMPT_CCCEV_VERIFIER,
                input_variables=["detailed_task", "format_instructions", "unstructured_data", "generated_cccev"]
            ),
            verbose=True
        )
        result_str = chain.run({
            "detailed_task": detailed_task,
            "format_instructions": CccevParser.get_format_instructions(),
            "unstructured_data": unstructured_data,
            "generated_cccev": generated_cccev
        })
        print(result_str)
        result = json.loads(result_str)
        return json.dumps(result["cccev"])
