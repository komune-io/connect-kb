import os

from PyPDF2 import PdfReader
from langchain import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

condense_question_template = """
Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language. 
Include the follow up instructions in the standalone question.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_question_template)

qa_template = """Your name is Tmate. 
A person will ask you a question and you will provide a helpful answer. 
Write the answer in the same language as the question. 
If you don't know the answer, just politely and concisely say that you don't know. Don't try to make up an answer. 
Use the following documents to answer the question:

{context}

Question: {question}
Helpful answer:
"""
QA_PROMPT = PromptTemplate(template=qa_template, input_variables=["question", "context"])


def get_chat_model_name():
    model_name = os.getenv('OPENAI_MODEL_NAME_CHAT')
    if model_name is None:
        model_name = 'gpt-3.5-turbo-16k'

    print(f"chat: {model_name}")
    return model_name


def get_extraction_model_name():
    model_name = os.getenv('OPENAI_MODEL_NAME_EXTRACTION')
    if model_name is None:
        model_name = 'gpt-4'

    print(f"extraction: {model_name}")
    return model_name


def get_pdf_text(pdf_docs):
    text = ""
    pdf_reader = PdfReader(pdf_docs)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text
