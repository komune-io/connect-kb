import os

from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain.schema import HumanMessage, AIMessage
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
If you don't know the answer, just politely say that you don't know. Don't try to make up an answer. 
Use the following context to answer the question:

{context}

Question: {question}
Helpful answer:
"""
QA_PROMPT = PromptTemplate(template=qa_template, input_variables=["question", "context"])


def get_model_name():
    load_dotenv()
    model_name = os.getenv('OPENAI_MODEL_NAME')
    if model_name is None:
        model_name = 'gpt-3.5-turbo'

    print(model_name)

    return model_name


def get_pdf_text(pdf_docs):
    text = ""
    pdf_reader = PdfReader(pdf_docs)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def get_vectorstore():
    load_dotenv()
    persist_directory = os.getenv('CHROMADB_PERSIST_DIRECTORY')

    return Chroma(
        persist_directory=persist_directory,
        embedding_function=OpenAIEmbeddings()
    )


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_conversation_chain(vectorstore: Chroma, metadata_filter, messages):
    model_name = get_model_name()
    llm = ChatOpenAI(model_name=model_name, temperature=0)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    memory.chat_memory = parse_chat_history(messages, metadata_filter)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs=metadata_filter),
        memory=memory,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT}
    )
    return conversation_chain


def parse_chat_history(messages, metadata):
    chat_history = ChatMessageHistory()
    for message in messages:
        if message["type"] == "HUMAN":
            chat_history.add_message(HumanMessage(
                content=message["content"],
                additional_kwargs={"additional_kwargs": metadata}
            ))
        if message["type"] == "AI":
            chat_history.add_message(AIMessage(
                content=message["content"],
                additional_kwargs={"additional_kwargs": metadata}
            ))
    return chat_history
