import json
from io import BytesIO

from dotenv import load_dotenv
from flask import Flask, request, Response
from flask_cors import CORS, cross_origin
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.graphs import Neo4jGraph
from waitress import serve

from services.api_tools import get_model_name, get_pdf_text
from services.chat.conversation_chain import ConversationChainBuilder
from services.graph.document_repository import DocumentRepository
from services.graph.graph_embedder import GraphEmbedder

load_dotenv()

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
app = Flask(__name__)
CORS(app)

LLM = ChatOpenAI(temperature=0, model_name=get_model_name())
EMBEDDER = OpenAIEmbeddings()

GRAPH = Neo4jGraph(url="neo4j://localhost:7687", username="neo4j", password="smartbsmartb")
DOCUMENT_REPOSITORY = DocumentRepository(GRAPH)
GRAPH_EMBEDDER = GraphEmbedder(embedder=EMBEDDER, graph=GRAPH)

CONVERSATION_CHAIN_BUILDER = ConversationChainBuilder(llm=LLM, embedder=EMBEDDER, document_repository=DOCUMENT_REPOSITORY)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['POST'])
@cross_origin()
def upload_file():
    # check if the post request has the file part
    if 'file' not in request.files:
        print("No file part")
        return Response("Missing `file` part", mimetype='text/plain', status=400)
    file = request.files['file']

    if 'path' not in request.form:
        print("No path provided")
        return Response("Missing `path` part", mimetype='text/plain', status=400)
    file_path = request.form["path"]

    # workaround: flask request.files['files'].filename fail on spaces
    filename = ''.join(str(file.headers).split("filename=")[1].split("\r")[0]).replace('"', '')
    if filename == '':
        filename = file_path.split('/')[-1]

    file_trace = [filename]

    if not file or not allowed_file(filename):
        file_trace.append(f"file not allowed (not in {ALLOWED_EXTENSIONS})")
        print(file_trace)
        return Response(str(file_trace), mimetype='text/plain', status=400)

    file_metadata = {}
    if 'metadata' in request.form:
        file_metadata.update(json.loads(request.form['metadata']))

    file_content = BytesIO(file.read())
    raw_text = get_pdf_text(file_content)

    doc_identifier = DOCUMENT_REPOSITORY.save(filename, raw_text, request.form["path"], file_metadata, file_trace)
    GRAPH_EMBEDDER.embed_document(doc_identifier)

    print(file_trace)
    return Response(str(file_trace), mimetype='text/plain', status=200)


@app.route('/ask', methods=['POST'])
@cross_origin()
def ask():
    if "question" not in request.json:
        print("No question param")
        return Response("Missing `question` param", mimetype='text/plain', status=400)

    if "targeted_files" not in request.json:
        print("No targeted_files param")
        return Response("Missing `targeted_files` param", mimetype='text/plain', status=400)

    if "messages" not in request.json:
        print("No messages param")
        return Response("Missing `messages` param", mimetype='text/plain', status=400)

    question = request.json["question"]
    file_paths: [str] = request.json["targeted_files"]
    messages = request.json["messages"]
    print("ask:", question, file_paths)

    chain = CONVERSATION_CHAIN_BUILDER.get_conversation_chain(file_paths, messages)
    response = chain({'question': question})
    return Response(response['answer'], mimetype='text/plain', status=200)


if __name__ == '__main__':
    serve(app, host="0.0.0.0", port=9999)
