import json
from io import BytesIO

from flask import Flask, flash, request, redirect, Response
from flask_cors import CORS, cross_origin
from waitress import serve

import api_tools

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
app = Flask(__name__)
CORS(app)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
@cross_origin()
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']

        # workaround: flask request.files['files'].filename fail on spaces
        filename = ''.join(str(request.files['file'].headers).split("filename=")[1].split("\r")[0]).replace('"', '')

        file_trace = []
        file_status = 500

        file_trace.append(filename)

        collection = vectorstore._collection
        existing_chunks_len = len(collection.get(where={"source": filename})['documents'])
        if existing_chunks_len > 0:
            collection.delete(where={"source": filename})
            file_trace.append(str(existing_chunks_len) + " existing chunks deleted")

        file_metadata = {"source": filename}
        if 'metadata' in request.form:
            file_metadata.update(json.loads(request.form['metadata']))

        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(filename):
            print("file found")

            file_content = BytesIO(file.read())

            raw_text = api_tools.get_pdf_text(file_content)

            # get the text chunks
            text_chunks = api_tools.get_text_chunks(raw_text)

            # create vector store
            metadata = []
            for _ in text_chunks:
                metadata.append(file_metadata.copy())
            vectorstore.add_texts(text_chunks, metadata)
            file_trace.append(str(len(text_chunks)) + " chunks vectorized")
            vectorstore.persist()
            file_status = 200
        else:
            file_trace.append("file not allowed")
            file_status = 400

        print(file_trace)
        return Response(str(file_trace), mimetype='text/plain', status=file_status)
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''


@app.route('/ask', methods=['POST'])
@cross_origin()
def ask():
    metadata_filter = {}
    if 'metadata' in request.json:
        metadata = request.json["metadata"]
        if 'targeted_files' in metadata:
            filenames = metadata['targeted_files']
            if len(filenames) == 1:
                metadata_filter = {"filter": {"source": filenames[0]}}
            elif len(filenames) > 1:
                sources = []
                for filename in filenames:
                    sources.append({"source": filename})
                metadata_filter = {"filter": {"$or": sources}}

    print("metadata_filter:", metadata_filter)
    response = api_tools.get_conversation_chain(vectorstore, metadata_filter, request.json["messages"])(
        {'question': request.json["question"]})
    return Response(response['answer'], mimetype='text/plain', status=200)


vectorstore = api_tools.get_vectorstore()

if __name__ == '__main__':
    serve(app, host="0.0.0.0", port=9999)
