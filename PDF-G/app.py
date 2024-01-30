from flask import Flask, render_template, request, jsonify
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os
import faiss

app = Flask(__name__)

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    query = request.form['query']

    # Check if the 'uploads' directory exists, and create it if not
    uploads_dir = os.path.join(os.getcwd(), 'uploads')
    os.makedirs(uploads_dir, exist_ok=True)

    uploaded_file = request.files['file']
    if uploaded_file:
        file_path = os.path.join(uploads_dir, uploaded_file.filename)
        uploaded_file.save(file_path)

        # Your existing code for processing the PDF
        pdf_reader = PdfReader(file_path)
        raw_text = ''
        for i, page in enumerate(pdf_reader.pages):
            content = page.extract_text()
            if content:
                raw_text += content

        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=800,
            chunk_overlap=200,
            length_function=len,
        )
        texts = text_splitter.split_text(raw_text)

        embeddings = OpenAIEmbeddings()
        document_search = FAISS.from_texts(texts, embeddings)

        chain = load_qa_chain(OpenAI(), chain_type="stuff")

        docs = document_search.similarity_search(query)
        answer = chain.run(input_documents=docs, question=query)

        return jsonify({'answer': answer})

    return jsonify({'error': 'No file uploaded'})

if __name__ == '__main__':
    app.run(debug=True)