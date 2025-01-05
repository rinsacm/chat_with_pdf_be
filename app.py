from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv
import os
from flask_cors import CORS
load_dotenv()
qa_chain=None

os.environ['AZURE_OPENAI_ENDPOINT']=os.getenv('OPENAI_URL')
os.environ['AZURE_OPENAI_API_KEY']=os.getenv('OPENAI_KEY')
os.environ['AZURE_OPENAI_API_VERSION']="2024-08-01-preview"

def get_chunk_dos_from_pdf(pdf_path):
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=30)
    split_documents = text_splitter.split_documents(documents)
    return split_documents
def get_embeddings_of_pdf(pdf_path):
    embeddings=AzureOpenAIEmbeddings(model="text-embedding-ada-002",chunk_size=1536,)
    print(embeddings)
    return embeddings

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
def create_pdf_vector_store_and_qachain(pdf_path):
    from langchain import hub
    global qa_chain
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    # See full prompt at https://smith.langchain.com/hub/rlm/rag-prompt

    from langchain_openai import AzureChatOpenAI
    prompt = hub.pull("rlm/rag-prompt")
    llm = AzureChatOpenAI(
        azure_deployment="gpt-4o",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        verbose=False,
        temperature=0.3,
    )

    vector_store=FAISS.from_documents(get_chunk_dos_from_pdf(pdf_path),get_embeddings_of_pdf(pdf_path))
    vector_store.save_local("faiss_index")
    new_vector_store = FAISS.load_local(
    "faiss_index", get_embeddings_of_pdf(pdf_path), allow_dangerous_deserialization=True,
    )
    retriever=new_vector_store.as_retriever()
    qa_chain = (
    {
        "context": retriever| format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
    )







from flask import Flask, request, jsonify
app = Flask(__name__)
CORS(app)
def create_directory_if_not_exists(dir_path):
    try:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Directory created or already exists: {dir_path}")
    except Exception as e:
        print(f"Error creating directory: {e}")

@app.route('/upload', methods=['POST'])
def upload_pdf():
    # Check if a file is provided
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the file temporarily
    create_directory_if_not_exists('uploads')
    pdf_path = os.path.join('uploads', file.filename)
    file.save(pdf_path)

    # Create the vector store for the uploaded PDF
    create_pdf_vector_store_and_qachain(pdf_path)

    return jsonify({"message": "PDF uploaded and processed successfully!"}), 200

@app.route('/ask', methods=['POST'])
def ask():
    # Get user query
    query = request.json.get("query")

    if not query:
        return jsonify({"error": "Query is required"}), 400

    # Process query through the Langchain QA chain
    response = qa_chain.invoke(input=query)

    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)