import os
from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_chroma import Chroma
from langchain_community.document_loaders.base_o365 import CHUNK_SIZE
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import requests
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from tempfile import NamedTemporaryFile
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()

# Constants
CHUNK_SIZE = 500
EMBEDDING_MODEL =  "Alibaba-NLP/gte-base-en-v1.5"
VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"
COLLECTION_NAME = "real_estate"
llm = None
vector_store = None

def initialize_components():
    global llm, vector_store

    if llm is None:
        llm = ChatGroq(model = "llama-3.3-70b-versatile", temperature = 0.9, max_tokens = 500 )
    if vector_store is None:
        ef = HuggingFaceEmbeddings(
            model_name = EMBEDDING_MODEL,
            model_kwargs = {"trust_remote_code": True},
        )
        vector_store = Chroma(
            collection_name = COLLECTION_NAME,
            embedding_function = ef,
            persist_directory= str(VECTORSTORE_DIR),
        )


def fetch_and_load_cnbc(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.5845.96 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch {url}: {response.status_code}")

    with NamedTemporaryFile(delete=False, suffix=".html") as f:
        f.write(response.content)
        temp_path = f.name

    loader = UnstructuredHTMLLoader(temp_path)
    docs = loader.load()
    # Overwrite metadata['source'] with actual URL
    for doc in docs:
        doc.metadata["source"] = url
    return docs

def process_urls(urls):
    """
    This function takes list of urls and stores into vectorDB
    :param urls: input urls
    :return:
    """
    yield "Iniializing Components..."
    initialize_components()
    yield "Resetting VectorStore..."
    vector_store.reset_collection()
    yield "Loading data..."
    # loader = UnstructuredURLLoader(urls)
    # ------Using HTML loader for avoiding brower errors------
    # Properly initialize docs
    raw_docs = []
    for url in urls:
        raw_docs.extend(fetch_and_load_cnbc(url))


    # spliting text -----------------------------
    yield "Splitting text into Chunks..."
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size = CHUNK_SIZE
    )
    split_docs = text_splitter.split_documents(raw_docs)
    # ------assigning unique ids- to docs ---------------
    yield "Adding Chunks into VectorStore..."
    uuids = [str(uuid4()) for _ in range(len(split_docs))]
    vector_store.add_documents(split_docs, ids = uuids)

    yield "Done adding docs into vector database...:)"

# ------------------Processing docs------------------------
# rag.py
from langchain_community.document_loaders import PyPDFLoader, TextLoader

def process_docs(files):
    """
    Takes a list of file paths and stores their content in ChromaDB.
    :param files: list of file paths (PDF or TXT)
    """
    yield "Initializing Components..."
    initialize_components()
    yield "Resetting VectorStore..."
    vector_store.reset_collection()

    yield "Adding Documents to VectorStore..."

    raw_docs = []
    for file_path in files:
        ext = file_path.split('.')[-1].lower()
        if ext == "pdf":
            loader = PyPDFLoader(file_path)
        elif ext == "txt":
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            yield f"Skipping unsupported file type: {file_path}"
            continue
        docs = loader.load()
        # Change source to filename instead of temp path
        for doc in docs:
            doc.metadata["source"] = Path(file_path).name
        raw_docs.extend(docs)



    # Split into chunks
    yield "Splitting text into Chunks..."
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=CHUNK_SIZE
    )
    split_docs = text_splitter.split_documents(raw_docs)

    # Add to vector store
    yield "Storing Chunks in VectorStore..."
    uuids = [str(uuid4()) for _ in range(len(split_docs))]
    vector_store.add_documents(split_docs, ids=uuids)

    yield "Done adding docs into vector database...:)"


def generate_answer(query):
    if not vector_store:
        raise RuntimeError("Vector database not initialized!!!")
    chain = RetrievalQAWithSourcesChain.from_llm(llm = llm, retriever = vector_store.as_retriever())
    result = chain.invoke({"question": query}, return_only_outputs=True)
    sources = result.get("sources","")
    return result['answer'], sources



if __name__ == '__main__':
    urls = [
        "https://www.cnbc.com/2024/12/21/how-the-federal-reserves-rate-policy-affects-mortgages.html",
        "https://www.cnbc.com/2024/12/20/why-mortgage-rates-jumped-despite-fed-interest-rate-cut.html",
    ]

    process_urls(urls)
    initialize_components()
    answer , sources = generate_answer("Tell me what was the 30 year fixed mortagate rate along with the date?")
    print(f'Answer: {answer}')
    print(f'Sources:{sources}')