from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_qdrant import QdrantVectorStore, Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from operator import itemgetter
from langchain.schema.output_parser import StrOutputParser
from pydantic import BaseModel, Field
from langchain.schema.runnable import RunnablePassthrough
from typing import Any, List, Union
from langchain_community.document_loaders import PyMuPDFLoader
# from langchain_community.vectorstores import Qdrant
from langchain_core.documents.base import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

llm_model = OllamaLLM(model="llama3.1:8b-instruct-q8_0")
# print(llm_model.invoke("Come up with ten names for a song about Parrots"))


# Load the document - here we are just using a protocol in a specific directory
file_path = './documents/protocol.pdf'
separate_pages = []             
loader = PyMuPDFLoader(file_path)
page = loader.load()
separate_pages.extend(page)
print(f"Number of separate pages: {len(separate_pages)}")

# OyMuPDFLoader loads pages into separate docs!
# This is a problem when we chunk because we only chunk individual
# documents.  We need ONE overall document so that the chunks can
# overlap between actual PDF pages.
document_string = ""
for page in separate_pages:
    document_string += page.page_content
print(f"Length of the document string: {len(document_string)}")

# Now let's chop it up into little chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)
text_chunks = text_splitter.split_text(document_string)
print(f"Number of chunks: {len(text_chunks)} ")
max_chunk_size = 0
for chunk in text_chunks:
    max_chunk_size = max(max_chunk_size, len(chunk))
print(f"Maximum chunk size: {max_chunk_size}")
document = [Document(page_content=chunk) for chunk in text_chunks]
print(f"Length of  document: {len(document)}")

# Get embedding model
embedding_model = OllamaEmbeddings(
    model="mxbai-embed-large",
)                         

# ### Parent Child Experiment
# # Make parent documents 2000 tokens and child split to 250 tokens
# parent_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=2000,
# )
# child_splitter = RecursiveCharacterTextSplitter(
#     chunk_size = 250,
# )

# client = QdrantClient(url="http://localhost:6333")
# client.create_collection(
#     collection_name="parent_protocol",
#     vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
# )

# parent_vectorstore = QdrantVectorStore(
#     client=client,
#     collection_name="parent_protocol",
#     embedding=embedding_model,
# )
# store = InMemoryStore()

# parent_retriever = ParentDocumentRetriever(
#     vectorstore=parent_vectorstore,
#     docstore=store,
#     child_splitter=child_splitter,
#     parent_splitter=parent_splitter,
# )
# parent_retriever.add_documents(document)


client = QdrantClient(url="http://localhost:6333")
if client.collection_exists("protocol_collection"):
    print("Collection exists")
    qdrant_vectorstore = QdrantVectorStore.from_existing_collection(
        embedding=embedding_model,
        collection_name="protocol_collection",
        url="http://localhost:6333"
    )
else: 
    print("Collection does not exist")
    qdrant_vectorstore = QdrantVectorStore.from_documents(
    documents=document,
    embedding=embedding_model,
    collection_name="protocol_collection",
    url="http://localhost:6333"
)
    
retriever = qdrant_vectorstore.as_retriever(search_kwargs={"k":10})

RAG_PROMPT_TEMPLATE = """\
<|start_header_id|>system<|end_header_id|>
You are a helpful assistant. You answer user questions based on provided context. 
If you can't answer the question with the provided context, say you don't know.<|eot_id|>

<|start_header_id|>user<|end_header_id|>

Context:
{context}

User Query:
{query}<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
"""

rag_prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

rag_chain = ({"context": itemgetter("query") | retriever, "query": itemgetter("query")} 
             | rag_prompt | llm_model)
# parent_rag_chain = ({"context": itemgetter("query") | parent_retriever, "query": itemgetter("query")} 
#              | rag_prompt | llm_model)

class Input(BaseModel):
    query: str

class Output(BaseModel):
    output: Any

app = FastAPI()


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


add_routes(
    app,
    rag_chain.with_types(input_type=Input, output_type=Output).with_config(
        {"run_name": "ProtocolRAG"}
    )
)

# add_routes(
#     app,
#     parent_rag_chain.with_types(input_type=Input, output_type=Output).with_config(
#         {"run_name": "ProtocolRAG"}
#     )
# )

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
