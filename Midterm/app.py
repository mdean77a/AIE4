# Prototype for Hugging Face upload
import os, tiktoken
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Qdrant
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from operator import itemgetter
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from IPython.display import Markdown, display
import chainlit as cl
from openai import OpenAI, AsyncOpenAI
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")
# Path to my directory containing PDF files
directory = "References/"

# List to store all the documents
all_docs = []

# Iterate through all the files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".pdf"):  # Check if the file is a PDF
        file_path = os.path.join(directory, filename)
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()
        all_docs.extend(docs)  # Append the loaded docs to my list

# Default behavior is to break PDF files into their pages
# Using tiktoken, I checked the token lengths of several representative pages and
# the lengths were always less than 1000 tokens, so INITIAL STRATEGY is to use
# each document as a single chunk and not further split.

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

page_split_vectorstore = Qdrant.from_documents(
    all_docs,
    embedding_model,
    location=":memory:",
    collection_name="page_split_collection",
)
page_split_retriever = page_split_vectorstore.as_retriever()

# ALTERNATIVE STRATEGY is to recombine all the pages into one string document and then
# split it.  The advantage of this approach is to have chunk overlap, which is not
# possible with my initial strategy.

one_document = ""
for doc in all_docs:
    one_document += doc.page_content

def tiktoken_len(text):
    tokens = tiktoken.encoding_for_model("gpt-4o").encode(
        text,
    )
    return len(tokens)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 800,
    chunk_overlap = 400,
    length_function = tiktoken_len,
)

split_chunks = text_splitter.split_text(one_document)

chunk_split_vectorstore = Qdrant.from_documents(
    text_splitter.create_documents(split_chunks),
    embedding_model,
    location=":memory:",
    collection_name="chunk_split_collection",
)

chunk_split_retriever = chunk_split_vectorstore.as_retriever()

llm = ChatOpenAI(model="gpt-4o")

rag_prompt_template = """\
You are a helpful and polite and cheerful assistant who answers questions based solely on the provided context. 
Use the context to answer the question and provide a  clear answer. Do not mention the document in your
response.
If there is no specific information
relevant to the question, then tell the user that you can't answer based on the context.

Context:
{context}

Question:
{question}
"""

rag_prompt = ChatPromptTemplate.from_template(rag_prompt_template)

## CREATE MY TWO RAG CHAINS

page_split_rag_chain = (
    {"context": itemgetter("question") | page_split_retriever, "question": itemgetter("question")}
    | rag_prompt | llm | StrOutputParser()
)

chunk_split_rag_chain = (
    {"context": itemgetter("question") | chunk_split_retriever, "question": itemgetter("question")}
    | rag_prompt | llm | StrOutputParser()
)

page_response = (page_split_rag_chain.invoke({"question": "List the ten major risks of AI?"}))
print(page_response)

@cl.on_chat_start
async def on_chat_start():
    welcome_message = f"""
# 🪙 Welcome to our Company's Responsible AI Development Question and Answer Chatbot!

As you know, AI is a hot topic, and our company is engaged in developing software 
applications.  We believe that we will inevitably be developing software that utilizes
AI, but people are rightfully concerned about the implications of using AI.  Nobody seems to understand
the riht way to think about building ethical and useful AI applications for enterprises.  

I am here to help you understand how the AI industry is evolving, especially as it relates to politics.
Many people believe that the best guidance is likely to come from the government, so I have thoroughly
read two important documents:

1.  White House Blueprint for an AI Bill of Rights, Making Automated Systems Work for the American People, October 2022
2.  Artificial Intelligence Risk Management Framework: Generative Artificial Intelligence Profile (NIST AI 600-1)

Ask me questions about AI, its implications, where the industry is heading, etc.  I will try to help!

"""
    
    await cl.Message(content=welcome_message).send()

    # Set the chain in the session
    cl.user_session.set("chain", page_split_rag_chain)


@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")
    result =  chain.invoke({"question":message.content})
    msg = cl.Message(content=result)
    await msg.send()

