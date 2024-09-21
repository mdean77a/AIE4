# Defaults
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
default_embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
default_location = ":memory:"
default_llm = ChatOpenAI(model="gpt-4o")