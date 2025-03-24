from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# Load text file
loader = TextLoader("docs/medical_notes.txt")  # ✅ make sure this file exists!
documents = loader.load()

# Split text into chunks
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Create embeddings and vector store
embeddings = OllamaEmbeddings(model="medllama2")
vectorstore = Chroma.from_documents(docs, embedding=embeddings)

# Load LLM
#llm = Ollama(model="medllama2")
llm = OllamaLLM(model="medllama2")


# Create QA chain
qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

# Ask a question
query = "What could cause stomach pain at night in a 35-year-old?"
answer = qa.invoke(query)
print("Answer:", answer)
