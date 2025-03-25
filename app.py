from fastapi import FastAPI, Request
from pydantic import BaseModel
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Load text file
loader = TextLoader("docs/medical_notes.txt")
documents = loader.load()

# Split text into chunks
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Create embeddings and vector store
embeddings = OllamaEmbeddings(model="medllama2")
vectorstore = Chroma.from_documents(docs, embedding=embeddings)

# Load LLM
llm = OllamaLLM(model="medllama2")

# Create QA chain
qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

# FastAPI app
app = FastAPI()

# Request model
class Question(BaseModel):
    query: str

# Endpoint to get answer from LLM
@app.post("/ask")
def ask_question(question: Question):
    answer = qa.invoke(question.query)
    return {"answer": answer}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)
