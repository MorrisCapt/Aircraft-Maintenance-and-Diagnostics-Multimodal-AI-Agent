# Import necessary libraries
import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from groq import Groq
#from langchain_groq import GroqClient  # Import GroqClient from langchain_groq
from langchain_groq import ChatGroq # Keep ChatGroq import as it is
import fitz  # PyMuPDF for PDF reading
import os
import shutil
import speech_recognition as sr
from tempfile import NamedTemporaryFile
import asyncio
import uvicorn
import os  # Import the os module


# Initialize FastAPI
app = FastAPI()

# Enable CORS for frontend interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the sentence transformer for embeddings
embedding_model_name = "multi-qa-MiniLM-L6-cos-v1"
embedder = SentenceTransformer(embedding_model_name)
# API Key
GROQ_API_KEY = "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"
#model = (model_name="mixtral-8x7b", show_progress=True) #Commented out as it's not used and causes an error

# Placeholder for LLM - Replace with actual LLM initialization
llm = None # This needs to be initialized with a real LLM

# Function to load and extract text from PDF files using PyMuPDF
def load_documents_from_pdfs(pdf_paths):
    documents = []
    for path in pdf_paths:
        doc_text = ""
        with fitz.open(path) as pdf:
            for page in pdf:
                doc_text += page.get_text()
        documents.append(doc_text)
    return documents

# Function to embed the documents
def embed_documents(documents):
    embeddings = embedder.encode(documents, show_progress_bar=True)
    return embeddings

# Function to query the Groq LLM for answers
def query_groq(query, document_embeddings, knowledge_base):
    query_embedding = embedder.encode([query])[0]
    similarities = []
    for doc_embedding in document_embeddings:
        similarity = torch.nn.functional.cosine_similarity(
            torch.tensor(query_embedding), torch.tensor(doc_embedding), dim=0
        )
        similarities.append(similarity.item())
    best_match_idx = similarities.index(max(similarities))
    best_match_doc = knowledge_base[best_match_idx]
    # Use llm instead of groq_model to make the query
    if llm: # Check if llm is initialized
        response = llm.ask(f"Question: {query}\nContext: {best_match_doc}")
        return response
    else:
        return "LLM not initialized."

# Load initial documents
@app.on_event("startup")
async def startup_event():
    pdf_paths = [
        "aviation_chunk_1.pdf", "aviation_chunk_2.pdf", "aviation_chunk_3.pdf",
        "aviation_chunk_4.pdf", "aviation_chunk_5.pdf", "aviation_chunk_6.pdf"
    ]
    global knowledge_base
    knowledge_base = load_documents_from_pdfs(pdf_paths)
    global document_embeddings
    document_embeddings = embed_documents(knowledge_base)

# Text query endpoint
@app.get("/query")
async def query_agent(query: str):
    answer = query_groq(query, document_embeddings, knowledge_base)
    return {"query": query, "answer": answer}

# Upload media (images/videos) endpoint
@app.post("/upload")
async def upload_media(file: UploadFile = File(...)):
    file_location = f"uploads/{file.filename}"
    os.makedirs("uploads", exist_ok=True)
    with open(file_location, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"filename": file.filename, "status": "uploaded"}

# Speech-to-text endpoint
@app.post("/speech")
async def speech_to_text(audio: UploadFile = File(...)):
    recognizer = sr.Recognizer()
    with NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
        tmp_audio.write(await audio.read())
        tmp_audio_path = tmp_audio.name

    with sr.AudioFile(tmp_audio_path) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            answer = query_groq(text, document_embeddings, knowledge_base)
            return {"transcription": text, "answer": answer}
        except sr.UnknownValueError:
            return JSONResponse(status_code=400, content={"error": "Could not understand audio"})
        except sr.RequestError as e:
            return JSONResponse(status_code=500, content={"error": f"Speech recognition error: {e}"})
def load_documents_from_pdfs(pdf_paths):
    documents = []
    for path in pdf_paths:
        doc_text = ""
        with fitz.open(path) as pdf:
            for page in pdf:
                doc_text += page.get_text()
        documents.append(doc_text)
    return documents

# Function to embed the documents
def embed_documents(documents):
    embeddings = embedder.encode(documents, show_progress_bar=True)
    return embeddings

# Function to query the Groq LLM for answers
def query_groq(query, document_embeddings, knowledge_base):
    query_embedding = embedder.encode([query])[0]
    similarities = []
    for doc_embedding in document_embeddings:
        similarity = torch.nn.functional.cosine_similarity(
            torch.tensor(query_embedding), torch.tensor(doc_embedding), dim=0
        )
        similarities.append(similarity.item())
    best_match_idx = similarities.index(max(similarities))
    best_match_doc = knowledge_base[best_match_idx]
    # Use llm instead of groq_model to make the query
    response = llm.ask(f"Question: {query}\nContext: {best_match_doc}") 
    return response

# Load initial documents
@app.on_event("startup")
async def startup_event():
    pdf_paths = [
        "aviation_chunk_1.pdf", "aviation_chunk_2.pdf", "aviation_chunk_3.pdf",
        "aviation_chunk_4.pdf", "aviation_chunk_5.pdf", "aviation_chunk_6.pdf"
    ]
    global knowledge_base
    knowledge_base = load_documents_from_pdfs(pdf_paths)
    global document_embeddings
    document_embeddings = embed_documents(knowledge_base)

# Text query endpoint
@app.get("/query")
async def query_agent(query: str):
    answer = query_groq(query, document_embeddings, knowledge_base)
    return {"query": query, "answer": answer}

# Upload media (images/videos) endpoint
@app.post("/upload")
async def upload_media(file: UploadFile = File(...)):
    file_location = f"uploads/{file.filename}"
    os.makedirs("uploads", exist_ok=True)
    with open(file_location, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"filename": file.filename, "status": "uploaded"}

# Speech-to-text endpoint
@app.post("/speech")
async def speech_to_text(audio: UploadFile = File(...)):
    recognizer = sr.Recognizer()
    with NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
        tmp_audio.write(await audio.read())
        tmp_audio_path = tmp_audio.name

    with sr.AudioFile(tmp_audio_path) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            answer = query_groq(text, document_embeddings, knowledge_base)
            return {"transcription": text, "answer": answer}
        except sr.UnknownValueError:
            return JSONResponse(status_code=400, content={"error": "Could not understand audio"})
        except sr.RequestError as e:
            return JSONResponse(status_code=500, content={"error": f"Speech recognition error: {e}"})

# Run the FastAPI app using uvicorn.run directly
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) # Use uvicorn.run