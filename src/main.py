import os
import sys
import uvicorn #type: ignore
from fastapi import FastAPI, HTTPException # type: ignore
from pydantic import BaseModel # type: ignore

from semantic_search import SemanticSearch
from rag import RAGSystem


# ===== Config =====


MODEL_NAME = "gpt-4o-mini"

# ===== FastAPI =====
app = FastAPI(
    title="RAG API",
    description="Ask questions over PDF documents using Retrieval-Augmented Generation",
    version="1.1.0"
)

# Global vars
rag_system: RAGSystem | None = None
semantic_app: SemanticSearch | None = None


# ===== Request models =====
class QuestionRequest(BaseModel):
    question: str

class FilePathRequest(BaseModel):
    file_path: str


@app.get("/")
def root():
    return {"message": "Hello world"}


@app.post("/set_document")
def set_document(req: FilePathRequest):
    """Set a new document path and rebuild RAG system"""
    global rag_system, semantic_app
    try:
        if semantic_app:
            semantic_app.close_connection()

        semantic_app = SemanticSearch(file_document_path=req.file_path)
        db = semantic_app.index_documents_to_weaviate()
        rag_system = RAGSystem(db=db, embedding=semantic_app.embeddings, model_name=MODEL_NAME)
        return {"message": f"Document indexed successfully from {req.file_path}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load document: {str(e)}")


@app.post("/ask")
def ask_question(req: QuestionRequest):
    """Ask a question using the current document"""
    if rag_system is None:
        raise HTTPException(status_code=400, detail="No document loaded. Please call /set_document first.")

    try:
        answer = rag_system.answer_question(req.question)
        docs = rag_system.find_similiar_chunk(req.question, top_k=3)
        return {"question": req.question, "answer": answer, "context": docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{str(e)}")


@app.on_event("shutdown")
def shutdown_event():
    if semantic_app:
        semantic_app.close_connection()
        print("Connection closed")







