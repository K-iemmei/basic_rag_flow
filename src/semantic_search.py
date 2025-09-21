import os
import weaviate # type: ignore
from langchain_weaviate.vectorstores import WeaviateVectorStore # type: ignore
from langchain_community.document_loaders import PyPDFLoader # type: ignore
from langchain_text_splitters import RecursiveCharacterTextSplitter # type: ignore
from langchain_huggingface import HuggingFaceEmbeddings # type: ignore
from langchain_experimental.text_splitter import SemanticChunker  # type: ignore

class SemanticSearch:
    def __init__(self, file_document_path, chunk_size=400, chunk_overlap=100, model_emb_name="BAAI/bge-m3"):
        self.file_document_path = file_document_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_emb_name = model_emb_name
        print("Loading model embedding ... ")
        self.embeddings = HuggingFaceEmbeddings(model_name=self.model_emb_name)
        print("Loading model embedding successfully ")
        self.weaviate_client = weaviate.connect_to_local()
        print("Connected to Weaviate successfully.")
        
    def _load_and_split_documents(self, chunk_type = "size"):
        print(f"Loading document from: {self.file_document_path}")
        loader = PyPDFLoader(self.file_document_path)
        documents = loader.load()
        if chunk_type == "semantic":
            text_splitter = SemanticChunker(self.embeddings)

        else:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                add_start_index=True
            )
        splits = text_splitter.split_documents(documents)
        print(f"Document split into {len(splits)} chunks.")
        return splits
    def index_documents_to_weaviate(self):
        splits = self._load_and_split_documents(chunk_type = "semantic")
        
        print("Indexing documents to Weaviate...")
        try:
            vector_store = WeaviateVectorStore.from_documents(
                documents=splits,
                embedding=self.embeddings,
                client=self.weaviate_client,
                index_name = "k_database",
                text_key = "k_key"
            )
            print("Indexing complete.")
            return vector_store
        except Exception as e:
            print(f"An error occurred during indexing: {e}")
            return None

    def close_connection(self):
        if self.weaviate_client and self.weaviate_client.is_connected():
            self.weaviate_client.close()
            print("Weaviate connection closed.")
    
if __name__ == "__main__":
    file_path = r"C:\Users\phamx\OneDrive\Documents\Project\Lab1\data\documents\trump.pdf"
    app = SemanticSearch(file_document_path=file_path)
    try:
        db = app.index_documents_to_weaviate()
        if db:
            query = "Who is Trump ?"
            print(f"\nSearching for documents similar to '{query}'...")
            docs = db.similarity_search(query)
            for i, doc in enumerate(docs):
                print("-" * 50)
                print(f"Document {i + 1}:")
                print(f"Content: {doc.page_content[:50]} ...")
            
    finally:
        app.close_connection()
