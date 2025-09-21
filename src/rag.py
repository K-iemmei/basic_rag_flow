import os
import sys
import weaviate  # type: ignore
from langchain_weaviate.vectorstores import WeaviateVectorStore  # type: ignore
from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
from semantic_search import * 
from langchain_openai import ChatOpenAI # type: ignore
from langchain_core.prompts import ChatPromptTemplate # type: ignore
from langchain_core.output_parsers import StrOutputParser # type: ignore
from langchain_core.runnables import RunnablePassthrough # type: ignore



def format_docs(docs):
    return "\n\n".join(doc for doc in docs)

class RAGSystem:
    def __init__(self, db,embedding, model_name):
        self.db = db
        self.model_name = model_name
        self.embedding = embedding
    def find_similiar_chunk(self, query, top_k = 5):
        if not self.db:
            print("Error: Weaviate vector store is not initialized.")
            return []

        print(f"\nSearching for the {top_k} most similar documents to '{query}'...")
        collection = self.db._client.collections.get(self.db._index_name)
        query_result = collection.query.hybrid(
            query,
            vector = self.embedding.embed_query(query),
            alpha = 0.5,
            limit = top_k,
            return_metadata = ["score"]
        )
        results = []
        for obj in query_result.objects:
            content = obj.properties.get("k_key") or obj.properties.get("text")  # ưu tiên lấy k_key, fallback sang text
            score = obj.metadata.score
            results.append(content)
            print("Content: " , content)
            print("Score: " , score)
            print("===========================================")
        print("Search successful !!")
        return results
    def get_rag_chain(self):
        if not self.db:
            print("Error: Weaviate vector store is not initialized.")
            return "Cannot answer the question without a database connection." 
    
        template = """
        You are a helpful assistant. Use the provided context to answer the question.
        Question: {question}
        Context: {context}
        Answer:
        """
        prompt = ChatPromptTemplate.from_template(template)
        llm = ChatOpenAI(model = self.model_name)
        rag_chain = (
            prompt 
            | llm
            | StrOutputParser()
        )
        return rag_chain
    def answer_question(self,question):
        retrieved_docs = self.find_similiar_chunk(question)
        context_string = format_docs(retrieved_docs)
        rag_chain = self.get_rag_chain()
        answer = rag_chain.invoke({"context": context_string, "question": question})
        print("Answer: ", answer)
        return answer

if __name__ == "__main__":
    file_path = r"C:\Users\phamx\OneDrive\Documents\Project\Lab1\data\documents\trump.pdf"
    app = SemanticSearch(file_document_path=file_path)
    try:
        db = app.index_documents_to_weaviate()
    except:
        print("Cant create db !!!")
        sys.exit()
    rag = RAGSystem(db= db,embedding = app.embeddings,model_name="gpt-4o-mini")
    rag.answer_question("Who delivers Harry’s first letter from Hogwarts ?")
    app.close_connection()
