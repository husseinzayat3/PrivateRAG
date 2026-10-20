from functools import cache
from config import OLLAMA_MODEL_NAME, OLLAMA_BASE_URL
from vectorstore import get_vectorstore
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def get_retriever():
    return get_vectorstore().as_retriever(search_kwargs={"k": 6})

# Cached so the chain (and the models behind it) is built once, not per question
@cache
def get_rag_chain():
    retriever = get_retriever()
    llm = OllamaLLM(model=OLLAMA_MODEL_NAME, base_url=OLLAMA_BASE_URL)
    
    template = """You are a highly intelligent and helpful expert assistant. 
Your task is to answer the user's question accurately and comprehensively based ONLY on the provided context below.

Instructions:
- Analyze the context carefully before answering.
- Provide a detailed and well-structured response.
- Use Markdown formatting (like bullet points, bold text, or code blocks) to make your answer easy to read.
- If the context contains multiple perspectives or steps, synthesize them clearly.
- If the answer is not contained in the context, politely state that you do not have enough information to answer, and do not make up facts.

Context:
{context}

Question: {question}

Expert Answer:"""
    prompt = PromptTemplate.from_template(template)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

def answer_question(question: str) -> str:
    try:
        chain = get_rag_chain()
        return chain.invoke(question)
    except Exception as e:
        return f"Error connecting to LLM or retrieving context. Make sure Ollama is running (`ollama run {OLLAMA_MODEL_NAME}`). Details: {e}"
