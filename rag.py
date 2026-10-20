import os
from functools import cache
from config import OLLAMA_MODEL_NAME, OLLAMA_BASE_URL, LLM_TEMPERATURE, RETRIEVAL_K
from vectorstore import get_vectorstore
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

CONDENSE_SYSTEM_PROMPT = """Given a chat history and the user's latest question, rewrite the question \
as a standalone question that can be understood without the chat history. \
Resolve pronouns and references like "it", "that" or "the second one" using the history. \
Do NOT answer the question. If it is already standalone, return it unchanged. \
Return only the rewritten question, with no preamble."""

ANSWER_SYSTEM_PROMPT = """You are a highly intelligent and helpful expert assistant.
Your task is to answer the user's question accurately and comprehensively based ONLY on the numbered context passages provided with the question.

Instructions:
- Analyze the context carefully before answering.
- Provide a detailed and well-structured response.
- Use Markdown formatting (like bullet points, bold text, or code blocks) to make your answer easy to read.
- If the context contains multiple perspectives or steps, synthesize them clearly.
- Cite the passages you use with their numbers in square brackets, e.g. [1] or [2][3], right after the statement they support.
- If the answer is not contained in the context, politely state that you do not have enough information to answer, and do not make up facts."""

ANSWER_USER_PROMPT = """Context:
{context}

Question: {question}"""

condense_prompt = ChatPromptTemplate.from_messages([
    ("system", CONDENSE_SYSTEM_PROMPT),
    MessagesPlaceholder("history"),
    ("human", "Latest question: {question}\n\nStandalone question:"),
])

answer_prompt = ChatPromptTemplate.from_messages([
    ("system", ANSWER_SYSTEM_PROMPT),
    MessagesPlaceholder("history"),
    ("human", ANSWER_USER_PROMPT),
])

# Cached per (model, temperature) so changing sidebar settings doesn't
# rebuild clients on every question
@cache
def get_llm(model=OLLAMA_MODEL_NAME, temperature=LLM_TEMPERATURE):
    return ChatOllama(model=model, base_url=OLLAMA_BASE_URL, temperature=temperature)

def to_messages(history):
    """Converts [{"role": ..., "content": ...}] chat history to LangChain messages."""
    return [
        HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"])
        for m in history
    ]

def source_label(doc):
    """Human-readable source, e.g. 'report.pdf, p. 3' or 'data.csv, row 12'."""
    meta = doc.metadata
    name = meta.get("title") or os.path.basename(str(meta.get("source", "unknown")))
    if "page_label" in meta:
        return f"{name}, p. {meta['page_label']}"
    if "page" in meta:
        return f"{name}, p. {int(meta['page']) + 1}"
    if "row" in meta:
        return f"{name}, row {int(meta['row']) + 1}"
    return name

def format_docs(docs):
    return "\n\n".join(
        f"[{i}] (Source: {source_label(doc)})\n{doc.page_content}"
        for i, doc in enumerate(docs, start=1)
    )

def condense_question(question, history, model=OLLAMA_MODEL_NAME):
    """Rewrites a follow-up question into a standalone one for retrieval."""
    if not history:
        return question
    chain = condense_prompt | get_llm(model, 0.0) | StrOutputParser()
    rewritten = chain.invoke({"history": to_messages(history), "question": question})
    return rewritten.strip().strip('"') or question

def retrieve(question, history=(), model=OLLAMA_MODEL_NAME, k=RETRIEVAL_K):
    """Returns (search_query, docs) for the question, using the chat history
    to resolve follow-ups."""
    search_query = condense_question(question, list(history), model)
    docs = get_vectorstore().similarity_search(search_query, k=k)
    return search_query, docs

def stream_answer(question, docs, history=(), model=OLLAMA_MODEL_NAME, temperature=LLM_TEMPERATURE):
    """Yields the answer token by token."""
    if not docs:
        yield "No documents are indexed yet. Upload or connect documents and click **Ingest Documents** first."
        return
    chain = answer_prompt | get_llm(model, temperature) | StrOutputParser()
    yield from chain.stream({
        "history": to_messages(history),
        "context": format_docs(docs),
        "question": question,
    })
