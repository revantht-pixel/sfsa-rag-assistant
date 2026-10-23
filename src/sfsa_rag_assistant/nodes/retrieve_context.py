# src/sfsa_rag_assistant/nodes/retrieve_context.py
"""
Node for retrieving relevant context from the vector database.

This node wraps the existing DocumentRetriever and integrates it into
the LangGraph workflow.
"""

import logging
import threading
from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document

from ..retrieval import DocumentRetriever
from ..app_config import settings
from ..states import SFSAAgenticState

logger = logging.getLogger(__name__)

# Building a DocumentRetriever loads the embedding model and the whole index, which took about
# 17 s on the production box -- a third of every answer when it was done per question. So one is
# built per (index, model) for the life of the process and reused.
_retrievers: Dict[Tuple[str, str], DocumentRetriever] = {}
_retrievers_lock = threading.Lock()        # also stops concurrent first questions each building their own

# Searches are short, and the embedding model's tokenizer can fail ("Already borrowed") when
# one instance is used by several threads at once, so searches take turns.
_search_lock = threading.Lock()


def _get_retriever(vectordb_path: Any, embedding_model_name: str) -> DocumentRetriever:
    """The shared retriever for this index and model; built on first use. A failed build is not cached."""
    key = (str(vectordb_path), embedding_model_name)
    with _retrievers_lock:
        retriever = _retrievers.get(key)
        if retriever is None:
            retriever = DocumentRetriever(vectordb_path=vectordb_path, embedding_model_name=embedding_model_name)
            _retrievers[key] = retriever
        return retriever


def clear_retriever_cache() -> None:
    """Forget the shared retrievers; the next question builds a fresh one (e.g. after the index is replaced)."""
    with _retrievers_lock:
        _retrievers.clear()


def retrieve_context(state: SFSAAgenticState) -> SFSAAgenticState:
    """
    Retrieve relevant documents from the vector database based on user query.
    
    This node:
    1. Extracts the user query from state (or uses refined_query if available)
    2. Uses DocumentRetriever to fetch relevant documents from FAISS
    3. Formats retrieved documents for both display and LLM consumption
    4. Updates state with retrieved documents, formatted context, and source metadata
    
    Parameters
    ----------
    state : SFSAAgenticState
        Current workflow state containing user_query or refined_query
    
    Returns
    -------
    SFSAAgenticState
        Updated state with:
        - retrieved_docs: List of LangChain Document objects
        - retrieved_formatted: Formatted string of retrieved context
        - sources: List of source metadata dicts
    """
    logger.info("=" * 60)
    logger.info("NODE: retrieve_context")
    logger.info("=" * 60)
    
    # Use contextualized_query for retrieval (rewritten with conversation context)
    # Falls back to user_query if contextualization hasn't run yet
    query = state.get("contextualized_query") or state.get("user_query")
    
    if not query:
        logger.error("No query found in state")
        return {
            "retrieved_docs": [],
            "retrieved_formatted": "",
            "sources": [],
            "error": "No query provided for retrieval"
        }
    
    logger.info(f"Query for retrieval: {query}")
    if state.get("contextualized_query") and state.get("contextualized_query") != state.get("user_query"):
        logger.info(f"  ↳ Original query: {state.get('user_query')}")
        logger.info(f"  ↳ Contextualized: {query}")
    
    try:
        # The shared DocumentRetriever for the configured index and model
        retriever_instance = _get_retriever(settings.get_vectordb_path(), settings.embedding_model)
        
        # Get retriever with configured search parameters
        retriever = retriever_instance.get_retriever(
            search_type="similarity",
            search_kwargs={"k": settings.retrieval_k}
        )
        
        logger.info(f"Retrieving top {settings.retrieval_k} documents...")
        
        # Retrieve documents
        with _search_lock:
            retrieved_docs: List[Document] = retriever.invoke(query)
        
        logger.info(f"Retrieved {len(retrieved_docs)} documents")
        
        # Format retrieved documents for LLM consumption
        formatted_context = _format_documents_for_llm(retrieved_docs)
        
        # Extract source metadata
        sources = _extract_source_metadata(retrieved_docs)
        
        # Log sample of retrieved content
        if retrieved_docs:
            logger.info(f"Sample (first doc): {retrieved_docs[0].page_content[:150]}...")
        
        return {
            "retrieved_docs": retrieved_docs,
            "retrieved_formatted": formatted_context,
            "sources": sources,
            "error": None  # Clear any previous errors
        }
    
    except Exception as e:
        logger.error(f"Error during retrieval: {e}", exc_info=True)
        return {
            "retrieved_docs": [],
            "retrieved_formatted": "",
            "sources": [],
            "error": f"Retrieval failed: {str(e)}",
            "failure": "retrieval"
        }


def _format_documents_for_llm(docs: List[Document]) -> str:
    """
    Format retrieved documents into a single string for LLM consumption.
    
    Documents are presented naturally without numbering to avoid LLM referencing them.
    
    Parameters
    ----------
    docs : List[Document]
        List of retrieved LangChain Document objects
    
    Returns
    -------
    str
        Formatted string with all document contents
    """
    if not docs:
        return "No relevant documents found."
    
    formatted_parts = []
    
    for doc in docs:
        # Just add the content without document numbers or labels
        # This prevents the LLM from saying "According to Document 1" etc.
        formatted_parts.append(doc.page_content)
        formatted_parts.append("")  # Blank line separator
    
    return "\n".join(formatted_parts)


def _extract_source_metadata(docs: List[Document]) -> List[Dict[str, Any]]:
    """
    Extract source metadata from retrieved documents for citation purposes.
    
    Parameters
    ----------
    docs : List[Document]
        List of retrieved LangChain Document objects
    
    Returns
    -------
    List[Dict[str, Any]]
        List of metadata dictionaries with source information
    """
    sources = []
    
    for doc in docs:
        source_entry = {
            "content_preview": doc.page_content[:200],  # First 200 chars
            **doc.metadata  # Include all metadata (file, page, etc.)
        }
        sources.append(source_entry)
    
    return sources


def _get_source_display(metadata: Dict[str, Any]) -> str:
    """
    Generate a human-readable source display string from metadata.
    
    Parameters
    ----------
    metadata : Dict[str, Any]
        Document metadata dictionary
    
    Returns
    -------
    str
        Formatted source string (e.g., "document.pdf, page 5")
    """
    parts = []
    
    # Common metadata fields to check
    if "source" in metadata:
        # Extract filename from path if it's a full path
        source = metadata["source"]
        if "/" in source:
            source = source.split("/")[-1]
        parts.append(source)
    
    if "page" in metadata:
        parts.append(f"page {metadata['page']}")
    elif "page_number" in metadata:
        parts.append(f"page {metadata['page_number']}")
    
    if not parts:
        return "unknown source"
    
    return ", ".join(parts)
