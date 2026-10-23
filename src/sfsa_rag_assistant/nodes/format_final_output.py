# src/sfsa_rag_assistant/nodes/format_final_output.py
"""
Node for formatting the final output with citations and metadata.

This is the terminal node that prepares the final response for the user,
including the answer, sources, and workflow metadata.
"""

import logging
from typing import List, Dict, Any
from datetime import datetime

from ..states import SFSAAgenticState, format_sources_for_display
from ..wiki_utils import extract_wiki_url_from_metadata

logger = logging.getLogger(__name__)


def format_final_output(state: SFSAAgenticState) -> SFSAAgenticState:
    """
    Format the final output with response, citations, and metadata.
    
    This node:
    1. Determines which response to use (current or combined)
    2. Formats sources and citations from vector DB and web search
    3. Adds workflow metadata (agents used, attempts, etc.)
    4. Creates structured final output
    
    Response selection logic:
    - If response_satisfactory: use current_response
    - If max attempts reached: combine all accumulated_responses
    
    Parameters
    ----------
    state : SFSAAgenticState
        Complete workflow state
    
    Returns
    -------
    SFSAAgenticState
        Updated state with:
        - final_response: str the answer text
        - final_output: Dict complete structured output with metadata
    """
    logger.info("=" * 60)
    logger.info("NODE: format_final_output")
    logger.info("=" * 60)
    
    # Step 1: Determine which response to use
    response_satisfactory = state.get("response_satisfactory", True)
    current_response = state.get("current_response", "")
    accumulated_responses = state.get("accumulated_responses", [])
    validation_attempts = state.get("validation_attempts", 0)
    max_attempts = state.get("max_validation_attempts", 2)
    
    if response_satisfactory:
        # Use the current satisfactory response
        final_response = current_response
        logger.info("Using current satisfactory response")
    elif validation_attempts >= max_attempts and len(accumulated_responses) > 1:
        # Max attempts reached - combine all responses
        final_response = _combine_responses(accumulated_responses)
        logger.info(f"Combining {len(accumulated_responses)} responses (max attempts reached)")
    else:
        # Fallback
        final_response = current_response or "I apologize, but I couldn't generate a response."
        logger.warning("Using fallback response logic")
    
    # Step 2: Gather sources
    sources = _gather_sources(state)
    
    # Step 3: Build workflow metadata
    metadata = _build_metadata(state)
    
    # Step 4: Format sources for display
    formatted_sources = format_sources_for_display(sources) if sources else "No sources available."
    
    # Step 5: Create final structured output
    final_output = {
        "query": state.get("user_query", ""),
        "response": final_response,
        "sources": sources,
        "formatted_sources": formatted_sources,
        "metadata": metadata,
        "timestamp": datetime.now().isoformat()
    }
    
    logger.info(f"Final response: {len(final_response)} chars")
    logger.info(f"Sources: {len(sources)} items")
    logger.info(f"Workflow: {metadata.get('workflow_path', 'unknown')}")
    
    return {
        "final_response": final_response,
        "final_output": final_output
    }


def _combine_responses(responses: List[str]) -> str:
    """
    Combine multiple responses when max validation attempts reached.
    
    Creates a composite answer that includes insights from all attempts.
    
    Parameters
    ----------
    responses : List[str]
        List of accumulated responses from multiple generation attempts
    
    Returns
    -------
    str
        Combined response with all attempts
    """
    if not responses:
        return "I apologize, but I couldn't generate a response."
    
    if len(responses) == 1:
        return responses[0]
    
    # Build combined response
    combined_parts = []
    
    combined_parts.append("Based on multiple refinement attempts, here is a comprehensive answer:\n")
    
    # Add the most recent (refined) response first
    combined_parts.append(responses[-1])
    
    # Optionally add note about previous attempts
    if len(responses) > 1:
        combined_parts.append(f"\n\n---\nNote: This response was refined through {len(responses)} generation attempts to provide the most complete answer possible.")
    
    return "\n".join(combined_parts)


def _gather_sources(state: SFSAAgenticState) -> List[Dict[str, Any]]:
    """
    Gather all sources from vector DB and web search.
    
    Parameters
    ----------
    state : SFSAAgenticState
        Current workflow state
    
    Returns
    -------
    List[Dict[str, Any]]
        Combined list of all sources with metadata
    """
    all_sources = []
    
    # Add vector DB sources
    vector_sources = state.get("sources", [])
    for source in vector_sources:
        # Construct wiki URL from metadata
        wiki_url = extract_wiki_url_from_metadata(source)
        
        all_sources.append({
            **source,
            "source_type": "vector_db",
            "wiki_url": wiki_url  # Add clickable wiki URL
        })
    
    # Add web search sources
    web_results = state.get("web_search_results", [])
    for result in web_results:
        all_sources.append({
            "title": result.get("title", ""),
            "url": result.get("url", ""),
            "content_preview": result.get("content", "")[:200],
            "score": result.get("score", 0.0),
            "source_type": "web_search"
        })
    
    return all_sources


def _build_metadata(state: SFSAAgenticState) -> Dict[str, Any]:
    """
    Build workflow metadata for transparency and debugging.
    
    Parameters
    ----------
    state : SFSAAgenticState
        Current workflow state
    
    Returns
    -------
    Dict[str, Any]
        Metadata about the workflow execution
    """
    # Determine workflow path
    context_sufficient = state.get("context_sufficient", True)
    used_web_search = state.get("web_search_results", []) != []
    validation_attempts = state.get("validation_attempts", 0)
    
    if context_sufficient and validation_attempts == 1:
        workflow_path = "simple_rag"
    elif not context_sufficient:
        workflow_path = "rag_with_web_search"
    elif validation_attempts > 1:
        workflow_path = "rag_with_refinement"
    else:
        workflow_path = "unknown"
    
    metadata = {
        "workflow_path": workflow_path,
        "agents_triggered": _get_triggered_agents(state),
        "query_contextualization": {
            "was_contextualized": bool(state.get("contextualized_query") and 
                                      state.get("contextualized_query") != state.get("user_query")),
            "original_query": state.get("user_query", ""),
            "contextualized_query": state.get("contextualized_query", ""),
            "reasoning": state.get("contextualization_reasoning", "")
        },
        "context_sufficiency": {
            "sufficient": state.get("context_sufficient", True),
            "reasoning": state.get("context_sufficiency_reasoning", "")
        },
        "validation": {
            "attempts": validation_attempts,
            "max_attempts": state.get("max_validation_attempts", 2),
            "final_status": "satisfactory" if state.get("response_satisfactory") else "max_attempts_reached",
            "reasoning": state.get("validation_reasoning", ""),
            "history": state.get("validation_history", [])
        },
        "retrieval": {
            "num_documents": len(state.get("retrieved_docs", [])),
            "vectordb_used": bool(state.get("retrieved_docs"))
        },
        "web_search": {
            "used": used_web_search,
            "query": state.get("web_search_query", "") if used_web_search else None,
            "num_results": len(state.get("web_search_results", []))
        },
        "errors": state.get("error"),
        "failure": state.get("failure")
    }
    
    return metadata


def _get_triggered_agents(state: SFSAAgenticState) -> List[str]:
    """
    Determine which agents were triggered in this workflow.
    
    Parameters
    ----------
    state : SFSAAgenticState
        Current workflow state
    
    Returns
    -------
    List[str]
        List of agent names that were triggered
    """
    agents = []
    
    # Agent 1 always runs
    agents.append("Agent 1: Context Sufficiency Checker")
    
    # Agent 2 runs if context insufficient
    if state.get("web_search_results"):
        agents.append("Agent 2: Web Search with Query Reformulation")
    
    # Agent 3 always runs
    agents.append("Agent 3: Response Validator")
    
    return agents
