# src/sfsa_rag_assistant/graph.py
"""
LangGraph workflow orchestration for SFSA Agentic RAG.

This module defines the complete workflow graph with all nodes and conditional edges,
implementing the 3-agent architecture for context-aware RAG with web search fallback
and response validation loops.
"""

import logging
from typing import Dict, Any, Literal, List
from langgraph.graph import StateGraph, END

from .states import SFSAAgenticState, initialize_state
from .nodes import (
    contextualize_query,
    retrieve_context,
    check_context_sufficiency,
    web_search,
    combine_context,
    generate_response,
    validate_response,
    format_final_output
)
from .app_config import settings

logger = logging.getLogger(__name__)


# ============================================================================
# CONDITIONAL ROUTING FUNCTIONS
# ============================================================================

def route_after_agent1(state: SFSAAgenticState) -> Literal["generate_response", "web_search"]:
    """
    Route after Agent 1 based on context sufficiency.
    
    - If sufficient: skip web search, go directly to generation
    - If insufficient: trigger web search to augment context
    
    Parameters
    ----------
    state : SFSAAgenticState
        Current workflow state with context_sufficient decision
    
    Returns
    -------
    Literal["generate_response", "web_search"]
        Next node to execute
    """
    if state.get("context_sufficient", True):
        logger.info("Agent 1: Context sufficient -> generate_response")
        return "generate_response"
    else:
        logger.info("Agent 1: Context insufficient -> web_search")
        return "web_search"


def route_after_agent3(state: SFSAAgenticState) -> Literal["generate_response", "format_final_output"]:
    """
    Route after Agent 3 based on response validation.
    
    - If satisfactory: proceed to final output
    - If needs refinement AND under max attempts: loop back to generation
    - If at max attempts: proceed to final output (combine all responses)
    
    Parameters
    ----------
    state : SFSAAgenticState
        Current workflow state with validation decision
    
    Returns
    -------
    Literal["generate_response", "format_final_output"]
        Next node to execute
    """
    response_satisfactory = state.get("response_satisfactory", True)
    validation_attempts = state.get("validation_attempts", 0)
    max_attempts = state.get("max_validation_attempts", settings.max_validation_attempts)
    
    if response_satisfactory:
        logger.info(f"Agent 3: Response satisfactory (attempt {validation_attempts}) -> format_final_output")
        return "format_final_output"
    
    if validation_attempts >= max_attempts:
        logger.info(f"Agent 3: Max attempts reached ({validation_attempts}/{max_attempts}) -> format_final_output")
        return "format_final_output"
    
    logger.info(f"Agent 3: Needs refinement (attempt {validation_attempts}/{max_attempts}) -> generate_response")
    return "generate_response"


# ============================================================================
# GRAPH BUILDER
# ============================================================================

def build_graph() -> StateGraph:
    """
    Build the complete LangGraph workflow.
    
    Architecture:
    ```
    START
      ↓
    retrieve_context
      ↓
    check_context_sufficiency (Agent 1)
      ↓
    ┌─────────────────┐
    │ sufficient?     │
    └─────────────────┘
         │         │
       Yes        No
         │         │
         │         ↓
         │    web_search (Agent 2)
         │         │
         │         ↓
         │    combine_context
         │         │
         └─────────┘
               ↓
         generate_response
               ↓
         validate_response (Agent 3)
               ↓
         ┌─────────────────┐
         │ satisfactory?   │
         └─────────────────┘
              │         │
            Yes        No (& < max attempts)
              │         │
              │         ↓ (refined_query)
              │      generate_response ↻
              │         │
              │         ↓
              │    validate_response ↻
              │         │
              └─────────┘
                   ↓
            format_final_output
                   ↓
                  END
    ```
    
    Returns
    -------
    StateGraph
        Compiled LangGraph workflow ready for execution
    """
    logger.info("Building LangGraph workflow...")
    
    # Initialize graph with state schema
    workflow = StateGraph(SFSAAgenticState)
    
    # ─── Add Nodes ──────────────────────────────────────────────────────────
    workflow.add_node("contextualize_query", contextualize_query)
    workflow.add_node("retrieve_context", retrieve_context)
    workflow.add_node("check_context_sufficiency", check_context_sufficiency)
    workflow.add_node("web_search", web_search)
    workflow.add_node("combine_context", combine_context)
    workflow.add_node("generate_response", generate_response)
    workflow.add_node("validate_response", validate_response)
    workflow.add_node("format_final_output", format_final_output)
    
    # ─── Set Entry Point ────────────────────────────────────────────────────
    workflow.set_entry_point("contextualize_query")
    
    # ─── Add Edges ──────────────────────────────────────────────────────────
    
    # Query contextualization happens first
    workflow.add_edge("contextualize_query", "retrieve_context")
    
    # Always proceed from retrieval to Agent 1
    workflow.add_edge("retrieve_context", "check_context_sufficiency")
    
    # Conditional routing after Agent 1
    workflow.add_conditional_edges(
        "check_context_sufficiency",
        route_after_agent1,
        {
            "generate_response": "generate_response",
            "web_search": "web_search"
        }
    )
    
    # Web search path: web_search → combine_context → generate_response
    workflow.add_edge("web_search", "combine_context")
    workflow.add_edge("combine_context", "generate_response")
    
    # Always validate after generation
    workflow.add_edge("generate_response", "validate_response")
    
    # Conditional routing after Agent 3
    workflow.add_conditional_edges(
        "validate_response",
        route_after_agent3,
        {
            "generate_response": "generate_response",  # Refinement loop
            "format_final_output": "format_final_output"
        }
    )
    
    # Terminal node
    workflow.add_edge("format_final_output", END)
    
    # ─── Compile Graph ──────────────────────────────────────────────────────
    logger.info("Graph compiled successfully")
    
    return workflow.compile()


# ============================================================================
# GRAPH EXECUTION
# ============================================================================

def run_workflow(
    user_query: str,
    conversation_history: List[Dict[str, str]] = None,
    max_validation_attempts: int = None
) -> Dict[str, Any]:
    """
    Execute the complete agentic RAG workflow for a user query.
    
    This is the main entry point for running the workflow. It:
    1. Initializes the state with the query
    2. Builds and compiles the graph
    3. Executes the workflow
    4. Returns the final output
    
    Parameters
    ----------
    user_query : str
        The user's question
    conversation_history : List[Dict[str, str]], optional
        Previous conversation turns for context
    max_validation_attempts : int, optional
        Override default max validation attempts (default: from settings)
    
    Returns
    -------
    Dict[str, Any]
        Final output with response, sources, and metadata
    
    Examples
    --------
    >>> result = run_workflow("What is steel casting?")
    >>> print(result["response"])
    >>> print(result["metadata"]["workflow_path"])
    """
    logger.info("=" * 80)
    logger.info("STARTING SFSA AGENTIC RAG WORKFLOW")
    logger.info("=" * 80)
    logger.info(f"Query: {user_query}")
    
    try:
        # Step 1: Initialize state
        initial_state = initialize_state(
            user_query=user_query,
            conversation_history=conversation_history or [],
            max_validation_attempts=max_validation_attempts or settings.max_validation_attempts
        )
        
        # Step 2: Build graph
        graph = build_graph()
        
        # Step 3: Execute workflow
        logger.info("Executing workflow...")
        final_state = graph.invoke(initial_state)
        
        # Step 4: Extract final output
        final_output = final_state.get("final_output")
        
        if not final_output:
            logger.error("No final_output in state")
            return {
                "query": user_query,
                "response": "Workflow completed but no output was generated.",
                "sources": [],
                "metadata": {"error": "No final_output in state", "failure": "no_output"}
            }
        
        logger.info("=" * 80)
        logger.info("WORKFLOW COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Workflow path: {final_output.get('metadata', {}).get('workflow_path', 'unknown')}")
        logger.info(f"Response length: {len(final_output.get('response', ''))} chars")
        logger.info(f"Sources: {len(final_output.get('sources', []))} items")
        
        return final_output
    
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}", exc_info=True)
        return {
            "query": user_query,
            "response": f"I apologize, but an error occurred: {str(e)}",
            "sources": [],
            "metadata": {
                "error": str(e),
                "workflow_path": "failed",
                "failure": "workflow"
            }
        }


# ============================================================================
# GRAPH UTILITIES
# ============================================================================

def get_graph_visualization() -> str:
    """
    Get a text representation of the graph structure.
    
    Currently returns a pre-formatted ASCII diagram. Could be extended
    to generate Mermaid or other formats.
    
    Returns
    -------
    str
        Text visualization of the workflow graph
    """
    return """
SFSA Agentic RAG Workflow Graph
================================

START
  ↓
retrieve_context
  ↓
check_context_sufficiency (Agent 1)
  ↓
┌─────────────────┐
│ sufficient?     │
└─────────────────┘
     │         │
   Yes        No
     │         │
     │         ↓
     │    web_search (Agent 2)
     │         │
     │         ↓
     │    combine_context
     │         │
     └─────────┘
           ↓
     generate_response
           ↓
     validate_response (Agent 3)
           ↓
     ┌─────────────────┐
     │ satisfactory?   │
     └─────────────────┘
          │         │
        Yes        No (& < max attempts)
          │         │
          │         ↓ (refined_query)
          │      generate_response ↻
          │         │
          │         ↓
          │    validate_response ↻
          │         │
          └─────────┘
               ↓
        format_final_output
               ↓
              END

Nodes: 7 total
- retrieve_context: Fetch documents from FAISS
- check_context_sufficiency: Agent 1 - Context evaluator
- web_search: Agent 2 - Query reformulation + Tavily search
- combine_context: Merge RAG + web results
- generate_response: LLM answer generation
- validate_response: Agent 3 - Quality validator
- format_final_output: Final formatting with citations

Conditional Edges: 2
1. After Agent 1: Route based on context_sufficient
2. After Agent 3: Route based on response_satisfactory + validation_attempts
"""


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    "build_graph",
    "run_workflow",
    "get_graph_visualization",
    "route_after_agent1",
    "route_after_agent3"
]
