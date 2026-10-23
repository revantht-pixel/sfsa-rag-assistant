# src/sfsa_rag_assistant/nodes/generate_response.py
"""
Node for generating responses using the LLM with available context.

This is the core generation node that produces answers using either:
- Retrieved context from vector DB (if Agent 1 deemed it sufficient)
- Combined context from vector DB + web search (if Agent 1 deemed insufficient)
- With original query or refined query (from Agent 3 validation loop)
"""

import logging
from typing import List, Dict, Any, Optional

from ..generation import TextGenerator
from ..app_config import settings
from ..states import SFSAAgenticState
from ..wiki_utils import extract_wiki_url_from_metadata

logger = logging.getLogger(__name__)


def generate_response(state: SFSAAgenticState) -> SFSAAgenticState:
    """
    Generate a response using the LLM with available context.
    
    This node:
    1. Selects the appropriate context (combined or retrieved only)
    2. Selects the appropriate query (refined or original)
    3. Builds a comprehensive prompt with conversation history
    4. Generates response using Ollama
    5. Tracks the response in accumulated_responses list
    6. Increments validation_attempts counter
    
    Context selection logic:
    - Use combined_context if available (web search was triggered)
    - Otherwise use retrieved_formatted (RAG only)
    
    Query selection logic:
    - Use refined_query if available (from Agent 3 loop)
    - Otherwise use user_query (original)
    
    Parameters
    ----------
    state : SFSAAgenticState
        Current workflow state with context and query
    
    Returns
    -------
    SFSAAgenticState
        Updated state with:
        - current_response: str latest generated response
        - accumulated_responses: List appended with new response
        - validation_attempts: int incremented counter
    """
    logger.info("=" * 60)
    logger.info("NODE: generate_response")
    logger.info("=" * 60)
    
    # Step 1: Select context (prefer combined, fallback to retrieved)
    context = state.get("combined_context") or state.get("retrieved_formatted", "")
    
    if not context:
        logger.error("No context available for generation")
        return {
            "current_response": "I apologize, but I don't have enough context to answer your question.",
            "error": "No context available for generation",
            "failure": _failure_after(state, "no_context")
        }
    
    # Step 2: Select query (prefer refined from Agent 3, then contextualized, fallback to original)
    query = state.get("refined_query") or state.get("contextualized_query") or state.get("user_query", "")
    
    if not query:
        logger.error("No query available for generation")
        return {
            "current_response": "I apologize, but I don't have a question to answer.",
            "error": "No query available for generation",
            "failure": _failure_after(state, "no_query")
        }
    
    # Step 3: Get conversation history
    conversation_history = state.get("conversation_history", [])
    
    # Step 4: Get current validation attempt number
    current_attempt = state.get("validation_attempts", 0) + 1
    
    logger.info(f"Context type: {'combined' if state.get('combined_context') else 'retrieved only'}")
    logger.info(f"Query type: {'refined' if state.get('refined_query') else 'original'}")
    logger.info(f"Context length: {len(context)} chars")
    logger.info(f"Query: {query}")
    logger.info(f"Validation attempt: {current_attempt}")
    
    try:
        # Step 5: Initialize TextGenerator with Ollama
        generator = TextGenerator(
            model_name=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=settings.ollama_temperature,
            max_tokens=settings.ollama_max_tokens,
            load_mode="ollama"
        )
        generator.load_model()
        
        # Step 6: Build comprehensive prompt
        prompt = _build_generation_prompt(
            query=query,
            context=context,
            conversation_history=conversation_history,
            is_refinement=(state.get("refined_query") is not None),
            source_catalog=build_source_catalog(state),
        )
        
        logger.info("Generating response with Ollama...")
        
        # Step 7: Generate response
        response = generator.invoke(prompt)
        
        logger.info(f"Generated response: {len(response)} chars")
        logger.info(f"Response preview: {response[:200]}...")
        
        # Step 8: Update accumulated responses
        accumulated_responses = state.get("accumulated_responses", [])
        accumulated_responses.append(response)
        
        return {
            "current_response": response,
            "accumulated_responses": accumulated_responses,
            "validation_attempts": current_attempt,
            "error": None,
            "failure": _failure_after(state, None)
        }
    
    except Exception as e:
        logger.error(f"Error during generation: {e}", exc_info=True)
        error_message = f"I apologize, but I encountered an error while generating a response: {str(e)}"
        return {
            "current_response": error_message,
            "error": f"Generation failed: {str(e)}",
            "failure": _failure_after(state, "generation")
        }


def _failure_after(state: SFSAAgenticState, failed: Optional[str]) -> Optional[str]:
    """
    The value of `failure` once this node has run.

    A retrieval failure is never cleared: an answer written without its sources
    is not a good answer, however well generation then goes. Any other failure
    is replaced by this attempt's outcome, so a generation retry that works
    clears an earlier generation failure.
    """
    if state.get("failure") == "retrieval":
        return "retrieval"
    return failed


def _build_generation_prompt(
    query: str,
    context: str,
    conversation_history: List[Dict[str, str]],
    is_refinement: bool = False,
    source_catalog: str = "",
) -> str:
    """
    Build a comprehensive prompt for response generation.
    
    Parameters
    ----------
    query : str
        The query to answer (original or refined)
    context : str
        The context to use (retrieved or combined)
    conversation_history : List[Dict[str, str]]
        Previous conversation turns
    is_refinement : bool
        Whether this is a refinement attempt (from Agent 3 loop)
    
    Returns
    -------
    str
        Formatted prompt for generation
    """
    prompt_parts = []
    
    # System instructions
    prompt_parts.append("""You are an expert assistant for the Steel Founders' Society of America (SFSA).

Your role is to provide accurate, detailed, and thorough answers about:
- Steel casting processes and techniques
- Foundry operations and best practices
- Metallurgy and material properties
- Quality control and defect prevention
- Industry standards and regulations
- Equipment and tooling

CRITICAL INSTRUCTIONS:
- Answer questions directly and naturally as if the knowledge is your own
- Provide THOROUGH and COMPLETE responses that address all aspects of the question
- Include relevant technical details, specific values, methods, or examples when available
- NEVER reference "documents", "context", "provided information", or "sources" in your response
- NEVER say phrases like "According to Document 1", "Based on the context", "The provided text shows", etc.
- Add in-text citations using square brackets that match the source numbers you were given, such as [1] or [2, 3]
- Place citations at the end of the sentence or clause they support
- Simply state facts and information as if you know them directly
- Be specific and technical when appropriate
- Use the most relevant information available to give a comprehensive answer
- If information is limited, acknowledge this naturally WITHOUT mentioning the context/documents
- Use clear, professional language
- Structure your response logically with paragraphs or bullet points as appropriate
- Stay strictly within the information available to you - do NOT make up information
- Avoid being unnecessarily verbose, but don't omit important details for brevity

Example of WRONG approach:
"According to Document 1, steel casting is..."
"Based on the provided context, the main applications are..."
"Web Result 1 from Tavily Search mentions that..."

Example of CORRECT approach:
"Steel casting is a manufacturing process used when high strength and design flexibility are required [1]."
"The main applications include transportation, construction, and mining [1, 2]."
"Modern steel foundries utilize Industry 4.0 technologies..."
""")
    
    # Add conversation history if available
    if conversation_history:
        prompt_parts.append("\nCONVERSATION HISTORY:")
        for turn in conversation_history[-3:]:  # Last 3 turns for context
            if turn.get("role") == "user":
                prompt_parts.append(f"User: {turn.get('content', '')}")
            elif turn.get("role") == "assistant":
                prompt_parts.append(f"Assistant: {turn.get('content', '')}")
        prompt_parts.append("")
    
    # Add refinement context if applicable
    if is_refinement:
        prompt_parts.append("\nNOTE: This is a refinement attempt. The previous response was deemed incomplete or unclear. Please provide a more thorough and complete answer, including any relevant details, specifics, or technical information that may have been missing.\n")
    
    # Add context
    if source_catalog:
        prompt_parts.append("\nSOURCE CATALOG:")
        prompt_parts.append(source_catalog)
        prompt_parts.append("")

    prompt_parts.append("\nCONTEXT:")
    prompt_parts.append(context)
    prompt_parts.append("")
    
    # Add the question
    prompt_parts.append(f"\nQUESTION: {query}")
    prompt_parts.append("\nANSWER:")
    
    return "\n".join(prompt_parts)


def build_source_catalog(state: SFSAAgenticState) -> str:
    """
    Build a numbered source catalog aligned with final source ordering.
    """
    entries: List[str] = []

    for index, source in enumerate(state.get("sources", []), start=1):
        filename = source.get("source", "N/A")
        page = source.get("page", "N/A")
        wiki_url = extract_wiki_url_from_metadata(source) or ""
        snippet = source.get("content_preview", "")
        entries.append(
            f"[{index}] SFSA Wiki | file={filename} | page={page} | url={wiki_url}\nSnippet: {snippet}"
        )

    offset = len(entries)
    for relative_index, result in enumerate(state.get("web_search_results", []), start=1):
        index = offset + relative_index
        title = result.get("title", "N/A")
        url = result.get("url", "")
        snippet = result.get("content", "")
        entries.append(f"[{index}] Web | title={title} | url={url}\nSnippet: {snippet}")

    return "\n\n".join(entries)
