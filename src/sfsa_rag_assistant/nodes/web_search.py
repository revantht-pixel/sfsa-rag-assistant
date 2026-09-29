# src/sfsa_rag_assistant/nodes/web_search.py
"""
Agent 2: Web Search with Query Reformulation

This agent reformulates the user query to be more search-engine friendly,
then performs a web search using Tavily API to find additional context.
"""

import logging
from typing import List, Dict, Any
from pydantic import BaseModel, Field

from ..generation import TextGenerator
from ..app_config import settings
from ..states import SFSAAgenticState

logger = logging.getLogger(__name__)


class QueryReformulation(BaseModel):
    """
    Pydantic model for Agent 2's query reformulation structured output.
    """
    search_query: str = Field(
        description="Reformulated search query optimized for web search engines"
    )
    reasoning: str = Field(
        description="Brief explanation of how the query was reformulated"
    )


def web_search(state: SFSAAgenticState) -> SFSAAgenticState:
    """
    Agent 2: Reformulate query and perform web search.
    
    This agent:
    1. Analyzes the user query and retrieved context
    2. Reformulates the query to be more search-engine friendly
    3. Performs web search using Tavily API (up to max_results)
    4. Formats the results for LLM consumption
    
    Parameters
    ----------
    state : SFSAAgenticState
        Current workflow state with user_query and retrieved context
    
    Returns
    -------
    SFSAAgenticState
        Updated state with:
        - web_search_query: str reformulated search query
        - web_search_results: List[Dict] raw Tavily results
        - web_search_formatted: str formatted results for LLM
    """
    logger.info("=" * 60)
    logger.info("AGENT 2: web_search")
    logger.info("=" * 60)

    if not settings.web_search_enabled:
        # Off by default. Skip everything -- including the reformulation LLM call --
        # so nothing about the question is sent anywhere. Same shape as "no results".
        logger.info("Web search is disabled (SFSA_WEB_SEARCH_ENABLED is not true); skipping")
        return {
            "web_search_query": "",
            "web_search_results": [],
            "web_search_formatted": "No web search results found.",
            "error": None
        }

    # Use contextualized query (or fall back to original if not available)
    user_query = state.get("contextualized_query") or state.get("user_query", "")
    retrieved_context = state.get("retrieved_formatted", "")
    
    if not user_query:
        logger.error("No user query found in state")
        return {
            "web_search_query": "",
            "web_search_results": [],
            "web_search_formatted": "",
            "error": "No user query for web search"
        }
    
    logger.info(f"Original query: {user_query}")
    
    try:
        # Step 1: Reformulate query for better web search
        reformulated_query = _reformulate_query(user_query, retrieved_context)
        
        logger.info(f"Reformulated query: {reformulated_query}")
        
        # Step 2: Perform web search with Tavily
        search_results = _search_tavily(reformulated_query)
        
        logger.info(f"Found {len(search_results)} web results")
        
        # Step 3: Format results for LLM
        formatted_results = _format_web_results(search_results)
        
        return {
            "web_search_query": reformulated_query,
            "web_search_results": search_results,
            "web_search_formatted": formatted_results,
            "error": None
        }
    
    except Exception as e:
        logger.error(f"Error in Agent 2: {e}", exc_info=True)
        return {
            "web_search_query": user_query,  # Use original query as fallback
            "web_search_results": [],
            "web_search_formatted": "Web search failed.",
            "error": f"Agent 2 failed: {str(e)}"
        }


def _reformulate_query(original_query: str, context: str) -> str:
    """
    Use Ollama to reformulate the query for better web search results.
    
    Parameters
    ----------
    original_query : str
        The original user query
    context : str
        The retrieved context from vector DB (to understand what's missing)
    
    Returns
    -------
    str
        Reformulated search query
    """
    try:
        # Initialize TextGenerator with Ollama
        generator = TextGenerator(
            model_name=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=settings.ollama_temperature,
            load_mode="ollama"
        )
        generator.load_model()
        
        # Configure for structured output
        structured_llm = generator.with_structured_output(QueryReformulation)
        
        # Build prompt for query reformulation
        prompt = _build_reformulation_prompt(original_query, context)
        
        logger.info("Invoking Agent 2 for query reformulation...")
        
        # Get structured reformulation
        result: QueryReformulation = structured_llm.invoke(prompt)
        
        logger.info(f"Reformulation reasoning: {result.reasoning}")
        
        return result.search_query
    
    except Exception as e:
        logger.error(f"Error reformulating query: {e}")
        # Fallback to original query if reformulation fails
        return original_query


def _build_reformulation_prompt(query: str, context: str) -> str:
    """
    Build the prompt for Agent 2 to reformulate the query.
    
    Parameters
    ----------
    query : str
        The original user query
    context : str
        The retrieved context (to understand gaps)
    
    Returns
    -------
    str
        Formatted prompt for query reformulation
    """
    # Truncate context if too long (keep first 1000 chars)
    context_preview = context[:1000] + "..." if len(context) > 1000 else context
    
    prompt = f"""You are a query reformulation expert for web search in the Steel Foundry domain.

Your task is to reformulate the user's query to make it more effective for web search engines, considering what information is missing from the existing context.

ORIGINAL USER QUERY:
{query}

EXISTING CONTEXT (from internal database):
{context_preview}

TASK:
Reformulate the query to:
1. Be more specific and search-engine friendly
2. Include relevant technical terms or industry keywords
3. Focus on finding information that complements or fills gaps in the existing context
4. Keep it concise (3-8 words typically)

EXAMPLES:
- "What is steel casting?" → "steel casting process foundry techniques"
- "How to reduce defects?" → "casting defect prevention methods steel foundry"
- "Tell me about heat treatment" → "steel heat treatment processes metallurgy"

Provide your reformulated search query and brief reasoning.
"""
    
    return prompt


def _search_tavily(query: str) -> List[Dict[str, Any]]:
    """
    Perform web search using Tavily API.
    
    Parameters
    ----------
    query : str
        The search query
    
    Returns
    -------
    List[Dict[str, Any]]
        List of search results with title, url, content, and score
    """
    if not settings.web_search_enabled:
        # Last line of defence: this function is the only place a query can leave the
        # server, so it refuses on its own even if some other path reaches it.
        logger.error("Refusing to call Tavily: web search is disabled")
        return []

    try:
        from tavily import TavilyClient

        # Initialize Tavily client
        tavily_api_key = settings.tavily_api_key
        if not tavily_api_key:
            logger.error("Tavily API key not configured")
            return []
        
        client = TavilyClient(api_key=tavily_api_key)
        
        # Perform search
        logger.info(f"Searching Tavily with max_results={settings.tavily_max_results}")
        
        response = client.search(
            query=query,
            max_results=settings.tavily_max_results,
            search_depth="basic",  # "basic" or "advanced"
            include_answer=False,  # We'll generate our own answer
            include_raw_content=False  # Don't need full HTML
        )
        
        # Extract results
        results = response.get("results", [])
        
        # Log results summary
        for i, result in enumerate(results, 1):
            logger.info(f"  Result {i}: {result.get('title', 'No title')} - {result.get('url', '')}")
        
        return results
    
    except ImportError:
        logger.error("tavily-python package not installed. Run: pip install tavily-python")
        return []
    except Exception as e:
        logger.error(f"Tavily search error: {e}")
        return []


def _format_web_results(results: List[Dict[str, Any]]) -> str:
    """
    Format Tavily search results for LLM consumption.
    
    Parameters
    ----------
    results : List[Dict[str, Any]]
        List of Tavily search results
    
    Returns
    -------
    str
        Formatted string with all web search results
    """
    if not results:
        return "No web search results found."
    
    formatted_parts = []
    
    for result in results:
        title = result.get("title", "No title")
        url = result.get("url", "")
        content = result.get("content", "")
        score = result.get("score", 0.0)
        
        # Format each result naturally without numbering
        # This prevents the LLM from saying "According to Web Result 1" etc.
        result_text = f"""{title}
Source: {url}
{content}
"""
        formatted_parts.append(result_text)
    
    # Separate results with blank lines
    return "\n\n".join(formatted_parts)
