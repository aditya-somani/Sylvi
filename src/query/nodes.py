import logging
from datetime import datetime
from typing import Dict, Any, List, Literal, Optional
from pydantic import BaseModel

# Import State representation
from src.query.state import QueryState

# Services (Lazy instantiated inside nodes to prevent global side-effects during imports)
from src.services.llm import LLMService
from src.services.vector_db import VectorDBService
from src.memory.profile import ProfileMemoryDB
from src.prompts import (
    INTENT_ROUTER_SYSTEM_PROMPT,
    CHITCHAT_SYSTEM_PROMPT,
    REMINDER_SYSTEM_PROMPT,
    QUERY_OPTIMIZER_SYSTEM_PROMPT,
    RAG_GENERATION_SYSTEM_PROMPT,
    FACT_DELETION_SYSTEM_PROMPT,
    SEARCH_DECIDER_SYSTEM_PROMPT
)

logger = logging.getLogger(__name__)


def _format_chat_history(chat_history: List[Dict[str, Any]], max_messages: int = 8, max_chars_per_msg: int = 200) -> str:
    """
    Formats recent chat history into a compact string for LLM context injection.
    Truncates individual messages and limits total message count for efficiency.
    """
    if not chat_history:
        return ""
    # Take only the most recent N messages
    recent = chat_history[-max_messages:]
    lines = []
    for msg in recent:
        role_name = "User" if msg["role"] == "user" else "Sylvi"
        content = msg.get("content", "")
        if len(content) > max_chars_per_msg:
            content = content[:max_chars_per_msg] + "..."
        lines.append(f"{role_name}: {content}")
    return "\n".join(lines)


# --- Pydantic Schemas for Structured Outputs ---

class IntentClassification(BaseModel):
    intent: Literal["chit_chat", "reminder", "profile_query", "retrieval"]

class ReminderExtractor(BaseModel):
    reminder_text: str
    trigger_time: str  # Format: YYYY-MM-DDTHH:MM:SS

class QueryOptimizer(BaseModel):
    search_query: str

class SearchDecider(BaseModel):
    needs_search: bool
    search_query: Optional[str]


# --- Query Nodes ---

def intent_router_node(state: QueryState) -> Dict[str, Any]:
    """
    Acts as the entry router. Classifies user query intent using structured output.
    This is designed to be extremely fast and lightweight.
    """
    query = state["query"]
    chat_history = state.get("chat_history") or []
    
    llm_service = LLMService()
    system_instruction = INTENT_ROUTER_SYSTEM_PROMPT
    
    # Use compact history (last 12 msgs, 150 chars each) to keep routing fast
    history_str = _format_chat_history(chat_history, max_messages=12, max_chars_per_msg=150)
        
    if history_str:
        prompt = f"Recent Chat History:\n{history_str}\n\nUser Message: {query}"
    else:
        prompt = f"User Message: {query}"
        
    classification = llm_service.generate_structured_groq(
        prompt=prompt,
        schema=IntentClassification,
        system_instruction=system_instruction,
        temperature=0.0
    )
    
    return {
        "intent": classification.intent
    }


def chitchat_node(state: QueryState) -> Dict[str, Any]:
    """Generates a fast, low-latency greeting/pleasantry response."""
    query = state["query"]
    llm_service = LLMService()
    
    answer = llm_service.generate_groq(
        prompt=query,
        system_instruction=CHITCHAT_SYSTEM_PROMPT,
        temperature=0.7
    )
    return {"answer": answer}


def reminder_node(state: QueryState) -> Dict[str, Any]:
    """Extracts reminder details using ChatGroq structured outputs and saves into PostgreSQL."""
    chat_id = state["chat_id"]
    query = state["query"]
    current_time = state["current_time"]
    chat_history = state.get("chat_history") or []
    
    llm_service = LLMService()
    system_instruction = REMINDER_SYSTEM_PROMPT.format(current_time=current_time)
    
    # Use last 16 messages for reminder context resolution
    history_str = _format_chat_history(chat_history, max_messages=16, max_chars_per_msg=200)
        
    if history_str:
        prompt = f"Recent Chat History:\n{history_str}\n\nUser Query: {query}"
    else:
        prompt = f"User Query: {query}"
        
    extracted = llm_service.generate_structured_groq(
        prompt=prompt,
        schema=ReminderExtractor,
        system_instruction=system_instruction,
        temperature=0.0
    )
    
    text = extracted.reminder_text
    time_str = extracted.trigger_time
    
    if not time_str:
        return {"answer": "I understood you wanted a reminder, but I couldn't resolve the exact time. Could you specify it?"}
        
    from datetime import timezone, timedelta
    ist = timezone(timedelta(hours=5, minutes=30))
    try:
        trigger_time = datetime.fromisoformat(time_str)
        if trigger_time.tzinfo is None:
            trigger_time = trigger_time.replace(tzinfo=ist)
    except Exception:
        trigger_time = datetime.now(ist)
        
    db = ProfileMemoryDB()
    db.add_reminder(chat_id, text, trigger_time)
    
    formatted_time = trigger_time.strftime("%A, %b %d at %I:%M %p IST")
    return {
        "answer": f"⏰ Done! I've set a reminder to '{text}' for {formatted_time}."
    }


def retrieval_node(state: QueryState) -> Dict[str, Any]:
    """
    Loads both profile facts from PostgreSQL AND matching chunks from Pinecone.
    Uses ChatGroq to optimize the query into search keywords before querying Pinecone.
    """
    query = state["query"]
    chat_history = state.get("chat_history") or []
    
    llm_service = LLMService()
    vector_db = VectorDBService()
    db = ProfileMemoryDB()
    
    # 1. Load PostgreSQL facts
    facts = db.get_all_facts(state["chat_id"])
    reminders = db.get_reminders_by_chat(state["chat_id"])
    
    # 2. Optimize query for semantic search
    system_instruction = QUERY_OPTIMIZER_SYSTEM_PROMPT
    
    # Format compact history (last 8 messages) to resolve pronouns in optimized queries
    history_str = _format_chat_history(chat_history, max_messages=8, max_chars_per_msg=200)
    if history_str:
        prompt = f"Recent Chat History:\n{history_str}\n\nRaw Query: {query}"
    else:
        prompt = f"Raw Query: {query}"
        
    optimized = llm_service.generate_structured_groq(
        prompt=prompt,
        schema=QueryOptimizer,
        system_instruction=system_instruction,
        temperature=0.0
    )
    
    # 3. Embed optimized query & search Pinecone
    query_vector = llm_service.embed_text(optimized.search_query)
    matches = vector_db.query_vectors(
        query_vector=query_vector,
        top_k=5,
        metadata_filter={"chat_id": state["chat_id"]}
    )
    
    return {
        "profile_facts": facts,
        "pinecone_context": matches,
        "active_reminders": reminders
    }


async def web_search_node(state: QueryState) -> Dict[str, Any]:
    """
    Decides if the query needs real-time web search based on loaded PostgreSQL/Pinecone facts
    and recent chat history context.
    If yes, updates Telegram status and runs a DuckDuckGo search.
    """
    query = state["query"]
    profile_facts = state.get("profile_facts") or []
    pinecone_context = state.get("pinecone_context") or []
    chat_history = state.get("chat_history") or []
    status_callback = state.get("status_callback")
    
    # Format current facts for the search decision
    facts_str = "No profile facts or memories available."
    if profile_facts or pinecone_context:
        facts_list = []
        for fact in profile_facts:
            facts_list.append(f"- Fact: {fact['fact']}")
        for match in pinecone_context:
            meta = match.get("metadata") or {}
            text = meta.get("text", "")
            facts_list.append(f"- Memory Document: {text}")
        facts_str = "\n".join(facts_list)
    
    # Include recent chat history so the search decider can resolve context
    # (e.g. user says "Weather" after saying "I am based in Jawad MP India")
    history_str = _format_chat_history(chat_history, max_messages=12, max_chars_per_msg=150)
        
    llm_service = LLMService()
    
    # Build prompt with all available context
    prompt_parts = [f"USER MEMORIES & FACTS:\n{facts_str}"]
    if history_str:
        prompt_parts.append(f"RECENT CONVERSATION:\n{history_str}")
    prompt_parts.append(f"USER QUERY: {query}")
    
    # Call decider
    decision = llm_service.generate_structured_groq(
        prompt="\n\n".join(prompt_parts),
        schema=SearchDecider,
        system_instruction=SEARCH_DECIDER_SYSTEM_PROMPT,
        temperature=0.0
    )
    
    if decision.needs_search and decision.search_query:
        # Trigger dynamic status callback on Telegram if registered
        if status_callback:
            try:
                await status_callback("🔍 Searching the web...")
            except Exception as callback_err:
                logger.warning(f"Failed to update Telegram status callback: {str(callback_err)}")
                
        # Run DuckDuckGo Search
        from src.services.search import WebSearchService
        search_service = WebSearchService()
        search_results = search_service.search(decision.search_query)
        
        return {
            "web_search_context": search_results
        }
        
    return {
        "web_search_context": []
    }


def generation_node(state: QueryState) -> Dict[str, Any]:
    """Synthesizes final RAG answer using ChatGroq."""
    llm_service = LLMService()
    
    query = state["query"]
    profile_facts = state.get("profile_facts") or []
    pinecone_context = state.get("pinecone_context") or []
    active_reminders = state.get("active_reminders") or []
    web_search_context = state.get("web_search_context") or []
    chat_history = state.get("chat_history") or []
    
    # Format PostgreSQL Profile section
    profile_section = "No stored profile facts found about the user."
    if profile_facts:
        profile_section = "\n".join(f"- {fact['fact']}" for fact in profile_facts)
        
    # Format Active Reminders section
    reminders_section = "No active pending reminders scheduled."
    if active_reminders:
        reminders_section = "\n".join(
            f"- '{r['reminder_text']}' scheduled for {r['trigger_time']}"
            for r in active_reminders
        )
        
    # Format Web Search results
    search_section = "No recent web search results available."
    if web_search_context:
        formatted_results = []
        for idx, result in enumerate(web_search_context):
            ref = f"[{idx + 1}] Source: Web Search - {result.get('title')}"
            link = result.get("link")
            if link:
                ref += f" ({link})"
            formatted_results.append(f"{ref}\nContent:\n{result.get('snippet')}\n")
        search_section = "\n---\n".join(formatted_results)
        
    # Format Pinecone context matches
    context_section = "No saved document context matches found in vector memory."
    if pinecone_context:
        formatted_matches = []
        for idx, match in enumerate(pinecone_context):
            meta = match.get("metadata") or {}
            text = meta.get("text", "")
            itype = meta.get("input_type", "document")
            timestamp = meta.get("timestamp", "unknown time")
            url = meta.get("source_url", "")
            
            ref = f"[{idx + 1}] Source: {itype} (Saved: {timestamp})"
            if url:
                ref += f" URL: {url}"
            formatted_matches.append(f"{ref}\nContent:\n{text}\n")
        context_section = "\n---\n".join(formatted_matches)
    
    # Format recent conversation context (last 20 messages for generation)
    history_section = "No recent conversation history."
    history_str = _format_chat_history(chat_history, max_messages=20, max_chars_per_msg=250)
    if history_str:
        history_section = history_str
        
    system_instruction = RAG_GENERATION_SYSTEM_PROMPT
    
    prompt = (
        f"USER PROFILE FACTS:\n"
        f"{profile_section}\n\n"
        f"ACTIVE PENDING REMINDERS:\n"
        f"{reminders_section}\n\n"
        f"RECENT CONVERSATION HISTORY:\n"
        f"{history_section}\n\n"
        f"WEB SEARCH RESULTS CONTEXT:\n"
        f"{search_section}\n\n"
        f"SAVED VECTOR DOCUMENTS CONTEXT:\n"
        f"{context_section}\n\n"
        f"USER QUERY: {query}\n"
    )
    
    answer = llm_service.generate_groq(
        prompt=prompt,
        system_instruction=system_instruction,
        temperature=0.3
    )
    
    return {
        "answer": answer
    }


class FactDeletionSelector(BaseModel):
    fact_id: Optional[int]  # ID of the fact to delete, or None if no match
    explanation: str        # Explanation of matching logic


def delete_fact_node(state: QueryState) -> Dict[str, Any]:
    # Resolves which fact the user wants to delete and removes it from PostgreSQL.
    query = state["query"]
    db = ProfileMemoryDB()
    facts = db.get_all_facts(state["chat_id"])
    
    if not facts:
        return {"answer": "You don't have any facts stored in your profile memory to delete."}
        
    llm_service = LLMService()
    
    # Format facts with IDs
    facts_list = "\n".join(f"ID {f['id']}: {f['fact']}" for f in facts)
    
    system_instruction = FACT_DELETION_SYSTEM_PROMPT.format(facts_list=facts_list)
    
    selection = llm_service.generate_structured_groq(
        prompt=f"User Forget Request: {query}",
        schema=FactDeletionSelector,
        system_instruction=system_instruction,
        temperature=0.0
    )
    
    if selection.fact_id is not None:
        # Find the text of the deleted fact
        deleted_text = next((f["fact"] for f in facts if f["id"] == selection.fact_id), "Unknown fact")
        success = db.delete_fact(selection.fact_id)
        if success:
            return {"answer": f"🗑️ I've forgotten that: \"{deleted_text}\""}
            
    return {"answer": "I couldn't find a matching fact in my memory to forget. Could you be more specific?"}
