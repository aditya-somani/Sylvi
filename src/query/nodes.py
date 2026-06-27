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

logger = logging.getLogger(__name__)


# --- Pydantic Schemas for Structured Outputs ---

class IntentClassification(BaseModel):
    intent: Literal["chit_chat", "reminder", "profile_query", "retrieval"]

class ReminderExtractor(BaseModel):
    reminder_text: str
    trigger_time: str  # Format: YYYY-MM-DDTHH:MM:SS

class QueryOptimizer(BaseModel):
    search_query: str


# --- Query Nodes ---

def intent_router_node(state: QueryState) -> Dict[str, Any]:
    """
    Acts as the entry router. Classifies user query intent using structured output.
    This is designed to be extremely fast and lightweight.
    """
    query = state["query"]
    llm_service = LLMService()
    
    system_instruction = (
        "You are the intent router for Sylvi, a stateful personal memory copilot.\n"
        "Your task is to classify the user's query into one of these intents:\n"
        "1. 'chit_chat': Simple greetings ('hi', 'hello', 'hey'), thank yous, bye, or basic polite banter.\n"
        "2. 'reminder': The user wants to schedule a reminder (e.g., 'remind me to...', 'set a reminder').\n"
        "3. 'profile_query': The user is asking about details they expect you to know about them personally, "
        "their preferences, or settings (e.g., 'What is my favorite language?', 'What do you know about me?').\n"
        "4. 'retrieval': The user is searching their saved documents, web links, or transcripts (e.g., 'What did I save about LangGraph?')."
    )
    
    classification = llm_service.generate_structured_groq(
        prompt=f"User Message: {query}",
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
        system_instruction="You are Sylvi, a friendly personal memory copilot. Give a brief, pleasant response.",
        temperature=0.7
    )
    return {"answer": answer}


def reminder_node(state: QueryState) -> Dict[str, Any]:
    """Extracts reminder details using ChatGroq structured outputs and saves into SQLite."""
    chat_id = state["chat_id"]
    query = state["query"]
    current_time = state["current_time"]
    
    llm_service = LLMService()
    
    system_instruction = (
        "You are an expert information extraction assistant.\n"
        f"Internal Clock Context (Current UTC/Local Time): {current_time}\n\n"
        "Extract the task description to be reminded of and resolve the trigger time to absolute YYYY-MM-DDTHH:MM:SS format.\n"
        "Make sure to correctly resolve relative offsets (like 'in 5 minutes' or 'tomorrow morning') by adding to the current time.\n"
        "- If user says 'in the afternoon', schedule for 14:00:00 of the corresponding day.\n"
        "- If user says 'tomorrow morning', schedule for 09:00:00 next day.\n"
    )
    
    extracted = llm_service.generate_structured_groq(
        prompt=f"User Query: {query}",
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


def profile_query_node(state: QueryState) -> Dict[str, Any]:
    """Loads facts from SQLite for direct personal profile queries."""
    db = ProfileMemoryDB()
    facts = db.get_all_facts()
    return {
        "profile_facts": facts
    }


def retrieval_node(state: QueryState) -> Dict[str, Any]:
    """
    Loads both profile facts from SQLite AND matching chunks from Pinecone.
    Uses ChatGroq to optimize the query into search keywords before querying Pinecone.
    """
    query = state["query"]
    
    llm_service = LLMService()
    vector_db = VectorDBService()
    db = ProfileMemoryDB()
    
    # 1. Load SQLite facts
    facts = db.get_all_facts()
    
    # 2. Optimize query for semantic search
    system_instruction = (
        "You are a search query optimizer. Given a user query, extract the core entities, keywords, "
        "and technical terms to create a search query optimized for vector database retrieval."
    )
    
    optimized = llm_service.generate_structured_groq(
        prompt=f"Raw Query: {query}",
        schema=QueryOptimizer,
        system_instruction=system_instruction,
        temperature=0.0
    )
    
    # 3. Embed optimized query & search Pinecone
    query_vector = llm_service.embed_text(optimized.search_query)
    matches = vector_db.query_vectors(query_vector=query_vector, top_k=5)
    
    return {
        "profile_facts": facts,
        "pinecone_context": matches
    }


def generation_node(state: QueryState) -> Dict[str, Any]:
    """Synthesizes final RAG answer using ChatGroq."""
    llm_service = LLMService()
    
    query = state["query"]
    profile_facts = state.get("profile_facts") or []
    pinecone_context = state.get("pinecone_context") or []
    
    # Format SQLite Profile section
    profile_section = "No stored profile facts found about the user."
    if profile_facts:
        profile_section = "\n".join(f"- {fact['fact']}" for fact in profile_facts)
        
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
        
    system_instruction = (
        "You are Sylvi, an intelligent, personal memory copilot. Your objective is to answer "
        "the user's query utilizing their structured profile facts and saved vector documents context.\n\n"
        "Rules:\n"
        "1. Give direct, helpful answers.\n"
        "2. If you base your answer on a saved document, cite the source index (e.g. [1]) and "
        "mention the source type (e.g., 'According to the article you saved...').\n"
        "3. If the context does not contain the answer, explain politely that you do not "
        "remember or have not saved that information."
    )
    
    prompt = (
        f"USER PROFILE FACTS:\n"
        f"{profile_section}\n\n"
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
    """Resolves which fact the user wants to delete and removes it from SQLite."""
    query = state["query"]
    db = ProfileMemoryDB()
    facts = db.get_all_facts()
    
    if not facts:
        return {"answer": "You don't have any facts stored in your profile memory to delete."}
        
    llm_service = LLMService()
    
    # Format facts with IDs
    facts_list = "\n".join(f"ID {f['id']}: {f['fact']}" for f in facts)
    
    system_instruction = (
        "You are a memory manager assistant. Your job is to analyze the user's request to 'forget' "
        "or 'delete' a fact about themselves, and match it against their current stored facts.\n\n"
        "Facts List:\n"
        f"{facts_list}\n\n"
        "Determine if any fact matches the deletion request. Return the ID of the matching fact. "
        "If no fact matches, return null for fact_id."
    )
    
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
