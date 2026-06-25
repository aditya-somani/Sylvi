import json
import logging
from datetime import datetime
from typing import Dict, Any, List

# Import State representation
from src.query.state import QueryState

# Services (Lazy instantiated inside nodes to prevent global side-effects during imports)
from src.services.llm import LLMService
from src.services.vector_db import VectorDBService
from src.memory.profile import ProfileMemoryDB

logger = logging.getLogger(__name__)

# --- In-Node Helper Functions ---

def parse_json_safely(text: str) -> Dict[str, Any]:
    """Cleans markdown code fences and parses JSON safely."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 2:
            cleaned = "\n".join(lines[1:-1]) if lines[-1].startswith("```") else "\n".join(lines[1:])
    try:
        return json.loads(cleaned.strip())
    except Exception as e:
        logger.error(f"Failed to parse JSON: {cleaned}. Error: {str(e)}")
        # Fallback to general retrieval in case of format failure
        return {"intent": "retrieval"}


# --- Query Nodes ---

def intent_router_node(state: QueryState) -> Dict[str, Any]:
    """
    Acts as the entry router. Classifies user query intent and parses
    relative time instructions (e.g. 'in 5 minutes', 'tomorrow morning')
    using the injected current_time clock context.
    """
    query = state["query"]
    current_time = state["current_time"]
    
    llm_service = LLMService()
    
    system_instruction = (
        "You are the intent router for Sylvi, a stateful personal memory copilot.\n"
        f"Internal Clock Context (Current UTC Time): {current_time}\n\n"
        "Your task is to classify the user's query and parse its parameters.\n\n"
        "Classify into one of these intents:\n"
        "1. 'chit_chat': Simple greetings ('hi', 'hello', 'hey'), thank yous, bye, or basic polite banter.\n"
        "2. 'reminder': The user wants to schedule a reminder (e.g., 'remind me to...', 'set a reminder at...', 'remind me in 10 minutes').\n"
        "3. 'profile_query': The user is asking about details they expect you to know about them personally, "
        "their preferences, or settings (e.g., 'What is my favorite language?', 'What do you know about me?').\n"
        "4. 'retrieval': The user is searching their saved documents, web links, or transcripts (e.g., 'What did I save about LangGraph?', 'Search my notes for Python').\n\n"
        "Strict Formatting:\n"
        "You MUST respond ONLY with a JSON object containing:\n"
        '- "intent": the classified intent string.\n'
        '- "reminder_text": (Only if intent is "reminder") The description of the task to be reminded of.\n'
        '- "trigger_time": (Only if intent is "reminder") The absolute trigger time in YYYY-MM-DDTHH:MM:SS format, '
        "calculated based on the user's relative expression and the Internal Clock Context.\n"
        "  - If user says 'in the afternoon', schedule for 14:00:00.\n"
        "  - If user says 'tomorrow morning', schedule for 09:00:00 next day.\n"
        "  - Make sure to correctly resolve relative offsets like 'in 5 minutes' by adding to the current time.\n"
    )
    
    prompt = f"User Message: {query}"
    
    response_text = llm_service.generate_groq(
        prompt=prompt,
        system_instruction=system_instruction,
        temperature=0.0
    )
    
    parsed = parse_json_safely(response_text)
    
    return {
        "intent": parsed.get("intent", "retrieval"),
        "reminder_details": parsed if parsed.get("intent") == "reminder" else None
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
    """Saves the parsed reminder into SQLite."""
    chat_id = state["chat_id"]
    details = state.get("reminder_details") or {}
    
    text = details.get("reminder_text", "Reminder notification")
    time_str = details.get("trigger_time")
    
    if not time_str:
        return {"answer": "I understood you wanted a reminder, but I couldn't resolve the exact time. Could you specify it?"}
        
    try:
        trigger_time = datetime.fromisoformat(time_str)
    except Exception:
        trigger_time = datetime.utcnow()
        
    db = ProfileMemoryDB()
    db.add_reminder(chat_id, text, trigger_time)
    
    formatted_time = trigger_time.strftime("%A, %b %d at %I:%M %p UTC")
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
    Retrieves facts and vectors concurrently.
    """
    query = state["query"]
    
    llm_service = LLMService()
    vector_db = VectorDBService()
    db = ProfileMemoryDB()
    
    # 1. Load SQLite facts
    facts = db.get_all_facts()
    
    # 2. Embed user query & search Pinecone
    query_vector = llm_service.embed_text(query)
    matches = vector_db.query_vectors(query_vector=query_vector, top_k=5)
    
    return {
        "profile_facts": facts,
        "pinecone_context": matches
    }


def generation_node(state: QueryState) -> Dict[str, Any]:
    """Synthesizes final RAG answer using Gemini 1.5 Flash."""
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
    
    answer = llm_service.generate_gemini(
        prompt=prompt,
        system_instruction=system_instruction,
        temperature=0.3
    )
    
    return {
        "answer": answer
    }
