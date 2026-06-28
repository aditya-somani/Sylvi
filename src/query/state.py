from typing import TypedDict, List, Dict, Any, Optional

class QueryState(TypedDict):
    """
    Optimized state container flowing through the LangGraph query/retrieval pipeline.
    """
    # --- Inputs ---
    # The raw message query from the user
    query: str
    
    # The Telegram chat/user ID
    chat_id: str
    
    # The absolute current local/UTC timestamp (internal clock context)
    # Passed in from the runner on startup (e.g., '2026-06-25T14:29:42')
    current_time: str

    # The recent chat history messages
    chat_history: Optional[List[Dict[str, str]]]

    # --- Routing & Intent ---
    # Determined by the Intent Router: 'chit_chat' | 'reminder' | 'retrieval' | 'profile_query'
    intent: Optional[str]

    # --- Retrieved Contexts ---
    # User facts loaded from PostgreSQL, containing 'id' and 'fact' keys
    profile_facts: Optional[List[Dict[str, Any]]]
    
    # Match chunks from Pinecone
    pinecone_context: Optional[List[Dict[str, Any]]]
    
    # Active reminders loaded from SQL for this user
    active_reminders: Optional[List[Dict[str, Any]]]
    
    # Web search results retrieved from DuckDuckGo
    web_search_context: Optional[List[Dict[str, Any]]]
    
    # Thread-safe status update callback function for Telegram UI
    status_callback: Optional[Any]

    # --- Parsed Details ---
    # Parsed reminder details if intent is 'reminder' (e.g., {'text': 'buy milk', 'trigger_time': '2026-06-25T17:00:00'})
    reminder_details: Optional[Dict[str, Any]]

    # --- Output ---
    # The final synthesized answer or confirmation message
    answer: Optional[str]
