from langgraph.graph import StateGraph, START, END

# Import State representation
from src.query.state import QueryState

# Import Nodes
from src.query.nodes import (
    intent_router_node,
    chitchat_node,
    reminder_node,
    retrieval_node,
    generation_node,
    delete_fact_node
)

# --- Graph Routing Logic ---

def route_intent(state: QueryState) -> str:
    """Routes execution based on the intent set by the router node."""
    val = state.get("intent")
    if val == "profile_query":
        return "retrieval"
    if val in ["chit_chat", "reminder", "retrieval", "delete_profile_fact"]:
        return str(val)
    return "retrieval" # Fallback


# --- Graph Compilation ---

builder = StateGraph(QueryState)

# Add all nodes
builder.add_node("intent_router", intent_router_node)
builder.add_node("chitchat", chitchat_node)
builder.add_node("reminder", reminder_node)
builder.add_node("retrieval", retrieval_node)
builder.add_node("generation", generation_node)
builder.add_node("delete_fact", delete_fact_node)

# Set entry point
builder.add_edge(START, "intent_router")

# Define conditional branching
builder.add_conditional_edges(
    "intent_router",
    route_intent,
    {
        "chit_chat": "chitchat",
        "reminder": "reminder",
        "retrieval": "retrieval",
        "delete_profile_fact": "delete_fact"
    }
)

# Terminate early for fast routes
builder.add_edge("chitchat", END)
builder.add_edge("reminder", END)
builder.add_edge("delete_fact", END)

# Converted routes flow to the generation node
builder.add_edge("retrieval", "generation")
builder.add_edge("generation", END)

# Compile the final graph
query_graph = builder.compile()
