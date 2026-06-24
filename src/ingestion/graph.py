from langgraph.graph import StateGraph, START, END

from src.ingestion.state import IngestionState
from src.ingestion.nodes import (
    text_processor_node,
    url_processor_node,
    voice_processor_node,
    image_processor_node,
    chunk_node,
    embed_node,
    upsert_node
)

# 1. Define the routing function
def route_by_input_type(state: IngestionState) -> str:
    """
    Evaluates the input_type and determines which processor node
    should be executed next.
    """
    val = state.get("input_type")
    if val in ["text", "link", "voice", "image"]:
        return val
    raise ValueError(f"Invalid input_type in state: {val}")


# 2. Build the stateful graph
builder = StateGraph(IngestionState)

# Add all processing nodes
builder.add_node("text_processor", text_processor_node)
builder.add_node("url_processor", url_processor_node)
builder.add_node("voice_processor", voice_processor_node)
builder.add_node("image_processor", image_processor_node)

builder.add_node("chunk_node", chunk_node)
builder.add_node("embed_node", embed_node)
builder.add_node("upsert_node", upsert_node)

# 3. Define the edges
# Route from START to the corresponding processor based on input_type
builder.add_conditional_edges(
    START,
    route_by_input_type,
    {
        "text": "text_processor",
        "link": "url_processor",
        "voice": "voice_processor",
        "image": "image_processor"
    }
)

# All parallel processors converge back to the chunking node
builder.add_edge("text_processor", "chunk_node")
builder.add_edge("url_processor", "chunk_node")
builder.add_edge("voice_processor", "chunk_node")
builder.add_edge("image_processor", "chunk_node")

# Linear flow for indexing: Chunk -> Embed -> Upsert -> END
builder.add_edge("chunk_node", "embed_node")
builder.add_edge("embed_node", "upsert_node")
builder.add_edge("upsert_node", END)

# 4. Compile the graph
ingestion_graph = builder.compile()
