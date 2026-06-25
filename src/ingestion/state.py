from typing import TypedDict, List, Dict, Any, Optional

class IngestionState(TypedDict):
    """
    Defines the shared state container flowing through the LangGraph ingestion pipeline.
    """
    # --- Inputs ---
    # Allowed: "text" | "link" | "voice" | "image"
    input_type: str
    
    # Can be raw text, a webpage URL, or a local file path to audio/image
    raw_content: str

    # --- Processed Data ---
    # The clean Markdown text body (extracted from URL, transcribed from audio, or captioned from image)
    processed_text: Optional[str]
    
    # Processed text divided into smaller semantic chunks
    chunks: Optional[List[str]]
    
    # High-dimensional embeddings corresponding to the chunks
    embeddings: Optional[List[List[float]]]
    
    # List of structured objects prepared for Pinecone upsert
    vector_payloads: Optional[List[Dict[str, Any]]]

    # --- Metadata ---
    # Common metadata (e.g., source: "telegram", timestamp, message_id)
    metadata: Optional[Dict[str, Any]]
