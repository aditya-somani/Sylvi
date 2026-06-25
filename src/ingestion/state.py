from typing import TypedDict, List, Dict, Any, Optional

class I have a doubt about how the `root_by_input_type` node classifies the input type. I don't see it calling any API or anything like that, but it seems to be using the `input_type` attribute, which is a built-in attribute of the `IngestionState` class.

I don't understand how `IngestionState` is classifying the input type, because I don't think it's going to do that automatically. How does it know which type of input it is? I have a doubt about how the `root_by_input_type` node classifies the input type. I don't see it calling any API or anything like that, but it seems to be using the `input_type` attribute, which is a built-in attribute of the `IngestionState` class.

I don't understand how `IngestionState` is classifying the input type, because I don't think it's going to do that automatically. How does it know which type of input it is? IngestionState(TypedDict):
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
