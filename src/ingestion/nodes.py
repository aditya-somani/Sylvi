import uuid
import logging
from datetime import datetime
from typing import Dict, Any, List

# LangGraph state representation
from src.ingestion.state import IngestionState

# Services (Lazy instantiated inside nodes to prevent global side-effects during imports)
from src.services.llm import LLMService
from src.services.vector_db import VectorDBService
from src.services.scrapers import WebScraperService

# Official langchain splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# --- Ingestion Nodes ---

def text_processor_node(state: IngestionState) -> Dict[str, Any]:
    """Processes plain text input."""
    logger.info("Running text_processor_node...")
    raw = state["raw_content"]
    return {
        "processed_text": raw
    }


async def url_processor_node(state: IngestionState) -> Dict[str, Any]:
    """Scrapes a URL and uses Gemini to summarize and de-noise it."""
    logger.info("Running url_processor_node...")
    url = state["raw_content"]
    
    # Lazy instantiate services to avoid import-time connection attempts
    scraper_service = WebScraperService()
    llm_service = LLMService()
    
    # 1. Scrape the URL
    scraped_content = await scraper_service.scrape_url(url)
    
    # 2. De-noise & Summarize using Gemini 1.5 Flash
    system_instruction = (
        "You are an expert content ingest engineer. Clean and structure the provided text "
        "into a dense, highly informative Markdown document. Extract all key facts, technical "
        "specifications, terms, names, dates, and core context. Ignore advertisements, "
        "navigation boilerplate, cookies, and irrelevant sidebar text. Do not add conversational fluff; "
        "output only the structured markdown."
    )
    prompt = f"Scraped Page Content:\n\n{scraped_content}"
    
    clean_markdown = llm_service.generate_gemini(
        prompt=prompt,
        system_instruction=system_instruction,
        temperature=0.1
    )
    
    # Store source URL in state metadata
    metadata = state.get("metadata")
    updated_metadata = dict(metadata) if metadata else {}
    updated_metadata["source_url"] = url
    
    return {
        "processed_text": clean_markdown,
        "metadata": updated_metadata
    }


def voice_processor_node(state: IngestionState) -> Dict[str, Any]:
    """Transcribes a local voice file using Groq Whisper-large-v3."""
    logger.info("Running voice_processor_node...")
    audio_path = state["raw_content"]
    
    llm_service = LLMService()
    transcript = llm_service.transcribe_voice(audio_path)
    
    return {
        "processed_text": f"[Voice Transcript - {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}]\n{transcript}"
    }


def image_processor_node(state: IngestionState) -> Dict[str, Any]:
    """Generates a detailed description of an image using Gemini Flash Vision."""
    logger.info("Running image_processor_node...")
    image_path = state["raw_content"]
    
    llm_service = LLMService()
    description = llm_service.describe_image(image_path)
    
    return {
        "processed_text": f"[Image Description]\n{description}"
    }


def chunk_node(state: IngestionState) -> Dict[str, Any]:
    """Splits processed text into smaller chunks using RecursiveCharacterTextSplitter."""
    logger.info("Running chunk_node...")
    text = state.get("processed_text", "")
    if not text:
        raise ValueError("No processed text found to chunk.")
        
    # Use standard RecursiveCharacterTextSplitter from langchain
    # Chunks are limited to 800 characters with a 100 character overlap
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks: List[str] = splitter.split_text(text)
    
    return {
        "chunks": chunks
    }


def embed_node(state: IngestionState) -> Dict[str, Any]:
    """Generates embeddings for each text chunk using Gemini Embeddings API."""
    logger.info("Running embed_node...")
    chunks = state.get("chunks")
    if not chunks:
        raise ValueError("No chunks found to embed.")
        
    llm_service = LLMService()
    embeddings: List[List[float]] = []
    
    for chunk in chunks:
        # Generates 768-dimensional embedding
        vector = llm_service.embed_text(chunk)
        embeddings.append(vector)
        
    return {
        "embeddings": embeddings
    }


def upsert_node(state: IngestionState) -> Dict[str, Any]:
    """Prepares and upserts vectors into Pinecone."""
    logger.info("Running upsert_node...")
    chunks = state.get("chunks")
    embeddings = state.get("embeddings")
    
    if not chunks or not embeddings or len(chunks) != len(embeddings):
        raise ValueError("Mismatch or missing elements between chunks and embeddings.")
        
    vector_db_service = VectorDBService()
    metadata = state.get("metadata")
    base_metadata = dict(metadata) if metadata else {}
    timestamp_str = datetime.utcnow().isoformat()
    
    pinecone_vectors: List[Dict[str, Any]] = []
    
    for idx, (chunk, vector) in enumerate(zip(chunks, embeddings)):
        unique_id = f"doc_{uuid.uuid4().hex[:12]}_{idx}"
        
        # Build Pinecone metadata schema
        vector_metadata = {
            "text": chunk,
            "input_type": state["input_type"],
            "timestamp": timestamp_str,
            "source": base_metadata.get("source", "telegram"),
            "message_id": base_metadata.get("message_id", ""),
            "source_url": base_metadata.get("source_url", "")
        }
        
        pinecone_vectors.append({
            "id": unique_id,
            "values": vector,
            "metadata": vector_metadata
        })
        
    # Write directly to Pinecone
    vector_db_service.upsert_vectors(pinecone_vectors)
    
    return {
        "vector_payloads": pinecone_vectors
    }
