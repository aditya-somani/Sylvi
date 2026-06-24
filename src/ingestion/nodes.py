import time
import uuid
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple

from src.ingestion.state import IngestionState
from src.services.llm import LLMService
from src.services.vector_db import VectorDBService
from src.services.scrapers import WebScraperService

logger = logging.getLogger(__name__)

# Initialize services
llm_service = LLMService()
vector_db_service = VectorDBService()
scraper_service = WebScraperService()


def _split_text_recursively(text: str, max_chars: int = 1000, overlap: int = 100) -> List[str]:
    """
    A lightweight, custom Recursive Character Text Splitter.
    Splits text by paragraphs, lines, sentences, or words to stay within max_chars,
    maintaining a sliding-window overlap.
    """
    if len(text) <= max_chars:
        return [text]

    separators = ["\n\n", "\n", ". ", " "]
    chunks = []
    
    # Simple recursive split implementation
    def split_helper(subtext: str):
        if len(subtext) <= max_chars:
            chunks.append(subtext)
            return

        for sep in separators:
            if sep in subtext:
                parts = subtext.split(sep)
                current_chunk = []
                current_len = 0
                
                for part in parts:
                    part_len = len(part) + len(sep)
                    if current_len + part_len > max_chars:
                        if current_chunk:
                            chunks.append(sep.join(current_chunk))
                        # Standard overlapping sliding window
                        overlap_text = current_chunk[-1] if current_chunk else ""
                        current_chunk = [overlap_text[-overlap:], part] if overlap_text else [part]
                        current_len = sum(len(c) for c in current_chunk) + len(sep) * (len(current_chunk) - 1)
                    else:
                        current_chunk.append(part)
                        current_len += part_len
                
                if current_chunk:
                    chunks.append(sep.join(current_chunk))
                return
        
        # Fallback if no separator matches
        for i in range(0, len(subtext), max_chars - overlap):
            chunks.append(subtext[i : i + max_chars])

    split_helper(text)
    # Clean up empty chunks and trim whitespace
    return [c.strip() for c in chunks if c.strip()]


# --- Nodes ---

def text_processor_node(state: IngestionState) -> Dict[str, Any]:
    """Processes plain text input."""
    start_time = time.time()
    logger.info("Running text_processor_node...")
    
    raw = state["raw_content"]
    latency = (time.time() - start_time) * 1000
    
    return {
        "processed_text": raw,
        "latency_ms": {"text_processor": latency}
    }


async def url_processor_node(state: IngestionState) -> Dict[str, Any]:
    """Scrapes a URL and uses Gemini to summarize and de-noise it."""
    start_time = time.time()
    logger.info("Running url_processor_node...")
    
    url = state["raw_content"]
    
    # 1. Scrape the URL (with readability-lxml cleaning)
    scrape_start = time.time()
    scraped_content = await scraper_service.scrape_url(url)
    scrape_duration = (time.time() - scrape_start) * 1000
    
    # 2. De-noise & Summarize using Gemini 1.5 Flash
    llm_start = time.time()
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
    llm_duration = (time.time() - llm_start) * 1000
    
    latency = (time.time() - start_time) * 1000
    
    # Store source URL in state metadata
    updated_metadata = state.get("metadata") or {}
    updated_metadata["source_url"] = url
    
    return {
        "processed_text": clean_markdown,
        "metadata": updated_metadata,
        "latency_ms": {
            "url_scraper_substep": scrape_duration,
            "url_llm_clean_substep": llm_duration,
            "url_processor": latency
        }
    }


def voice_processor_node(state: IngestionState) -> Dict[str, Any]:
    """Transcribes a local voice file using Groq Whisper-large-v3."""
    start_time = time.time()
    logger.info("Running voice_processor_node...")
    
    audio_path = state["raw_content"]
    
    transcript = llm_service.transcribe_voice(audio_path)
    
    latency = (time.time() - start_time) * 1000
    
    return {
        "processed_text": f"[Voice Transcript - {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}]\n{transcript}",
        "latency_ms": {"voice_processor": latency}
    }


def image_processor_node(state: IngestionState) -> Dict[str, Any]:
    """Generates a detailed description of an image using Gemini Flash Vision."""
    start_time = time.time()
    logger.info("Running image_processor_node...")
    
    image_path = state["raw_content"]
    
    description = llm_service.describe_image(image_path)
    
    latency = (time.time() - start_time) * 1000
    
    return {
        "processed_text": f"[Image Description]\n{description}",
        "latency_ms": {"image_processor": latency}
    }


def chunk_node(state: IngestionState) -> Dict[str, Any]:
    """Splits processed text into smaller semantic chunks."""
    start_time = time.time()
    logger.info("Running chunk_node...")
    
    text = state.get("processed_text", "")
    if not text:
        raise ValueError("No processed text found to chunk.")
        
    # Split text into chunks of max 800 characters with 100 characters overlap
    chunks = _split_text_recursively(text, max_chars=800, overlap=100)
    
    latency = (time.time() - start_time) * 1000
    
    # Merge existing state latency dict if present
    latencies = state.get("latency_ms") or {}
    latencies["chunker"] = latency
    
    return {
        "chunks": chunks,
        "latency_ms": latencies
    }


def embed_node(state: IngestionState) -> Dict[str, Any]:
    """Generates embeddings for each text chunk using Gemini Embeddings API."""
    start_time = time.time()
    logger.info("Running embed_node...")
    
    chunks = state.get("chunks")
    if not chunks:
        raise ValueError("No chunks found to embed.")
        
    embeddings = []
    for chunk in chunks:
        # Generates 768-dimensional embedding
        vector = llm_service.embed_text(chunk)
        embeddings.append(vector)
        
    latency = (time.time() - start_time) * 1000
    
    latencies = state.get("latency_ms") or {}
    latencies["embedder"] = latency
    
    return {
        "embeddings": embeddings,
        "latency_ms": latencies
    }


def upsert_node(state: IngestionState) -> Dict[str, Any]:
    """Prepares and upserts vectors into Pinecone."""
    start_time = time.time()
    logger.info("Running upsert_node...")
    
    chunks = state.get("chunks")
    embeddings = state.get("embeddings")
    
    if not chunks or not embeddings or len(chunks) != len(embeddings):
        raise ValueError("Mismatch or missing elements between chunks and embeddings.")
        
    base_metadata = state.get("metadata") or {}
    timestamp_str = datetime.utcnow().isoformat()
    
    pinecone_vectors = []
    
    for idx, (chunk, vector) in enumerate(zip(chunks, embeddings)):
        unique_id = f"doc_{uuid.uuid4().hex[:12]}_{idx}"
        
        # Build Pinecone metadata schema
        # We store the raw text inside metadata so we can display it upon retrieval
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
    
    latency = (time.time() - start_time) * 1000
    
    latencies = state.get("latency_ms") or {}
    latencies["upserter"] = latency
    
    return {
        "vector_payloads": pinecone_vectors,
        "latency_ms": latencies
    }
