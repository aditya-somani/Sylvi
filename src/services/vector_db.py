import logging
from typing import List, Dict, Any, Optional
from pinecone import Pinecone
from src.config import settings

logger = logging.getLogger(__name__)

class VectorDBService:
    """
    Service wrapper for Pinecone Vector Database.
    Implements indexing (upserting) and metadata-filtered querying.
    """
    def __init__(self):
        self.api_key = settings.PINECONE_API_KEY
        self.index_name = settings.PINECONE_INDEX_NAME
        
        # Initialize Pinecone client once and cache the index instance.
        # This reuses connection pools and avoids repeated index resolution calls,
        # directly improving query latency (p50/p99 optimization).
        self.pc = Pinecone(api_key=self.api_key)
        self.index = self.pc.Index(self.index_name)

    def upsert_vectors(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Upserts a list of vectors into the Pinecone index.
        
        Each item in `items` should be structured as:
        {
            "id": "unique-chunk-id",
            "values": [0.1, -0.2, ...], # 768-dim embedding
            "metadata": {
                "text": "original chunk content",
                "source": "telegram",
                "type": "link | voice | text | image",
                "timestamp": "2026-06-24...",
                "url": "https://..." # optional
            }
        }
        """
        try:
            logger.info(f"Upserting {len(items)} vectors to index: {self.index_name}")
            # Pinecone expects list of dicts: {"id": str, "values": list, "metadata": dict}
            response = self.index.upsert(vectors=items)
            return response
        except Exception as e:
            logger.error(f"Failed to upsert vectors to Pinecone: {str(e)}")
            raise e

    def query_vectors(
        self, 
        query_vector: List[float], 
        top_k: int = 5, 
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Queries Pinecone for the most similar vectors.
        
        Parameters:
        - query_vector: The embedding vector of the search query.
        - top_k: Number of documents to retrieve.
        - metadata_filter: Pinecone metadata filter dict (e.g. {"type": "voice"}).
        """
        try:
            # We set include_values=False because we don't need the dense embeddings
            # returned in the response payload. This saves network bandwidth and reduces latency.
            response = self.index.query(
                vector=query_vector,
                top_k=top_k,
                filter=metadata_filter,
                include_metadata=True,
                include_values=False
            )
            
            # Extract matches into a clean list of dicts
            results = []
            for match in response.get("matches", []):
                results.append({
                    "id": match["id"],
                    "score": match["score"],
                    "metadata": match.get("metadata", {})
                })
            return results
        except Exception as e:
            logger.error(f"Failed to query vectors from Pinecone: {str(e)}")
            raise e
