import os
import itertools
from typing import List, Optional, Iterator, Any
from PIL import Image
from google import genai
from google.genai import types
from groq import Groq
from src.config import settings

class APIKeyRotator:
    """
    A helper class that rotates through a list of API keys in a round-robin fashion.
    This acts as a basic load balancer to distribute requests across multiple accounts
    and bypass free-tier rate limits.
    """
    def __init__(self, raw_keys: str, name_for_errors: str):
        self.name_for_errors: str = name_for_errors
        self.keys: List[str] = [k.strip() for k in raw_keys.split(",") if k.strip()]
        
        if not self.keys:
            raise ValueError(f"No API keys found for: {name_for_errors}")
            
        self._rotator: Iterator[str] = itertools.cycle(self.keys)
        
    def get_next_key(self) -> str:
        """Retrieves the next available API key in the sequence."""
        return next(self._rotator)
        
    def get_key_count(self) -> int:
        """Returns the number of loaded API keys."""
        return len(self.keys)


class LLMService:
    """
    Wraps the external calls to LLM providers (Gemini and Groq).
    Implements round-robin key rotation for load balancing.
    Uses the latest google-genai SDK for Gemini operations.
    """
    def __init__(self) -> None:
        self.gemini_keys: APIKeyRotator = APIKeyRotator(settings.GEMINI_API_KEYS, "GEMINI_API_KEYS")
        self.groq_keys: APIKeyRotator = APIKeyRotator(settings.GROQ_API_KEYS, "GROQ_API_KEYS")

    def _get_gemini_client(self) -> genai.Client:
        """
        Dynamically configures and returns the latest Google GenAI Client
        using the next API key in rotation.
        """
        api_key: str = self.gemini_keys.get_next_key()
        return genai.Client(api_key=api_key)

    def _get_groq_client(self) -> Groq:
        """
        Dynamically initializes and returns a Groq client
        using the next API key in rotation.
        """
        api_key: str = self.groq_keys.get_next_key()
        return Groq(api_key=api_key)

    def embed_text(self, text: str) -> List[float]:
        """
        Generates 768-dimensional embeddings using Gemini's text-embedding-004 model.
        """
        if not text.strip():
            raise ValueError("Cannot embed empty text.")
            
        client: genai.Client = self._get_gemini_client()
        response: Any = client.models.embed_content(
            model="text-embedding-004",
            contents=text
        )
        
        # Cast/check return payload to prevent Pylance dynamic resolution warnings
        embeddings = getattr(response, "embeddings", None)
        if embeddings and len(embeddings) > 0:
            values = getattr(embeddings[0], "values", None)
            if values:
                return values
        raise ValueError("Failed to retrieve embeddings from Gemini API response.")

    def generate_gemini(
        self, 
        prompt: str, 
        system_instruction: Optional[str] = None,
        temperature: float = 0.2
    ) -> str:
        """
        Generates text using Gemini 1.5 Flash.
        """
        client: genai.Client = self._get_gemini_client()
        
        config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_instruction
        )
        
        response: Any = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt,
            config=config
        )
        
        text = getattr(response, "text", None)
        if text is not None:
            return str(text)
        raise ValueError("Failed to retrieve text from Gemini generation response.")

    def generate_groq(
        self, 
        prompt: str, 
        system_instruction: Optional[str] = None,
        temperature: float = 0.2
    ) -> str:
        """
        Generates text using Groq's Llama 3 (llama3-70b-8192).
        """
        client: Groq = self._get_groq_client()
        
        messages: List[Dict[str, str]] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
            
        messages.append({"role": "user", "content": prompt})
        
        completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=messages,
            temperature=temperature
        )
        
        content = completion.choices[0].message.content
        if content is not None:
            return str(content)
        raise ValueError("Failed to retrieve text from Groq completion response.")

    def transcribe_voice(self, file_path: str) -> str:
        """
        Transcribes an audio file using Groq's Whisper-large-v3.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
            
        client: Groq = self._get_groq_client()
        
        with open(file_path, "rb") as file:
            translation = client.audio.transcriptions.create(
                file=(os.path.basename(file_path), file.read()),
                model="whisper-large-v3",
                response_format="text"
            )
        return str(translation).strip()

    def describe_image(self, file_path: str) -> str:
        """
        Generates a detailed semantic description of an image.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Image file not found: {file_path}")
            
        client: genai.Client = self._get_gemini_client()
        image = Image.open(file_path)
        
        prompt = (
            "Analyze this image in detail. Generate a rich, descriptive, and comprehensive summary "
            "of what is shown. Include any text visible in the image (OCR), describe the objects, "
            "actions, style, colors, and key context. This summary will be used in a search engine "
            "to retrieve this image, so make it highly detailed and use descriptive keywords."
        )
        
        response: Any = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=[image, prompt]
        )
        
        text = getattr(response, "text", None)
        if text is not None:
            return str(text)
        raise ValueError("Failed to retrieve text from Gemini image description response.")
