import os
import random
import itertools
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from groq import Groq

class APIKeyRotator:
    """
    A helper class that rotates through a list of API keys in a round-robin fashion.
    This acts as a basic load balancer to distribute requests across multiple accounts
    and bypass free-tier rate limits.
    """
    def __init__(self, env_var_name: str):
        self.env_var_name = env_var_name
        # Read the raw string, e.g., "key1,key2,key3"
        raw_keys = os.getenv(env_var_name, "")
        
        # Split by comma and strip whitespaces, filtering out empty strings
        self.keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
        
        if not self.keys:
            raise ValueError(f"No API keys found for environment variable: {env_var_name}")
            
        # itertools.cycle yields keys infinitely in a round-robin order
        self._rotator = itertools.cycle(self.keys)
        
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
    """
    def __init__(self):
        # Initialize key rotators for both providers
        self.gemini_keys = APIKeyRotator("GEMINI_API_KEYS")
        self.groq_keys = APIKeyRotator("GROQ_API_KEYS")

    def _get_gemini_client(self) -> Any:
        """
        Dynamically configures and returns the Google Generative AI client
        using the next API key in rotation.
        """
        api_key = self.gemini_keys.get_next_key()
        genai.configure(api_key=api_key)
        return genai

    def _get_groq_client(self) -> Groq:
        """
        Dynamically initializes and returns a Groq client
        using the next API key in rotation.
        """
        api_key = self.groq_keys.get_next_key()
        return Groq(api_key=api_key)

    def embed_text(self, text: str) -> List[float]:
        """
        Generates 768-dimensional embeddings using Gemini's text-embedding-004 model.
        """
        if not text.strip():
            raise ValueError("Cannot embed empty text.")
            
        client = self._get_gemini_client()
        response = client.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_document"
        )
        return response["embedding"]

    def generate_gemini(
        self, 
        prompt: str, 
        system_instruction: Optional[str] = None,
        temperature: float = 0.2
    ) -> str:
        """
        Generates text using Gemini 1.5 Flash. 
        Best suited for heavy summarization, scraping de-noising, and large context windows.
        """
        client = self._get_gemini_client()
        
        config = {
            "temperature": temperature,
        }
        
        model = client.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=config,
            system_instruction=system_instruction
        )
        
        response = model.generate_content(prompt)
        return response.text

    def generate_groq(
        self, 
        prompt: str, 
        system_instruction: Optional[str] = None,
        temperature: float = 0.2
    ) -> str:
        """
        Generates text using Groq's Llama 3 (llama3-70b-8192).
        Best suited for low-latency, fast chat replies and routing decisions.
        """
        client = self._get_groq_client()
        
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
            
        messages.append({"role": "user", "content": prompt})
        
        completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=messages,
            temperature=temperature
        )
        
        return completion.choices[0].message.content

    def transcribe_voice(self, file_path: str) -> str:
        """
        Transcribes an audio file (.ogg, .mp3, .wav) using Groq's Whisper-large-v3.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
            
        client = self._get_groq_client()
        
        with open(file_path, "rb") as file:
            translation = client.audio.transcriptions.create(
                file=(os.path.basename(file_path), file.read()),
                model="whisper-large-v3",
                response_format="text"
            )
        return str(translation).strip()
