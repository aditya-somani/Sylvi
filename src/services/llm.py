import os
import base64
import itertools
from typing import List, Optional, Iterator, Any, Type
from groq import Groq
from google import genai
from google.genai import types
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from src.config import settings
from src.prompts import IMAGE_DESCRIPTION_PROMPT

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
    Wraps the external calls to LLM providers (ChatGroq and Gemini Embeddings).
    Implements round-robin key rotation for load balancing.
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

    def _get_groq_chat(self, model: Optional[str] = None, temperature: float = 0.2) -> ChatGroq:
        """
        Dynamically configures and returns a ChatGroq client.
        """
        api_key: str = self.groq_keys.get_next_key()
        model_name = model or settings.GROQ_TEXT_MODEL
        return ChatGroq(
            api_key=api_key,
            model=model_name,
            temperature=temperature
        )

    def embed_text(self, text: str) -> List[float]:
        """
        Generates 768-dimensional embeddings using Gemini's configured model.
        """
        if not text.strip():
            raise ValueError("Cannot embed empty text.")
            
        client: genai.Client = self._get_gemini_client()
        response: Any = client.models.embed_content(
            model=settings.GEMINI_EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(output_dimensionality=768)
        )
        
        embeddings = getattr(response, "embeddings", None)
        if embeddings and len(embeddings) > 0:
            values = getattr(embeddings[0], "values", None)
            if values:
                return values
        raise ValueError("Failed to retrieve embeddings from Gemini API response.")

    def generate_groq(
        self, 
        prompt: str, 
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
        model: Optional[str] = None
    ) -> str:
        """
        Generates standard text completion using ChatGroq.
        """
        chat = self._get_groq_chat(model=model, temperature=temperature)
        messages = []
        if system_instruction:
            messages.append(("system", system_instruction))
        messages.append(("user", prompt))
        
        response = chat.invoke(messages)
        return str(response.content)

    def generate_structured_groq(
        self,
        prompt: str,
        schema: Type[BaseModel],
        system_instruction: Optional[str] = None,
        temperature: float = 0.0,
        model: Optional[str] = None
    ) -> Any:
        """
        Generates a structured Pydantic object from ChatGroq.
        """
        chat = self._get_groq_chat(model=model, temperature=temperature)
        structured_llm = chat.with_structured_output(schema)
        
        messages = []
        if system_instruction:
            messages.append(("system", system_instruction))
        messages.append(("user", prompt))
        
        return structured_llm.invoke(messages)

    def describe_image(self, file_path: str) -> str:
        """
        Generates a detailed semantic description of an image using Groq Vision.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Image file not found: {file_path}")
            
        with open(file_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
        
        prompt = IMAGE_DESCRIPTION_PROMPT
        
        chat = self._get_groq_chat(model=settings.GROQ_VISION_MODEL, temperature=0.2)
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                },
            ]
        )
        
        response = chat.invoke([message])
        return str(response.content)

    def transcribe_voice(self, audio_path: str) -> str:
        """
        Transcribes a local audio file using Groq Whisper.
        Uses the groq SDK directly since langchain-groq does not expose
        the audio transcription endpoint.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        api_key: str = self.groq_keys.get_next_key()
        client = Groq(api_key=api_key)

        with open(audio_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model=settings.GROQ_WHISPER_MODEL,
                file=audio_file,
                response_format="text",
            )

        # When response_format="text", the SDK returns the transcript string directly
        transcript = str(transcription).strip()
        if not transcript:
            raise RuntimeError(f"Whisper returned an empty transcript for: {audio_path}")

        return transcript
