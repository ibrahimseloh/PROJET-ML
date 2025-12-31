"""Service Gemini pour appels LLM avec streaming"""
import google.generativeai as genai
from typing import Generator
import asyncio


class GeminiService:
    """Service pour appels à l'API Gemini avec streaming"""
    
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        """Initialise le service"""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.model_name = model
    
    def generate(self, prompt: str, temperature: float = 0.5, max_tokens: int = 2048) -> str:
        """Génère une réponse complète (non-streaming) - SIMPLE ET DIRECT"""
        import sys
        try:
            print(f"  🔍 [GEMINI] Sending request to API...", flush=True)
            sys.stdout.flush()
            print(f"  🔍 [GEMINI] Model: {self.model_name}", flush=True)
            print(f"  🔍 [GEMINI] Prompt length: {len(prompt)}", flush=True)
            sys.stdout.flush()
            
            print(f"  🔍 [GEMINI] Calling generate_content()...", flush=True)
            sys.stdout.flush()
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    top_p=0.9,
                    top_k=40,
                )
            )
            
            print(f"  🔍 [GEMINI] Response received! Length: {len(response.text)}", flush=True)
            sys.stdout.flush()
            return response.text
            
        except Exception as e:
            error_msg = str(e)
            print(f"  ❌ [GEMINI] Error: {type(e).__name__}: {error_msg[:100]}", flush=True)
            sys.stdout.flush()
            
            # Détect quota errors
            if "429" in error_msg or "exceeded" in error_msg.lower() or "quota" in error_msg.lower():
                return "❌ Erreur: Quota API dépassé. Vérifiez votre facturation Google Cloud ou attendez demain."
            
            return f"❌ Erreur Gemini: {error_msg[:100]}"
    
    def stream(self, prompt: str, temperature: float = 0.5, max_tokens: int = 1024) -> Generator[str, None, None]:
        """Streaming des réponses token-by-token - OPTIMISÉ"""
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    top_p=0.9,
                    top_k=40,
                ),
                stream=True
            )
            
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            yield f"Erreur Gemini: {str(e)}"
    
    async def stream_async(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2048) -> Generator[str, None, None]:
        """Streaming asynchrone"""
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
                stream=True
            )
            
            for chunk in response:
                if chunk.text:
                    yield chunk.text
                    await asyncio.sleep(0)
        except Exception as e:
            yield f"Erreur Gemini: {str(e)}"
