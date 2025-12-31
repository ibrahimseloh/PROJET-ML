"""Configuration centrale pour l'application"""
import os
from dataclasses import dataclass
from typing import Optional
import google.generativeai as genai

@dataclass
class GeminiConfig:
    """Configuration Gemini API"""
    api_key: str
    model: str = "gemini-2.5-flash"
    temperature: float = 0.7
    max_output_tokens: int = 2048
    timeout: int = 30
    
    def initialize_client(self):
        """Initialise le client Gemini"""
        genai.configure(api_key=self.api_key)
        return genai.GenerativeModel(
            self.model,
            generation_config=genai.types.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
            )
        )
    
    @classmethod
    def from_env(cls, api_key: Optional[str] = None) -> 'GeminiConfig':
        """Charge depuis variables d'environnement ou input"""
        key = api_key or os.getenv('GEMINI_API_KEY')
        if not key:
            raise ValueError("GEMINI_API_KEY not provided")
        return cls(api_key=key)
    
    def validate(self) -> bool:
        """Valide la clé API"""
        try:
            genai.configure(api_key=self.api_key)
            # Test avec une requête simple
            test_model = genai.GenerativeModel(self.model)
            test_model.generate_content("Test")
            return True
        except Exception as e:
            print(f"Validation error: {e}")
            return False
