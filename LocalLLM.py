import requests
import os
from dotenv import load_dotenv

load_dotenv()
OLLAMA_URL = os.getenv("OLLAMA_URL")

class LocalLLM:
  def __init__(self, model=None):
    self.model = model or os.getenv("OLLAMA_MODEL", "qwen3.5:9b")
    
  def _query(self, prompt, system=None):
    payload = {
      "model": self.model,
      "prompt": prompt,
      "stream": False
    }
    
    if system:
      payload["system"] = system
      
    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    return response.json()["response"]
  
  def analyze_text(self, text):
    prompt = f"Przenaalizuj tekst i podaj kluczowe wnioski:\n\n{text}"
    return self._query(prompt)
  
  def ask(self, question, context=None):
    prompt = ""
    if context:
      prompt += f"Kontekst:\n{context}\n\n"
    prompt += f"Pytanie: {question}"
    return self._query(prompt)