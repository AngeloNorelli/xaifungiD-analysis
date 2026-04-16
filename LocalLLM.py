import requests
import os
import time
from dotenv import load_dotenv

load_dotenv()
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "120"))
OLLAMA_RETRIES = int(os.getenv("OLLAMA_RETRIES", "2"))
OLLAMA_BACKOFF = float(os.getenv("OLLAMA_BACKOFF", "1.0"))

class LocalLLM:
  def __init__(self, model=None):
    self.model = model or os.getenv("OLLAMA_MODEL", "qwen3.5:9b")

  def _query(self, prompt, system=None):
    if system:
      prompt = f"{system.strip()}\n\n{prompt}"
    payload = {
      "model": self.model,
      "prompt": prompt,
      "stream": False,
      "temperature": float(os.getenv("OLLAMA_TEMPERATURE", "0.7")),
      "max_tokens": int(os.getenv("OLLAMA_MAX_TOKENS", "2048"))
    }
      
    for attempt in range(1, OLLAMA_RETRIES + 1):
      try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
        response.raise_for_status()
        
        try:
          data = response.json()
        except ValueError:
          return response.text
        
        return data.get("response") or data.get("output") or response.text
      except requests.exceptions.RequestException:
        if attempt == OLLAMA_RETRIES:
          raise
        time.sleep(OLLAMA_BACKOFF * attempt)
  
  def ask(self, question, context=None, system=None):
    prompt = ""
    if context:
      prompt += f"Kontekst:\n{context}\n\n"
    prompt += f"Pytanie: {question}"
    return self._query(prompt, system=system)