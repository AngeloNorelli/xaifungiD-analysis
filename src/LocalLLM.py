import requests
import os
import time
import json
import re
from dotenv import load_dotenv

load_dotenv()
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "120"))
OLLAMA_RETRIES = int(os.getenv("OLLAMA_RETRIES", "2"))
OLLAMA_BACKOFF = float(os.getenv("OLLAMA_BACKOFF", "1.0"))

FIELD_ALIASES = {
  "engagement_level": ["engagement", "engagement_level", "engagement_score"],
  "confidence_level": ["confidence", "confidence_level", "confidence_score"],
  "communication_style": ["style", "communication_level", "style_score"]
}
RESPONSE_KEYS = ['response', 'output', 'thinking']

class LocalLLM:
  def __init__(self, model=None):
    self.model = model or os.getenv("OLLAMA_MODEL", "qwen3.5:9b")

  def _find_code_fences(self, text):
    match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text, re.IGNORECASE)
    if match:
      candidate = match.group(1).strip()
      try:
        parsed = json.loads(candidate)
        return json.dumps(parsed, ensure_ascii=False)
      except Exception:
        return None

  def _find_brackets(self, text):
    for opener, closer in (('{', '}'), ('[', ']')):
      opens = [m.start() for m in re.finditer(re.escape(opener), text)]
      closes = [m.start() for m in re.finditer(re.escape(closer), text)]
      if not opens or not closes:
        continue
      for start in reversed(opens):
        for end in (p for p in closes if p > start):
          candidate = text[start:end+1].strip()
          try:
            parsed = json.loads(candidate)
            return json.dumps(parsed, ensure_ascii=False)
          except Exception:
            continue
      return None
   
  def _find_braces_regex(self, text):
    for pattern in (r'\{[\s\S]*?\}', r'\[[\s\S]*?\]'):
      matches = re.findall(pattern, text)
      for piece in reversed(matches):
        try:
          parsed = json.loads(piece)
          return json.dumps(parsed, ensure_ascii=False)
        except Exception:
          continue
      return None

  def _extract_json_from_response(self, text):
    if not text:
      return None
    
    if self._find_code_fences(text):
      return self._find_code_fences(text)
    
    try:
      parsed = json.loads(text.strip())
      return json.dumps(parsed, ensure_ascii=False)
    except Exception:
      pass
    
    if self._find_brackets(text):
      return self._find_brackets(text)
    
    if self._find_braces_regex(text):
      return self._find_braces_regex(text)
    return None

  def _find_structured_key(self, data, keys):
    for key in keys:
      if key not in data:
        continue
      val = data[key]
      if isinstance(val, (dict, list)):
        return json.dumps(val, ensure_ascii=False)
      if isinstance(val, str):
        try:
          parsed = json.loads(val)
          return json.dumps(parsed, ensure_ascii=False)
        except Exception:
          extracted = self._extract_json_from_response(val)
          if extracted is not None:
            return extracted
    return None
  
  def _post(self, pyaload):
    response = requests.post(OLLAMA_URL, json=pyaload, timeout=OLLAMA_TIMEOUT)
    response.raise_for_status()
    return response
  
  def _process_response(self, response):
    try:
      data = response.json()
    except ValueError:
      extracted = self._extract_json_from_response(response.text)
      return extracted if extracted is not None else ""
    
    extracted = self._find_structured_key(data, RESPONSE_KEYS)
    if extracted is not None:
      return extracted
    
    extracted = self._extract_json_from_response(response.text)
    return extracted if extracted is not None else ""

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
        response = self._post(payload)
        return self._process_response(response)
      except requests.exceptions.RequestException:
        if attempt == OLLAMA_RETRIES:
          raise
        time.sleep(OLLAMA_BACKOFF * attempt)
        
  def _get_first_match(self, data, keys):
    for key in keys:
      if key in data:
        return data[key]
    return None
        
  def _normalize_response(self, raw_json_str):
    try:
      data = json.loads(raw_json_str)
    except Exception:
      return None
    
    def safe_float(val):
      try:
        return float(val)
      except (ValueError, TypeError):
        return None
    
    participant_id = data.get("participant_id") or data.get("participant") or data.get("id")
    
    profile = []
    for target_field, aliases in FIELD_ALIASES.items():
      value = self._get_first_match(data, aliases)
      profile[target_field] = safe_float(value)
      
    return {
      "participant_id": participant_id,
      "profile": profile
    }
  
  def ask(self, question, context=None, system=None):
    prompt = ""
    if context:
      prompt += f"Kontekst:\n{context}\n\n"
    prompt += f"Pytanie: {question}"
    return self._query(prompt, system=system)