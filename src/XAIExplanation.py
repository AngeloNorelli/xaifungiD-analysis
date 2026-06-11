import json
from collections import defaultdict
from src.LocalLLM import LocalLLM


class XAIExplanation:
  def __init__(self, llm=None, participant_id=None, llm_responses=None, transcript=None):
    self.llm = llm or LocalLLM()
    self.participant_id = participant_id
    self.llm_responses = llm_responses
    self.transcript = transcript

  def _extract_transcript(self, transcript_path, participant_id):
    try:
      with open(transcript_path, 'r') as f:
        grouped = defaultdict(list)
        for line in f:
          line = line.strip()
          if not line:
            continue
          record = json.loads(line)
          if record.get("source_file").startswith(participant_id):
            grouped[record.get("slide_id")].append(record)
        self.transcript = dict(grouped)
    except Exception as e:
      print(f"Error reading transcript: {e}")
    return None

  def _extract_llm_responses_for_participant(self, response_path, participant_id):
    try:
      with open(response_path, 'r') as f:
        data = json.load(f)
        if participant_id not in data:
          return
        records = []
        for p_id, record in data.items():
          if p_id == participant_id:
            for slide_id, slide_record in record.items():
              records.append({
                "slide_id": slide_id,
                "evaluation": slide_record
              })              
        self.llm_responses = records
    except Exception as e:
      print(f"Error reading LLM response: {e}")
    return None
  
  def _get_two_evaluations(self, first_index):
    if not self.llm_responses or len(self.llm_responses) < 2:
      return None, None
    return self.llm_responses[first_index], self.llm_responses[first_index + 1]
  
  def _get_transcript_for_evaluation(self, slide_id):
    if not self.transcript or slide_id not in self.transcript:
      return None
    return self.transcript[slide_id]
  
  def _generate_explanation(self, transcript, data, system=None):
    prompt = """
    Wytłumacz różnicę między dwoma ocenami dla uczestnika na podstawie poniższego transkryptu i danych.
    Skup się na kluczowych czynnikach, które doprowadziły do zmiany oceny.
    Podaj wgląd w to, co mogło spowodować zmianę i jak to się odnosi do zachowania lub odpowiedzi uczestnika.
    Postara się nie rozpisywać, ale również sbróbuj podać konkretne elementy trnaskryptu, 
    które mogą być istotne dla tej zmiany (max 3 zadania na kategorię).
    Pierwsza Ocena: {data1}
    Druga Ocena: {data2}
    Transkrypt: {transcript}
    """
    prompt = prompt.format(transcript=transcript, data1=data[0], data2=data[1])
    if system:
      prompt = f"{system.strip()}\n\n{prompt}"

    response = self.llm._query(prompt, system=system)
    try:
      response_data = json.loads(response)
      return response_data.get("response", response)
    except Exception:
      return response
  
  def explain(self, llm_response_path, transcript_path, participant_id):
    self._extract_transcript(transcript_path, participant_id)
    self._extract_llm_responses_for_participant(llm_response_path, participant_id)
    
    if not self.llm_responses or not self.transcript:
      return []
    
    explanations = []
    for i in range(len(self.llm_responses) - 1):
      eval1, eval2 = self._get_two_evaluations(i)
      if not eval1 or not eval2:
        continue
      slide1_id = eval1.get("slide_id")
      slide2_id = eval2.get("slide_id")
      transcript1 = self._get_transcript_for_evaluation(slide1_id)
      transcript2 = self._get_transcript_for_evaluation(slide2_id)
      if not transcript1 or not transcript2:
        continue
      explanation = self._generate_explanation([transcript1, transcript2], [eval1.get("evaluation"), eval2.get("evaluation")])
      explanations.append({
        "slide_pair": (slide1_id, slide2_id),
        "explanation": explanation
      })
    return explanations
