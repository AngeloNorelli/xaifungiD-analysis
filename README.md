# XAI-FUNGI Analysis

Repozytorium zawiera narzędzia do analizy transkrypcji z badania użytkowników dotyczącego zrozumiałości wyjaśnień modeli uczenia maszynowego (XAI). Projekt jest forkiem [sbobek/xaifungi-analysis](https://github.com/sbobek/xaifungi-analysis).

Pełna analiza dostępna jest w `analysis.ipynb`. Dodatkowe skrypty i narzędzia znajdują się w `src/`.

---

## Struktura danych źródłowych

Transkrypcje pobierane są automatycznie z Zenodo (plik `TRANSCRIPTS.zip`). Każda transkrypcja to plik CSV o nazwie `[RR]_[SS]_[NN].csv` z następującymi kolumnami:

| Kolumna | Opis |
|---------|------|
| `speaker_id` | Identyfikator mówiącego (uczestnik lub badacz) |
| `slide_id` | ID slajdu aktualnie wyświetlanego uczestnikowi |
| `question_id` | ID pytania zadanego przez badacza (jeśli dotyczy) |
| `problem_id` | ID zadania klasyfikacyjnego (jeśli dotyczy) |
| `text` | Transkrypcja wypowiedzi |

### Specjalne wartości `slide_id`

| Wartość | Znaczenie |
|---------|-----------|
| `__S00__` | Początek właściwej części wywiadu (analiza slajdów) |
| `__S01__`–`__S14__` | Kolejne slajdy z wizualizacjami wyjaśnień |
| `__S15__` | Koniec sekcji analizy slajdów |
| `__S88__` | Początek sekcji zadań klasyfikacyjnych (problem solving) |
| `__S99__` | Sekcja swobodnego układania kolejności slajdów przez uczestnika |

---

## Pipeline przetwarzania

```
TRANSCRIPTS.zip (Zenodo)
        │
        ▼
  TranscriptParser                  ← src/TranscriptParser.py
  (parse_file_grouped_by_slide)
        │
        ▼
  parsed_transcripts.jsonl          ← ~1 781 rekordów (uczestnik × slajd)
        │
        ▼
  LocalLLM.ask()                    ← src/LocalLLM.py
  (prompt per slajd per uczestnik)
        │
        ▼
  llm_responses.json                ← metryki per slajd per uczestnik
        │
        ▼
  plot_participant_profile_slide_by_slide()    ← analysis.ipynb
  (wizualizacja dynamiki w czasie)
        │
        ▼
  XAIExplanation.explain()          ← src/XAIExplanation.py
  (wyjaśnienie zmian między slajdami)
```

---

## Moduł parsowania — `TranscriptParser`

**Plik:** `src/TranscriptParser.py`

Klasa `TranscriptParser` przetwarza surowe pliki CSV z transkrypcjami na ustrukturyzowane rekordy JSONL gotowe do analizy przez LLM.

### Główne metody

| Metoda | Opis |
|--------|------|
| `parse_file_grouped_by_slide(path)` | Generator rekordów: agreguje wypowiedzi uczestnika per slajd z jednego pliku |
| `parse_csv_file(path)` | Generator wierszy: przetwarza każdy wiersz CSV na słownik atrybutów |
| `parse_all(input_dir, output_path)` | Przetwarza cały katalog i zapisuje wynik do pliku JSONL |
| `parse_file_to_jsonl(path, out_fh)` | Zapisuje rekordy z jednego pliku do otwartego deskryptora |

### Atrybuty wyliczane per wiersz (`parse_csv_file`)

| Atrybut | Typ | Opis |
|---------|-----|------|
| `source_file` | str | Nazwa pliku źródłowego |
| `speaker_id` | str | Identyfikator mówiącego |
| `role` | str | `"participant"` lub `"researcher"` (na podstawie prefiksu ID) |
| `slide_id` | str\|None | ID aktualnego slajdu |
| `question_id` | str\|None | ID pytania (jeśli wiersz jest odpowiedzią na pytanie) |
| `problem_id` | str\|None | ID zadania klasyfikacyjnego |
| `cleaned_text` | str | Oczyszczony tekst (normalizacja białych znaków, usunięcie zewnętrznych cudzysłowów) |
| `token_count` | int | Liczba tokenów (słów) w wypowiedzi |
| `sentence_count` | int | Szacunkowa liczba zdań |
| `is_question` | bool | Czy wypowiedź zawiera pytanie |
| `is_slide_marker` | bool | Czy wiersz jest znacznikiem zmiany slajdu |
| `contains_digits` | bool | Czy wypowiedź zawiera liczby |

### Uruchomienie z linii poleceń

```bash
# Przetworzenie całego katalogu transcripts/
python -m src.TranscriptParser --input transcripts/ --output parsed_transcripts.jsonl
```

---

## Moduł LLM — `LocalLLM`

**Plik:** `src/LocalLLM.py`

Klasa `LocalLLM` obsługuje komunikację z lokalnym modelem językowym via [Ollama](https://ollama.com/). Domyślny model to `qwen3:5b`.

### Konfiguracja (zmienne środowiskowe / `.env`)

| Zmienna | Domyślna wartość | Opis |
|---------|-----------------|------|
| `OLLAMA_URL` | `http://localhost:11434/api/generate` | Adres endpointu Ollama |
| `OLLAMA_MODEL` | `qwen3.5:9b` | Nazwa modelu |
| `OLLAMA_TIMEOUT` | `120` | Timeout zapytania w sekundach |
| `OLLAMA_RETRIES` | `2` | Liczba ponownych prób przy błędzie |
| `OLLAMA_BACKOFF` | `1.0` | Czas oczekiwania między próbami (mnożnik) |
| `OLLAMA_TEMPERATURE` | `0.7` | Temperatura generowania |
| `OLLAMA_MAX_TOKENS` | `2048` | Maksymalna długość odpowiedzi |

### Główne metody

| Metoda | Opis |
|--------|------|
| `ask(question, context, system)` | Wysyła pytanie do modelu z opcjonalnym kontekstem i wiadomością systemową |
| `_extract_json_from_response(text)` | Wieloetapowa ekstrakcja JSON z odpowiedzi modelu |
| `_normalize_response(raw_json_str)` | Normalizuje odpowiedź do ujednoliconego formatu z obsługą aliasów pól |

### Obsługa aliasów pól

Model może zwracać metryki pod różnymi nazwami. `LocalLLM` automatycznie rozpoznaje następujące aliasy:

| Pole docelowe | Akceptowane aliasy |
|--------------|-------------------|
| `engagement_level` | `engagement`, `engagement_level`, `engagement_score` |
| `confidence_level` | `confidence`, `confidence_level`, `confidence_score` |
| `communication_style` | `style`, `communication_level`, `style_score` |

---

## Metryki wydobywane przez LLM

Dla każdego uczestnika i każdego slajdu model językowy zwraca trzy metryki w skali **0.0–1.0**:

### `engagement_level` — poziom zaangażowania

Mierzy, na ile aktywnie uczestnik angażował się w dany slajd. Podstawą oceny są:
- liczba wypowiedzi i ich łączna długość (`utterance_count`, `token_count`),
- liczba zdań i pytań zadanych przez uczestnika,
- obecność spontanicznych komentarzy (nie tylko odpowiedzi na pytania badacza).

| Wartość | Interpretacja |
|---------|--------------|
| 0.0–0.3 | Minimalne zaangażowanie — krótkie, lakoniczne odpowiedzi |
| 0.4–0.6 | Umiarkowane zaangażowanie — standardowa interakcja |
| 0.7–1.0 | Wysokie zaangażowanie — rozbudowane komentarze, pytania zwrotne |

### `confidence_level` — poziom pewności siebie

Mierzy, z jaką pewnością uczestnik wypowiadał się na temat treści slajdu. Podstawą oceny są:
- używanie form twierdzących vs. wyrażeń wątpienia (`"chyba"`, `"nie jestem pewien"`, `"wydaje mi się"`),
- kompletność i stanowczość odpowiedzi,
- reagowanie na pytania bez ociągania.

| Wartość | Interpretacja |
|---------|--------------|
| 0.0–0.3 | Niska pewność — dominują wahania, pytania zwrotne do badacza |
| 0.4–0.6 | Umiarkowana pewność — mieszanka stwierdzeń i wątpliwości |
| 0.7–1.0 | Wysoka pewność — stanowcze, precyzyjne sformułowania |

### `communication_style` — styl komunikacji

Mierzy stopień techniczności/formalności wypowiedzi uczestnika. Podstawą oceny są:
- używanie terminologii specjalistycznej vs. języka potocznego,
- długość i złożoność zdań,
- odwoływanie się do konkretnych danych (liczb, nazw algorytmów, cech modelu).

| Wartość | Interpretacja |
|---------|--------------|
| 0.0–0.3 | Styl potoczny/opisowy — brak terminologii technicznej |
| 0.4–0.6 | Styl mieszany — sporadyczne użycie terminologii |
| 0.7–1.0 | Styl techniczny/formalny — precyzyjne pojęcia, odwołania do danych |

### Format odpowiedzi LLM

```json
{
  "participant_id": "DR_IT_05",
  "profile": {
    "engagement_level": 0.82,
    "confidence_level": 0.65,
    "communication_style": 0.74
  }
}
```

### Prompt używany do analizy

```
Jesteś asystentem, który odpowiada WYŁĄCZNIE poprawnym JSON-em,
bez żadnych dodatkowych komentarzy.

Potrzebuję abyś przeanalizował tekst uczestnika '[participant_id]'
i skupił się na wydobyciu profilu uczestnika na bazie dynamiki jego wypowiedzi:
  - długość wypowiedzi,
  - reakcja na zadania,
  - sposób wypowiedzi.
```

Kontekstem przekazywanym do modelu jest agregacja wypowiedzi uczestnika ze wszystkich slajdów, w formacie:
```
Slajd __S01__ - DR_IT_05: [aggregated_text]
Slajd __S02__ - DR_IT_05: [aggregated_text]
...
```

---

## Wizualizacja dynamiki — `plot_participant_profile_slide_by_slide`

**Plik:** `analysis.ipynb` (Cell 9)

Funkcja generuje wykres liniowy przedstawiający zmianę wszystkich trzech metryk w czasie (slajd po slajdzie) dla jednego uczestnika. Umożliwia obserwację:
- momentów wzrostu/spadku zaangażowania lub pewności siebie,
- ogólnego trendu przez cały wywiad.

---

## Wyjaśnianie zmian — `XAIExplanation`

**Plik:** `src/XAIExplanation.py`

Klasa `XAIExplanation` analizuje **różnice w metrykach między kolejnymi slajdami** dla danego uczestnika. Dla każdej pary sąsiednich slajdów model językowy generuje narracyjne wyjaśnienie, co mogło spowodować zmianę w ocenie.

### Metody

| Metoda | Opis |
|--------|------|
| `explain(llm_response_path, transcript_path, participant_id)` | Główna metoda: ładuje dane, porównuje kolejne oceny i zwraca listę wyjaśnień |
| `_generate_explanation(transcript, data)` | Wysyła do LLM prompt z transkrypcją i dwoma ocenami do porównania |

### Format wyjściowy

```json
[
  {
    "slide_pair": ["__S02__", "__S03__"],
    "explanation": "Uczestnik wyraźnie zwiększył pewność siebie przechodząc do slajdu S03..."
  }
]
```

---

## Pliki danych

| Plik | Opis |
|------|------|
| `parsed_transcripts.jsonl` | Wypowiedzi uczestników pogrupowane per slajd (~1 781 rekordów) |
| `llm_responses.json` | Oceny LLM per slajd per uczestnik (podejście slajd po slajdzie) |
| `llm_responses_all_slide_at_once.json` | Oceny LLM gdy wszystkie slajdy uczestnika podawane są jednocześnie |
