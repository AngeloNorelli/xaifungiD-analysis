"""
Parser dla katalogu `transcripts/`.

Klasa TranscriptParser udostępnia metody:
- parse_file_grouped_by_slide(path) -> generator rekordów (agregacja uczestnik+slajd)
- parse_csv_file(path) -> generator wierszy z atrybutami
- parse_all(input_dir, output_path) -> zapisuje wszystkie rekordy do JSONL
- parse_file_to_jsonl(path, output_fh) -> zapisuje rekordy z jednego pliku do otwartego fh

Przykład użycia z innego pliku:
from parser import TranscriptParser
p = TranscriptParser()
for rec in p.parse_file_grouped_by_slide('transcripts/DR_IT_05.csv'):
    print(rec)

"""

from __future__ import annotations
import argparse
import csv
import json
import os
import re
from typing import Dict, Iterable, Generator, Optional

TRANSCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "transcripts")

RE_MULTISPACE = re.compile(r"\s+", flags=re.UNICODE)
SENTENCE_END_RE = re.compile(r"[\.\?!]\s+|\n")


class TranscriptParser:
    """Parser plików z transkrypcjami.

    Główna metoda: parse_file_grouped_by_slide(path) -> generator rekordów.
    """

    def __init__(self, transcripts_dir: Optional[str] = None):
        self.transcripts_dir = transcripts_dir or TRANSCRIPTS_DIR

    @staticmethod
    def clean_text(s: str) -> str:
        if s is None:
            return ""
        s = s.strip()
        if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
            s = s[1:-1].strip()
        s = RE_MULTISPACE.sub(" ", s)
        return s

    @staticmethod
    def count_tokens(s: str) -> int:
        if not s:
            return 0
        return len(s.split())

    @staticmethod
    def count_sentences(s: str) -> int:
        if not s:
            return 0
        parts = re.split(r"(?<=[\.\?!])\s+", s)
        count = 0
        for p in parts:
            # jest zdanie jeśli występują litery/numbers
            if re.search(r"[A-Za-z0-9ąćęłńóśżźĄĆĘŁŃÓŚŻŹ]", p):
                count += 1
        return max(1, count) if s.strip() else 0

    def row_to_attributes(self, row: Dict[str, str], source_file: str, row_index: int) -> Dict:
        speaker_id = (row.get("speaker_id") or row.get("speaker") or "").strip()
        slide_id = (row.get("slide_id") or "").strip()
        question_id = (row.get("question_id") or "").strip()
        problem_id = (row.get("problem_id") or "").strip()
        raw_text = (row.get("text") or row.get("") or "").strip()

        cleaned = self.clean_text(raw_text)

        token_count = self.count_tokens(cleaned)
        sentence_count = self.count_sentences(cleaned)

        is_question = False
        if question_id.startswith("__Q") or "?" in cleaned:
            is_question = True

        is_slide_marker = slide_id.startswith("__S") if slide_id else False

        contains_digits = bool(re.search(r"\d", cleaned))

        speaker_prefix = speaker_id.split("_")[0] if speaker_id else ""

        prefix_upper = speaker_prefix.upper()
        role = "participant"
        if prefix_upper.startswith("DR") or prefix_upper.startswith("RS"):
            role = "researcher"

        attributes = {
            "source_file": os.path.basename(source_file),
            "row_index": row_index,
            "speaker_id": speaker_id,
            "speaker_prefix": speaker_prefix,
            "role": role,
            "slide_id": slide_id if slide_id else None,
            "question_id": question_id if question_id else None,
            "problem_id": problem_id if problem_id else None,
            "raw_text": raw_text,
            "cleaned_text": cleaned,
            "token_count": token_count,
            "sentence_count": sentence_count,
            "is_question": is_question,
            "is_slide_marker": is_slide_marker,
            "contains_digits": contains_digits,
        }
        return attributes

    def parse_csv_file(self, path: str) -> Generator[Dict, None, None]:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            reader = csv.DictReader(fh)
            for i, row in enumerate(reader):
                yield self.row_to_attributes(row, path, i)

    def parse_file_grouped_by_slide(self, path: str) -> Generator[Dict, None, None]:
        """Parsuje jeden plik i zwraca zgrupowane wypowiedzi uczestników per aktualny slajd.

        Każdy rekord zawiera: source_file, slide_id, participant_id, aggregated_text, utterance_count, token_count, sentence_count, first_row, last_row
        """
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            reader = csv.DictReader(fh)
            current_slide = None
            buffers: Dict[str, Dict] = {}
            for i, row in enumerate(reader):
                attrs = self.row_to_attributes(row, path, i)
                # jeśli wiersz wskazuje zmianę slajdu -> flush poprzedni
                if attrs.get("slide_id"):
                    if current_slide is not None:
                        for pid, buf in buffers.items():
                            agg_text = " ".join(buf['texts']).strip()
                            if not agg_text:
                                continue
                            rec = {
                                "source_file": os.path.basename(path),
                                "slide_id": current_slide,
                                "participant_id": pid,
                                "aggregated_text": agg_text,
                                "utterance_count": buf['utter_count'],
                                "token_count": self.count_tokens(agg_text),
                                "sentence_count": self.count_sentences(agg_text),
                                "first_row": buf.get('first_row'),
                                "last_row": buf.get('last_row'),
                            }
                            yield rec
                    current_slide = attrs.get("slide_id")
                    buffers = {}
                # jeśli to wypowiedź uczestnika i mamy aktywny slajd -> buforuj
                if attrs.get("role") == "participant" and current_slide is not None:
                    text = attrs.get("cleaned_text", "").strip()
                    if text:
                        pid = attrs.get("speaker_id") or "unknown"
                        if pid not in buffers:
                            buffers[pid] = {"texts": [], "utter_count": 0, "first_row": i, "last_row": i}
                        buffers[pid]["texts"].append(text)
                        buffers[pid]["utter_count"] += 1
                        buffers[pid]["last_row"] = i
            # flush ostatni slajd
            if current_slide is not None:
                for pid, buf in buffers.items():
                    agg_text = " ".join(buf['texts']).strip()
                    if not agg_text:
                        continue
                    rec = {
                        "source_file": os.path.basename(path),
                        "slide_id": current_slide,
                        "participant_id": pid,
                        "aggregated_text": agg_text,
                        "utterance_count": buf['utter_count'],
                        "token_count": self.count_tokens(agg_text),
                        "sentence_count": self.count_sentences(agg_text),
                        "first_row": buf.get('first_row'),
                        "last_row": buf.get('last_row'),
                    }
                    yield rec

    def parse_file_to_jsonl(self, path: str, out_fh) -> int:
        """Zapisuje rekordy z jednego pliku do już otwartego file-handle. Zwraca liczbę zapisanych rekordów."""
        count = 0
        for rec in self.parse_file_grouped_by_slide(path):
            out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
        return count

    def parse_all(self, input_dir: Optional[str], output_path: str) -> Dict[str, int]:
        input_dir = input_dir or self.transcripts_dir
        total_files = 0
        total_rows = 0
        with open(output_path, "w", encoding="utf-8") as out_fh:
            for root, _, files in os.walk(input_dir):
                for fn in files:
                    if fn.lower().endswith('.csv'):
                        path = os.path.join(root, fn)
                        total_files += 1
                        total_rows += self.parse_file_to_jsonl(path, out_fh)
        return {"files": total_files, "rows": total_rows}


def main():
    parser = argparse.ArgumentParser(description="Parse transcripts CSV files to JSONL with attributes for LLM.")
    parser.add_argument("--input", "-i", default=None, help="katalog z plikami CSV (domyślnie ./transcripts)")
    parser.add_argument("--file", "-f", default=None, help="konkretny plik CSV do przetworzenia (ścieżka względna lub bezwzględna)")
    parser.add_argument("--output", "-o", default=os.path.join(os.path.dirname(__file__), "parsed_transcripts.jsonl"), help="plik wyjściowy JSONL")
    args = parser.parse_args()

    parser_obj = TranscriptParser()
    input_dir = args.input or parser_obj.transcripts_dir
    output = args.output

    if args.file:
        if not os.path.isfile(args.file):
            print(f"Błąd: plik '{args.file}' nie istnieje.")
            return
        print(f"Parsowanie pliku: {args.file}\nZapis do: {output}")
        with open(output, "w", encoding="utf-8") as out_fh:
            count = parser_obj.parse_file_to_jsonl(args.file, out_fh)
        print(f"Skończono. Zapisano {count} rekordów z pliku {args.file}.")
        return

    if not os.path.isdir(input_dir):
        print(f"Błąd: katalog wejściowy '{input_dir}' nie istnieje.")
        return

    print(f"Parsowanie plików CSV z: {input_dir}\nZapis do: {output}")
    stats = parser_obj.parse_all(input_dir, output)
    print(f"Skończono. Przetworzono {stats['rows']} wierszy w {stats['files']} plikach.")


if __name__ == '__main__':
    main()
