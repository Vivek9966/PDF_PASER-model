#!/usr/bin/env python3

import json
import sys
import datetime
import re
import pdfplumber
import argparse
from pathlib import Path
from llama_cpp import Llama
import pytesseract
from pdf2image import convert_from_path
from PIL import Image


class PDFHeadingExtractor:
    def __init__(self, min_size=14, bold_threshold=20):
        self.min_size = min_size
        self.bold_threshold = bold_threshold

    def is_bold_font(self, font_name):
        if not font_name:
            return False
        bold_indicators = ['Bold', 'bold', 'Black', 'black', 'Heavy', 'heavy',
                           'Demi', 'demi', 'SemiBold', 'Semibold', 'Medium']
        return any(indicator in font_name for indicator in bold_indicators)

    def calculate_heading_score(self, font_name, size):
        score = 0
        score += size * 1.5
        if self.is_bold_font(font_name):
            score += 25
            if 'Black' in font_name:
                score += 10
            elif 'Heavy' in font_name:
                score += 8
            elif 'Demi' in font_name or 'Semi' in font_name:
                score += 5
        return score

    def extract_headings(self, pdf_path):
        headings = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    words = page.extract_words(extra_attrs=["fontname", "size"])
                    if not words:
                        headings += self._extract_headings_with_ocr(pdf_path, page_num)
                        continue
                    lines = {}
                    for word in words:
                        y_key = round(word["top"])
                        if y_key not in lines:
                            lines[y_key] = []
                        lines[y_key].append(word)
                    for y_key, line_words in lines.items():
                        line_text = " ".join(w["text"] for w in line_words)
                        font_name = line_words[0]["fontname"]
                        size = line_words[0]["size"]
                        score = self.calculate_heading_score(font_name, size)
                        if size >= self.min_size or score >= self.bold_threshold:
                            headings.append({
                                "text": line_text.strip(),
                                "font_name": font_name,
                                "size": size,
                                "score": score,
                                "page": page_num + 1,
                                "is_bold": self.is_bold_font(font_name)
                            })
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}", file=sys.stderr)
        return headings

    def _extract_headings_with_ocr(self, pdf_path, page_num):
        headings = []
        try:
            images = convert_from_path(
                pdf_path,
                first_page=page_num + 1,
                last_page=page_num + 1,
                dpi=300
            )
            if not images:
                return headings
            text = pytesseract.image_to_string(images[0])
            for line in text.split('\n'):
                line = line.strip()
                if line:
                    headings.append({
                        "text": line,
                        "font_name": "OCR",
                        "size": 0,
                        "score": 40,
                        "page": page_num + 1,
                        "is_bold": False
                    })
        except Exception as e:
            print(f"OCR error on page {page_num + 1}: {str(e)}", file=sys.stderr)
        return headings

    def extract_text_after_heading(self, pdf_path, heading_page, heading_text, max_lines=20):
        try:
            with pdfplumber.open(pdf_path) as pdf:
                page = pdf.pages[heading_page - 1]
                text = page.extract_text()
                if text:
                    lines = text.split('\n')
                    heading_index = None
                    for i, line in enumerate(lines):
                        if heading_text in line:
                            heading_index = i
                            break
                    if heading_index is not None:
                        start = heading_index + 1
                        end = min(len(lines), start + max_lines)
                        return "\n".join(lines[start:end])
                return self._extract_text_with_ocr(pdf_path, heading_page, heading_text, max_lines)
        except Exception as e:
            print(f"Text extraction error: {str(e)}", file=sys.stderr)
            return ""

    def _extract_text_with_ocr(self, pdf_path, page_num, heading_text, max_lines):
        try:
            images = convert_from_path(
                pdf_path,
                first_page=page_num,
                last_page=page_num,
                dpi=300
            )
            if not images:
                return ""
            text = pytesseract.image_to_string(images[0])
            lines = text.split('\n')
            heading_index = None
            for i, line in enumerate(lines):
                if heading_text in line:
                    heading_index = i
                    break
            if heading_index is None:
                return ""
            start = heading_index + 1
            end = min(len(lines), start + max_lines)
            return "\n".join(lines[start:end])
        except Exception as e:
            print(f"OCR extraction error: {str(e)}", file=sys.stderr)
            return ""


class TinyLlamaRanker:
    def __init__(self):
        try:
            model_path = Path("./tinyllama-1.1b-chat-v1.0.Q4_0.gguf")
            if not model_path.exists():
                raise FileNotFoundError(f"TinyLlama model file not found: {model_path.absolute()}")
            self.llm = Llama(
                model_path=str(model_path),
                n_ctx=2048,
                n_threads=4,
                verbose=False
            )
        except Exception as e:
            print(f"ERROR: Could not load TinyLlama model: {e}", file=sys.stderr)
            raise

    def _safe_prompt(self, prompt, max_tokens=500, stop=None):
        try:
            response = self.llm(
                prompt,
                max_tokens=max_tokens,
                stop=stop or ["</s>", "HEADINGS:", "Rules:", "INSTRUCTIONS:"],
                temperature=0.1,
                top_p=0.9
            )
            return response["choices"][0]["text"].strip()
        except Exception as e:
            print(f"LLM processing error: {str(e)}", file=sys.stderr)
            return ""

    def score_heading(self, persona, job, heading_info):
        prompt = f"""You are a {persona} with the job: "{job}"
Rate the relevance of the following document heading for completing your job.
HEADING TEXT: '{heading_info['text']}'
DOCUMENT: {heading_info['document']}
PAGE: {heading_info['page']}
FONT SIZE: {heading_info['size']:.1f}
IS BOLD: {heading_info['is_bold']}
INSTRUCTIONS:
1. Assign a RELEVANCE SCORE from 0 to 100.
2. OUTPUT ONLY the NUMBER representing the score."""
        try:
            response = self._safe_prompt(prompt, max_tokens=100)
            score_match = re.search(r'\d+', response)
            if score_match:
                score = int(score_match.group(0))
                return max(0, min(100, score))
            return 0
        except Exception:
            return 0

    def refine_subsection_text(self, persona, job, heading, raw_text):
        if not raw_text.strip():
            return ""
        prompt = f"""You are a {persona} with the job: "{job}"
Refine the text below focusing on job relevance. Remove fluff, keep key points.
HEADING: "{heading}"
RAW TEXT:
{raw_text}
INSTRUCTIONS:
1. Remove irrelevant details and examples
2. Summarize key points concisely
3. Retain specific data, strategies, actions
4. Use bullet points if appropriate
5. Output ONLY the refined text
REFINED TEXT:"""
        return self._safe_prompt(prompt, max_tokens=1000)

    def rank_headings(self, persona, job, all_headings):
        headings_to_score = all_headings
        if len(headings_to_score) > 50:
            headings_to_score = sorted(headings_to_score, key=lambda x: x["score"], reverse=True)[:50]
        scored_headings = []
        for heading in headings_to_score:
            relevance_score = self.score_heading(persona, job, heading)
            heading["relevance_score"] = relevance_score
            scored_headings.append(heading)
        scored_headings.sort(key=lambda x: x["relevance_score"], reverse=True)
        top_ranked = scored_headings[:15]
        result = [
            {
                "text": h["text"],
                "rank": i + 1,
                "page": h["page"],
                "document": h.get("document", "Unknown")
            }
            for i, h in enumerate(top_ranked)
        ]
        if not result or all(h.get("relevance_score", -1) <= 0 for h in scored_headings[:10]):
            fallback_sorted = sorted(all_headings, key=lambda x: x.get("score", 0), reverse=True)
            result = [
                {"text": h["text"], "rank": i + 1, "page": h["page"], "document": h.get("document", "Unknown")}
                for i, h in enumerate(fallback_sorted[:10])
            ]
        return result


class DocumentAnalyzer:
    def __init__(self, pdf_folder):
        self.pdf_folder = Path(pdf_folder)
        if not self.pdf_folder.exists():
            raise FileNotFoundError(f"PDF folder not found: {pdf_folder}")
        self.heading_extractor = PDFHeadingExtractor()
        self.ranking_model = TinyLlamaRanker()

    def find_pdfs(self):
        return list(self.pdf_folder.glob("*.pdf"))

    def analyze(self, input_json_path):
        try:
            with open(input_json_path, 'r', encoding='utf-8') as f:
                input_data = json.load(f)
            persona = input_data["persona"]["role"]
            job = input_data["job_to_be_done"]["task"]
            documents = input_data["documents"]
            document_filenames = [doc["filename"] for doc in documents]
        except (KeyError, json.JSONDecodeError) as e:
            raise ValueError(f"Invalid input JSON: {str(e)}")

        pdf_files = self.find_pdfs()
        if not pdf_files:
            return {"error": "No PDF files found"}

        pdf_map = {f.name: f for f in pdf_files}
        all_headings = []

        for doc in documents:
            filename = doc["filename"]
            if filename not in pdf_map:
                continue
            pdf_path = pdf_map[filename]
            headings = self.heading_extractor.extract_headings(str(pdf_path))
            for heading in headings:
                heading["document"] = filename
            all_headings.extend(headings)

        if not all_headings:
            return {"error": "No headings found in documents"}

        ranked_headings = self.ranking_model.rank_headings(persona, job, all_headings)

        extracted_sections = [
            {
                "document": r["document"],
                "section_title": r["text"],
                "importance_rank": r["rank"],
                "page_number": r["page"]
            }
            for r in ranked_headings
        ]

        subsection_analysis = []
        for section in extracted_sections:
            doc_name = section["document"]
            page_num = section["page_number"]
            heading_text = section["section_title"]
            if doc_name not in pdf_map:
                continue
            pdf_path = pdf_map[doc_name]
            raw_text = self.heading_extractor.extract_text_after_heading(
                str(pdf_path),
                page_num,
                heading_text
            )
            refined_text = self.ranking_model.refine_subsection_text(
                persona,
                job,
                heading_text,
                raw_text
            ) if raw_text else ""
            subsection_analysis.append({
                "document": doc_name,
                "refined_text": refined_text,
                "page_number": page_num
            })

        return {
            "metadata": {
                "input_documents": document_filenames,
                "persona": persona,
                "job_to_be_done": job,
                "processing_timestamp": datetime.datetime.now().isoformat()
            },
            "extracted_sections": extracted_sections,
            "subsection_analysis": subsection_analysis
        }


def main():
    parser = argparse.ArgumentParser(description='PDF Heading Analyzer with OCR')
    parser.add_argument('--folder', '-f', required=True, help='Folder containing PDFs')
    parser.add_argument('--input', '-i', required=True, help='Path to input JSON configuration file')
    parser.add_argument('--output', '-o', help='Output file (default: stdout)')
    args = parser.parse_args()

    try:
        analyzer = DocumentAnalyzer(args.folder)
        result = analyzer.analyze(args.input)
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
        else:
            print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"ERROR: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
