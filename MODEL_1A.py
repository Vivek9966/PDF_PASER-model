import pdfplumber
from collections import defaultdict
import re
import json # Import json module

class PDFTextAnalyzer:
    """
    Analyzes PDF text to identify headings, structure, and prominence based on
    font properties and spatial arrangement.
    """

    def __init__(self, pdf_path):
        """
        Initializes the analyzer with the path to the PDF.

        Args:
            pdf_path (str): The path to the PDF file.
        """
        self.pdf_path = pdf_path
        # Instance variables to store intermediate and final results
        self.all_extracted_words = []
        self.unique_words = []
        self.merged_words = []
        self.filtered_words = []
        self.final_sorted_words = []
        self.ranked_words = []
        # Instance variable for JSON output
        self.json_output = {}
        # Instance variable to store the extracted title text to avoid duplicates
        self.extracted_title_text = ""

    # --- Helper methods (detect_bold_from_font, clean_text_advanced, etc.) remain the same ---
    # (Include all helper methods from Pasted_Text_1753630558386.txt here)
    # detect_bold_from_font, remove_character_repetition, insert_spaces_before_capitals,
    # clean_text_advanced, calculate_comprehensive_score, group_characters

    def detect_bold_from_font(self, font_name):
        """Checks if a font name indicates bold styling."""
        if not font_name:
            return False
        bold_indicators = [
            'Bold', 'bold', 'Black', 'black', 'Heavy', 'heavy',
            'Demi', 'demi', 'SemiBold', 'Semibold', 'Medium'
        ]
        return any(indicator in font_name for indicator in bold_indicators)

    def remove_character_repetition(self, text):
        """Removes excessive repeated characters, allowing common double letters."""
        if len(text) < 2:
            return text

        result = []
        i = 0
        while i < len(text):
            current_char = text[i]
            count = 1
            # Count consecutive occurrences
            while i + count < len(text) and text[i + count] == current_char:
                count += 1

            # Allow up to 2 consecutive identical characters
            chars_to_add = min(count, 2)
            result.append(current_char * chars_to_add)

            i += count

        cleaned = ''.join(result)

        # Handle extreme cases
        if len(cleaned) > 3:
            unique_chars = set(cleaned.lower())
            if len(unique_chars) == 1 and len(cleaned) > 10:
                return cleaned[0] * 2
        return cleaned

    def insert_spaces_before_capitals(self, text):
        """Inserts spaces before capital letters following lowercase/digits."""
        if not text:
            return text
        return re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', text)

    def clean_text_advanced(self, text):
        """Applies advanced text cleaning routines."""
        if not text:
            return text

        text = self.insert_spaces_before_capitals(text)
        cleaned = self.remove_character_repetition(text)
        cleaned = ' '.join(cleaned.split())
        if len(cleaned.strip()) < 1:
            return ""
        return cleaned.strip()

    def calculate_comprehensive_score(self, item):
        """Calculates a prominence score based on font and size."""
        font_name = item.get('fontname', '') or ''
        size = item.get('size', 0)
        score = 0
        if self.detect_bold_from_font(font_name):
            score += 20
        score += size * 1.5
        if 'Black' in font_name:
            score += 25
        elif 'Heavy' in font_name:
            score += 20
        elif 'Demi' in font_name or 'Semi' in font_name:
            score += 15
        elif 'Medium' in font_name:
            score += 10
        if size > 20:
            score += 10
        elif size > 16:
            score += 5
        return score

    def group_characters(self, chars):
        """Groups individual characters into words/segments based on proximity."""
        if not chars:
            return []
        groups = []
        current_group = {
            'text': '',
            'fontname': '',
            'size': 0,
            'x_positions': [],
            'y_positions': []
        }
        sorted_chars = sorted(chars, key=lambda c: (c.get('y0', 0), c.get('x0', 0)))
        for char in sorted_chars:
            text = char.get('text', '')
            if not text or text.isspace():
                continue
            fontname = char.get('fontname', '')
            size = char.get('size', 0)
            x0 = char.get('x0', 0)
            y0 = char.get('y0', 0)
            if not current_group['text']:
                current_group = {
                    'text': text,
                    'fontname': fontname,
                    'size': size,
                    'x_positions': [x0],
                    'y_positions': [y0]
                }
            else:
                last_x = current_group['x_positions'][-1] if current_group['x_positions'] else 0
                avg_size = current_group['size']
                x_distance = abs(x0 - last_x)
                size_diff = abs(size - avg_size)
                if x_distance < avg_size * 2 and size_diff < 2:
                    current_group['text'] += text
                    current_group['x_positions'].append(x0)
                    current_group['y_positions'].append(y0)
                else:
                    if len(current_group['text'].strip()) >= 1:
                        groups.append(current_group)
                    current_group = {
                        'text': text,
                        'fontname': fontname,
                        'size': size,
                        'x_positions': [x0],
                        'y_positions': [y0]
                    }
        if current_group['text'] and len(current_group['text'].strip()) >= 1:
            groups.append(current_group)
        return groups

    # --- Modified extract_words_advanced with better title logic and duplicate prevention ---
    def extract_words_advanced(self):
        """Extracts words and character groups with detailed properties."""
        self.all_extracted_words = []
        seen_combinations = set()
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                # --- Improved Title Extraction ---
                first_page_title = "Untitled Document"
                self.extracted_title_text = "" # Reset instance variable
                if pdf.pages:
                    first_page = pdf.pages[0]
                    first_page_words = first_page.extract_words(x_tolerance=3, y_tolerance=3)

                    if first_page_words:
                        # Strategy 1: Look for lines with name-like characteristics
                        # (This is heuristic and might need tuning)
                        # Sort lines by vertical position (top) to get the first line(s)
                        first_page_lines = first_page.extract_text_lines(x_tolerance=3, y_tolerance=3)
                        if first_page_lines:
                             # Get the first line object
                             first_line_obj = sorted(first_page_lines, key=lambda l: l.get('top', float('inf')))[0]
                             first_line_text = first_line_obj.get('text', '').strip()
                             # Basic cleaning
                             cleaned_first_line = self.clean_text_advanced(first_line_text)
                             if len(cleaned_first_line) > 3 and not cleaned_first_line.isdigit(): # Heuristic check
                                  first_page_title = cleaned_first_line
                                  self.extracted_title_text = cleaned_first_line.lower()
                             # else, fall back to size-based method

                        # Strategy 2: Fallback to largest text if line method didn't work well
                        # or if first_page_title is still the default
                        if first_page_title == "Untitled Document":
                            # Sort by size descending and take the first one as a simple title guess
                            largest_word_on_first_page = max(first_page_words, key=lambda w: w.get('size', 0))
                            temp_title = self.clean_text_advanced(largest_word_on_first_page.get('text', 'Untitled Document')).strip()
                            if temp_title and len(temp_title) > 3 and not temp_title.isdigit():
                                first_page_title = temp_title
                                self.extracted_title_text = temp_title.lower()
                            # Ensure a title
                            if not first_page_title:
                                first_page_title = "Untitled Document"

                self.json_output['title'] = first_page_title
                self.json_output['outline'] = []
                # --- End Improved Title Extraction ---

                for page_num, page in enumerate(pdf.pages):
                    try:
                        words = page.extract_words(
                            x_tolerance=3,
                            y_tolerance=3,
                            keep_blank_chars=False,
                            use_text_flow=True
                        )
                        for word in words:
                            text = word['text'].strip()
                            if not text or len(text) < 1:
                                continue
                            cleaned_text = self.clean_text_advanced(text)
                            if not cleaned_text or len(cleaned_text.strip()) < 1:
                                continue

                            # --- Prevent Title Duplication ---
                            # Check if the cleaned text is the title (case-insensitive substring check)
                            # This prevents the full name or parts of it appearing in the outline again.
                            if self.extracted_title_text and self.extracted_title_text in cleaned_text.lower():
                                # print(f"DEBUG: Skipping potential title duplicate: '{cleaned_text}'")
                                continue # Skip adding this word
                            # --- End Prevent Title Duplication ---

                            combo_key = (cleaned_text.lower().strip(), page_num)
                            if combo_key in seen_combinations:
                                continue
                            seen_combinations.add(combo_key)
                            font_bold = self.detect_bold_from_font(word.get('fontname', ''))
                            size = word.get('size', 0)
                            size_bold = size > 12
                            boldness_score = self.calculate_comprehensive_score(word)
                            self.all_extracted_words.append({
                                'text': cleaned_text,
                                'font_name': word.get('fontname', '') or 'Unknown',
                                'size': round(size, 2),
                                'font_bold': font_bold,
                                'size_bold': size_bold,
                                'boldness_score': round(boldness_score, 2),
                                'page': page_num + 1
                            })
                        chars = page.chars
                        if chars:
                            char_groups = self.group_characters(chars)
                            for group in char_groups:
                                if len(group['text'].strip()) < 1:
                                    continue
                                cleaned_text = self.clean_text_advanced(group['text'])
                                if not cleaned_text or len(cleaned_text.strip()) < 1:
                                    continue

                                # --- Prevent Title Duplication (for char groups too) ---
                                if self.extracted_title_text and self.extracted_title_text in cleaned_text.lower():
                                    # print(f"DEBUG: Skipping potential title duplicate (char group): '{cleaned_text}'")
                                    continue # Skip adding this group
                                # --- End Prevent Title Duplication ---

                                combo_key = (cleaned_text.lower().strip(), page_num, 'char_group')
                                if combo_key in seen_combinations:
                                    continue
                                seen_combinations.add(combo_key)
                                font_bold = self.detect_bold_from_font(group['fontname'])
                                size = group['size']
                                size_bold = size > 12
                                boldness_score = self.calculate_comprehensive_score(group)
                                self.all_extracted_words.append({
                                    'text': cleaned_text,
                                    'font_name': group['fontname'] or 'Unknown',
                                    'size': round(size, 2),
                                    'font_bold': font_bold,
                                    'size_bold': size_bold,
                                    'boldness_score': round(boldness_score, 2),
                                    'page': page_num + 1
                                })
                    except Exception as e:
                        print(f"Warning: Error processing page {page_num + 1}: {e}")
                        continue
        except FileNotFoundError:
            print(f"Error: PDF file '{self.pdf_path}' not found.")
            raise
        except Exception as e:
            print(f"Error opening PDF: {e}")
            raise
    # --- End Modified extract_words_advanced ---

    def remove_duplicate_entries(self):
        """Removes duplicate text entries, keeping the one with the highest score."""
        seen_texts = {}
        self.unique_words = []
        for word in self.all_extracted_words:
            text_key = word['text'].lower().strip()
            if text_key in seen_texts:
                if word['boldness_score'] > seen_texts[text_key]['boldness_score']:
                    seen_texts[text_key] = word
            else:
                seen_texts[text_key] = word
        self.unique_words = list(seen_texts.values())

    def merge_consecutive_similar_entries(self):
        """
        Merges consecutive entries with the same boldness_score and size,
        BUT ONLY IF THEY ARE ON PAGE 1.
        Assumes the list is already sorted (e.g., by page then score).
        """
        if not self.unique_words:
            self.merged_words = []
            return

        word_list = self.unique_words # Use instance variable
        merged_list = []
        current_merged_entry = word_list[0].copy()

        for i in range(1, len(word_list)):
            next_entry = word_list[i]

            on_page_1 = current_merged_entry['page'] == 1 and next_entry['page'] == 1
            same_page = current_merged_entry['page'] == next_entry['page']
            same_score = current_merged_entry['boldness_score'] == next_entry['boldness_score']
            same_size = abs(current_merged_entry['size'] - next_entry['size']) < 0.1

            if on_page_1 and same_page and same_score and same_size:
                current_merged_entry['text'] += ' ' + next_entry['text']
            else:
                merged_list.append(current_merged_entry)
                current_merged_entry = next_entry.copy()

        merged_list.append(current_merged_entry)
        self.merged_words = merged_list

    def filter_small_non_bold(self, min_size=14.0):
        """Filters out entries that are small and not font-bold."""
        self.filtered_words = []
        for word in self.merged_words: # Use instance variable
            if word['size'] >= min_size or word['font_bold']:
                self.filtered_words.append(word)

    def assign_ranks(self):
        """Assigns H1, H2, ... ranks based on size within each page."""
        if not self.filtered_words: # Use instance variable
            self.ranked_words = []
            return

        words_list = self.filtered_words
        words_by_page = defaultdict(list)
        for word in words_list:
            words_by_page[word['page']].append(word)

        ranked_words = []
        first_page_processed = False

        for page_num in sorted(words_by_page.keys()):
            page_words = words_by_page[page_num]
            page_words_sorted = sorted(page_words, key=lambda w: w['size'], reverse=True)

            current_rank_level = 1
            previous_size = None
            for word in page_words_sorted:
                if previous_size is None or abs(word['size'] - previous_size) > 0.1:
                    if page_num == 1 and not first_page_processed:
                        word['rank'] = 'T' # Title rank for the largest on first page
                        first_page_processed = True
                    else:
                        word['rank'] = f"H{current_rank_level}"
                    current_rank_level += 1
                else:
                    word['rank'] = f"H{current_rank_level - 1}"

                previous_size = word['size']
                ranked_words.append(word)

        self.ranked_words = ranked_words

    def generate_json_output(self):
        """Generates the JSON output structure from the ranked words."""
        outline = []
        # Use the final ranked results
        sorted_words = self.ranked_words

        # Sort words primarily by page, then by their original boldness score descending
        words_for_json = sorted(sorted_words, key=lambda w: (w['page'], -w['boldness_score']))

        for word in words_for_json:
            # Map the rank
            raw_rank = word.get('rank', 'H1')
            if raw_rank == 'T':
                mapped_level = 'H1' # Treat Title as H1
            elif raw_rank.startswith('H'):
                mapped_level = raw_rank
            else:
                mapped_level = 'H1'

            outline.append({
                "level": mapped_level,
                "text": word['text'],
                "page": word['page']
            })

        # Use the title extracted in extract_words_advanced
        # self.json_output['title'] is already set
        self.json_output['outline'] = outline
        return self.json_output

    def save_json_output(self, output_file_path):
        """Saves the generated JSON output to a file."""
        if not self.json_output:
            print("Warning: No JSON output generated yet. Run analyze() and generate_json_output() first.")
            return

        try:
            with open(output_file_path, 'w', encoding='utf-8') as json_file:
                json.dump(self.json_output, json_file, indent=4, ensure_ascii=False)
            print(f"JSON output successfully saved to '{output_file_path}'")
        except Exception as e:
            print(f"Error saving JSON output to '{output_file_path}': {e}")


    def analyze(self):
        """Performs the full analysis pipeline."""
        print(f"Analyzing PDF: {self.pdf_path}")
        try:
            self.extract_words_advanced()
            if not self.all_extracted_words:
                print("No text found in the PDF.")
                return []

            self.remove_duplicate_entries()
            print(f"DEBUG: After deduplication: {len(self.unique_words)} entries.")

            sorted_for_merge = sorted(self.unique_words, key=lambda x: (x['page'], -x['boldness_score']))
            self.unique_words = sorted_for_merge
            self.merge_consecutive_similar_entries()
            print(f"DEBUG: After merging (Page 1 only): {len(self.merged_words)} entries.")

            self.filter_small_non_bold(min_size=14.0)
            print(f"DEBUG: After filtering (<14 & non-bold): {len(self.filtered_words)} entries.")

            # Sort the final filtered list by page first, then by boldness score descending
            self.final_sorted_words = sorted(self.filtered_words, key=lambda x: (x['page'], -x['boldness_score']))
            print(f"DEBUG: Final sorted list: {len(self.final_sorted_words)} entries.")

            self.assign_ranks()
            print(f"DEBUG: Ranked list: {len(self.ranked_words)} entries.")

            return self.ranked_words
        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            return []

    # --- print_analysis method can be included from Pasted_Text_1753630558386.txt if needed ---
    # For brevity, it's omitted here as the focus is on JSON output.
    # --- End print_analysis ---


def main(pdf_loc, pdf_name):
    """Main execution function."""
    pdf_file_path = f"{pdf_loc}"
    output_json_path = f"output_outline_{pdf_name}.json"
    try:
        analyzer = PDFTextAnalyzer(pdf_file_path)
        ranked_words = analyzer.analyze()

        if ranked_words:
            json_data = analyzer.generate_json_output()
            print("\nGenerated JSON Output:")
            print(json.dumps(json_data, indent=2))
            analyzer.save_json_output(output_json_path)
        else:
            print("Analysis failed or produced no results.")
    except Exception as e:
        print(f"Error in main execution: {e}")

if __name__ == "__main__":
    # Test with the provided resume
    main("Vivek_resume_do_not_change_ (2).pdf", "Vivek_resume_do_not_change_ (1).pdf")

    # Example for batch processing (uncomment if needed)

    from pathlib import Path
    folder_path = Path("Adobe-India-Hackathon25/Challenge_1b/Collection 1/PDFs")
    for pdf_file in folder_path.glob("*.pdf"):
        print(f"\n--- Processing: {pdf_file.name} ---")
        main(str(pdf_file.absolute()), pdf_file.name)

