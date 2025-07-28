import pdfplumber
from collections import defaultdict
import re
import json
import os
from pathlib import Path
from difflib import SequenceMatcher # Import added for enhanced repetition removal

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

    def detect_bold_from_font(self, font_name):
        """Checks if a font name indicates bold styling."""
        if not font_name:
            return False
        bold_indicators = [
            'Bold', 'bold', 'Black', 'black', 'Heavy', 'heavy',
            'Demi', 'demi', 'SemiBold', 'Semibold', 'Medium'
        ]
        return any(indicator in font_name for indicator in bold_indicators)

    # --- START OF ENHANCED METHODS ---
    def remove_character_repetition(self, text):
        """
        Removes excessive repeated characters.
        Tries to detect interleaved repetition (e.g., aabbcc -> abc)
        and falls back to removing consecutive duplicates.
        """
        if len(text) < 2:
            return text

        # Heuristic: Check for interleaved pattern (like every 2nd char is a repeat)
        # This is common with the described issue.
        # We'll compare the string with itself shifted by 1 or 2 positions.
        # If there's a high similarity, it might indicate interleaved repetition.
        # This is a bit of a guess, but often effective.

        # Try removing every 2nd character if it matches the 1st, 3rd matches 1st, etc.
        # Pattern: A A B B C C -> A B C
        if len(text) >= 4:
            candidate1 = text[::2] # Take every 2nd char starting from 0
            # Check if the odd positions are mostly similar to even positions
            # Simple check: compare lengths and a sample
            if len(candidate1) > 1:
                # More robust check using SequenceMatcher for similarity
                candidate2 = text[1::2] # Take every 2nd char starting from 1
                # If candidate1 and candidate2 are very similar, it's likely duplication
                # Ensure comparison length matches
                min_len = min(len(candidate1), len(candidate2))
                if min_len > 0:
                    similarity = SequenceMatcher(None, candidate1[:min_len], candidate2[:min_len]).ratio()
                    # Adjust threshold as needed, 0.75 seems reasonable
                    if similarity > 0.75 and len(candidate1) >= len(text) * 0.4:
                        # Assume candidate1 (or candidate2) is the base text
                        # Let's take candidate1 as the cleaned version
                        cleaned = candidate1
                        # Apply consecutive deduplication to the result just in case
                        cleaned = self._remove_consecutive_duplicates(cleaned)
                        return cleaned.strip()

        # Try removing every 3rd character if it matches the 1st, 4th matches 2nd, etc.
        # Pattern: A B A B C D C D -> A B C D (less common but possible)
        # This is more complex, let's stick to the simpler heuristic for now.
        # If the first heuristic didn't trigger, fall back to consecutive deduplication.

        # Fallback: Remove consecutive duplicates (original logic, improved slightly)
        return self._remove_consecutive_duplicates(text)

    def _remove_consecutive_duplicates(self, text):
        """Helper to remove consecutive duplicate characters."""
        if len(text) < 2:
            return text
        result = [text[0]]
        for i in range(1, len(text)):
            if text[i] != text[i-1]:
                result.append(text[i])
            # Optional: Allow up to N consecutive, but for garbled text, removing all
            # consecutive duplicates after the first is usually better.
            # The original logic allowed 2, but for this case, 1 is likely better.
        cleaned = ''.join(result)

        # Handle extreme cases of single character repetition
        if len(cleaned) > 3:
            unique_chars = set(cleaned.lower())
            if len(unique_chars) == 1 and len(cleaned) > 10:
                return cleaned[0] * 2 # Return like "aa"
        return cleaned

    def insert_spaces_before_capitals(self, text):
        """Inserts spaces before capital letters following lowercase/digits."""
        if not text:
            return text
        # Improved regex to avoid adding space before the very first character if it's capital
        # and to handle sequences better. The original regex r'([a-z0-9])([A-Z])' is fine.
        # Using raw string notation (r'...') explicitly for clarity.
        return re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', text)

    def clean_text_advanced(self, text):
        """Applies advanced text cleaning routines."""
        if not text:
            return text
        # Apply the enhanced repetition removal first
        text = self.remove_character_repetition(text)
        # Then insert spaces
        text = self.insert_spaces_before_capitals(text)
        # Split and rejoin to normalize whitespace
        cleaned = ' '.join(text.split())
        if len(cleaned.strip()) < 1:
            return ""
        return cleaned.strip()
    # --- END OF ENHANCED METHODS ---

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

    def extract_words_advanced(self):
        """Extracts words and character groups with detailed properties."""
        self.all_extracted_words = []
        seen_combinations = set()
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                # --- REMOVE the initial title extraction logic from here ---
                # We will determine the title later after analysis
                # ---
                self.json_output['title'] = "Untitled Document" # Placeholder
                self.json_output['outline'] = []

                for page_num, page in enumerate(pdf.pages):
                    # ... (rest of the word and char extraction loop remains the same) ...
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
        """Assigns H1, H2, ... ranks based on size within each page, up to H4."""
        if not self.filtered_words: # Use instance variable
            self.ranked_words = []
            return
        words_list = self.filtered_words
        words_by_page = defaultdict(list)
        for word in words_list:
            words_by_page[word['page']].append(word)
        ranked_words = []
        # Store the title text to avoid ranking it again as H1
        title_text_lower = self.json_output.get('title', '').lower().strip()
        max_heading_level = 4 # Define the maximum heading level (H4)

        for page_num in sorted(words_by_page.keys()):
            page_words = words_by_page[page_num]
            # Sort words by size descending for ranking
            page_words_sorted = sorted(page_words, key=lambda w: w['size'], reverse=True)

            # --- Logic to determine Title (only for Page 1) ---
            # If we are on page 1 and the title hasn't been definitively set from page 1 content yet
            # (e.g., it was just the placeholder), find the highest scoring item on page 1 as the title.
            if page_num == 1 and (self.json_output['title'] == "Untitled Document" or not self.json_output['title']):
                 if page_words_sorted:
                    # The highest scoring item on page 1 becomes the title
                    potential_title_item = page_words_sorted[0]
                    self.json_output['title'] = potential_title_item['text']
                    title_text_lower = potential_title_item['text'].lower().strip()
                    # Mark this item so it's not ranked as H1 again
                    potential_title_item['_is_title'] = True
            # ---

            current_rank_level = 1
            previous_size = None
            for word in page_words_sorted:
                # Skip items marked as the title
                if word.get('_is_title', False):
                    ranked_words.append(word)
                    previous_size = word['size']
                    continue

                # Assign 'T' rank only if explicitly marked earlier (shouldn't happen now with new logic)
                # Or if it's page 1 and hasn't been processed, but we handle title above now.
                # So, generally, just assign H ranks starting from H1.
                # Ensure we don't exceed H4
                if previous_size is None or abs(word['size'] - previous_size) > 0.1:
                    if current_rank_level <= max_heading_level: # Limit to H4
                        word['rank'] = f"H{current_rank_level}"
                        current_rank_level += 1
                    else:
                         word['rank'] = f"H{max_heading_level}" # Cap at H4
                else:
                    # Assign the same rank as the previous item
                    # Make sure the rank doesn't exceed H4 even for same-size items following H4 items
                    last_rank_level = max(1, current_rank_level - 1)
                    capped_rank_level = min(last_rank_level, max_heading_level)
                    word['rank'] = f"H{capped_rank_level}"

                previous_size = word['size']
                ranked_words.append(word)
        self.ranked_words = ranked_words

    def generate_json_output(self):
        """Generates the JSON output structure from the ranked words."""
        outline = []
        # Use the final ranked results
        sorted_words = self.ranked_words
        # Sort words primarily by page, then by their original boldness score descending
        # to maintain the order within the page as determined by the analysis pipeline
        words_for_json = sorted(sorted_words, key=lambda w: (w['page'], -w['boldness_score']))
        for word in words_for_json:
            # Skip items explicitly marked as the title for the outline
            if word.get('_is_title', False):
                continue

            # Map the rank from our analysis (T, H1, H2...) to the desired level (H1, H2, H3...)
            # We can treat 'T' (Title) as H1 or a special level.
            # For simplicity, let's map T -> H1, H1 -> H1, H2 -> H2, etc.
            # This assumes H1 is the highest level heading on a page after T.
            raw_rank = word.get('rank', 'H1')  # Default to H1 if no rank?

            # Ensure rank conforms to H1-H4 or T
            if raw_rank == 'T':  # In case T is still used elsewhere
                mapped_level = 'H1'
            elif raw_rank.startswith('H') and raw_rank[1:].isdigit():
                level_num = int(raw_rank[1:])
                if level_num >= 1 and level_num <= 4:
                    mapped_level = raw_rank
                else:
                    mapped_level = 'H4'  # Safety cap
            else:
                mapped_level = 'H1'  # Default fallback

            outline.append({
                "level": mapped_level,
                "text": word['text'],
                "page": word['page']
            })
        # The title should now be correctly set by assign_ranks or the placeholder if nothing found
        # self.json_output['title'] is set
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
                # Ensure title is set even if no text
                if self.json_output['title'] == "Untitled Document":
                     # Try one last time if pdfplumber could get *any* text
                     try:
                         with pdfplumber.open(self.pdf_path) as pdf:
                             if pdf.pages and pdf.pages[0].extract_text():
                                 largest_word_on_first_page = max(pdf.pages[0].extract_words(), key=lambda w: w.get('size', 0), default=None)
                                 if largest_word_on_first_page:
                                    tentative_title = self.clean_text_advanced(largest_word_on_first_page.get('text', ''))
                                    if tentative_title:
                                        self.json_output['title'] = tentative_title
                     except:
                         pass # Ignore errors in last-ditch title attempt
                return []
            self.remove_duplicate_entries()
            print(f"DEBUG: After deduplication: {len(self.unique_words)} entries.")
            sorted_for_merge = sorted(self.unique_words, key=lambda x: (x['page'], -x['boldness_score']))
            # Temporarily store sorted list for merging
            self.unique_words = sorted_for_merge
            self.merge_consecutive_similar_entries()
            print(f"DEBUG: After merging (Page 1 only): {len(self.merged_words)} entries.")
            self.filter_small_non_bold(min_size=14.0)
            print(f"DEBUG: After filtering (<14 & non-bold): {len(self.filtered_words)} entries.")
            # Sort the final filtered list by page first, then by boldness score descending
            self.final_sorted_words = sorted(self.filtered_words, key=lambda x: (x['page'], -x['boldness_score']))
            print(f"DEBUG: Final sorted list: {len(self.final_sorted_words)} entries.")

            # --- Determine Title AFTER main filtering ---
            # Find the item with the highest boldness_score on page 1 from the final list
            page_1_items = [word for word in self.final_sorted_words if word['page'] == 1]
            if page_1_items:
                 # Sort by boldness score descending
                 page_1_items_sorted = sorted(page_1_items, key=lambda w: w['boldness_score'], reverse=True)
                 self.json_output['title'] = page_1_items_sorted[0]['text']
            # ---

            self.assign_ranks() # This will now use the determined title to avoid duplication
            print(f"DEBUG: Ranked list: {len(self.ranked_words)} entries.")
            return self.ranked_words
        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            import traceback
            traceback.print_exc() # Print stack trace for debugging
            return []

    def print_analysis(self, max_entries_per_page=30):
        """Prints the analysis results."""
        sorted_words = self.ranked_words # Use the final ranked results
        if not sorted_words:
            print("No words to display.")
            return
        print("PDF TEXT ANALYSIS (Ranked by Size within Page)")
        print("=" * 85)
        print(f"{'Page':<6} {'Rank':<6} {'Score':<8} {'F-Bold':<8} {'Size':<8} {'Text':<45}")
        print("-" * 85)
        words_by_page = defaultdict(list)
        for word in sorted_words:
            words_by_page[word['page']].append(word)
        total_displayed = 0
        for page_num in sorted(words_by_page.keys()):
            page_words = words_by_page[page_num]
            print(f"\n--- Page {page_num} ---")
            for i, word in enumerate(page_words[:max_entries_per_page]):
                if total_displayed >= 100:
                     break
                rank = word.get('rank', 'N/A')
                text_display = word['text'][:43] + "..." if len(word['text']) > 43 else word['text']
                print(f"       {rank:<6} {word['boldness_score']:<8} "
                      f"{'YES' if word['font_bold'] else 'NO':<8} "
                      f"{word['size']:<8} "
                      f"{text_display:<45}")
                total_displayed += 1
            if len(page_words) > max_entries_per_page:
                 print(f"       ... (showing top {max_entries_per_page} entries for page {page_num})")
        total_words = len(sorted_words)
        font_bold_words = sum(1 for word in sorted_words if word['font_bold'])
        size_bold_words = sum(1 for word in sorted_words if word['size_bold'])
        print("\n" + "=" * 85)
        print("SUMMARY STATISTICS (After Merging, Filtering, Ranking):")
        print(f"Total unique text entries analyzed: {total_words}")
        print(f"Font-bold text entries detected: {font_bold_words}")
        print(f"Size-bold text entries detected (>12): {size_bold_words}")
        if total_words > 0:
            font_percentage = (font_bold_words / total_words) * 100
            size_percentage = (size_bold_words / total_words) * 100
            print(f"Percentage of font-bold text: {font_percentage:.2f}%")
            print(f"Percentage of size-bold text (>12): {size_percentage:.2f}%")
        print("\nPOTENTIALLY PROMINENT TEXT (by Page):")
        print("-" * 50)
        prominent_text = [word for word in sorted_words if word['boldness_score'] > 25]
        prominent_by_page = defaultdict(list)
        for word in prominent_text:
            prominent_by_page[word['page']].append(word)
        if prominent_text:
            for page_num in sorted(prominent_by_page.keys()):
                page_prominent = prominent_by_page[page_num]
                print(f"\n--- Page {page_num} ---")
                page_prominent_sorted = sorted(page_prominent, key=lambda w: w['size'], reverse=True)
                for i, word in enumerate(page_prominent_sorted[:15]):
                    rank = word.get('rank', 'N/A')
                    text_display = word['text'][:40] + "..." if len(word['text']) > 40 else word['text']
                    bold_indicators = []
                    if word['font_bold']:
                        bold_indicators.append("Font-Bold")
                    if word['size_bold']:
                        bold_indicators.append("Large(>12)")
                    if rank == 'T':
                         bold_indicators.append("TITLE")
                    indicators_str = ", ".join(bold_indicators) if bold_indicators else "Standard"
                    print(f"    [{rank}] ({word['size']:.2f}pt) {text_display} ({indicators_str})")
        else:
            print("No prominently bold text detected.")

def main(pdf_loc,pdf_name):
    """Main execution function."""
    pdf_file_path = f"{pdf_loc}"  # Update path as needed
    output_json_path = f"output_outline{pdf_name}.json" # Define output JSON file path
    analyzer = PDFTextAnalyzer(pdf_file_path)
    ranked_words = analyzer.analyze()
    if ranked_words:
        json_data = analyzer.generate_json_output()
        print("\nGenerated JSON Output:")
        print(json.dumps(json_data, indent=2))
        analyzer.save_json_output(output_json_path)
    else:
        print("Analysis failed or produced no results.")

if __name__ == "__main__":
   """ folder_path = Path("Adobe-India-Hackathon25/Challenge_1b/Collection 1/PDFs")
    for pdf_file in folder_path.glob("*.pdf"):
        print(f"Processing: {pdf_file.name}")
        main(pdf_file.absolute(),pdf_file.name)"""
   main("file03.pdf","file03.pdf")
