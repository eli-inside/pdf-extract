"""
Tests comparing OCR extraction vs text extraction for accuracy validation.

These tests use text-based PDFs as ground truth:
1. Extract text using pymupdf4llm (ground truth)
2. Convert PDF pages to images
3. OCR the images using our extraction pipeline
4. Compare OCR output to text output using multiple metrics
"""

import os
import re
import pytest
from pathlib import Path
from difflib import SequenceMatcher
from pdf2image import convert_from_path
import pytesseract

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from pdf_to_text import (
    extract_with_pymupdf,
    extract_with_ocr,
    clean_text,
)


# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "data"

# Two-column test PDFs (arXiv/ACL papers on legal NLP)
TWO_COLUMN_PDFS = [
    "legal_nlp_domain_2023.pdf",
    "legalbert_casehold.pdf",
    "llm_law_survey.pdf",
]

# Thresholds for validation
WORD_COUNT_THRESHOLD = 0.75  # OCR should capture at least 75% of words
SIMILARITY_THRESHOLD = 0.60  # Overall text similarity should be at least 60%
KEY_PHRASE_THRESHOLD = 0.70  # At least 70% of key phrases should be found


def extract_key_phrases(text: str, num_phrases: int = 10) -> list[str]:
    """
    Extract key phrases from text for validation.

    Extracts:
    - First substantial sentence (likely title/abstract)
    - Several sentences from the body
    - Some multi-word phrases
    """
    # Clean and normalize text
    text = re.sub(r'\s+', ' ', text).strip()

    # Split into sentences
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 30]

    phrases = []

    # Get first substantial sentence (often contains title/key info)
    if sentences:
        phrases.append(sentences[0][:100])  # First 100 chars of first sentence

    # Get sentences from different parts of the document
    if len(sentences) > 5:
        # From early part
        phrases.append(sentences[2][:80] if len(sentences) > 2 else "")
        # From middle
        mid = len(sentences) // 2
        phrases.append(sentences[mid][:80])
        # From later part
        phrases.append(sentences[-3][:80] if len(sentences) > 3 else "")

    # Extract some multi-word phrases (3-5 words) that are likely distinctive
    words = text.split()
    for i in range(0, min(len(words) - 4, 1000), 100):  # Sample every 100 words
        phrase = ' '.join(words[i:i+4])
        if len(phrase) > 15:  # Only meaningful phrases
            phrases.append(phrase)

    # Filter out empty phrases and return unique ones
    phrases = [p for p in phrases if p and len(p) > 10]
    return phrases[:num_phrases]


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    # Lowercase
    text = text.lower()
    # Remove special characters except basic punctuation
    text = re.sub(r'[^\w\s.,;:!?-]', ' ', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def calculate_word_overlap(text1: str, text2: str) -> float:
    """Calculate the ratio of words from text1 that appear in text2."""
    words1 = set(normalize_text(text1).split())
    words2 = set(normalize_text(text2).split())

    if not words1:
        return 0.0

    overlap = words1.intersection(words2)
    return len(overlap) / len(words1)


def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate sequence similarity between two texts."""
    norm1 = normalize_text(text1)
    norm2 = normalize_text(text2)

    # Use SequenceMatcher for similarity
    return SequenceMatcher(None, norm1, norm2).ratio()


def check_key_phrases(phrases: list[str], text: str) -> tuple[float, list[str]]:
    """
    Check what percentage of key phrases are found in the text.
    Returns (ratio_found, list_of_missing_phrases).
    """
    text_normalized = normalize_text(text)

    found = 0
    missing = []

    for phrase in phrases:
        phrase_normalized = normalize_text(phrase)
        if phrase_normalized in text_normalized:
            found += 1
        else:
            # Check for partial match (at least 70% of phrase words)
            phrase_words = set(phrase_normalized.split())
            text_words = set(text_normalized.split())
            if phrase_words and len(phrase_words.intersection(text_words)) / len(phrase_words) >= 0.7:
                found += 1
            else:
                missing.append(phrase[:50] + "..." if len(phrase) > 50 else phrase)

    ratio = found / len(phrases) if phrases else 1.0
    return ratio, missing


def ocr_pdf_pages(pdf_path: str, dpi: int = 300) -> str:
    """
    Convert PDF to images and OCR each page.
    This simulates what would happen with a scanned PDF.
    """
    images = convert_from_path(pdf_path, dpi=dpi)

    all_text = []
    for i, image in enumerate(images):
        # Use tesseract with automatic page segmentation
        text = pytesseract.image_to_string(image, lang='eng', config='--oem 3 --psm 3')
        all_text.append(text)

    return '\n\n'.join(all_text)


class TestOCRAccuracy:
    """Test suite for validating OCR accuracy against text extraction."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Verify test data exists."""
        assert TEST_DATA_DIR.exists(), f"Test data directory not found: {TEST_DATA_DIR}"

    @pytest.mark.parametrize("pdf_name", TWO_COLUMN_PDFS)
    def test_two_column_ocr_word_count(self, pdf_name):
        """Test that OCR captures sufficient words compared to text extraction."""
        pdf_path = TEST_DATA_DIR / pdf_name
        if not pdf_path.exists():
            pytest.skip(f"Test PDF not found: {pdf_path}")

        # Extract text (ground truth)
        text_output = extract_with_pymupdf(str(pdf_path))
        text_output = clean_text(text_output)

        # OCR the PDF (convert to images first)
        ocr_output = ocr_pdf_pages(str(pdf_path))
        ocr_output = clean_text(ocr_output)

        # Count words
        text_words = len(text_output.split())
        ocr_words = len(ocr_output.split())

        # Calculate ratio
        word_ratio = ocr_words / text_words if text_words > 0 else 0

        print(f"\n{pdf_name}:")
        print(f"  Text extraction: {text_words} words")
        print(f"  OCR extraction: {ocr_words} words")
        print(f"  Word ratio: {word_ratio:.2%}")

        assert word_ratio >= WORD_COUNT_THRESHOLD, \
            f"OCR word count ({ocr_words}) is less than {WORD_COUNT_THRESHOLD:.0%} of text extraction ({text_words})"

    @pytest.mark.parametrize("pdf_name", TWO_COLUMN_PDFS)
    def test_two_column_ocr_key_phrases(self, pdf_name):
        """Test that OCR captures key phrases from the document."""
        pdf_path = TEST_DATA_DIR / pdf_name
        if not pdf_path.exists():
            pytest.skip(f"Test PDF not found: {pdf_path}")

        # Extract text (ground truth)
        text_output = extract_with_pymupdf(str(pdf_path))
        text_output = clean_text(text_output)

        # Extract key phrases from text
        key_phrases = extract_key_phrases(text_output)

        # OCR the PDF
        ocr_output = ocr_pdf_pages(str(pdf_path))
        ocr_output = clean_text(ocr_output)

        # Check key phrases
        ratio_found, missing = check_key_phrases(key_phrases, ocr_output)

        print(f"\n{pdf_name}:")
        print(f"  Key phrases extracted: {len(key_phrases)}")
        print(f"  Phrases found in OCR: {ratio_found:.2%}")
        if missing:
            print(f"  Missing phrases: {missing[:3]}...")

        assert ratio_found >= KEY_PHRASE_THRESHOLD, \
            f"Only {ratio_found:.0%} of key phrases found in OCR (threshold: {KEY_PHRASE_THRESHOLD:.0%})"

    @pytest.mark.parametrize("pdf_name", TWO_COLUMN_PDFS)
    def test_two_column_ocr_similarity(self, pdf_name):
        """Test overall text similarity between OCR and text extraction."""
        pdf_path = TEST_DATA_DIR / pdf_name
        if not pdf_path.exists():
            pytest.skip(f"Test PDF not found: {pdf_path}")

        # Extract text (ground truth)
        text_output = extract_with_pymupdf(str(pdf_path))
        text_output = clean_text(text_output)

        # OCR the PDF
        ocr_output = ocr_pdf_pages(str(pdf_path))
        ocr_output = clean_text(ocr_output)

        # Calculate similarity
        similarity = calculate_similarity(text_output, ocr_output)
        word_overlap = calculate_word_overlap(text_output, ocr_output)

        print(f"\n{pdf_name}:")
        print(f"  Sequence similarity: {similarity:.2%}")
        print(f"  Word overlap: {word_overlap:.2%}")

        assert similarity >= SIMILARITY_THRESHOLD, \
            f"Text similarity ({similarity:.2%}) is below threshold ({SIMILARITY_THRESHOLD:.0%})"

class TestOCRPipelineIntegration:
    """Integration tests for the full OCR pipeline."""

    @pytest.mark.parametrize("pdf_name", TWO_COLUMN_PDFS[:2])  # Test with first 2 PDFs
    def test_ocr_pipeline(self, pdf_name):
        """Test the extract_with_ocr function produces readable output."""
        pdf_path = TEST_DATA_DIR / pdf_name
        if not pdf_path.exists():
            pytest.skip(f"Test PDF not found: {pdf_path}")

        # Run OCR pipeline (uses Gemini vision for layout detection)
        ocr_output = extract_with_ocr(str(pdf_path), dpi=200)
        ocr_output = clean_text(ocr_output)

        # Basic sanity checks
        word_count = len(ocr_output.split())

        print(f"\n{pdf_name}:")
        print(f"  OCR word count: {word_count}")
        print(f"  Sample (first 200 chars): {ocr_output[:200]}...")

        assert word_count > 100, f"OCR output too short: {word_count} words"

        # Check that output doesn't have obvious interleaving (alternating short fragments)
        lines = ocr_output.split('\n')
        short_lines = sum(1 for line in lines if 0 < len(line.strip()) < 10)
        short_line_ratio = short_lines / len(lines) if lines else 0

        assert short_line_ratio < 0.3, \
            f"Too many short lines ({short_line_ratio:.0%}), possible column interleaving"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
