#!/usr/bin/env python3
"""
PDF to Text Extraction Tool
Uses OCR with AI-powered layout detection for reliable text extraction.

Usage:
    python3 pdf_to_text.py input.pdf [output.txt]

If output not specified, writes to input.txt

Pipeline:
1. Render PDF pages to images (bypasses font encoding issues)
2. Use Gemini Pro vision to detect page layout (headers, columns, footers)
3. Use Tesseract OCR to extract text from each region
4. Use Gemini Flash Lite to fix OCR errors paragraph by paragraph

Features:
- Handles both text-based and scanned PDFs uniformly via OCR
- Auto-detects single vs dual-column layout
- AI vision detects header/footer regions for clean extraction
- LLM post-processing fixes OCR errors and drop caps
- Strips common academic headers/footers

Dependencies:
    pip install pdfplumber pdf2image pytesseract google-genai pymupdf4llm
    brew install tesseract poppler  (macOS)
    apt install tesseract-ocr poppler-utils  (Linux)

    Also requires GEMINI_API_KEY or GOOGLE_API_KEY environment variable
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path


def check_dependencies():
    """Check if required tools are installed."""
    missing = []

    # Check tesseract
    try:
        subprocess.run(['tesseract', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        missing.append('tesseract-ocr')

    # Check pdftotext
    try:
        subprocess.run(['pdftotext', '-v'], capture_output=True)
    except FileNotFoundError:
        missing.append('poppler-utils')

    # Check Python packages
    try:
        import pdfplumber
    except ImportError:
        missing.append('pdfplumber (pip)')

    try:
        from pdf2image import convert_from_path
    except ImportError:
        missing.append('pdf2image (pip)')

    try:
        import pytesseract
    except ImportError:
        missing.append('pytesseract (pip)')

    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print("\nInstall with:")
        print("  brew install tesseract poppler  (macOS)")
        print("  apt install tesseract-ocr poppler-utils  (Linux)")
        print("  pip install pdfplumber pdf2image pytesseract")
        sys.exit(1)


def is_text_based_pdf(pdf_path):
    """Check if PDF has extractable text or is image-based."""
    result = subprocess.run(
        ['pdftotext', '-l', '2', str(pdf_path), '-'],
        capture_output=True,
        text=True,
        timeout=30
    )
    text = result.stdout.strip()
    # If we get substantial text, it's text-based
    words = len(text.split())
    return words > 50


def extract_with_pymupdf(pdf_path):
    """Extract text from PDF using pymupdf4llm (handles multi-column layouts).

    This is an alternative to OCR for text-based PDFs. Use --force-text to enable.
    Note: May have font encoding issues with some copy-protected PDFs.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Extracted text as string
    """
    import pymupdf4llm

    print("Extracting text with PyMuPDF...")
    print("Consider using the pymupdf_layout package for a greatly improved page layout analysis.")
    text = pymupdf4llm.to_markdown(str(pdf_path))
    return text


def fix_shifted_font_encoding(text):
    """Fix text from PDFs with shifted font encodings (copy-protection).

    Some PDFs use fonts with character codes shifted by a fixed offset
    to prevent simple text copying. This function detects and fixes such text.

    Args:
        text: Extracted text that may contain shifted characters

    Returns:
        Text with shifted encodings fixed
    """
    # Detect sequences of unicode replacement characters (���)
    # These indicate the extractor couldn't map the font encoding
    replacement_char = '\ufffd'

    if replacement_char not in text:
        return text

    print("Removing corrupted font characters...")

    # Count how many we're removing
    count = text.count(replacement_char)

    # Remove replacement characters but preserve structure
    # These are typically in title/header sections with copy-protection
    cleaned = text.replace(replacement_char, '')

    # Clean up any resulting empty bold/italic markers
    cleaned = re.sub(r'\*\*\s*\*\*', '', cleaned)  # Empty bold
    cleaned = re.sub(r'_\s*_', '', cleaned)  # Empty italic
    cleaned = re.sub(r'####\s*\n', '', cleaned)  # Empty headers
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)  # Excessive newlines

    print(f"  Removed {count} corrupted characters")

    return cleaned


def extract_with_ocr(pdf_path, dpi=200):
    """Extract text from PDF by rendering to images and using OCR.

    This approach bypasses font encoding issues by reading what's visually
    on the page. Works for both text-based and scanned PDFs.

    Pipeline:
    1. Render PDF pages to images
    2. Use Gemini Pro vision to detect layout (header, body columns, footer)
    3. Use Tesseract OCR to extract text from each region
    4. Combine in proper reading order

    Args:
        pdf_path: Path to the PDF file
        dpi: Resolution for rendering (200 is good balance of speed/quality)
    """
    from pdf2image import convert_from_path
    import pytesseract

    print(f"Rendering PDF to images at {dpi} DPI...")
    images = convert_from_path(pdf_path, dpi=dpi)
    print(f"  {len(images)} pages rendered")

    # Use Gemini Pro vision to detect layout (header/footer regions AND column structure)
    page_regions = detect_page_regions_with_vision(images)

    print("Extracting text with OCR...")
    all_text = []

    for i, image in enumerate(images):
        print(f"  Page {i+1}/{len(images)}...", end='', flush=True)

        width, height = image.size

        # Get layout info for this page
        layout = page_regions[i] if i < len(page_regions) else {
            'header_end': 0.0, 'footer_start': 1.0, 'is_dual_column': False, 'gutter_ratio': 0.5
        }
        header_end_y = int(height * layout['header_end'])
        footer_start_y = int(height * layout['footer_start'])
        is_dual_column = layout['is_dual_column']
        gutter_ratio = layout['gutter_ratio']

        page_parts = []

        # OCR header region (full-width)
        if header_end_y > 0:
            header_region = image.crop((0, 0, width, header_end_y))
            header_config = r'--oem 3 --psm 3'
            header_text = pytesseract.image_to_string(header_region, lang='eng', config=header_config)
            header_text = clean_ocr_text(header_text)
            if header_text.strip():
                page_parts.append(header_text)

        # OCR body region
        body_top = header_end_y
        body_bottom = footer_start_y

        if body_top < body_bottom:
            if is_dual_column:
                # Split into left and right columns
                gutter_x = int(width * gutter_ratio)
                margin = int(width * 0.01)

                left_half = image.crop((0, body_top, gutter_x - margin, body_bottom))
                right_half = image.crop((gutter_x + margin, body_top, width, body_bottom))

                # OCR each column
                config = r'--oem 3 --psm 4'
                left_text = pytesseract.image_to_string(left_half, lang='eng', config=config)
                right_text = pytesseract.image_to_string(right_half, lang='eng', config=config)

                left_text = clean_ocr_text(left_text)
                right_text = clean_ocr_text(right_text)

                if left_text.strip():
                    page_parts.append(left_text)
                if right_text.strip():
                    page_parts.append(right_text)
            else:
                # Single column - OCR entire body
                body_region = image.crop((0, body_top, width, body_bottom))
                body_config = r'--oem 3 --psm 3'
                body_text = pytesseract.image_to_string(body_region, lang='eng', config=body_config)
                body_text = clean_ocr_text(body_text)
                if body_text.strip():
                    page_parts.append(body_text)

        # OCR footer region (full-width)
        if footer_start_y < height:
            footer_region = image.crop((0, footer_start_y, width, height))
            footer_config = r'--oem 3 --psm 3'
            footer_text = pytesseract.image_to_string(footer_region, lang='eng', config=footer_config)
            footer_text = clean_ocr_text(footer_text)
            if footer_text.strip():
                page_parts.append(footer_text)

        page_text = "\n\n".join(page_parts)
        all_text.append(page_text)
        print(" done")

    return '\n\n'.join(all_text)


def get_gemini_api_key():
    """Get Gemini API key from environment or global_env.sh.

    Raises:
        ValueError: If no API key is found.
    """
    import os
    import re

    api_key = os.environ.get('GOOGLE_API_KEY') or os.environ.get('GEMINI_API_KEY')
    if not api_key:
        env_file = os.path.expanduser('~/global_env.sh')
        if os.path.exists(env_file):
            with open(env_file) as f:
                content = f.read()
                match = re.search(r'(?:setenv|export)\s+(?:GOOGLE|GEMINI)_API_KEY\s*[=\s]\s*["\']([^"\']+)["\']', content)
                if match:
                    api_key = match.group(1)

    if not api_key:
        raise ValueError(
            "Gemini API key not found. Set GEMINI_API_KEY or GOOGLE_API_KEY "
            "environment variable, or add it to ~/global_env.sh"
        )

    return api_key


def detect_page_regions_with_vision(images):
    """Detect page layout regions and column structure using Gemini vision.

    Sends page images to Gemini to visually identify:
    - Header, body, and footer regions
    - Whether the body is single or dual-column
    - Gutter position for dual-column pages

    Args:
        images: List of PIL images (page renderings)

    Returns:
        List of dicts with:
            - header_end: float (0-1 ratio of page height)
            - footer_start: float (0-1 ratio of page height)
            - is_dual_column: bool
            - gutter_ratio: float (0-1 ratio of page width, center of gutter)
    """
    import json
    from io import BytesIO
    from google import genai as genai_module
    from google.genai import types

    api_key = get_gemini_api_key()
    client = genai_module.Client(api_key=api_key)

    page_regions = []

    print(f"Detecting layout with Gemini Pro vision ({len(images)} pages)...")

    for page_idx, image in enumerate(images):
        print(f"  Page {page_idx+1}/{len(images)}...", end='', flush=True)

        # Convert PIL image to bytes
        img_buffer = BytesIO()
        image.save(img_buffer, format='PNG')
        img_bytes = img_buffer.getvalue()

        prompt = """Analyze this PDF page image and identify the layout.

Look for:
1. HEADER: Top portion with title, authors, abstract, etc. (full-width or centered content)
2. BODY: Main text content area
3. FOOTER: Bottom content (page numbers, copyright, etc.)
4. COLUMNS: Is the body text in one column or two columns?

Return ONLY a JSON object with:
- header_end: percentage (0-100) of page height where header ends
- footer_start: percentage (0-100) of page height where footer begins
- is_dual_column: true if body has two columns, false if single column
- gutter_ratio: if dual-column, the horizontal position (0-100) of the gap between columns

Example for dual-column page: {"header_end": 30, "footer_start": 95, "is_dual_column": true, "gutter_ratio": 50}
Example for single-column page: {"header_end": 10, "footer_start": 95, "is_dual_column": false, "gutter_ratio": 50}

Notes:
- Side-by-side author names are part of HEADER, not body columns
- Academic papers typically have dual columns; books are usually single column
- The gutter is the vertical white space between columns"""

        response = client.models.generate_content(
            model='gemini-3-pro-preview',
            contents=[prompt, types.Part.from_bytes(data=img_bytes, mime_type='image/png')]
        )
        result_text = response.text.strip()

        # Extract JSON from response
        if '{' not in result_text:
            raise ValueError(f"Gemini vision did not return JSON for page {page_idx + 1}: {result_text[:100]}")

        json_str = result_text[result_text.index('{'):result_text.rindex('}')+1]
        result = json.loads(json_str)
        page_regions.append({
            'header_end': result.get('header_end', 0) / 100,
            'footer_start': result.get('footer_start', 100) / 100,
            'is_dual_column': result.get('is_dual_column', False),
            'gutter_ratio': result.get('gutter_ratio', 50) / 100
        })
        print(" done")

    return page_regions


def clean_ocr_text(text):
    """Clean up OCR artifacts from extracted text."""
    if not text:
        return ""

    lines = text.strip().split('\n')
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()

        # Remove lines that are just 1-2 non-digit characters (OCR noise)
        if len(stripped) <= 2 and not stripped.isdigit():
            if not any(c.isalpha() for c in stripped):
                continue

        cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)


def clean_text(text, custom_headers=None):
    """Clean extracted text - remove headers, footers, fix spacing."""
    lines = text.split('\n')

    # First pass: detect repeated lines that differ only in trailing numbers
    # These are likely running headers like "ARTICLE TITLE 179", "ARTICLE TITLE 181"
    from collections import Counter

    def normalize_for_repetition(line):
        """Remove trailing numbers and normalize for comparison."""
        stripped = line.strip()
        # Remove trailing page numbers (1-3 digits at end)
        normalized = re.sub(r'\s+\d{1,3}\s*$', '', stripped)
        return normalized

    # Count occurrences of normalized lines
    normalized_counts = Counter()
    for line in lines:
        stripped = line.strip()
        if stripped and len(stripped) > 10:  # Only check substantial lines
            normalized = normalize_for_repetition(line)
            if normalized:
                normalized_counts[normalized] += 1

    # Lines that appear 3+ times (with different page numbers) are likely headers
    repeated_headers = {norm for norm, count in normalized_counts.items() if count >= 3}

    # Default patterns to remove (common academic journal artifacts)
    remove_patterns = [
        r'^\s*\d{1,3}\s*$',  # Standalone page numbers
        r'^Copyright\s*[©®]?\s*\d{4}',
        r'^\s*Access\s+provided\s+by',
        r'^DOI:\s*10\.',
        r'^https?://',  # URLs on their own line
        r'^Published by .* Press',
        # Generic academic running headers: "Author / Journal Vol (Year) Pages"
        r'^.{1,60}\s*/\s*.{10,80}\d{1,4}\s*\(\d{4}\)\s*\d+[–-]\d+',
        r'^\d{1,3}\s+.{1,60}\s*/\s*.{10,80}\(\d{4}\)',  # Page number at start
        r'.{10,80}\d+[–-]\d+\s+\d{1,3}\s*$',  # Page number at end
    ]

    # Add custom header patterns if provided
    if custom_headers:
        remove_patterns.extend(custom_headers)

    patterns = [re.compile(p, re.IGNORECASE) for p in remove_patterns]

    cleaned = []
    for line in lines:
        stripped = line.strip()

        # Skip empty lines at edges, keep internal ones
        if not stripped:
            if cleaned:
                cleaned.append('')
            continue

        # Check if this is a detected repeated header
        normalized = normalize_for_repetition(line)
        if normalized in repeated_headers:
            continue

        # Check static removal patterns
        should_remove = False
        for p in patterns:
            if p.match(stripped):
                should_remove = True
                break

        if not should_remove:
            cleaned.append(line)

    # Trim edges
    while cleaned and not cleaned[0].strip():
        cleaned.pop(0)
    while cleaned and not cleaned[-1].strip():
        cleaned.pop()

    result = '\n'.join(cleaned)

    # Clean up excessive whitespace
    result = re.sub(r'\n{4,}', '\n\n\n', result)

    return result


def correct_with_llm(text):
    """Use Gemini to fix extraction errors and ensure proper reading order.

    Sends text paragraph-by-paragraph to Gemini to:
    - Fix drop caps that got separated (e.g., "F rebels... I the" → "IF rebels...the")
    - Fix OCR-induced typos
    - Correct words split across line breaks
    - Remove corrupted characters from embedded fonts
    - Preserve original structure and meaning

    Args:
        text: Raw extracted text to correct

    Returns:
        Corrected text
    """
    from google import genai as genai_module

    api_key = get_gemini_api_key()
    client = genai_module.Client(api_key=api_key)

    # Split text into paragraphs (separated by blank lines)
    paragraphs = re.split(r'\n\s*\n', text)
    # Filter out empty paragraphs
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    print(f"Correcting with LLM ({len(paragraphs)} paragraphs)...")

    corrected_paragraphs = []
    previous_context = ""

    for i, paragraph in enumerate(paragraphs):
        print(f"  Paragraph {i+1}/{len(paragraphs)}...", end='', flush=True)

        prompt = f"""Fix errors in this text extracted from a PDF document.

Instructions:
1. Fix drop caps that got separated (e.g., if you see "F rebels face" followed by "# I" on the next line, merge to "IF rebels face")
2. Fix OCR typos (e.g., "rn" misread as "m", "l" as "1", etc.)
3. Remove corrupted characters (sequences of unicode replacement characters like ���)
4. Correct words that were incorrectly split across lines
5. Preserve the original structure and meaning
6. Do NOT add any new content or commentary
7. Do NOT remove meaningful content
8. Return ONLY the corrected text, nothing else

{f"Previous context (for continuity): ...{previous_context[-200:]}" if previous_context else ""}

Text to correct:
{paragraph}"""

        response = client.models.generate_content(
            model='gemini-flash-lite-latest',
            contents=prompt
        )
        corrected = response.text.strip()
        corrected_paragraphs.append(corrected)
        previous_context = corrected
        print(" done")

    # Join paragraphs with double newlines
    return '\n\n'.join(corrected_paragraphs)


def main():
    parser = argparse.ArgumentParser(
        description='Extract text from PDF files using OCR with AI-powered layout detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s document.pdf                    # Extract with OCR + LLM correction
    %(prog)s document.pdf output.txt         # Specify output file
    %(prog)s document.pdf --no-llm           # Skip LLM (faster but lower quality)
    %(prog)s document.pdf --dpi 400          # Higher quality OCR
    %(prog)s document.pdf --force-text       # Use direct text extraction (may have font issues)

Pipeline:
    1. Render PDF pages to images (bypasses font encoding issues)
    2. Use Gemini Pro vision to detect page layout (headers, columns, footers)
    3. Use Tesseract OCR to extract text from each region
    4. Use Gemini Flash Lite to fix OCR errors paragraph by paragraph
        """
    )
    parser.add_argument('input', help='Input PDF file')
    parser.add_argument('output', nargs='?', help='Output text file (default: input.txt)')
    parser.add_argument('--dpi', type=int, default=200, help='DPI for rendering (default: 200)')
    parser.add_argument('--force-text', action='store_true', help='Use direct text extraction instead of OCR (may have font encoding issues)')
    parser.add_argument('--headers', nargs='*', help='Additional header patterns to remove (regex)')
    parser.add_argument('--no-clean', action='store_true', help='Skip header/footer cleaning')
    parser.add_argument('--no-llm', action='store_true', help='Skip LLM post-processing (faster but lower quality)')

    args = parser.parse_args()

    # Check dependencies
    check_dependencies()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        sys.exit(1)

    output_path = Path(args.output) if args.output else input_path.with_suffix('.txt')

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")

    # Extract text - default is OCR-based pipeline which:
    # 1. Renders PDF to images (bypasses font encoding issues)
    # 2. Uses Gemini Pro vision to detect page layout
    # 3. Uses Tesseract OCR to extract text
    if args.force_text:
        # Use direct text extraction (may have font encoding issues)
        print("Mode: Direct text extraction (--force-text)")
        text = extract_with_pymupdf(input_path)
        text = fix_shifted_font_encoding(text)
    else:
        # Default: OCR-based extraction (works for text and scanned PDFs)
        is_text_pdf = is_text_based_pdf(input_path)
        print(f"Detected: {'text-based' if is_text_pdf else 'image-based (scanned)'} PDF")
        print("Using OCR pipeline (Gemini Pro vision + Tesseract)...")
        text = extract_with_ocr(input_path, dpi=args.dpi)

    # Clean text
    if not args.no_clean:
        print("Cleaning headers/footers...")
        text = clean_text(text, custom_headers=args.headers)

    # LLM post-processing to fix extraction errors (drop caps, OCR typos, etc.)
    if not args.no_llm:
        print("Correcting extraction errors with LLM...")
        text = correct_with_llm(text)

    # Write output
    output_path.write_text(text, encoding='utf-8')

    words = len(text.split())
    lines = text.count('\n') + 1
    print(f"\nComplete: {lines} lines, {words} words")
    print(f"Saved to: {output_path}")


if __name__ == '__main__':
    main()
