#!/usr/bin/env python3
"""
PDF to Text Extraction Tool
Handles both text-based and image-based (scanned) PDFs, single and dual-column layouts.

Usage:
    python3 pdf_to_text.py input.pdf [output.txt]

If output not specified, writes to input.txt

Features:
- Auto-detects if PDF needs OCR or has extractable text
- Auto-detects single vs dual-column layout
- Handles dual-column PDFs by reading left column then right column
- Strips common academic headers/footers
- Configurable DPI for OCR quality

Dependencies:
    pip install pdfplumber pdf2image pytesseract
    brew install tesseract poppler  (macOS)
    apt install tesseract-ocr poppler-utils  (Linux)
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
    try:
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
    except Exception:
        return False


def detect_columns(pdf_path):
    """Detect if PDF has a two-column layout by analyzing word positions.

    Academic papers often have:
    - Full-width header/title at top
    - Two-column body text below

    We detect this by looking for a vertical gap (gutter) where no words exist.

    Returns:
        tuple: (is_dual_column: bool, gutter_center: float or None)
               gutter_center is the x-position of the center of the gutter as a ratio (0-1)
    """
    import pdfplumber

    with pdfplumber.open(pdf_path) as pdf:
        # Check first 2 pages
        pages_to_check = min(2, len(pdf.pages))

        for i in range(pages_to_check):
            page = pdf.pages[i]

            # Extract words with positions
            words = page.extract_words(
                x_tolerance=3,
                y_tolerance=3,
                keep_blank_chars=False
            )

            if not words or len(words) < 20:
                continue

            page_width = page.width
            page_height = page.height
            page_center = page_width / 2

            # Focus on body area (skip header ~30% and footer ~15%)
            body_top = page_height * 0.30
            body_bottom = page_height * 0.85

            body_words = [w for w in words
                          if w['top'] > body_top and w['bottom'] < body_bottom]

            if len(body_words) < 20:
                continue

            # Find the distribution of word positions
            # In a two-column layout, there should be a clear gap in the middle

            # Get the rightmost edge of left-side words and leftmost edge of right-side words
            left_words = [w for w in body_words if w['x1'] < page_center]
            right_words = [w for w in body_words if w['x0'] > page_center]

            if len(left_words) < 10 or len(right_words) < 10:
                continue

            # Find where left column ends and right column starts
            left_edge = max(w['x1'] for w in left_words)
            right_edge = min(w['x0'] for w in right_words)

            # Calculate the gap between columns
            gap = right_edge - left_edge

            # In a two-column layout, there should be a clear gutter (at least 3% of page width)
            min_gutter = page_width * 0.03

            if gap > min_gutter:
                # Calculate gutter center as a ratio of page width
                gutter_center = (left_edge + right_edge) / 2 / page_width
                return True, gutter_center

        return False, None


def extract_single_column_text(pdf_path):
    """Extract text from single-column text-based PDF using pdftotext."""
    print("Extracting text (single-column, text-based)...")
    result = subprocess.run(
        ['pdftotext', '-layout', str(pdf_path), '-'],
        capture_output=True,
        text=True,
        timeout=120
    )
    return result.stdout


def extract_single_column_ocr(pdf_path, dpi=300):
    """Extract text from single-column image-based PDF using OCR."""
    from pdf2image import convert_from_path
    import pytesseract

    print(f"Extracting text via OCR at {dpi} DPI (single-column, image-based)...")

    images = convert_from_path(pdf_path, dpi=dpi)
    print(f"Processing {len(images)} pages...")

    all_text = []

    # PSM 3 = Fully automatic page segmentation
    config = r'--oem 3 --psm 3'

    for i, image in enumerate(images):
        print(f"  Page {i+1}/{len(images)}...", end='', flush=True)
        text = pytesseract.image_to_string(image, lang='eng', config=config)
        all_text.append(text.strip())
        print(" done")

    return '\n\n--- PAGE BREAK ---\n\n'.join(all_text)


def extract_with_pymupdf(pdf_path):
    """Extract text from PDF using PyMuPDF's multi-column detection.

    Uses pymupdf4llm which handles multi-column layouts automatically.
    This is the recommended approach for text-based PDFs with complex layouts.

    Args:
        pdf_path: Path to the PDF file
    """
    import pymupdf4llm

    print("Extracting text with PyMuPDF (auto column detection)...")

    # pymupdf4llm.to_markdown handles multi-column layouts automatically
    md_text = pymupdf4llm.to_markdown(str(pdf_path))

    return md_text


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


def detect_page_regions_with_vision(images, gutter_ratio=0.5):
    """Detect header, body, and footer regions using Gemini vision.

    Sends page images to Gemini to visually identify layout regions.

    Args:
        images: List of PIL images (page renderings)
        gutter_ratio: Expected gutter position as ratio of page width (0-1)

    Returns:
        List of tuples: (header_end_ratio, footer_start_ratio) for each page
        Both are 0-1 ratios of page height.
    """
    import json
    from io import BytesIO
    from google import genai as genai_module
    from google.genai import types

    api_key = get_gemini_api_key()
    client = genai_module.Client(api_key=api_key)

    page_regions = []

    for page_idx, image in enumerate(images):
        # Convert PIL image to bytes
        img_buffer = BytesIO()
        image.save(img_buffer, format='PNG')
        img_bytes = img_buffer.getvalue()

        prompt = """Analyze this PDF page image and identify the layout regions.

Look for:
1. HEADER: The top portion with title, authors, affiliations, abstract heading, etc.
   This is content that spans the full width or is centered.
2. BODY: The main content area with two columns of text.
3. FOOTER: Any full-width content at the bottom (page numbers, copyright, etc.)

Return ONLY a JSON object with:
- header_end: percentage (0-100) of page height where the header ends and columns begin
- footer_start: percentage (0-100) of page height where footer begins

Example: {"header_end": 35, "footer_start": 95}

Important: Authors displayed side-by-side are part of the HEADER, not body columns.
The body columns contain the main article text (paragraphs, sections, etc.)."""

        response = client.models.generate_content(
            model='gemini-3-pro-preview',
            contents=[prompt, types.Part.from_bytes(data=img_bytes, mime_type='image/png')]
        )
        result_text = response.text.strip()

        # Extract JSON from response
        if '{' in result_text:
            json_str = result_text[result_text.index('{'):result_text.rindex('}')+1]
            result = json.loads(json_str)
            header_end = result.get('header_end', 0) / 100
            footer_start = result.get('footer_start', 100) / 100
            page_regions.append((header_end, footer_start))
        else:
            # Fallback if JSON parsing fails
            page_regions.append((0.0, 1.0))

    return page_regions


def extract_dual_column_ocr(pdf_path, dpi=300, gutter_ratio=0.5):
    """Extract text from dual-column image-based PDF using OCR with image splitting.

    Handles page layout by detecting regions using Gemini vision:
    - Header: Full-width content at top (OCR as single region)
    - Body: Two-column content in middle (OCR left then right)
    - Footer: Full-width content at bottom (OCR as single region)

    Args:
        pdf_path: Path to the PDF file
        dpi: DPI for rendering pages to images
        gutter_ratio: Position of the gutter as a ratio of page width (0-1)
    """
    from pdf2image import convert_from_path
    import pytesseract

    print(f"Extracting text via OCR at {dpi} DPI (dual-column, split at {gutter_ratio:.1%})...")

    images = convert_from_path(pdf_path, dpi=dpi)
    print(f"Processing {len(images)} pages...")

    # Use Gemini vision to detect page regions (header, footer)
    print("Detecting page layout with vision model...")
    page_regions = detect_page_regions_with_vision(images, gutter_ratio)

    all_text = []

    for i, image in enumerate(images):
        print(f"  Page {i+1}/{len(images)}...", end='', flush=True)

        width, height = image.size
        gutter_x = int(width * gutter_ratio)
        margin = int(width * 0.01)

        # Get region boundaries for this page
        header_ratio, footer_ratio = page_regions[i] if i < len(page_regions) else (0.0, 1.0)
        header_end_y = int(height * header_ratio)
        footer_start_y = int(height * footer_ratio)

        page_parts = []

        # OCR header region as single full-width region (if exists)
        if header_end_y > 0:
            header_region = image.crop((0, 0, width, header_end_y))
            header_config = r'--oem 3 --psm 3'  # Full page segmentation for header
            header_text = pytesseract.image_to_string(header_region, lang='eng', config=header_config)
            header_text = clean_ocr_text(header_text)
            if header_text.strip():
                page_parts.append(header_text)

        # OCR two-column body region
        body_top = header_end_y
        body_bottom = footer_start_y
        if body_top < body_bottom:
            # Crop left and right columns from body region only
            left_half = image.crop((0, body_top, gutter_x - margin, body_bottom))
            right_half = image.crop((gutter_x + margin, body_top, width, body_bottom))

            # OCR each half with single-column mode
            config = r'--oem 3 --psm 4'

            left_text = pytesseract.image_to_string(left_half, lang='eng', config=config)
            right_text = pytesseract.image_to_string(right_half, lang='eng', config=config)

            # Clean up each column
            left_text = clean_ocr_text(left_text)
            right_text = clean_ocr_text(right_text)

            # Add columns in reading order
            if left_text.strip():
                page_parts.append(left_text)
            if right_text.strip():
                page_parts.append(right_text)

        # OCR footer region as single full-width region (if exists)
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

    return '\n\n--- PAGE BREAK ---\n\n'.join(all_text)


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


def correct_with_llm(text, chunk_size=6000):
    """Use Gemini to fix extraction errors and ensure proper reading order.

    Sends text in chunks to Gemini to:
    - Fix drop caps that got separated (e.g., "F rebels... I the" → "IF rebels...the")
    - Fix OCR-induced typos
    - Correct words split across line breaks
    - Remove corrupted characters from embedded fonts
    - Preserve original structure and meaning

    Args:
        text: Raw extracted text to correct
        chunk_size: Maximum characters per chunk (to fit in context window)

    Returns:
        Corrected text
    """
    from google import genai as genai_module

    api_key = get_gemini_api_key()
    client = genai_module.Client(api_key=api_key)

    # Split text into chunks with overlap
    chunks = []
    overlap = 200  # Character overlap between chunks
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap

    print(f"Correcting with LLM ({len(chunks)} chunks)...")

    corrected_chunks = []
    previous_context = ""

    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}/{len(chunks)}...", end='', flush=True)

        prompt = f"""Fix errors in this text extracted from a PDF document.

Instructions:
1. Fix drop caps that got separated (e.g., if you see "F rebels face" followed by "# I" on the next line, merge to "IF rebels face")
2. Fix OCR typos (e.g., "rn" misread as "m", "l" as "1", etc.)
3. Remove corrupted characters (sequences of unicode replacement characters like ���)
4. Correct words that were incorrectly split across lines
5. Preserve the original structure, paragraphs, and meaning
6. Do NOT add any new content or commentary
7. Do NOT remove meaningful content
8. Return ONLY the corrected text, nothing else

{f"Previous context (for continuity): ...{previous_context[-300:]}" if previous_context else ""}

Text to correct:
{chunk}"""

        response = client.models.generate_content(
            model='gemini-3.0-flash-preview',  # Flash for speed, Pro for vision
            contents=prompt
        )

        corrected = response.text
        corrected_chunks.append(corrected)
        previous_context = corrected

        print(" done")

    # Join chunks, handling overlaps
    result = corrected_chunks[0]
    for chunk in corrected_chunks[1:]:
        # Skip the overlap portion that duplicates previous chunk
        result += chunk[overlap:]

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Extract text from PDF files (handles text/scanned, single/dual-column)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s document.pdf                    # Auto-detect and process with LLM
    %(prog)s document.pdf output.txt         # Specify output file
    %(prog)s document.pdf --no-llm           # Skip LLM (faster but lower quality)
    %(prog)s document.pdf --dpi 400          # Higher quality OCR
    %(prog)s document.pdf --force-ocr        # Force OCR even for text PDFs
        """
    )
    parser.add_argument('input', help='Input PDF file')
    parser.add_argument('output', nargs='?', help='Output text file (default: input.txt)')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for OCR (default: 300)')
    parser.add_argument('--force-ocr', action='store_true', help='Force OCR even if text is extractable')
    parser.add_argument('--force-text', action='store_true', help='Force text extraction even if PDF appears scanned')
    parser.add_argument('--single-column', action='store_true', help='Force single-column extraction')
    parser.add_argument('--dual-column', action='store_true', help='Force dual-column extraction')
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

    # Detect PDF type (text vs image-based)
    if args.force_ocr:
        is_text = False
        print("Mode: Force OCR")
    elif args.force_text:
        is_text = True
        print("Mode: Force text extraction")
    else:
        is_text = is_text_based_pdf(input_path)
        print(f"Detected: {'text-based' if is_text else 'image-based (scanned)'} PDF")

    # Extract text based on detected/forced settings
    if is_text:
        # For text-based PDFs, use PyMuPDF which handles multi-column layouts automatically
        text = extract_with_pymupdf(input_path)
    else:
        # For image-based (scanned) PDFs, detect column layout for OCR
        gutter_ratio = 0.5  # Default to center split
        if args.single_column:
            is_dual_column = False
            print("Layout: Force single-column")
        elif args.dual_column:
            is_dual_column = True
            print("Layout: Force dual-column")
        else:
            is_dual_column, detected_gutter = detect_columns(input_path)
            if is_dual_column and detected_gutter:
                gutter_ratio = detected_gutter
            print(f"Layout: {'dual-column' if is_dual_column else 'single-column'}")

        if is_dual_column:
            # For scanned dual-column PDFs, use OCR with column splitting
            text = extract_dual_column_ocr(input_path, dpi=args.dpi, gutter_ratio=gutter_ratio)
        else:
            # Single-column image-based: use OCR
            text = extract_single_column_ocr(input_path, dpi=args.dpi)

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
