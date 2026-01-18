#!/usr/bin/env python3
"""
PDF to Text Extraction Tool
Uses Gemini Pro vision for reliable text extraction from PDF images.

Usage:
    python3 pdf_to_text.py input.pdf              # Output to stdout
    python3 pdf_to_text.py input.pdf -o out.txt   # Output to file

Progress messages go to stderr, text output goes to stdout (or file with -o).

Pipeline:
1. Render PDF pages to images (bypasses font encoding issues)
2. Use Gemini Pro vision to extract text from each page
3. Clean headers/footers

Features:
- Handles both text-based and scanned PDFs uniformly
- Gemini Pro vision handles multi-column layouts automatically
- Strips common academic headers/footers

Dependencies:
    pip install -r requirements.txt
    brew install poppler  (macOS)
    apt install poppler-utils  (Linux)

Requires GEMINI_API_KEY or GOOGLE_API_KEY environment variable.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path


def check_dependencies():
    """Check if required tools are installed."""
    missing = []

    # Check pdftotext (used for PDF type detection)
    try:
        subprocess.run(['pdftotext', '-v'], capture_output=True)
    except FileNotFoundError:
        missing.append('poppler-utils')

    # Check Python packages
    try:
        from pdf2image import convert_from_path
    except ImportError:
        missing.append('pdf2image (pip)')

    try:
        from google import genai
    except ImportError:
        missing.append('google-genai (pip)')

    if missing:
        print(f"Missing dependencies: {', '.join(missing)}", file=sys.stderr)
        print("\nInstall with:", file=sys.stderr)
        print("  brew install poppler  (macOS)", file=sys.stderr)
        print("  apt install poppler-utils  (Linux)", file=sys.stderr)
        print("  pip install pdf2image google-genai", file=sys.stderr)
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

    print("Extracting text with PyMuPDF...", file=sys.stderr)
    print("Consider using the pymupdf_layout package for a greatly improved page layout analysis.", file=sys.stderr)
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

    print("Removing corrupted font characters...", file=sys.stderr)

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

    print(f"  Removed {count} corrupted characters", file=sys.stderr)

    return cleaned


def extract_with_ocr(pdf_path, dpi=200, stream=None, custom_headers=None):
    """Extract text from PDF by rendering to images and using Gemini Pro vision.

    This approach bypasses font encoding issues by reading what's visually
    on the page. Works for both text-based and scanned PDFs.

    Pipeline:
    1. Render PDF pages to images
    2. Send each page to Gemini Pro vision for text extraction
    3. Clean each page (remove headers/footers)
    4. Stream or combine pages

    Args:
        pdf_path: Path to the PDF file
        dpi: Resolution for rendering (200 is sufficient for Gemini Pro)
        stream: Optional file-like object to stream output to (enables per-page output)
        custom_headers: Optional list of regex patterns to remove

    Returns:
        If stream is None: Extracted text as string
        If stream is provided: tuple of (line_count, word_count)
    """
    from pdf2image import convert_from_path
    from io import BytesIO
    from google import genai as genai_module
    from google.genai import types

    print(f"Rendering PDF to images at {dpi} DPI...", file=sys.stderr)
    images = convert_from_path(pdf_path, dpi=dpi)
    print(f"  {len(images)} pages rendered", file=sys.stderr)

    api_key = get_gemini_api_key()
    client = genai_module.Client(api_key=api_key)

    prompt = """Extract ALL the text from this PDF page image, preserving the reading order.
Return the text organized as:
1. First any header/running head at the top
2. Then the main body text (read top to bottom, left to right for multi-column layouts)
3. Finally any footnotes at the bottom
Just return the extracted text, nothing else."""

    print("Extracting text with Gemini Pro vision...", file=sys.stderr)
    all_text = []
    total_words = 0
    total_lines = 0

    for i, image in enumerate(images):
        print(f"  Page {i+1}/{len(images)}...", end='', flush=True, file=sys.stderr)

        # Convert PIL image to bytes
        img_buffer = BytesIO()
        image.save(img_buffer, format='PNG')
        img_bytes = img_buffer.getvalue()

        response = client.models.generate_content(
            model='gemini-3-pro-preview',
            contents=[prompt, types.Part.from_bytes(data=img_bytes, mime_type='image/png')]
        )
        page_text = (response.text or "").strip()

        # Clean each page immediately
        page_text = clean_page(page_text, custom_headers)

        if stream is not None:
            # Streaming mode: write immediately
            if page_text:
                stream.write(page_text)
                stream.write('\n\n')
                stream.flush()
                total_words += len(page_text.split())
                total_lines += page_text.count('\n') + 1
        else:
            all_text.append(page_text)

        print(" done", file=sys.stderr)

    if stream is not None:
        return (total_lines, total_words)
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


def clean_page(text, custom_headers=None):
    """Clean a single page's text - remove headers, footers, fix spacing.

    Args:
        text: Text from a single page
        custom_headers: Optional list of regex patterns to remove

    Returns:
        Cleaned text
    """
    lines = text.split('\n')

    # Patterns to remove (common academic journal artifacts)
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


# Alias for backwards compatibility with tests
clean_text = clean_page


def main():
    parser = argparse.ArgumentParser(
        description='Extract text from PDF files using Gemini Pro vision',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s document.pdf                    # Stream to stdout
    %(prog)s document.pdf -o output.txt      # Save to file
    %(prog)s document.pdf --force-text       # Use direct text extraction (may have font issues)

Pipeline:
    1. Render PDF pages to images (bypasses font encoding issues)
    2. Use Gemini Pro vision to extract text from each page
    3. Clean headers/footers from each page
    4. Stream cleaned text to stdout (or save to file)
        """
    )
    parser.add_argument('input', help='Input PDF file')
    parser.add_argument('-o', '--output', help='Output text file (default: stdout)')
    parser.add_argument('--dpi', type=int, default=200, help='DPI for rendering (default: 200)')
    parser.add_argument('--force-text', action='store_true', help='Use direct text extraction instead of OCR (may have font encoding issues)')
    parser.add_argument('--headers', nargs='*', help='Additional header patterns to remove (regex)')

    args = parser.parse_args()

    # Check dependencies
    check_dependencies()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Input:  {input_path}", file=sys.stderr)
    if args.output:
        print(f"Output: {args.output}", file=sys.stderr)
    else:
        print("Output: stdout", file=sys.stderr)

    # Extract text - default is Gemini Pro vision pipeline which:
    # 1. Renders PDF to images (bypasses font encoding issues)
    # 2. Uses Gemini Pro vision to extract text from each page
    # 3. Cleans each page (removes headers/footers)
    if args.force_text:
        # Use direct text extraction (may have font encoding issues)
        print("Mode: Direct text extraction (--force-text)", file=sys.stderr)
        text = extract_with_pymupdf(input_path)
        text = fix_shifted_font_encoding(text)
        text = clean_page(text, custom_headers=args.headers)
        if args.output:
            Path(args.output).write_text(text, encoding='utf-8')
            print(f"\nSaved to: {args.output}", file=sys.stderr)
        else:
            print(text)
        words = len(text.split())
        lines = text.count('\n') + 1
    else:
        # Default: Gemini Pro vision extraction (works for text and scanned PDFs)
        is_text_pdf = is_text_based_pdf(input_path)
        print(f"Detected: {'text-based' if is_text_pdf else 'image-based (scanned)'} PDF", file=sys.stderr)
        print("Using Gemini Pro vision for text extraction...", file=sys.stderr)

        if args.output:
            # Buffer for file output
            text = extract_with_ocr(input_path, dpi=args.dpi, custom_headers=args.headers)
            Path(args.output).write_text(text, encoding='utf-8')
            print(f"\nSaved to: {args.output}", file=sys.stderr)
            words = len(text.split())
            lines = text.count('\n') + 1
        else:
            # Stream directly to stdout
            lines, words = extract_with_ocr(input_path, dpi=args.dpi, stream=sys.stdout, custom_headers=args.headers)

    print(f"Complete: {lines} lines, {words} words", file=sys.stderr)


if __name__ == '__main__':
    main()
