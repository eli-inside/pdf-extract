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

    try:
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
    except Exception:
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


def detect_column_regions(pdf_path, gutter_ratio=0.5):
    """Detect where full-width header ends and two-column body begins on each page.

    Analyzes word positions to find where the gutter gap first appears consistently,
    indicating the transition from full-width header to two-column body.

    Args:
        pdf_path: Path to the PDF file
        gutter_ratio: Expected gutter position as ratio of page width (0-1)

    Returns:
        List of floats: header_end_ratio for each page (0-1, as fraction of page height)
    """
    import pdfplumber

    header_ratios = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_width = page.width
            page_height = page.height

            # Define gutter zone (center ± 3% of page width - tighter zone)
            gutter_left = page_width * (gutter_ratio - 0.03)
            gutter_right = page_width * (gutter_ratio + 0.03)

            words = page.extract_words(x_tolerance=3, y_tolerance=3)
            if not words:
                header_ratios.append(0.0)
                continue

            # Divide page into horizontal bands (2% of page height each)
            band_height = page_height * 0.02
            num_bands = int(page_height / band_height)

            # For each band, check if any word center falls in the gutter zone
            band_has_gutter_word = [False] * num_bands

            for word in words:
                word_center_x = (word['x0'] + word['x1']) / 2
                word_center_y = (word['top'] + word['bottom']) / 2
                band_idx = min(int(word_center_y / band_height), num_bands - 1)

                if gutter_left < word_center_x < gutter_right:
                    band_has_gutter_word[band_idx] = True

            # Find the END of the contiguous header region
            # Header = area where gutter words appear; ends when 3+ consecutive bands are empty
            first_gutter_band = None
            header_end_band = 0

            for band_idx, has_word in enumerate(band_has_gutter_word):
                if has_word:
                    if first_gutter_band is None:
                        first_gutter_band = band_idx
                    header_end_band = band_idx + 1  # Header extends through this band

            # Now find where the header actually ends (3+ consecutive empty bands after header starts)
            if first_gutter_band is not None:
                consecutive_empty = 0
                for band_idx in range(first_gutter_band, num_bands):
                    if not band_has_gutter_word[band_idx]:
                        consecutive_empty += 1
                        if consecutive_empty >= 3:
                            # Header ends at the start of this empty region
                            header_end_band = band_idx - 2
                            break
                    else:
                        consecutive_empty = 0
                        header_end_band = band_idx + 1

                header_end_ratio = min((header_end_band * band_height) / page_height, 0.4)
            else:
                # No gutter words found - no header, columns start at top
                header_end_ratio = 0.0

            header_ratios.append(header_end_ratio)

    return header_ratios


def extract_dual_column_ocr(pdf_path, dpi=300, gutter_ratio=0.5):
    """Extract text from dual-column image-based PDF using OCR with image splitting.

    Handles full-width headers by detecting and OCRing them separately from the
    two-column body. This prevents headers from being split and jumbled.

    Args:
        pdf_path: Path to the PDF file
        dpi: DPI for rendering pages to images
        gutter_ratio: Position of the gutter as a ratio of page width (0-1)
    """
    from pdf2image import convert_from_path
    import pytesseract

    print(f"Extracting text via OCR at {dpi} DPI (dual-column, split at {gutter_ratio:.1%})...")

    # Detect header regions for each page
    header_ratios = detect_column_regions(pdf_path, gutter_ratio)

    images = convert_from_path(pdf_path, dpi=dpi)
    print(f"Processing {len(images)} pages...")

    all_text = []

    for i, image in enumerate(images):
        print(f"  Page {i+1}/{len(images)}...", end='', flush=True)

        width, height = image.size
        gutter_x = int(width * gutter_ratio)
        margin = int(width * 0.01)

        # Get header boundary for this page
        header_ratio = header_ratios[i] if i < len(header_ratios) else 0.0
        header_end_y = int(height * header_ratio)

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
        if body_top < height:
            # Crop left and right columns from body region only
            left_half = image.crop((0, body_top, gutter_x - margin, height))
            right_half = image.crop((gutter_x + margin, body_top, width, height))

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


def main():
    parser = argparse.ArgumentParser(
        description='Extract text from PDF files (handles text/scanned, single/dual-column)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s document.pdf                    # Auto-detect everything
    %(prog)s document.pdf output.txt         # Specify output file
    %(prog)s document.pdf --dpi 400          # Higher quality OCR
    %(prog)s document.pdf --single-column    # Force single-column mode
    %(prog)s document.pdf --dual-column      # Force dual-column mode
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

    # Detect column layout
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

    # Extract text based on detected/forced settings
    if is_text:
        # For text-based PDFs, use PyMuPDF which handles multi-column layouts
        text = extract_with_pymupdf(input_path)
    elif is_dual_column:
        # For scanned dual-column PDFs, use OCR with column splitting
        text = extract_dual_column_ocr(input_path, dpi=args.dpi, gutter_ratio=gutter_ratio)
    else:
        # Single-column image-based: use OCR
        text = extract_single_column_ocr(input_path, dpi=args.dpi)

    # Clean text
    if not args.no_clean:
        print("Cleaning headers/footers...")
        text = clean_text(text, custom_headers=args.headers)

    # Write output
    output_path.write_text(text, encoding='utf-8')

    words = len(text.split())
    lines = text.count('\n') + 1
    print(f"\nComplete: {lines} lines, {words} words")
    print(f"Saved to: {output_path}")


if __name__ == '__main__':
    main()
