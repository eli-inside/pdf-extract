#!/usr/bin/env python3
"""
PDF to Text Extraction Tool
Handles both text-based and image-based (scanned) PDFs.

Usage:
    python3 pdf_to_text.py input.pdf [output.txt]
    
If output not specified, writes to input_extracted.txt

Features:
- Auto-detects if PDF needs OCR or has extractable text
- Strips common academic headers/footers
- Configurable DPI for OCR quality
- Progress reporting

Dependencies:
    pip install pdfplumber pdf2image pytesseract
    apt install tesseract-ocr poppler-utils
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
        subprocess.run(['pdftotext', '-v'], capture_output=True, stderr=subprocess.STDOUT)
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
        print("  sudo apt install tesseract-ocr poppler-utils")
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
        # If we get substantial text (more than just whitespace/headers), it's text-based
        words = len(text.split())
        return words > 50
    except Exception:
        return False

def extract_with_pdftotext(pdf_path):
    """Extract text from text-based PDF using pdftotext."""
    print("Using pdftotext (text-based PDF)...")
    result = subprocess.run(
        ['pdftotext', '-layout', str(pdf_path), '-'],
        capture_output=True,
        text=True,
        timeout=120
    )
    return result.stdout

def extract_with_ocr(pdf_path, dpi=400):
    """Extract text from image-based PDF using OCR."""
    from pdf2image import convert_from_path
    import pytesseract
    
    print(f"Using OCR at {dpi} DPI (image-based PDF)...")
    
    images = convert_from_path(pdf_path, dpi=dpi)
    print(f"Processing {len(images)} pages...")
    
    all_text = []
    
    # PSM 4 = single column, works better for academic papers
    config = r'--oem 3 --psm 4'
    
    for i, image in enumerate(images):
        print(f"  OCR page {i+1}/{len(images)}...", end='', flush=True)
        try:
            text = pytesseract.image_to_string(image, lang='eng', config=config)
            all_text.append(text)
            print(" done")
        except Exception as e:
            print(f" ERROR: {e}")
            all_text.append(f"[OCR ERROR on page {i+1}]")
    
    return '\n\n'.join(all_text)

def clean_text(text, custom_headers=None):
    """Clean extracted text - remove headers, footers, fix spacing."""
    lines = text.split('\n')
    cleaned = []
    
    # Default patterns to remove (common academic journal artifacts)
    remove_patterns = [
        r'^\s*\d{1,3}\s*$',  # Standalone page numbers
        r'^Copyright\s*[©®]?\s*\d{4}',
        r'^\s*Access\s+provided\s+by',
        r'^DOI:\s*10\.',
        r'^http://muse\.jhu\.edu',
        r'^Published by .* Press',
    ]
    
    # Add custom header patterns if provided
    if custom_headers:
        remove_patterns.extend(custom_headers)
    
    patterns = [re.compile(p, re.IGNORECASE) for p in remove_patterns]
    
    for line in lines:
        stripped = line.strip()
        
        # Skip empty lines at edges, keep internal ones
        if not stripped:
            if cleaned:
                cleaned.append('')
            continue
        
        # Check removal patterns
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
        description='Extract text from PDF files (handles both text and scanned PDFs)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s document.pdf
    %(prog)s document.pdf output.txt
    %(prog)s document.pdf --dpi 300 --force-ocr
    %(prog)s document.pdf --headers "RUNNING HEADER" "AUTHOR NAME"
        """
    )
    parser.add_argument('input', help='Input PDF file')
    parser.add_argument('output', nargs='?', help='Output text file (default: input_extracted.txt)')
    parser.add_argument('--dpi', type=int, default=400, help='DPI for OCR (default: 400)')
    parser.add_argument('--force-ocr', action='store_true', help='Force OCR even if text is extractable')
    parser.add_argument('--force-text', action='store_true', help='Force pdftotext even if PDF appears scanned')
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
    
    # Determine extraction method
    if args.force_ocr:
        use_ocr = True
    elif args.force_text:
        use_ocr = False
    else:
        use_ocr = not is_text_based_pdf(input_path)
        print(f"Detected: {'image-based (scanned)' if use_ocr else 'text-based'} PDF")
    
    # Extract text
    if use_ocr:
        text = extract_with_ocr(input_path, dpi=args.dpi)
    else:
        text = extract_with_pdftotext(input_path)
    
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

