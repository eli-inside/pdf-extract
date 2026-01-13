# pdf-extract

A robust PDF text extraction tool that handles both text-based and scanned (image-based) PDFs.

## Features

- **Auto-detection**: Automatically determines if a PDF needs OCR or has extractable text
- **OCR support**: Uses Tesseract for scanned documents
- **Header/footer cleaning**: Strips common academic journal artifacts
- **Configurable**: DPI settings, custom header patterns, force modes

## Installation

```bash
# System dependencies
sudo apt install tesseract-ocr poppler-utils

# Python dependencies
pip install pdfplumber pdf2image pytesseract
```

## Usage

```bash
# Basic usage - auto-detects PDF type
python pdf_to_text.py document.pdf

# Specify output file
python pdf_to_text.py document.pdf output.txt

# Force OCR (for problematic text PDFs)
python pdf_to_text.py document.pdf --force-ocr

# Higher DPI for better OCR quality
python pdf_to_text.py document.pdf --dpi 600

# Remove custom headers (regex patterns)
python pdf_to_text.py document.pdf --headers "RUNNING HEADER" "Author Name"
```

## Options

| Option | Description |
|--------|-------------|
| `--dpi N` | DPI for OCR (default: 400) |
| `--force-ocr` | Force OCR even if text is extractable |
| `--force-text` | Force pdftotext even if PDF appears scanned |
| `--headers PATTERN...` | Additional header patterns to remove |
| `--no-clean` | Skip header/footer cleaning |

## How It Works

1. **Detection**: Extracts a sample with `pdftotext`. If fewer than 50 words, assumes scanned.
2. **Extraction**: 
   - Text PDFs: Uses `pdftotext -layout` for accurate text extraction
   - Scanned PDFs: Converts to images at specified DPI, runs Tesseract OCR
3. **Cleaning**: Removes page numbers, copyright notices, DOIs, and other artifacts

## Origin

Built for extracting academic papers for research purposes. Handles the messy reality of PDFs from JSTOR, Project MUSE, and similar sources.

## License

MIT
