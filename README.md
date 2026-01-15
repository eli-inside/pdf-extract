# pdf-extract

A robust PDF text extraction tool that handles text-based and scanned PDFs, with automatic multi-column layout detection.

## Features

- **Auto-detection**: Automatically determines if a PDF needs OCR or has extractable text
- **Multi-column support**: Detects and properly orders two-column academic paper layouts
- **Smart header removal**: Automatically detects and removes running headers/footers
- **OCR support**: Uses Tesseract for scanned documents
- **Configurable**: DPI settings, custom header patterns, force modes

## Installation

```bash
# System dependencies (macOS)
brew install tesseract poppler

# System dependencies (Linux)
sudo apt install tesseract-ocr poppler-utils

# Python dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Basic usage - auto-detects PDF type and column layout
python pdf_to_text.py document.pdf

# Specify output file
python pdf_to_text.py document.pdf output.txt

# Force OCR (for problematic text PDFs)
python pdf_to_text.py document.pdf --force-ocr

# Higher DPI for better OCR quality
python pdf_to_text.py document.pdf --dpi 400

# Force single or dual column mode
python pdf_to_text.py document.pdf --single-column
python pdf_to_text.py document.pdf --dual-column

# Add custom header patterns to remove (regex)
python pdf_to_text.py document.pdf --headers "RUNNING HEADER" "Author Name"
```

## Options

| Option | Description |
|--------|-------------|
| `--dpi N` | DPI for OCR (default: 300) |
| `--force-ocr` | Force OCR even if text is extractable |
| `--force-text` | Force text extraction even if PDF appears scanned |
| `--single-column` | Force single-column extraction |
| `--dual-column` | Force dual-column extraction |
| `--headers PATTERN...` | Additional header patterns to remove (regex) |
| `--no-clean` | Skip header/footer cleaning |

## How It Works

1. **Detection**:
   - Checks if PDF has extractable text using pdftotext
   - Analyzes word positions to detect two-column layouts

2. **Extraction**:
   - Text PDFs: Uses PyMuPDF (pymupdf4llm) which automatically handles multi-column layouts
   - Scanned single-column: Full-page OCR with Tesseract
   - Scanned dual-column: Splits page at detected gutter, OCRs each half separately

3. **Cleaning**:
   - Detects running headers by finding lines that repeat across pages (with different page numbers)
   - Removes standalone page numbers, DOIs, copyright notices, and other artifacts

## Dependencies

- **pymupdf4llm**: Primary text extraction with automatic column handling
- **pdfplumber**: Column layout detection
- **pytesseract**: OCR for scanned documents
- **pdf2image**: PDF to image conversion for OCR
- **poppler** (pdftotext): Quick text detection

## License

MIT
