# pdf-extract

A PDF text extraction tool designed for audiobook conversion. Uses OCR with AI-powered layout detection to reliably extract text from academic papers, including those with copy-protection or complex multi-column layouts.

## Features

- **OCR-first pipeline**: Renders PDFs to images and uses OCR, bypassing font encoding issues
- **AI layout detection**: Gemini Pro vision detects headers, columns, and footers per page
- **Multi-column support**: Properly orders two-column academic paper layouts
- **LLM post-processing**: Gemini Flash Lite fixes OCR errors (drop caps, typos, etc.)
- **Smart header removal**: Automatically detects and removes running headers/footers
- **Configurable**: DPI settings, skip LLM, force text mode

## Installation

```bash
# System dependencies (macOS)
brew install tesseract poppler

# System dependencies (Linux)
sudo apt install tesseract-ocr poppler-utils

# Python dependencies
pip install -r requirements.txt

# Required: Gemini API key
export GEMINI_API_KEY="your-key-here"  # or GOOGLE_API_KEY
```

## Usage

```bash
# Basic usage - OCR pipeline with AI layout detection
python pdf_to_text.py document.pdf

# Specify output file
python pdf_to_text.py document.pdf output.txt

# Higher DPI for better OCR quality
python pdf_to_text.py document.pdf --dpi 300

# Skip LLM post-processing (faster)
python pdf_to_text.py document.pdf --no-llm

# Force direct text extraction (may have font issues)
python pdf_to_text.py document.pdf --force-text

# Add custom header patterns to remove (regex)
python pdf_to_text.py document.pdf --headers "RUNNING HEADER" "Author Name"

# Skip header/footer cleaning
python pdf_to_text.py document.pdf --no-clean
```

## Options

| Option | Description |
|--------|-------------|
| `--dpi N` | DPI for rendering (default: 200) |
| `--force-text` | Use direct text extraction instead of OCR |
| `--no-llm` | Skip LLM post-processing (faster but lower quality) |
| `--no-clean` | Skip header/footer cleaning |
| `--headers PATTERN...` | Additional header patterns to remove (regex) |

## How It Works

1. **Rendering**: PDF pages are rendered to images at the specified DPI (default 200)

2. **Layout Detection**: Gemini Pro vision analyzes each page image to identify:
   - Header region (full-width content at top)
   - Body region (single or dual-column)
   - Footer region (full-width content at bottom)
   - Column gutter position for dual-column pages

3. **OCR Extraction**: Tesseract extracts text from each region in proper reading order:
   - Header → Left column → Right column → Footer

4. **Cleaning**: Removes running headers/footers by detecting:
   - Lines that repeat across pages (with different page numbers)
   - Standalone page numbers, DOIs, copyright notices

5. **LLM Correction** (optional): Gemini Flash Lite fixes:
   - Drop caps that got separated (e.g., "F rebels" → "IF rebels")
   - OCR typos (e.g., "rn" misread as "m")
   - Corrupted unicode characters

## Dependencies

- **google-genai**: Gemini API for vision layout detection and LLM post-processing
- **pdf2image**: PDF to image conversion for OCR
- **pytesseract**: OCR engine for text extraction
- **poppler** (pdftotext): Quick text detection
- **pymupdf4llm**: Alternative direct text extraction

## License

MIT
