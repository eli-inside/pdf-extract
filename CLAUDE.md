# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PDF text extraction tool that handles text-based and scanned PDFs, with special handling for multi-column academic papers.

## Development Setup

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# System dependencies (macOS)
brew install tesseract poppler

# System dependencies (Linux)
sudo apt install tesseract-ocr poppler-utils
```

## Running the Tool

```bash
# Basic extraction
python3 pdf_to_text.py document.pdf

# With output file
python3 pdf_to_text.py document.pdf output.txt

# Force OCR mode
python3 pdf_to_text.py document.pdf --force-ocr

# Higher DPI for OCR
python3 pdf_to_text.py document.pdf --dpi 600
```

## Architecture

The extraction pipeline in `pdf_to_text.py` follows this decision tree:

1. **Detection Phase**
   - `is_text_based_pdf()`: Checks if PDF has extractable text (>50 words via pdftotext)
   - `detect_columns()`: Analyzes word positions with pdfplumber to find column gutter

2. **Extraction Phase** (based on detection results)
   - Text-based PDFs → `extract_with_pymupdf()`: Uses pymupdf4llm for automatic multi-column handling
   - Scanned single-column → `extract_single_column_ocr()`: Full-page OCR with tesseract
   - Scanned dual-column → `extract_dual_column_ocr()`: Splits image at gutter, OCRs each half

3. **Cleaning Phase**
   - `clean_text()`: Removes running headers/footers using:
     - Repetition detection (lines appearing 3+ times with different page numbers)
     - Static patterns (page numbers, DOIs, copyright notices)

## Key Dependencies

- **pymupdf4llm**: Primary extractor for text-based PDFs - handles multi-column layouts automatically
- **pdfplumber**: Used for column detection via word position analysis
- **pytesseract/pdf2image**: OCR pipeline for scanned documents
- **pdftotext** (poppler): Quick text detection to determine if OCR is needed
