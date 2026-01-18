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

## Design Principles

**Text Extraction vs OCR**:
- **Text-based PDFs**: Use pymupdf4llm for direct text extraction. No OCR needed. The library handles multi-column layouts automatically.
- **Image-based (scanned) PDFs**: Use OCR with Tesseract. Vision AI (Gemini) detects layout regions for proper column handling.

**Structure over Content**: All detection and processing must be based on document structure (word positions, layout geometry), not content (specific words like "ABSTRACT", "Introduction"). Documents vary widely and may not contain expected section headers.

**Everything is Dynamic and Optional**:
- Text vs. scanned (OCR) - detected per document
- Single vs. two-column layout - detected per document
- Headers - detected per page via vision AI (top region where text spans full width)
- Footers - detected per page via vision AI (bottom region where text spans full width)
- Column gutter position - detected dynamically from word positions

**Reading Order Matters**: Output must follow natural reading order for audiobook use:
1. Header (full-width, top of page)
2. Left column (top to bottom)
3. Right column (top to bottom)
4. Footer (full-width, bottom of page)

## Architecture

The extraction pipeline in `pdf_to_text.py` follows this decision tree:

1. **Detection Phase**
   - `is_text_based_pdf()`: Checks if PDF has extractable text (>50 words via pdftotext)
   - `detect_columns()`: Analyzes word positions with pdfplumber to find column gutter
   - `detect_page_regions()`: Finds header/footer boundaries based on where text crosses the gutter

2. **Extraction Phase** (based on detection results)
   - Text-based PDFs → `extract_with_pymupdf()`: Uses pymupdf4llm for automatic multi-column handling
   - Scanned single-column → `extract_single_column_ocr()`: Full-page OCR with tesseract
   - Scanned dual-column → `extract_dual_column_ocr()`: OCRs header, left column, right column, footer separately

3. **Cleaning Phase**
   - `clean_text()`: Removes running headers/footers using:
     - Repetition detection (lines appearing 3+ times with different page numbers)
     - Static patterns (page numbers, DOIs, copyright notices)

## Key Dependencies

- **pymupdf4llm**: Primary extractor for text-based PDFs - handles multi-column layouts automatically
- **pdfplumber**: Used for column detection via word position analysis
- **pytesseract/pdf2image**: OCR pipeline for scanned documents
- **pdftotext** (poppler): Quick text detection to determine if OCR is needed

## Architecture Best Practices

- **TDD (Test-Driven Development)** - write the tests first; the implementation code isn't done until the tests pass.
- **DRY (Don't Repeat Yourself)** – eliminate duplicated logic by extracting shared utilities and modules.
- **Separation of Concerns** – each module should handle one distinct responsibility.
- **Single Responsibility Principle (SRP)** – every class/module/function/file should have exactly one reason to change.
- **Clear Abstractions & Contracts** – expose intent through small, stable interfaces and hide implementation details.
- **Low Coupling, High Cohesion** – keep modules self-contained, minimize cross-dependencies.
- **Scalability & Statelessness** – design components to scale horizontally and prefer stateless services when possible.
- **Observability & Testability** – build in logging, metrics, tracing, and ensure components can be unit/integration tested.
- **KISS (Keep It Simple, Sir)** - keep solutions as simple as possible.
- **YAGNI (You're Not Gonna Need It)** – avoid speculative complexity or over-engineering.
- **Don't Swallow Errors** by catching exceptions, silently filling in required but missing values or adding timeouts when something hangs unexpectedly. All of those are exceptions that should be thrown so that the errors can be seen, root causes can be found and fixes can be applied.
- **No Placeholder Code** - we're building production code here, not toys.
- **No Comments for Removed Functionality** - the source is not the place to keep history of what's changed; it's the place to implement the current requirements only.
- **Layered Architecture** - organize code into clear tiers where each layer depends only on the one(s) below it, keeping logic cleanly separated.
- **Prefer Non-Nullable Variables** when possible; use nullability sparingly.
- **Prefer Async Notifications** when possible over inefficient polling.
- **Consider First Principles** to assess your current architecture against the one you'd use if you started over from scratch.
- **Eliminate Race Conditions** that might cause dropped or corrupted data.
- **Write for Maintainability** so that the code is clear and readable and easy to maintain by future developers.
