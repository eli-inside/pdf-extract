# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PDF text extraction tool designed for audiobook conversion. Uses OCR with AI-powered layout detection to bypass font encoding issues common in copy-protected academic PDFs.

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

# Required: Gemini API key
export GEMINI_API_KEY="your-key-here"  # or GOOGLE_API_KEY
```

## Running the Tool

```bash
# Basic extraction (uses OCR pipeline by default)
python3 pdf_to_text.py document.pdf

# With output file
python3 pdf_to_text.py document.pdf output.txt

# Higher DPI for better OCR quality
python3 pdf_to_text.py document.pdf --dpi 300

# Skip LLM post-processing (faster but lower quality)
python3 pdf_to_text.py document.pdf --no-llm

# Force direct text extraction (may have font encoding issues)
python3 pdf_to_text.py document.pdf --force-text
```

## Design Principles

**OCR-First Approach**: The default pipeline renders PDFs to images and uses OCR. This bypasses font encoding issues that cause garbled text in some copy-protected PDFs.

**AI-Powered Layout Detection**: Gemini Pro vision analyzes each page image to detect:
- Header regions (full-width content at top)
- Body regions (single or dual-column)
- Footer regions (full-width content at bottom)
- Column gutter position for dual-column layouts

**LLM Post-Processing**: Gemini Flash Lite fixes common OCR errors:
- Drop caps that got separated (e.g., "F rebels" → "IF rebels")
- OCR typos (e.g., "rn" misread as "m")
- Corrupted unicode characters

**Reading Order Matters**: Output follows natural reading order for audiobook use:
1. Header (full-width, top of page)
2. Left column (top to bottom)
3. Right column (top to bottom)
4. Footer (full-width, bottom of page)

## Architecture

The extraction pipeline in `pdf_to_text.py`:

1. **Rendering Phase**
   - `convert_from_path()`: Renders PDF pages to images at specified DPI (default 200)

2. **Layout Detection Phase**
   - `detect_page_regions_with_vision()`: Uses Gemini Pro vision to identify header/body/footer regions and column structure per page

3. **Extraction Phase**
   - `extract_with_ocr()`: Uses Tesseract to OCR each region in proper reading order

4. **Cleaning Phase**
   - `clean_text()`: Removes running headers/footers using:
     - Repetition detection (lines appearing 3+ times with different page numbers)
     - Static patterns (page numbers, DOIs, copyright notices)

5. **LLM Correction Phase** (optional, can disable with --no-llm)
   - `correct_with_llm()`: Fixes OCR errors paragraph by paragraph using Gemini Flash Lite

## Key Dependencies

- **google-genai**: Gemini API for vision layout detection and LLM post-processing
- **pdf2image**: Renders PDF pages to images for OCR
- **pytesseract**: OCR engine for text extraction
- **pdftotext** (poppler): Quick text detection to identify PDF type
- **pymupdf4llm**: Alternative direct text extraction (use with --force-text)

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
