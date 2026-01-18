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
brew install poppler

# System dependencies (Linux)
sudo apt install poppler-utils

# Required: Gemini API key
export GEMINI_API_KEY="your-key-here"  # or GOOGLE_API_KEY
```

## Running the Tool

```bash
# Extract to stdout (progress goes to stderr)
python3 pdf_to_text.py document.pdf

# Save to file
python3 pdf_to_text.py document.pdf -o output.txt

# Higher DPI for better quality
python3 pdf_to_text.py document.pdf --dpi 300

# Force direct text extraction (may have font encoding issues)
python3 pdf_to_text.py document.pdf --force-text

# Pipe to another command
python3 pdf_to_text.py document.pdf 2>/dev/null | head -100
```

## Design Principles

**OCR-First Approach**: The default pipeline renders PDFs to images and uses OCR. This bypasses font encoding issues that cause garbled text in some copy-protected PDFs.

**Gemini Pro Vision Text Extraction**: Each page is sent to Gemini Pro vision which:
- Extracts all text from the page image
- Handles multi-column layouts automatically
- Preserves proper reading order (headers → body → footnotes)

**Reading Order Matters**: Output follows natural reading order for audiobook use:
1. Header (full-width, top of page)
2. Left column (top to bottom)
3. Right column (top to bottom)
4. Footer (full-width, bottom of page)

## Architecture

The extraction pipeline in `pdf_to_text.py`:

1. **Rendering Phase**
   - `convert_from_path()`: Renders PDF pages to images at specified DPI (default 200)

2. **Extraction Phase**
   - `extract_with_ocr()`: Sends each page image to Gemini Pro vision for text extraction

3. **Cleaning Phase**
   - `clean_text()`: Removes running headers/footers using:
     - Repetition detection (lines appearing 3+ times with different page numbers)
     - Static patterns (page numbers, DOIs, copyright notices)

## Key Dependencies

- **google-genai**: Gemini API for vision text extraction
- **pdf2image**: Renders PDF pages to images
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
