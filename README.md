# pdf-extract

A PDF text extraction tool designed for audiobook conversion. Uses Gemini Pro vision to reliably extract text from academic papers, including those with copy-protection or complex multi-column layouts.

## Features

- **Gemini Pro vision extraction**: Renders PDFs to images and uses Gemini Pro for text extraction
- **Handles multi-column layouts**: Gemini Pro vision automatically handles complex layouts
- **Bypasses font encoding issues**: Works with copy-protected PDFs that have garbled text extraction
- **Smart header removal**: Automatically detects and removes running headers/footers
- **Configurable**: DPI settings, force text mode

## Installation

```bash
# System dependencies (macOS)
brew install poppler

# System dependencies (Linux)
sudo apt install poppler-utils

# Python dependencies
pip install -r requirements.txt

# Required: Gemini API key
export GEMINI_API_KEY="your-key-here"  # or GOOGLE_API_KEY
```

## Usage

```bash
# Extract to stdout (progress goes to stderr)
python pdf_to_text.py document.pdf

# Save to file
python pdf_to_text.py document.pdf -o output.txt

# Higher DPI for better quality
python pdf_to_text.py document.pdf --dpi 300

# Force direct text extraction (may have font issues)
python pdf_to_text.py document.pdf --force-text

# Add custom header patterns to remove (regex)
python pdf_to_text.py document.pdf --headers "RUNNING HEADER" "Author Name"

# Pipe to another command
python pdf_to_text.py document.pdf 2>/dev/null | head -100
```

## Options

| Option | Description |
|--------|-------------|
| `-o, --output FILE` | Output file (default: stdout) |
| `--dpi N` | DPI for rendering (default: 200) |
| `--force-text` | Use direct text extraction instead of OCR |
| `--headers PATTERN...` | Additional header patterns to remove (regex) |

## How It Works

1. **Rendering**: PDF pages are rendered to images at the specified DPI (default 200)

2. **Extraction & Cleaning**: For each page:
   - Gemini Pro vision extracts text from the page image
   - Handles multi-column layouts automatically
   - Preserves proper reading order (headers → body → footnotes)
   - Cleans the page (removes page numbers, DOIs, copyright notices)
   - Streams the cleaned text to stdout immediately

## Dependencies

- **google-genai**: Gemini API for vision text extraction
- **pdf2image**: PDF to image conversion
- **poppler** (pdftotext): Quick text detection
- **pymupdf4llm**: Alternative direct text extraction

## License

MIT
