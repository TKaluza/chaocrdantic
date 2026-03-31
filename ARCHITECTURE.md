# chaocrdantic Architecture

## Overview

`chaocrdantic` is an importable OCR library built around an OpenAI-compatible multimodal endpoint and the stock `pydantic_ai` OpenAI integration. The core document flow is:

1. Load a PDF or image.
2. Render PDF pages to RGB images.
3. Send each page image plus the OCR prompt to the model server.
4. Parse the returned HTML into cleaned HTML, Markdown, layout blocks, and extracted images.
5. Assemble a document-level `OCRResult`.

## Main Components

- `chaocrdantic/image_utils.py`
  Renders PDFs and normalizes images for inference.
- `chaocrdantic/agent.py`
  Owns model calls, retry behavior, concurrency, and result assembly using `pydantic_ai.Agent` plus `OpenAIChatModel`.
- `chaocrdantic/output.py`
  Parses raw model HTML into structured page output.
- `chaocrdantic/document_renderer.py`
  Merges page output into document markdown with front matter and page breaks.
- `chaocrdantic/api.py`
  Exposes simple `ocr_file`, `ocr_file_async`, `ocr_image`, and `ocr_image_async` helpers.
- `chaocrdantic/cli.py`
  Wraps the same library path for command-line use.

## Runtime Defaults

- Base URL: `http://127.0.0.1:12434/v1`
- Model: `chandra-ocr-2-vllm`
- Max output tokens: `12384`
- Max workers: `10`
- Request timeout: `500s`
- PDF DPI: `192`
- Inference max size: `3072x2048`
- Inference min size: `1792x28`

## Async Story

The library supports async document processing directly:

- `ChaocrdanticAgent.run_pages_async(...)`
- `ChaocrdanticAgent.run_file_async(...)`
- `ocr_file_async(...)`
- `ocr_image_async(...)`

The synchronous wrappers call those async implementations via `asyncio.run(...)`.

## Result Types

- `OCRPageResult`
  Raw HTML, cleaned HTML, markdown, dimensions, layout blocks, extracted image metadata, and error state for one page.
- `OCRResult`
  Ordered page results plus merged document renderers and extracted image persistence.

## Notes

- The package still exports compatibility aliases `ChandraOCRAgent` and `ChandraOCRSettings` for migration only.
- The active library in this repository is `chaocrdantic`; older repo history and experiment notes are not part of the primary runtime path.
