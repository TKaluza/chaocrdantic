# chaocrdantic

`chaocrdantic` is a standalone OCR library and CLI refactored from Datalab's [Chandra](https://github.com/datalab-to/chandra) project. It keeps the agent-driven OCR flow, reimplements the orchestration on top of `pydantic_ai`, and returns structured per-page and document-level HTML, Markdown, layout blocks, and extracted figure crops.

The default target is a local OpenAI-compatible server at `http://127.0.0.1:12434/v1` using model `chandra-ocr-2-parallel`.

## Attribution

- Upstream OCR project: [datalab-to/chandra](https://github.com/datalab-to/chandra)
- Upstream license: [Apache-2.0](https://github.com/datalab-to/chandra/blob/master/LICENSE)

`chaocrdantic` is intentionally not kept in sync with upstream Chandra. It is a separate codebase that preserves attribution to the original project and model while moving forward independently.

## Install

```bash
uv sync --dev
```

## Library Usage

Synchronous:

```python
from chaocrdantic import ocr_file

result = ocr_file("document.pdf")
print(result.markdown)
```

Asynchronous:

```python
import asyncio

from chaocrdantic import ocr_file_async


async def main() -> None:
    result = await ocr_file_async("document.pdf")
    print(result.markdown)


asyncio.run(main())
```

Reusable agent:

```python
from chaocrdantic import ChaocrdanticAgent

agent = ChaocrdanticAgent()
result = agent.run_file("document.pdf")
```

Reusable async agent:

```python
import asyncio

from chaocrdantic import ChaocrdanticAgent


async def main() -> None:
    agent = ChaocrdanticAgent()
    result = await agent.run_file_async("document.pdf")
    print(result.pages[0].markdown)


asyncio.run(main())
```

## CLI

```bash
uv run chaocrdantic <document.pdf> --output-dir output/run --save-html --save-json
```

Useful flags:

- `--pages 0-2,5`
- `--max-output-tokens 12384`
- `--max-image-width 2048 --max-image-height 1536`
- `--max-workers 4`
- `--no-layout`
- `--clean-markdown`

## Clean Markdown

`--clean-markdown` is an opt-in mode for multi-page PDFs with repeated page headers or footers. Before OCR, chaocrdantic analyzes the rendered page images, clusters similar page layouts, detects stable header, main, and footer bands per layout, then:

- OCRs each layout's header and footer once.
- OCRs only the main content region for every page.
- Stores header/footer text and layout IDs in the document YAML front matter.
- Leaves the visible page body Markdown free of repeated header/footer text.

For documents below `CLEAN_MARKDOWN_MIN_PAGES` pages, currently `2`, chaocrdantic logs a warning and uses the standard pipeline. If layout analysis fails, it also warns and falls back to the standard pipeline rather than aborting the OCR run.

Example:

```bash
uv run chaocrdantic report.pdf --clean-markdown -o output/run --save-json
```

The Markdown front matter includes per-page layout anchors and document-level layout summaries:

```yaml
---
doc_id: "report"
ocr_engine: "chaocrdantic"
ocr_model: "chandra-ocr-2-parallel"
pages:
  - page: 1
    dimensions: {dpi: 192, width: 1654, height: 2339}
    layout: 1
    header: "Quarterly Report Q1 2026"
    footer: "Page 1 of 12"
layouts:
  - id: 1
    pages: [1, 2, 3]
    representative_page: 1
    header: "Quarterly Report Q1 2026"
    footer: "Page N of 12"
---
```

## Configuration

Settings are loaded from `.env` and `CHAOCRDANTIC_*` environment variables.

The repository includes a prefilled template at `.env.example`. A shared singleton is available for imports:

```python
from chaocrdantic import settings

print(settings.BASE_URL)
```

Common overrides:

- `CHAOCRDANTIC_BASE_URL`
- `CHAOCRDANTIC_MODEL_NAME`
- `CHAOCRDANTIC_MAX_OUTPUT_TOKENS`
- `CHAOCRDANTIC_MAX_WORKERS`
- `CHAOCRDANTIC_INFERENCE_MAX_IMAGE_WIDTH`
- `CHAOCRDANTIC_INFERENCE_MAX_IMAGE_HEIGHT`
- `CHAOCRDANTIC_CLEAN_MARKDOWN`
- `CHAOCRDANTIC_CLEAN_MARKDOWN_LAYOUT_THRESHOLD`

Defaults:

- `BASE_URL=http://127.0.0.1:12434/v1`
- `MODEL_NAME=chandra-ocr-2-parallel`
- `MAX_OUTPUT_TOKENS=12384`
- `TEMPERATURE=0.0`
- `TOP_P=0.1`
- `MAX_WORKERS=10`
- `REQUEST_TIMEOUT=500s`
- PDF render DPI `192`
- `CLEAN_MARKDOWN=false`
- `CLEAN_MARKDOWN_MIN_PAGES=2`
- `CLEAN_MARKDOWN_LAYOUT_THRESHOLD=0.72`

## Output Model

`OCRResult` contains:

- `pages`: ordered `OCRPageResult` entries
- `layouts`: clean-markdown layout summaries, when `--clean-markdown` is active
- `markdown`: merged document markdown
- `html`: merged cleaned HTML
- `has_errors`: whether any page failed
- `save_extracted_images(...)`: writes figure crops to `<document>_assets/`

Each `OCRPageResult` may include `layout_id`, `header_text`, and `footer_text` when clean-markdown mode is active. These fields are omitted from JSON output when unset.

## Development

Run tests:

```bash
uv run python -m pytest -q
```
