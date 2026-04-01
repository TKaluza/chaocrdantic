# chaocrdantic

`chaocrdantic` is a standalone OCR library and CLI refactored from Datalab's [Chandra](https://github.com/datalab-to/chandra) project. It keeps the agent-driven OCR flow, reimplements the orchestration on top of `pydantic_ai`, and returns structured per-page and document-level HTML, Markdown, layout blocks, and extracted figure crops.

The default target is a local vLLM-compatible server at `http://127.0.0.1:12434/v1` using model `chandra-ocr-2-vllm`.

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

Defaults:

- `BASE_URL=http://127.0.0.1:12434/v1`
- `MODEL_NAME=chandra-ocr-2-vllm`
- `MAX_OUTPUT_TOKENS=12384`
- `TEMPERATURE=0.0`
- `TOP_P=0.1`
- `MAX_WORKERS=10`
- `REQUEST_TIMEOUT=500s`
- PDF render DPI `192`

## Output Model

`OCRResult` contains:

- `pages`: ordered `OCRPageResult` entries
- `markdown`: merged document markdown
- `html`: merged cleaned HTML
- `has_errors`: whether any page failed
- `save_extracted_images(...)`: writes figure crops to `<document>_assets/`

## Development

Run tests:

```bash
uv run python -m pytest -q tests/test_agent.py tests/test_output.py
```
