# Chandra OCR System Architecture

## Overview

Chandra OCR is a layout-aware vision-language OCR system that converts PDF documents and images into structured HTML, Markdown, and JSON. The system sends page images to a multimodal model (either a local HuggingFace model or a remote vLLM-compatible server), receives HTML annotated with layout bounding boxes, and then parses that HTML into structured output formats.

The codebase is split into:

- **`chandra/`** — the core Python library (importable, provides a clean API)
- **`chandra/scripts/`** — entry points (CLI, Streamlit web app, Flask screenshot server, vLLM launch helper)
- **`chandra_ocr_pydantic/`** — a pydantic-ai-based OCR library using an OpenAI-compatible provider

---

## Libraries Used

| Library | Role |
|---|---|
| `pypdfium2` | Renders PDF pages to PIL images at configurable DPI |
| `Pillow` (PIL) | Image loading, resizing, cropping, format conversion |
| `filetype` | Detects whether an input file is PDF or image |
| `openai` | OpenAI-compatible client for vLLM inference (chat completions with vision) |
| `beautifulsoup4` | Parses the raw HTML output from the model to extract layout blocks |
| `markdownify` | Converts HTML to Markdown with custom math/chemistry extensions |
| `pydantic` | Data validation; used for `Settings` (via `pydantic-settings`) |
| `pydantic-settings` | Loads configuration from environment variables and `.env` files |
| `python-dotenv` | Locates and loads the `local.env` file |
| `six` | Unicode text compatibility in the `Markdownify` processor |
| `click` | CLI framework for the `chandra` command |
| `streamlit` | Web demo application (optional `app` extra) |
| `flask` | Screenshot-ready visualization server |
| `torch` / `transformers` / `accelerate` | HuggingFace local model inference (optional `hf` extra) |
| `pydantic-ai` | Agent framework for the `chandra_ocr_pydantic` module |

---

## Prompts

### Prompt Definitions (`chandra/prompts.py`)

Two core prompts are defined. Both share a common `PROMPT_ENDING` block that specifies allowed HTML tags, allowed attributes, and formatting guidelines.

#### `PROMPT_ENDING` (shared suffix for both prompts)

Instructs the model to:
- Use only a specific set of HTML tags (`math`, `table`, `div`, `p`, `img`, etc.)
- Render math in KaTeX-compatible LaTeX inside `<math>` tags
- Handle tables with `colspan`/`rowspan`
- Describe images in `alt` attributes; convert charts to data and diagrams to Mermaid
- Mark form checkboxes and radio buttons
- Join lines into `<p>` paragraphs
- Use `<chem>` tags for chemical formulas

#### `OCR_LAYOUT_PROMPT`

Used for layout-aware OCR. Tells the model to:
- Convert the image to HTML divided into **layout blocks**, each as a `<div>`
- Each div must have `data-bbox="x0 y0 x1 y1"` (normalized 0–1000) and `data-label` attributes
- Recognized labels: `Caption`, `Footnote`, `Equation-Block`, `List-Group`, `Page-Header`, `Page-Footer`, `Image`, `Section-Header`, `Table`, `Text`, `Complex-Block`, `Code-Block`, `Form`, `Table-Of-Contents`, `Figure`, `Chemical-Block`, `Diagram`, `Bibliography`, `Blank-Page`

**Key**: This is the default prompt used in production (CLI, Streamlit app, Flask app).

#### `OCR_PROMPT`

A simpler fallback that asks the model to OCR the image to HTML without layout block structure.

#### Prompt Selection

```python
PROMPT_MAPPING = {
    "ocr_layout": OCR_LAYOUT_PROMPT,
    "ocr": OCR_PROMPT,
}
```

A `BatchInputItem` can either carry a pre-built `prompt` string or a `prompt_type` key that resolves through this mapping at inference time.

---

## Data Flow

```
Input File (PDF or image)
        │
        ▼
chandra/input.py: load_file()
  ├── PDF → pypdfium2 renders each page to PIL Image (at IMAGE_DPI, min MIN_PDF_IMAGE_DIM)
  └── Image → Pillow loads and upscales if below MIN_IMAGE_DIM
        │
        ▼  List[PIL.Image]
        │
        ▼
chandra/model/schema.py: BatchInputItem(image=img, prompt_type="ocr_layout")
        │
        ▼
chandra/model/__init__.py: InferenceManager.generate(batch)
        │
        ├── method="vllm" → chandra/model/vllm.py: generate_vllm()
        │     ├── scale_to_fit() — resize each image to fit within (3072×2048) grid
        │     ├── image_to_base64() — encode as PNG base64
        │     ├── openai.OpenAI.chat.completions.create()  (multimodal, vision)
        │     │     message content = [image_url block, text block (prompt)]
        │     ├── Retry logic: detect_repeat_token() triggers re-generation
        │     └── ThreadPoolExecutor for parallel page processing
        │
        └── method="hf" → chandra/model/hf.py: generate_hf()
              ├── scale_to_fit() — same resizing
              ├── processor.apply_chat_template() — builds tokenized inputs
              └── model.generate() — local HuggingFace inference
        │
        ▼  List[GenerationResult(raw=html_string, token_count, error)]
        │
        ▼
chandra/output.py  (per page)
  ├── parse_markdown(raw) → Markdown string
  │     └── parse_html() → strips blank pages, handles image/figure srcs
  │           └── Markdownify.convert() — custom math/chem/table converters
  ├── parse_html(raw) → cleaned HTML string
  ├── parse_chunks(raw, image) → List[dict] with {bbox, label, content}
  │     └── parse_layout() → List[LayoutBlock] with pixel-space bboxes
  └── extract_images(raw, chunks, image) → dict[filename → PIL.Image]
        │
        ▼
BatchOutputItem(markdown, html, chunks, raw, page_box, token_count, images, error)
        │
        ▼
chandra/scripts/cli.py: save_merged_output()
  ├── {stem}.md   — concatenated Markdown for all pages
  ├── {stem}.html — concatenated HTML for all pages
  ├── {stem}_metadata.json — per-page stats (tokens, chunks, images)
  └── extracted image files (.webp)
```

---

## Entry Points

### CLI (`chandra`)

**Script**: `chandra/scripts/cli.py:main`
**Registered as**: `chandra` console script

```bash
chandra INPUT_PATH OUTPUT_PATH [OPTIONS]
```

Options: `--method [hf|vllm]`, `--page-range`, `--max-output-tokens`, `--max-workers`, `--max-retries`, `--include-images/--no-images`, `--include-headers-footers/--no-headers-footers`, `--save-html/--no-html`, `--batch-size`, `--paginate_output`

Processes all supported files in a directory or a single file. Defaults to `method=vllm` and `batch_size=28`.

### Streamlit App (`chandra_app`)

**Script**: `chandra/scripts/run_app.py:main` → launches `chandra/scripts/app.py`
**Registered as**: `chandra_app` console script

Interactive web demo. Lets the user upload a PDF/image, select a page, choose between `hf` and `vllm` backends, and view OCR results with layout overlays.

### Flask Screenshot Server (`chandra_screenshot`)

**Script**: `chandra/scripts/screenshot_app.py:main` (Flask app on port 8503)
**Registered as**: `chandra_screenshot` console script

REST endpoint `POST /process` accepts `{file_path, page_number}`, runs OCR, and returns `{image_base64, blocks, html, markdown}` suitable for screenshot-quality visualizations.

### vLLM Server Launcher (`chandra_vllm`)

**Script**: `chandra/scripts/vllm.py:main`
**Registered as**: `chandra_vllm` console script

Launches the official `vllm/vllm-openai` Docker container with GPU-specific settings (`--max-num-seqs`, `--max_num_batched_tokens`) scaled by available VRAM.

### pydantic-ai OCR Library (`chandra_ocr_pydantic`)

**Package**: `chandra_ocr_pydantic/`

A self-contained library that uses `pydantic_ai` to orchestrate OCR. See the section below.

---

## Configuration (`chandra/settings.py`)

Settings are loaded from environment variables or `local.env` via `pydantic-settings`:

| Setting | Default | Description |
|---|---|---|
| `IMAGE_DPI` | 192 | DPI for PDF rasterization |
| `MIN_PDF_IMAGE_DIM` | 1024 | Minimum pixel dimension for PDF pages |
| `MIN_IMAGE_DIM` | 1536 | Minimum pixel dimension for standalone images |
| `MODEL_CHECKPOINT` | `datalab-to/chandra-ocr-2` | HuggingFace model ID |
| `TORCH_DEVICE` | None | Device for HF inference (`cuda`, `cpu`, etc.) |
| `MAX_OUTPUT_TOKENS` | 12384 | Max tokens per page |
| `BBOX_SCALE` | 1000 | Coordinate space for bounding boxes (0–1000) |
| `VLLM_API_KEY` | `EMPTY` | API key for vLLM server |
| `VLLM_API_BASE` | `http://localhost:8000/v1` | vLLM server base URL |
| `VLLM_MODEL_NAME` | `chandra` | Model name served by vLLM |
| `MAX_VLLM_RETRIES` | 6 | Retry attempts on repeat-token detection or error |

---

## `chandra_ocr_pydantic/` — pydantic-ai OCR Library

A new module providing an alternative inference path using the `pydantic_ai` framework. It targets an OpenAI-compatible provider at `http://127.0.0.1:12434/` with model `chandra-ocr-2`.

### Design

- Uses `pydantic_ai.Agent` with an `OpenAIModel` pointed at the custom provider
- Accepts a PDF file path (or image path) as input
- Converts each page to a base64-encoded PNG (reusing `chandra.input` utilities)
- Sends each page image plus the OCR prompt to the agent via `agent.run_sync()`
- Returns a `pydantic` model (`OCRResult`) containing per-page and document-level results
- Designed as an importable library with a minimal CLI

See `chandra_ocr_pydantic/README.md` for usage instructions.
