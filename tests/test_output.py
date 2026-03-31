from pathlib import Path

from PIL import Image

from chaocrdantic.document_renderer import PAGE_BREAK
from chaocrdantic.models import ExtractedImage, LayoutBlock, OCRPageResult, OCRResult, PageDimensions
from chaocrdantic.output import extract_images, parse_chunks, parse_html, parse_markdown


SAMPLE_HTML = """
<div data-bbox="0 0 1000 40" data-label="Page-Header">HEADER</div>
<div data-bbox="0 50 1000 90" data-label="Section-Header"><h1>Title</h1></div>
<div data-bbox="0 100 1000 200" data-label="Text"><p>Hello <math>x^2</math></p></div>
<div data-bbox="100 210 900 500" data-label="Figure"><img alt="alt text" title="Rendered Figure"/></div>
<div data-bbox="200 510 600 540" data-label="Caption"><p>Fig. 1</p></div>
<div data-bbox="0 550 1000 650" data-label="Table"><table><tr><th>A</th></tr><tr><td>B</td></tr></table></div>
<div data-bbox="0 900 1000 960" data-label="Page-Footer">1/1</div>
""".strip()


def _page_result(page_number: int) -> OCRPageResult:
    image = Image.new("RGB", (1000, 1000), "white")
    chunks = parse_chunks(SAMPLE_HTML, image)
    cropped, meta = extract_images(SAMPLE_HTML, chunks, image, page_number=page_number)
    page = OCRPageResult(
        page_number=page_number,
        raw_html=SAMPLE_HTML,
        markdown=parse_markdown(SAMPLE_HTML, page_number=page_number),
        html=parse_html(SAMPLE_HTML, page_number=page_number),
        dimensions=PageDimensions(dpi=200, width=1000, height=1000),
        token_count=123,
        layout_blocks=[LayoutBlock(**chunk) for chunk in chunks],
        extracted_images=[ExtractedImage(**item) for item in meta],
        error=False,
    )
    page.set_extracted_images(cropped)
    return page


def test_parse_html_strips_headers_and_footers():
    cleaned = parse_html(SAMPLE_HTML, include_headers_footers=False, page_number=0)
    assert "HEADER" not in cleaned
    assert "1/1" not in cleaned
    assert "<h1>Title</h1>" in cleaned


def test_document_markdown_uses_front_matter_page_breaks_and_local_assets(tmp_path):
    result = OCRResult(
        file_path=str(tmp_path / "sample.pdf"),
        ocr_engine="chaocrdantic",
        ocr_model="test-model",
        num_pages=2,
        pages=[_page_result(0), _page_result(1)],
    )

    markdown = result.markdown

    assert markdown.startswith("---\n")
    assert "ocr_model: \"test-model\"" in markdown
    assert PAGE_BREAK in markdown
    assert "<figure id=\"fig-p1-0\">" in markdown
    assert "sample_assets/page-01-img-4.webp" in markdown
    assert "<table><tr><th>A</th></tr><tr><td>B</td></tr></table>" in markdown

    assets_dir = result.save_extracted_images(tmp_path)
    assert (assets_dir / "page-01-img-4.webp").exists()


def test_golden_file_shape():
    ideal = Path("ideal_DE102022115220A1.md").read_text(encoding="utf-8")
    assert ideal.startswith("---\n")
    assert ideal.count("  - page: ") == 10
    assert ideal.count(PAGE_BREAK) == 9
    assert ideal.count("<figure") >= 1
    assert ideal.count("<table") >= 1


def test_parse_markdown_strips_model_emitted_page_break_markup():
    html = """
    <div data-bbox="0 0 1000 200" data-label="Text">
      <div style="break-before: page;"></div>
      <p>Visible text</p>
    </div>
    """.strip()

    markdown = parse_markdown(html, page_number=0)

    assert PAGE_BREAK not in markdown
    assert "Visible text" in markdown


def test_document_markdown_preserves_error_pages_with_page_breaks(tmp_path):
    ok_page = _page_result(0)
    error_page = OCRPageResult(
        page_number=1,
        raw_html="",
        markdown="",
        html="",
        dimensions=PageDimensions(dpi=200, width=1000, height=1000),
        token_count=0,
        layout_blocks=[],
        extracted_images=[],
        error=True,
        error_message="status_code: 500, model_name: chandra-ocr-2, body: {'message': 'Context size has been exceeded.'}",
    )
    result = OCRResult(
        file_path=str(tmp_path / "sample.pdf"),
        ocr_engine="chaocrdantic",
        ocr_model="test-model",
        num_pages=2,
        pages=[ok_page, error_page],
    )

    markdown = result.markdown

    assert PAGE_BREAK in markdown
    assert "OCR error on page 2" in markdown
    assert "Context size has been exceeded." in markdown
