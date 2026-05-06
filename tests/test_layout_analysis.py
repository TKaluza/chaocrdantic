from PIL import Image, ImageDraw

from chaocrdantic.image_utils import RenderedPage
from chaocrdantic.layout_analysis import analyze_layouts


def _page(page_number: int, *, body_offset: int = 0, variant: str = "a") -> RenderedPage:
    image = Image.new("RGB", (600, 800), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 600, 54), fill="black")
    draw.rectangle((0, 746, 600, 800), fill="black")
    if variant == "a":
        for idx in range(4):
            y = 140 + body_offset + idx * 90
            draw.rectangle((90, y, 510, y + 22), fill="black")
    else:
        draw.rectangle((70, 130, 160, 650), fill="black")
        draw.rectangle((240, 130, 520, 650), fill="black")
    return RenderedPage(page_number=page_number, image=image, dpi=200)


def test_detects_repeated_header_footer_and_single_cluster():
    pages = [_page(0, body_offset=0), _page(1, body_offset=12), _page(2, body_offset=24)]

    result = analyze_layouts(
        pages,
        max_analysis_width=300,
        stable_threshold=0.08,
        min_section_height=12,
        section_padding=8,
    )

    assert len(result.layouts) == 1
    layout = result.layouts[0]
    assert layout.header_box is not None
    assert layout.footer_box is not None
    assert layout.header_box[1] == 0
    assert layout.header_box[3] < layout.main_box[1] + 5
    assert layout.footer_box[1] > layout.main_box[1]
    assert all(page.layout_id == 1 for page in result.pages)
    assert all(page.main_box[1] > 0 and page.main_box[3] < 800 for page in result.pages)


def test_clusters_different_layouts():
    pages = [_page(0, variant="a"), _page(1, variant="b")]

    result = analyze_layouts(
        pages,
        max_analysis_width=300,
        layout_threshold=0.98,
        stable_threshold=0.08,
        min_section_height=12,
    )

    assert len(result.layouts) == 2
    assert {page.layout_id for page in result.pages} == {1, 2}


def test_single_page_analysis_returns_non_outlier_layout():
    result = analyze_layouts(
        [_page(0)],
        max_analysis_width=300,
        min_section_height=12,
    )

    assert len(result.layouts) == 1
    assert result.layouts[0].is_outlier is False
    assert result.pages[0].header_box is not None
    assert result.pages[0].footer_box is not None
