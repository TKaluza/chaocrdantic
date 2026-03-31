"""CLI for chaocrdantic."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import click

from chaocrdantic.agent import ChaocrdanticAgent
from chaocrdantic.config import ChaocrdanticSettings


def _parse_pages(pages_str: Optional[str]) -> Optional[List[int]]:
    if not pages_str:
        return None
    result: List[int] = []
    for part in pages_str.split(","):
        part = part.strip()
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            result.extend(range(int(start_s), int(end_s) + 1))
        else:
            result.append(int(part))
    return sorted(set(result))


@click.command(name="chaocrdantic")
@click.argument("input_path", type=click.Path(exists=True))
@click.option("--output-dir", "-o", default=None, help="Directory to write output files.")
@click.option("--pages", default=None, help="Comma-separated 0-based pages or ranges.")
@click.option("--no-layout", is_flag=True, default=False, help="Use plain OCR instead of layout-aware OCR.")
@click.option("--base-url", default=None, help="Override the OpenAI-compatible provider base URL.")
@click.option("--model", default=None, help="Override the model name.")
@click.option("--save-html", is_flag=True, default=False, help="Also save HTML output.")
@click.option("--save-json", is_flag=True, default=False, help="Also save JSON output.")
@click.option("--max-output-tokens", type=int, default=None, help="Maximum output tokens per page.")
@click.option("--max-image-width", type=int, default=None, help="Maximum inference image width.")
@click.option("--max-image-height", type=int, default=None, help="Maximum inference image height.")
@click.option("--max-workers", type=int, default=None, help="Maximum concurrent page requests.")
@click.option("--max-retries", type=int, default=None, help="Maximum retries per page.")
@click.option(
    "--include-images/--no-images",
    default=True,
    help="Include extracted images and figure references in Markdown output.",
)
@click.option(
    "--include-headers-footers/--no-headers-footers",
    default=False,
    help="Include page headers and footers in output.",
)
def main(
    input_path: str,
    output_dir: Optional[str],
    pages: Optional[str],
    no_layout: bool,
    base_url: Optional[str],
    model: Optional[str],
    save_html: bool,
    save_json: bool,
    max_output_tokens: Optional[int],
    max_image_width: Optional[int],
    max_image_height: Optional[int],
    max_workers: Optional[int],
    max_retries: Optional[int],
    include_images: bool,
    include_headers_footers: bool,
) -> None:
    settings_kwargs = {
        "INCLUDE_IMAGES": include_images,
        "INCLUDE_HEADERS_FOOTERS": include_headers_footers,
    }
    if base_url:
        settings_kwargs["BASE_URL"] = base_url
    if model:
        settings_kwargs["MODEL_NAME"] = model
    if max_output_tokens is not None:
        settings_kwargs["MAX_OUTPUT_TOKENS"] = max_output_tokens
    if max_image_width is not None:
        settings_kwargs["INFERENCE_MAX_IMAGE_WIDTH"] = max_image_width
    if max_image_height is not None:
        settings_kwargs["INFERENCE_MAX_IMAGE_HEIGHT"] = max_image_height
    if max_workers is not None:
        settings_kwargs["MAX_WORKERS"] = max_workers
    if max_retries is not None:
        settings_kwargs["MAX_RETRIES"] = max_retries

    settings = ChaocrdanticSettings(**settings_kwargs)
    agent = ChaocrdanticAgent(settings=settings, use_layout=not no_layout)
    page_range = _parse_pages(pages)

    click.echo(
        f"Processing: {input_path}  "
        f"(model={settings.MODEL_NAME}, base_url={settings.BASE_URL})",
        err=True,
    )
    if page_range is not None:
        click.echo(f"Pages: {page_range}", err=True)

    result = agent.run_file(input_path, page_range=page_range)

    if result.has_errors:
        failed = [str(p.page_number) for p in result.pages if p.error]
        click.echo(
            f"Warning: {len(failed)} page(s) had errors: {', '.join(failed)}",
            err=True,
        )
        for page in result.pages:
            if page.error and page.error_message:
                click.echo(f"  page {page.page_number}: {page.error_message}", err=True)

    markdown = result.render_markdown(include_images=include_images)
    html = result.render_html()

    if output_dir is None:
        click.echo(markdown)
        return

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    stem = Path(input_path).stem

    md_file = out_path / f"{stem}.md"
    md_file.write_text(markdown, encoding="utf-8")
    click.echo(f"Saved Markdown: {md_file}", err=True)

    if include_images:
        assets_dir = result.save_extracted_images(out_path)
        click.echo(f"Saved extracted images: {assets_dir}", err=True)

    if save_html:
        html_file = out_path / f"{stem}.html"
        html_file.write_text(html, encoding="utf-8")
        click.echo(f"Saved HTML: {html_file}", err=True)

    if save_json:
        json_file = out_path / f"{stem}.json"
        json_file.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        click.echo(f"Saved JSON: {json_file}", err=True)

    click.echo(
        f"Done. {result.num_pages} page(s) processed. "
        f"Output: {out_path}",
        err=True,
    )


if __name__ == "__main__":
    main()
