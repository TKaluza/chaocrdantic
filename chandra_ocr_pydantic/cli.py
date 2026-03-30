"""
Minimal CLI for chandra_ocr_pydantic.

Usage examples::

    # OCR a single PDF, print Markdown to stdout
    python -m chandra_ocr_pydantic document.pdf

    # Save output to a directory
    python -m chandra_ocr_pydantic document.pdf --output-dir ./results

    # Process only pages 0 and 2
    python -m chandra_ocr_pydantic document.pdf --pages 0,2

    # Use plain OCR (no layout blocks)
    python -m chandra_ocr_pydantic document.pdf --no-layout

    # Connect to a different server
    python -m chandra_ocr_pydantic document.pdf --base-url http://10.0.0.1:12434/v1

Entry point registered as ``chandra_ocr_pydantic`` console script in
pyproject.toml (add it there to make ``chandra_ocr_pydantic`` available on
PATH after installation).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import click

from chandra_ocr_pydantic.agent import ChandraOCRAgent
from chandra_ocr_pydantic.config import ChandraOCRSettings


def _parse_pages(pages_str: Optional[str]) -> Optional[List[int]]:
    """Parse a comma-separated page string like '0,2,4-6' into a list of ints."""
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


@click.command(name="chandra_ocr_pydantic")
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "--output-dir",
    "-o",
    default=None,
    help=(
        "Directory to write output files. "
        "If omitted, Markdown is printed to stdout."
    ),
)
@click.option(
    "--pages",
    default=None,
    help=(
        "Comma-separated 0-based page numbers or ranges to process "
        "(e.g. '0,2,4-6'). Default: all pages."
    ),
)
@click.option(
    "--no-layout",
    is_flag=True,
    default=False,
    help="Use plain OCR prompt instead of layout-aware prompt.",
)
@click.option(
    "--base-url",
    default=None,
    help=(
        "Base URL for the OpenAI-compatible provider. "
        "Defaults to http://127.0.0.1:12434/v1 "
        "(or CHANDRA_PYDANTIC_BASE_URL env var)."
    ),
)
@click.option(
    "--model",
    default=None,
    help=(
        "Model name to use. "
        "Defaults to chandra-ocr-2 "
        "(or CHANDRA_PYDANTIC_MODEL_NAME env var)."
    ),
)
@click.option(
    "--save-html",
    is_flag=True,
    default=False,
    help="Also save HTML output alongside Markdown.",
)
@click.option(
    "--save-json",
    is_flag=True,
    default=False,
    help="Also save full structured JSON output (OCRResult).",
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
) -> None:
    """
    Run pydantic-ai based OCR on INPUT_PATH (PDF or image).

    By default the result Markdown is printed to stdout.
    Use --output-dir to save files instead.
    """
    # Build settings, overriding via CLI flags if provided
    settings_kwargs = {}
    if base_url:
        settings_kwargs["BASE_URL"] = base_url
    if model:
        settings_kwargs["MODEL_NAME"] = model

    settings = ChandraOCRSettings(**settings_kwargs)
    agent = ChandraOCRAgent(settings=settings, use_layout=not no_layout)

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

    if output_dir is None:
        # Print Markdown to stdout
        click.echo(result.markdown)
        return

    # Save to output directory
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    stem = Path(input_path).stem

    md_file = out_path / f"{stem}.md"
    md_file.write_text(result.markdown, encoding="utf-8")
    click.echo(f"Saved Markdown: {md_file}", err=True)

    if save_html:
        html_file = out_path / f"{stem}.html"
        html_file.write_text(result.html, encoding="utf-8")
        click.echo(f"Saved HTML: {html_file}", err=True)

    if save_json:
        json_file = out_path / f"{stem}.json"
        json_file.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        click.echo(f"Saved JSON: {json_file}", err=True)

    # Summary
    click.echo(
        f"Done. {result.num_pages} page(s) processed. "
        f"Output: {out_path}",
        err=True,
    )


if __name__ == "__main__":
    main()
