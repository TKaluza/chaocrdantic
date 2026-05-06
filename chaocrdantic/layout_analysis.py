"""Layout clustering and repeated header/footer detection for rendered pages."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from chaocrdantic.image_utils import RenderedPage


Box = tuple[int, int, int, int]


@dataclass
class PageLayout:
    page_number: int
    layout_id: int
    header_box: Box | None
    main_box: Box
    footer_box: Box | None
    is_outlier: bool


@dataclass
class LayoutCluster:
    layout_id: int
    page_numbers: list[int]
    header_box: Box | None
    main_box: Box
    footer_box: Box | None
    is_outlier: bool
    representative_page: int


@dataclass
class LayoutAnalysisResult:
    pages: list[PageLayout]
    layouts: list[LayoutCluster]


@dataclass
class _AnalysisPage:
    rendered: RenderedPage
    gray: np.ndarray
    target_gray: np.ndarray
    black: np.ndarray


@dataclass
class _SectionSet:
    header_box: Box | None
    main_box: Box
    footer_box: Box | None


def _odd(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 else value + 1


def normalize_gray(gray: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(gray, (0.5, 99.5))
    if hi <= lo:
        return gray.copy()
    norm = (gray.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return np.clip(norm, 0, 255).astype(np.uint8)


def preprocess_blackness(
    gray: np.ndarray,
    *,
    black_gamma: float = 1.7,
) -> np.ndarray:
    denoised = cv2.medianBlur(gray, 3)
    denoised = cv2.GaussianBlur(denoised, (5, 5), 0)
    ink = (255.0 - denoised.astype(np.float32)) / 255.0
    return np.power(np.clip(ink, 0.0, 1.0), black_gamma).astype(np.float32)


def edge_weight(height: int, width: int) -> np.ndarray:
    yy = np.abs(np.linspace(-1.0, 1.0, height, dtype=np.float32))[:, None]
    xx = np.abs(np.linspace(-1.0, 1.0, width, dtype=np.float32))[None, :]
    edge = np.maximum(np.power(yy, 1.4), np.power(xx, 1.4))
    return 0.20 + 0.80 * edge


def align_pages(
    black_pages: list[np.ndarray],
    max_shift_ratio: float = 0.035,
) -> list[np.ndarray]:
    if not black_pages:
        return []

    height, width = black_pages[0].shape
    weights = edge_weight(height, width)
    weighted = [(page * weights).astype(np.float32) for page in black_pages]
    ref = np.median(np.stack(weighted[: min(3, len(weighted))], axis=0), axis=0).astype(np.float32)
    window = cv2.createHanningWindow((width, height), cv2.CV_32F)
    max_shift = min(height, width) * max_shift_ratio

    aligned: list[np.ndarray] = []
    for index, page in enumerate(black_pages):
        try:
            shift, response = cv2.phaseCorrelate(ref, weighted[index], window)
            dx, dy = float(shift[0]), float(shift[1])
        except cv2.error:
            dx, dy, response = 0.0, 0.0, 0.0
        if response < 0.03 or abs(dx) > max_shift or abs(dy) > max_shift:
            dx = 0.0
            dy = 0.0

        matrix = np.float32([[1, 0, -dx], [0, 1, -dy]])
        aligned_page = cv2.warpAffine(
            page,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        aligned.append(aligned_page.astype(np.float32))
    return aligned


def _normalized_feature(values: np.ndarray) -> np.ndarray:
    feature = values.astype(np.float32).reshape(-1)
    feature -= float(feature.mean())
    norm = float(np.linalg.norm(feature))
    if norm <= 1e-6:
        return feature
    return feature / norm


def layout_feature(image: np.ndarray) -> np.ndarray:
    height, _ = image.shape
    top = image[: max(1, int(height * 0.22)), :]
    bottom = image[int(height * 0.78) :, :]

    parts: list[np.ndarray] = []
    for band in (top, bottom):
        coarse = cv2.resize(band, (32, 10), interpolation=cv2.INTER_AREA)
        row_profile = cv2.resize(band.sum(axis=1)[:, None], (1, 48), interpolation=cv2.INTER_AREA)
        col_profile = cv2.resize(band.sum(axis=0)[None, :], (64, 1), interpolation=cv2.INTER_AREA)
        parts.extend(
            [
                coarse.reshape(-1) * 0.5,
                row_profile.reshape(-1) * 1.5,
                col_profile.reshape(-1) * 0.7,
            ]
        )

    return _normalized_feature(np.concatenate(parts))


def cluster_layouts(
    aligned_pages: list[np.ndarray],
    *,
    layout_threshold: float = 0.72,
) -> list[list[int]]:
    clusters: list[list[int]] = []
    feature_groups: list[list[np.ndarray]] = []
    centroids: list[np.ndarray] = []
    for idx, image in enumerate(aligned_pages):
        feature = layout_feature(image)
        best_idx = -1
        best_score = -1.0
        for cluster_idx, centroid in enumerate(centroids):
            score = float(np.dot(feature, centroid))
            if score > best_score:
                best_score = score
                best_idx = cluster_idx
        if best_idx >= 0 and best_score >= layout_threshold:
            clusters[best_idx].append(idx)
            feature_groups[best_idx].append(feature)
            centroids[best_idx] = _normalized_feature(np.mean(np.stack(feature_groups[best_idx], axis=0), axis=0))
        else:
            clusters.append([idx])
            feature_groups.append([feature])
            centroids.append(feature.copy())
    clusters.sort(key=min)
    return clusters


def smooth_profile(profile: np.ndarray, window: int = 9) -> np.ndarray:
    window = _odd(window)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(profile.astype(np.float32), kernel, mode="same")


def close_mask(mask: np.ndarray, size: int = 9, gap: int | None = None) -> np.ndarray:
    size = _odd(gap if gap is not None else size)
    src = (mask.astype(np.uint8) * 255)[:, None]
    kernel = np.ones((size, 1), dtype=np.uint8)
    closed = cv2.morphologyEx(src, cv2.MORPH_CLOSE, kernel)
    return closed[:, 0] > 0


def mask_groups(mask: np.ndarray, min_height: int = 1) -> list[tuple[int, int]]:
    groups: list[tuple[int, int]] = []
    start: int | None = None
    for idx, active in enumerate(mask):
        if active and start is None:
            start = idx
        elif not active and start is not None:
            if idx - start >= min_height:
                groups.append((start, idx))
            start = None
    if start is not None and len(mask) - start >= min_height:
        groups.append((start, len(mask)))
    return groups


def weighted_row_profile(
    stability: np.ndarray,
) -> np.ndarray:
    _, width = stability.shape
    x = np.abs(np.linspace(-1.0, 1.0, width, dtype=np.float32))
    x_weight = 0.35 + 0.65 * np.power(x, 1.35)
    weighted = stability * x_weight[None, :]
    return weighted.sum(axis=1) / float(x_weight.sum())


def detect_sections(
    aligned_pages: list[np.ndarray],
    *,
    black_threshold: float = 0.12,
    recurrence: float = 0.55,
    stable_threshold: float = 0.10,
    max_band_ratio: float = 0.35,
    min_section_height: int = 24,
    section_padding: int = 18,
) -> _SectionSet:
    if not aligned_pages:
        raise ValueError("detect_sections requires at least one page")

    stack = np.stack(aligned_pages, axis=0)
    page_count, height, width = stack.shape
    median = np.median(stack, axis=0).astype(np.float32)

    if page_count == 1:
        recurrent = stack[0] > black_threshold
    else:
        required = max(2, int(np.ceil(page_count * recurrence)))
        recurrent = (stack > black_threshold).sum(axis=0) >= required

    stability = np.where(recurrent, median, 0.0).astype(np.float32)
    profile = weighted_row_profile(stability)
    profile = smooth_profile(profile, max(9, height // 120))
    max_profile = float(profile.max()) if profile.size else 0.0
    active = profile >= (max_profile * stable_threshold) if max_profile > 0 else np.zeros(height, dtype=bool)
    active = close_mask(active, gap=max(min_section_height, height // 24))

    top_limit = int(height * max_band_ratio)
    bottom_start = height - top_limit
    padding = max(0, section_padding)

    header_end = 0
    top_groups = mask_groups(active[:top_limit], min_section_height)
    top_anchor_limit = max(min_section_height, int(height * 0.16))
    anchored_top = [group for group in top_groups if group[0] <= top_anchor_limit]
    if anchored_top:
        header_end = min(top_limit, anchored_top[0][1] + padding)

    footer_start = height
    bottom_groups = mask_groups(active[bottom_start:], min_section_height)
    bottom_anchor = int(height * 0.84)
    anchored_bottom = [
        (start + bottom_start, end + bottom_start)
        for start, end in bottom_groups
        if end + bottom_start >= bottom_anchor
    ]
    if anchored_bottom:
        footer_start = max(0, anchored_bottom[-1][0] - padding)

    min_main_height = max(min_section_height, height // 10)
    if footer_start - header_end < min_main_height:
        if header_end > height - footer_start:
            header_end = max(0, footer_start - min_main_height)
        else:
            footer_start = min(height, header_end + min_main_height)

    header = (0, 0, width, int(header_end)) if header_end > 0 else None
    footer = (0, int(footer_start), width, height) if footer_start < height else None

    return _SectionSet(
        header_box=header,
        main_box=(0, int(header_end), width, int(footer_start)),
        footer_box=footer,
    )


def detect_layout_section_sets(
    aligned_pages_by_cluster: list[list[np.ndarray]],
    *,
    black_threshold: float = 0.12,
    recurrence: float = 0.55,
    stable_threshold: float = 0.10,
    max_band_ratio: float = 0.35,
    min_section_height: int = 24,
    section_padding: int = 18,
) -> list[_SectionSet]:
    section_sets: list[_SectionSet] = []
    multi_page_clusters = [cluster_pages for cluster_pages in aligned_pages_by_cluster if len(cluster_pages) >= 2]
    detection_clusters = multi_page_clusters or aligned_pages_by_cluster
    detected_by_identity: dict[int, _SectionSet] = {}

    for cluster_pages in detection_clusters:
        sections = detect_sections(
            cluster_pages,
            black_threshold=black_threshold,
            recurrence=recurrence,
            stable_threshold=stable_threshold,
            max_band_ratio=max_band_ratio,
            min_section_height=min_section_height,
            section_padding=section_padding,
        )
        detected_by_identity[id(cluster_pages)] = sections
        section_sets.append(sections)

    if len(detection_clusters) == len(aligned_pages_by_cluster):
        return section_sets

    fallback_cluster = max(multi_page_clusters, key=len)
    fallback = detected_by_identity[id(fallback_cluster)]
    return [
        detected_by_identity.get(id(cluster_pages), fallback)
        for cluster_pages in aligned_pages_by_cluster
    ]


def _analysis_gray(page: RenderedPage, *, max_analysis_width: int) -> np.ndarray:
    gray = normalize_gray(np.asarray(page.image.convert("L")))
    if gray.shape[1] <= max_analysis_width:
        return gray
    scale = max_analysis_width / float(gray.shape[1])
    target = (max_analysis_width, max(1, int(round(gray.shape[0] * scale))))
    return cv2.resize(gray, target, interpolation=cv2.INTER_AREA)


def _resize_to_shape(gray: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if gray.shape[:2] == shape:
        return gray
    return cv2.resize(gray, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)


def _scale_box(box: Box | None, *, analysis_shape: tuple[int, int], image_size: tuple[int, int]) -> Box | None:
    if box is None:
        return None
    analysis_h, analysis_w = analysis_shape
    image_w, image_h = image_size
    sx = image_w / float(analysis_w)
    sy = image_h / float(analysis_h)
    x0, y0, x1, y1 = box
    return (
        max(0, min(image_w, int(round(x0 * sx)))),
        max(0, min(image_h, int(round(y0 * sy)))),
        max(0, min(image_w, int(round(x1 * sx)))),
        max(0, min(image_h, int(round(y1 * sy)))),
    )


def analyze_layouts(
    pages: list[RenderedPage],
    *,
    max_analysis_width: int = 900,
    black_gamma: float = 1.7,
    black_threshold: float = 0.12,
    recurrence: float = 0.55,
    stable_threshold: float = 0.10,
    max_band_ratio: float = 0.35,
    min_section_height: int = 24,
    section_padding: int = 18,
    layout_threshold: float = 0.72,
    min_pages_for_outliers: int = 4,
) -> LayoutAnalysisResult:
    if not pages:
        raise ValueError("analyze_layouts requires at least one page")

    rendered_grays = [_analysis_gray(page, max_analysis_width=max_analysis_width) for page in pages]
    target_shape = rendered_grays[0].shape[:2]
    analysis_pages = [
        _AnalysisPage(
            rendered=page,
            gray=gray,
            target_gray=_resize_to_shape(gray, target_shape),
            black=preprocess_blackness(_resize_to_shape(gray, target_shape), black_gamma=black_gamma),
        )
        for page, gray in zip(pages, rendered_grays)
    ]

    aligned_black_pages = align_pages([page.black for page in analysis_pages])
    cluster_indices = cluster_layouts(aligned_black_pages, layout_threshold=layout_threshold)

    aligned_by_cluster: list[list[np.ndarray]] = []
    for indices in cluster_indices:
        aligned_by_cluster.append([aligned_black_pages[idx] for idx in indices])

    section_sets = detect_layout_section_sets(
        aligned_by_cluster,
        black_threshold=black_threshold,
        recurrence=recurrence,
        stable_threshold=stable_threshold,
        max_band_ratio=max_band_ratio,
        min_section_height=min_section_height,
        section_padding=section_padding,
    )

    page_layouts: list[PageLayout] = []
    clusters: list[LayoutCluster] = []
    has_multi_page_layout = any(len(indices) >= 2 for indices in cluster_indices)

    for layout_idx, (indices, sections) in enumerate(zip(cluster_indices, section_sets), start=1):
        cluster_pages = [analysis_pages[idx].rendered for idx in indices]
        is_outlier = len(pages) >= min_pages_for_outliers and has_multi_page_layout and len(indices) == 1
        representative_page = cluster_pages[0].page_number
        scaled_header = _scale_box(
            sections.header_box,
            analysis_shape=target_shape,
            image_size=cluster_pages[0].image.size,
        )
        scaled_main = _scale_box(
            sections.main_box,
            analysis_shape=target_shape,
            image_size=cluster_pages[0].image.size,
        )
        scaled_footer = _scale_box(
            sections.footer_box,
            analysis_shape=target_shape,
            image_size=cluster_pages[0].image.size,
        )
        assert scaled_main is not None
        clusters.append(
            LayoutCluster(
                layout_id=layout_idx,
                page_numbers=[page.page_number for page in cluster_pages],
                header_box=scaled_header,
                main_box=scaled_main,
                footer_box=scaled_footer,
                is_outlier=is_outlier,
                representative_page=representative_page,
            )
        )

        for page in cluster_pages:
            header_box = _scale_box(sections.header_box, analysis_shape=target_shape, image_size=page.image.size)
            main_box = _scale_box(sections.main_box, analysis_shape=target_shape, image_size=page.image.size)
            footer_box = _scale_box(sections.footer_box, analysis_shape=target_shape, image_size=page.image.size)
            assert main_box is not None
            layout = PageLayout(
                page_number=page.page_number,
                layout_id=layout_idx,
                header_box=header_box,
                main_box=main_box,
                footer_box=footer_box,
                is_outlier=is_outlier,
            )
            page_layouts.append(layout)

    page_layouts.sort(key=lambda page: page.page_number)
    return LayoutAnalysisResult(
        pages=page_layouts,
        layouts=clusters,
    )
