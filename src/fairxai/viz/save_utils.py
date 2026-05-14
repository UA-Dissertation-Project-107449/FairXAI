"""Small plotting save/layout helpers."""

from __future__ import annotations

from pathlib import Path


def save_figure(fig, output_file, dpi: int = 300):
    """Save with dissertation-friendly bounding box settings."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.25)
    return output_file


def heatmap_size(row_labels, n_cols: int, min_width: float = 10, min_height: float = 5):
    """Landscape-first heatmap sizing that leaves room for labels."""
    labels = [str(label) for label in row_labels]
    max_label_len = max([len(label) for label in labels] or [0])
    width = max(min_width, n_cols * 2.4 + max_label_len * 0.10 + 2.0)
    height = max(min_height, len(labels) * 0.45 + 2.2)
    return width, height
