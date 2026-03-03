#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def plot_matrix_heatmap(
    table,
    title: str,
    out_path: str | Path,
    y_label: str = "",
    x_label: str = "",
    cmap_name: str = "YlGnBu",
    decimals: int = 3,
    dpi: int = 160,
    col_order: Optional[list] = None,
) -> None:
    import matplotlib.pyplot as plt

    table = table.sort_index()
    if col_order is not None:
        table = table.reindex(col_order, axis=1)
    else:
        table = table.reindex(sorted(table.columns), axis=1)
    values = table.to_numpy(dtype=float)

    rows, cols = values.shape
    w = max(8.0, 1.5 * cols)
    h = max(4.5, 0.6 * rows)
    fig, ax = plt.subplots(figsize=(w, h))

    finite = values[np.isfinite(values)]
    if finite.size > 0:
        vmin = float(np.nanmin(finite))
        vmax = float(np.nanmax(finite))
        if vmax <= vmin:
            vmax = vmin + 1e-8
    else:
        vmin, vmax = 0.0, 1.0

    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color="#D9D9D9")
    masked = np.ma.masked_invalid(values)
    im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(np.arange(cols))
    ax.set_xticklabels([str(c) for c in table.columns], rotation=30, ha="right")
    ax.set_yticks(np.arange(rows))
    ax.set_yticklabels([str(r) for r in table.index])

    thresh = vmin + 0.6 * (vmax - vmin)
    fmt = "{:." + str(decimals) + "f}"
    for i in range(rows):
        for j in range(cols):
            v = values[i, j]
            if np.isfinite(v):
                txt = fmt.format(v)
                color = "white" if v >= thresh else "black"
            else:
                txt = "NA"
                color = "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("value")
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()

