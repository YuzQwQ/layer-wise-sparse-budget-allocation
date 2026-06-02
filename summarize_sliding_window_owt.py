"""Summarize R25 sliding-window / Streaming-style diagnostic on OpenWebText."""

from __future__ import annotations

import sys

import summarize_sliding_window_wikitext as base


if __name__ == "__main__":
    base.LOCAL_RESULTS_DIR = base.Path("results") / "round25-phase25-sliding-window-openwebtext-diagnostic"
    base.FORMAL_RUN_PREFIX = "r25_sliding_owt"
    base.DATASET_SHORT_NAME = "OpenWebText"
    if len(sys.argv) == 1:
        sys.argv.extend(
            [
                "--summary-prefix",
                "round25_sliding_window_openwebtext",
                "--summary-title",
                "R25 Sliding-window / Streaming-style Diagnostic on OpenWebText",
            ]
        )
    base.main()
