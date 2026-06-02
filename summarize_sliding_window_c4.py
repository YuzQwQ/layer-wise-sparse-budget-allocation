"""Summarize R25 sliding-window / Streaming-style diagnostic on C4."""

from __future__ import annotations

import sys

import summarize_sliding_window_wikitext as base


if __name__ == "__main__":
    base.LOCAL_RESULTS_DIR = base.Path("results") / "round25-phase25-sliding-window-c4-diagnostic"
    base.FORMAL_RUN_PREFIX = "r25_sliding_c4"
    base.DATASET_SHORT_NAME = "C4"
    if len(sys.argv) == 1:
        sys.argv.extend(
            [
                "--summary-prefix",
                "round25_sliding_window_c4",
                "--summary-title",
                "R25 Sliding-window / Streaming-style Diagnostic on C4",
            ]
        )
    base.main()
