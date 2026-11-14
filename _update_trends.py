#!/usr/bin/env python3
"""
Utility: fetch and cache trending topics.

Run this whenever you have internet to refresh the local trends cache.
The Gradio UI and CLI can then suggest topics even when offline.

    python _update_trends.py
"""
import sys
import os
from pathlib import Path

# Project root is this file's parent — add it to path so imports work
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

from trends.fetcher import update_trends


def main():
    data = update_trends()
    all_trends = data.get("all", [])
    print(f"\nCached {len(all_trends)} trending topics.\n")
    for i, t in enumerate(all_trends[:15], 1):
        traffic = t.get("traffic", "")
        suffix = f"  ({traffic})" if traffic else ""
        print(f"  {i:2d}. {t['title']}{suffix}")
    print()


if __name__ == "__main__":
    main()
