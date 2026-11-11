#!/usr/bin/env python3
"""
Utility: fetch and cache trending topics.

Run this whenever you have internet to refresh the local trends cache.
The Gradio UI and CLI can then suggest topics even when offline.

    python server/_update_trends.py
"""
import sys
from pathlib import Path

_SERVER = Path(__file__).resolve().parent
if str(_SERVER) not in sys.path:
    sys.path.insert(0, str(_SERVER))

import os
os.chdir(_SERVER.parent)

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
