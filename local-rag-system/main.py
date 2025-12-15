#!/usr/bin/env python3
"""
Local RAG System - Main Entry Point
====================================

A fully local Retrieval-Augmented Generation system.

Usage:
    python main.py                    # Interactive mode
    python main.py --help             # Show help
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.cli import main

if __name__ == '__main__':
    main()

