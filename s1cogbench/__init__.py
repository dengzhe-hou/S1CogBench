"""One entry point for the 24 S1CogBench tasks in 27 evaluated modes.

The package dispatches each mode to the original task runner and scorer that
produced the accepted results. It adds no prompt, history, sample, seed or
scoring logic of its own.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = '0.1.0'
