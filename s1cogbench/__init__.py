"""One entry point for the 24 S1CogBench tasks in 27 evaluated modes.

The package dispatches each mode to the original task runner and scorer that
produced the accepted results. Selected action rules and their sampling conventions are documented
separately from source prompts, histories, inputs and scoring targets.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = '0.2.0rc1'
