"""Ingest adapters. Each one yields Samples from some place your writing exists.

Adding a register means adding one of these, not touching the pipeline.
"""
from .claude_export import from_claude_export
from .textfiles import from_textfiles, from_markdown_dir
from .mbox import from_mbox

__all__ = [
    "from_claude_export", "from_textfiles", "from_markdown_dir", "from_mbox",
]
