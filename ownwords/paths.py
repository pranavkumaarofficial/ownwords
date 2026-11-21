"""Where the corpus and profiles live.

Outside the source tree on purpose. The corpus is someone's private writing and
the profiles are derived from it; keeping them in a per-user data directory
means a `git add .` in a checkout of this repo cannot pick them up, whatever
.gitignore says.

    OWNWORDS_HOME          if set, used as is
    %APPDATA%\\ownwords     Windows
    $XDG_DATA_HOME/ownwords, else ~/.local/share/ownwords
"""
from __future__ import annotations

import os
from pathlib import Path


def data_dir() -> Path:
    env = os.environ.get("OWNWORDS_HOME")
    if env:
        return Path(env).expanduser()
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA") or Path.home())
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return base / "ownwords"


def corpus_db() -> Path:
    return data_dir() / "corpus.db"


def profile_path(register: str, ext: str = "json") -> Path:
    return data_dir() / "profiles" / ("%s.%s" % (register, ext))
