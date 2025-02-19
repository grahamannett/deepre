import tomllib
from functools import cache
from pathlib import Path
from typing import Any


@cache
def read_toml_file(toml_file: str | Path) -> dict[str, Any]:
    """
    Load and cache TOML file contents.

    Args:
        toml_file (str | Path): Path to the TOML file

    Returns:
        dict[str, Any]: Parsed TOML content
    """
    with open(toml_file, "rb") as f:
        data = tomllib.load(f)
    return data
