"""Central logging configuration.

The project logs through :mod:`logging` rather than ``print`` so that library
code stays silent when embedded (for example behind the future MCP server) and
only the CLI decides where messages go.
"""

from __future__ import annotations

import logging
import sys

_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def configure_logging(verbose: bool = False) -> None:
    """Install a single stderr log handler on the root logger.

    Args:
        verbose: If true, log at ``DEBUG`` instead of ``INFO``.
    """
    level = logging.DEBUG if verbose else logging.INFO
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(handler)
    root.setLevel(level)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
