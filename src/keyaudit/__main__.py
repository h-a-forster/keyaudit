"""Allow ``python -m keyaudit``."""

from __future__ import annotations

import sys

from keyaudit.cli import main

sys.exit(main())
