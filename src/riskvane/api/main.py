"""Convenience runner: python -m riskvane.api.main"""

from __future__ import annotations

import uvicorn

from riskvane.api.app import create_app


def main() -> None:
    uvicorn.run(create_app(), host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
