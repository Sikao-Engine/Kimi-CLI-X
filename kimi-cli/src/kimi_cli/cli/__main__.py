from __future__ import annotations

import sys
from collections.abc import Sequence

from kimi_cli.cli import cli


def main(argv: Sequence[str] | None = None) -> int | str | None:
    from kimi_cli.utils.environment import GitBashNotFoundError
    from kimi_cli.utils.proxy import normalize_proxy_env

    normalize_proxy_env()

    try:
        if argv is None:
            return cli()
        return cli(args=list(argv))
    except SystemExit as exc:
        return exc.code
    except GitBashNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1



if __name__ == "__main__":
    raise SystemExit(main())
