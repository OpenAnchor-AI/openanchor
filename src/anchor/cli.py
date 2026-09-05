"""Command-line entry point for Anchor."""
from __future__ import annotations

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="anchor")
    parser.add_argument("--version", action="store_true", help="show the Anchor version")
    sub = parser.add_subparsers(dest="command")

    serve = sub.add_parser("serve", help="run the API gateway")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8088)
    serve.add_argument("--reload", action="store_true")

    sub.add_parser("config", help="print and validate worker configuration")
    sub.add_parser("smoke", help="run the end-to-end smoke test")

    cron = sub.add_parser("cron", help="run cron management commands")
    cron.add_argument("args", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.version:
        from anchor import __version__
        print(__version__)
        return 0
    if args.command == "serve":
        import uvicorn
        uvicorn.run("anchor.server:app", host=args.host, port=args.port, reload=args.reload)
        return 0
    if args.command == "config":
        from anchor.config import verify_runtime
        result = verify_runtime()
        for key, value in result.items():
            print(f"{key}: {value}")
        return 0 if result.get("ok", True) else 1
    if args.command == "smoke":
        from anchor.smoke import main as smoke_main
        return smoke_main()
    if args.command == "cron":
        from anchor.cron import main as cron_main
        old_argv = sys.argv
        try:
            sys.argv = ["anchor cron", *args.args]
            return cron_main()
        finally:
            sys.argv = old_argv

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
