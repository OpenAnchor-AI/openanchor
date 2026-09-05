"""Allow ``python -m anchor`` to invoke the same CLI as the ``anchor`` script."""
from anchor.cli import main
import sys

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
