"""5-regex query classifier (Day 6)."""
import re
from typing import Literal

QueryType = Literal["code", "debug", "cn", "en", "math", "chat",
                    "creative", "reasoning", "vision", "agent", "summarize", "translate"]

PATTERNS = [
    ("vision",    re.compile(r"\b(image|图片|照片|photo|screenshot|picture|jpeg|jpg|png|"
                             r"describe (the|this|an?) (image|photo|picture)|"
                             r"what (is|are) in|what.s in|看看)\b", re.I)),
    ("cn",        re.compile(r"[\u4e00-\u9fff]")),
    ("debug",     re.compile(r"(TypeError|ValueError|KeyError|AttributeError|RuntimeError|"
                             r"ImportError|SyntaxError|\bexception\b|\btraceback\b|"
                             r"\bstack trace\b|\brace condition\b|\bmemory leak\b|"
                             r"\bwhy (does|isn't|did|is)\b|\bbug\b|\bdebug\b|\bfix the\b)", re.I)),
    ("agent",     re.compile(r"\b(use (the )?tools?|call (the )?tool|create file|read file|"
                             r"write file|run the command|edit the file|open the file)\b", re.I)),
    ("math",      re.compile(r"(\b(solve|equation|integral|derivative|matrix|prove|theorem|"
                             r"sqrt|sin|cos|tan|log|limit)\b|\$\$.+\$\$)")),
    ("code",      re.compile(r"(\bdef \b|\bclass \b|\bfunction \b|\bimport \b|\breturn \b|"
                             r"\bimplement \b|\brefactor \b|\bpython\b|\blinked list\b|"
                             r"=>|`[^`]+`|\.py\b|\.js\b|\.ts\b|pyproject\.toml)")),
    ("reasoning", re.compile(r"\b(analyze|compare|evaluate|reason|argue|implication|"
                             r"therefore|conclusion)\b", re.I)),
]


def classify(query: str) -> str:
    """Return one of 12 query types. Regex rules in priority order. Default 'en'."""
    for label, pat in PATTERNS:
        if pat.search(query):
            return label
    return "en"


def classify_to_tier(query_type: str) -> str:
    """Single-product routing: always the auto lane.

    query_type is still used by hard-rules / head priors; it no longer selects
    a basic/premium/ultra product SKU.
    """
    return "auto"
