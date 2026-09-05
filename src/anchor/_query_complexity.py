"""Heuristic query complexity scorer for next_tier() routing decisions.

Returns float in [0.0, 1.0]:
    0.0-0.3  easy chat / very short
    0.3-0.6  medium reasoning / standard coding
    0.6-1.0  hard design / multi-step / architect keywords
"""

_HARD_KW = (
    "design", "architect", "distributed", "consensus", "raft", "crdt",
    "consistent hash", "load balance", "scalable", "fault tolerant",
    "设计", "架构", "推导", "证明", "复杂", "分布式",
)

_MEDIUM_KW = (
    "analyze", "compare", "evaluate", "explain", "implement",
    "实现", "分析", "比较", "summarize",
)

_EASY_KW = ("hi", "hello", "thanks", "what time", "where is", "how are you")


def query_complexity(query: str) -> float:
    ql = (query or "").lower()
    if len(ql) < 20 and any(k in ql for k in _EASY_KW):
        return 0.1
    if any(k in ql for k in _HARD_KW):
        return 0.9
    if any(k in ql for k in _MEDIUM_KW):
        return 0.5
    if len(ql) < 50:
        return 0.2
    return 0.4
