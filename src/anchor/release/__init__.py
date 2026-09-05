"""Release namespace: gating + circuit-breaker primitives used by the ship
gates (run_phase2 / run_phase3) and the admin ops module.

This package holds the "release engineering" building blocks; it is
intentionally separate from `anchor/` core routing so that the gate logic
can evolve without touching the request hot path.
"""
