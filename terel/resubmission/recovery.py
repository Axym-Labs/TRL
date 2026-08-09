"""Compatibility aliases for the renamed canonical validation entrypoint."""

from .canonical_validation import load_validation_plan as load_recovery_plan
from .canonical_validation import main, resolve_candidate, run_candidate

__all__ = ["load_recovery_plan", "main", "resolve_candidate", "run_candidate"]


if __name__ == "__main__":
    main()
