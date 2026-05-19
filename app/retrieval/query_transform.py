from __future__ import annotations


class RuleBasedQueryTransformer:
    """Rule-based subquery generator reserved for Phase 5."""

    def expand(self, query: str) -> list[str]:
        compact = query.strip()
        if not compact:
            return []
        return [compact]
