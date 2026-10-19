from __future__ import annotations

import re


class RuleBasedQueryTransformer:
    """Deterministic query variants for local multi-query and corrective RAG."""

    def expand(self, query: str) -> list[str]:
        compact = query.strip()
        if not compact:
            return []
        comparison = re.sub(r"^(?:so sánh|so sanh|compare)\s+", "", compact, flags=re.IGNORECASE)
        if comparison == compact:
            return [compact]
        parts = re.split(r"\s+(?:và|va|với|versus|vs\.?|and)\s+", comparison, maxsplit=1, flags=re.IGNORECASE)
        return list(dict.fromkeys([compact, *[p for p in parts if p.strip()]]))[:3]

    def rewrite(self, query: str) -> str:
        compact = " ".join(query.strip().split())
        if not compact:
            return ""
        rewritten = re.sub(
            r"^(hãy\s+)?(cho\s+tôi\s+biết|giải\s+thích|thông\s+tin\s+về)\s+",
            "",
            compact,
            flags=re.IGNORECASE,
        ).strip(" ?.!")
        if rewritten and rewritten != compact.strip(" ?.!"):
            return rewritten
        return f"{compact.strip(' ?.!')} y khoa"
