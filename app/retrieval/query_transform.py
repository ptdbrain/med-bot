from __future__ import annotations

import re


class RuleBasedQueryTransformer:
    """Deterministic query variants for local multi-query and corrective RAG."""

    def expand(self, query: str) -> list[str]:
        compact = query.strip()
        if not compact:
            return []
        variants = [
            compact,
            f"giải thích y khoa {compact}",
            f"tài liệu hướng dẫn về {compact}",
        ]
        return list(dict.fromkeys(variants))

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
