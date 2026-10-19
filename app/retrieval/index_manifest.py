from dataclasses import asdict, dataclass

from app.ingestion.identity import identity_hash


@dataclass(frozen=True)
class IndexManifest:
    schema_version: str
    embedding_fingerprint: str
    chunker_fingerprint: str
    extraction_fingerprint: str
    vector_dimension: int

    def to_dict(self):
        return asdict(self)


def require_compatible(actual: IndexManifest, requested: IndexManifest) -> None:
    if (
        actual.embedding_fingerprint != requested.embedding_fingerprint
        or actual.vector_dimension != requested.vector_dimension
    ):
        raise ValueError("Index embedding fingerprint mismatch; build a new index generation")


def embedding_fingerprint(settings) -> str:
    return identity_hash(
        [
            settings.embedding_provider,
            settings.embedding_model,
            settings.embedding_revision,
            str(settings.embedding_dimension),
            settings.http_embedding_fingerprint,
            "l2",
            "title-section-body-v1",
        ]
    )
