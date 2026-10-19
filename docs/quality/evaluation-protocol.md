# Evaluation Protocol

## Dataset Governance

Use only synthetic or authorized learning material. Each case carries a stable evidence-group ID, expected action, language, evidence spans, required claims, and reviewer provenance. Split by evidence group before paraphrases are authored so near-duplicate questions cannot cross dev/test. Unreviewed labels may validate code paths only and must produce `not_evaluated`, never a passing quality score.

## Retrieval

Compare BM25-only, dense-only, hybrid, hybrid+lexical, and hybrid+cross-encoder on the same corpus revision, chunker configuration, query cases, and evaluator version. Resolve gold evidence by stable block/page span and quote hash; filename or document membership alone is not relevance. Report Recall@5/10, Hit@5, MRR, nDCG@10, denominators, language/extraction slices, failures, and p50/p95 latency. Include duplicate evidence only once per ranking metric.

## Answer Quality

Keep safety cases separate. Report answerable coverage and correctness, false refusal, unanswerable refusal precision/recall, citation-ID validity, human citation correctness, claim faithfulness, disclaimer, format, and support-verifier false accepts/rejects separately. Unknown verifier outcomes remain in the denominator. Provider failures are failures, not refusals. Audit each factual claim including claims without citations.

## OCR And Chunking

Measure character/word error rate against reviewed text references and separately inspect numbers, units, negation, tables, and Vietnamese diacritics. Confidence is only a provider score. Report chunk token p50/p95/max, empty/tiny/duplicate counts, extraction provenance, truncation, and source-span validity. Rebuild indexes when extraction, chunker, or embedding fingerprint changes.

## Tuning And Held-Out Test

Tune chunk size/overlap, fusion, reranker candidate count, final k, and relevance thresholds on dev in staged experiments. Record config hashes, model revisions, corpus/index hashes, hardware, latency, and rationale. Freeze the release candidate before one held-out test run. If it fails, record FAIL; improvements require dev iteration and a newly collected held-out set.

## Interpretation

Targets in the project plan are acceptance objectives, not measured results or guarantees. A small synthetic set cannot establish clinical accuracy. Results apply only to the documented corpus, versions, profile, and tested hardware.
